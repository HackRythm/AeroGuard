import os
# Fix OpenMP duplicate linking issue in Windows python environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
import joblib

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Baseline Features (includes absolute lat/lon coordinates)
BASELINE_NODE_FEATURES = [
    'latitude', 'longitude', 'altitude', 'groundspeed', 'heading', 'vertical_rate',
    'acceleration', 'turn_rate', 'vertical_acceleration', 'climb_rate_change',
    'estimated_doppler_hz', 'estimated_rss_dbm', 'estimated_snr_db'
]

# Coordinate-Free / Behavior-Oriented Features (removes lat/lon, adds sin/cos heading)
COORD_FREE_NODE_FEATURES = [
    'altitude', 'groundspeed', 'heading_sin', 'heading_cos', 'vertical_rate',
    'acceleration', 'turn_rate', 'vertical_acceleration', 'climb_rate_change',
    'estimated_doppler_hz', 'estimated_rss_dbm', 'estimated_snr_db'
]

EDGE_FEATURES = [
    'distance', 'relative_altitude', 'relative_velocity', 'relative_heading'
]

PROXIMITY_THRESHOLD_M = 100000.0  # 100 km for edge connections

# ==============================================================================
# FEATURE PREPARATION & EDGE VERIFICATION
# ==============================================================================
def add_coordinate_free_features(df):
    """
    Adds sin/cos heading transformations to construct a coordinate-free representation.
    """
    df = df.copy()
    heading_rad = np.radians(df['heading'])
    df['heading_sin'] = np.sin(heading_rad)
    df['heading_cos'] = np.cos(heading_rad)
    return df

def verify_edge_attributes(df):
    """
    Empirically verifies existing 4D edge feature values, shapes, and checks for NaN/inf.
    """
    print("\n==================== VERIFYING EXISTING EDGE ATTRIBUTES ====================")
    grouped = df.groupby('timestamp')
    num_edges_total = 0
    nan_count = 0
    inf_count = 0
    edge_attr_samples = []

    for ts, group in grouped:
        n = len(group)
        if n > 1:
            x_c, y_c = group['x'].values, group['y'].values
            alts, vels, heads = group['altitude'].values, group['groundspeed'].values, group['heading'].values
            dx = x_c[:, None] - x_c[None, :]
            dy = y_c[:, None] - y_c[None, :]
            dist_matrix = np.sqrt(dx**2 + dy**2)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    h_dist = dist_matrix[i, j]
                    if h_dist <= PROXIMITY_THRESHOLD_M:
                        num_edges_total += 1
                        alt_diff = alts[i] - alts[j]
                        vel_diff = vels[i] - vels[j]
                        head_diff = (heads[i] - heads[j] + 180.0) % 360.0 - 180.0
                        raw_edge = [h_dist, alt_diff, vel_diff, head_diff]
                        edge_attr_samples.append(raw_edge)

    edge_attr_arr = np.array(edge_attr_samples)
    if len(edge_attr_arr) > 0:
        nan_count = np.isnan(edge_attr_arr).sum()
        inf_count = np.isinf(edge_attr_arr).sum()
        
        print(f"Total Edges Inspected: {num_edges_total}")
        print(f"Edge Attribute Array Shape: {edge_attr_arr.shape}")
        print(f"NaN Count in Edge Features: {nan_count}")
        print(f"Inf Count in Edge Features: {inf_count}")
        print("4D Edge Attributes Statistics:")
        for idx, col in enumerate(EDGE_FEATURES):
            vals = edge_attr_arr[:, idx]
            print(f"  - {col:<20}: Min={vals.min():.2f}, Max={vals.max():.2f}, Mean={vals.mean():.2f}, Std={vals.std():.2f}")
    else:
        print("[WARNING] No edges found under 100 km threshold during verification.")
    
    assert nan_count == 0 and inf_count == 0, "Edge attribute array contains NaN or Inf values!"
    print("Edge attribute verification PASSED cleanly.\n")

# ==============================================================================
# DYNAMIC GRAPH SNAPSHOT LOADER
# ==============================================================================
def build_graph_snapshots(df, node_features_list, scaler_node, scaler_edge, is_train=True):
    """
    Groups dataframe by timestamp, standardizes features, and constructs PyG Data snapshots.
    """
    snapshots = []
    scaled_nodes = scaler_node.transform(df[node_features_list])
    df_scaled = df.copy()
    df_scaled[node_features_list] = scaled_nodes

    grouped = df_scaled.groupby('timestamp')
    
    for ts, group in grouped:
        n = len(group)
        if n == 0:
            continue
            
        group = group.reset_index(drop=True)
        node_feats = group[node_features_list].values
        labels = group['label'].values
        
        edge_index = []
        edge_attr = []
        
        if n > 1:
            x_coords = group['x'].values
            y_coords = group['y'].values
            alts = group['altitude'].values
            vels = group['groundspeed'].values
            heads = group['heading'].values
            
            dx = x_coords[:, None] - x_coords[None, :]
            dy = y_coords[:, None] - y_coords[None, :]
            dist_matrix = np.sqrt(dx**2 + dy**2)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    h_dist = dist_matrix[i, j]
                    if h_dist <= PROXIMITY_THRESHOLD_M:
                        alt_diff = alts[i] - alts[j]
                        vel_diff = vels[i] - vels[j]
                        head_diff = (heads[i] - heads[j] + 180.0) % 360.0 - 180.0
                        
                        raw_edge = [h_dist, alt_diff, vel_diff, head_diff]
                        edge_index.append([i, j])
                        edge_attr.append(raw_edge)
                        
        if len(edge_index) > 0:
            edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_scaled = scaler_edge.transform(edge_attr)
            edge_attr_t = torch.tensor(edge_attr_scaled, dtype=torch.float)
        else:
            edge_index_t = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_t = torch.zeros((0, len(EDGE_FEATURES)), dtype=torch.float)
            
        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            y=torch.tensor(labels, dtype=torch.long)
        )
        snapshots.append(data)
        
    return snapshots

# ==============================================================================
# GNN MODEL DEFINITIONS
# ==============================================================================
class OriginalAeroGuardGCN(nn.Module):
    """
    Original Baseline GCN model using standard GCNConv (ignores edge attributes).
    """
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.classifier = nn.Linear(16, 2)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        logits = self.classifier(x)
        return logits

class EdgeAwareGNN(nn.Module):
    """
    Improved Edge-Aware GNN using GATv2Conv with edge_dim=4.
    Explicitly incorporates edge_attr into message passing.
    """
    def __init__(self, num_features, edge_dim=4):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, 32, heads=2, edge_dim=edge_dim, concat=False)
        self.conv2 = GATv2Conv(32, 16, heads=2, edge_dim=edge_dim, concat=False)
        self.classifier = nn.Linear(16, 2)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        logits = self.classifier(x)
        return logits

# ==============================================================================
# THRESHOLD CALIBRATION HELPER
# ==============================================================================
def calibrate_threshold(y_true, y_probs, model_name="Model"):
    """
    Sweeps thresholds 0.01 to 0.99 on validation set to find threshold maximizing F1 score.
    """
    best_thresh = 0.5
    best_f1 = -1.0
    
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
            
    print(f"Calibration [{model_name}]: Best Validation Threshold = {best_thresh:.2f} (Val F1 = {best_f1:.4f})")
    return float(best_thresh), float(best_f1)

# ==============================================================================
# MAIN TRAINING & EVALUATION PIPELINE
# ==============================================================================
def train_and_evaluate():
    # 1. Load Datasets
    train_path = "data/aero_guard_train.csv"
    test_path = "data/aero_guard_test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train or test CSV files not found. Run dataset generator first.")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Add coordinate-free features
    df_train = add_coordinate_free_features(df_train)
    df_test = add_coordinate_free_features(df_test)
    
    # 2. Audit Existing Edge Attributes
    verify_edge_attributes(df_train)
    
    # 3. Chronological Train/Validation Split for Threshold Calibration
    # 80% train, 20% validation chronologically by timestamp
    unique_timestamps = np.sort(df_train['timestamp'].unique())
    split_idx = int(len(unique_timestamps) * 0.8)
    train_timestamps = set(unique_timestamps[:split_idx])
    val_timestamps = set(unique_timestamps[split_idx:])
    
    df_train_sub = df_train[df_train['timestamp'].isin(train_timestamps)].reset_index(drop=True)
    df_val_sub = df_train[df_train['timestamp'].isin(val_timestamps)].reset_index(drop=True)
    
    print(f"Chronological Train Split: {len(df_train_sub)} rows ({len(train_timestamps)} timestamps)")
    print(f"Chronological Val Split:   {len(df_val_sub)} rows ({len(val_timestamps)} timestamps)")
    print(f"Test Split (Jan 2 OOD):    {len(df_test)} rows")
    
    # 4. Fit Feature Scalers on Training Subset
    scaler_node_baseline = StandardScaler()
    scaler_node_baseline.fit(df_train_sub[BASELINE_NODE_FEATURES])
    
    scaler_node_coordfree = StandardScaler()
    scaler_node_coordfree.fit(df_train_sub[COORD_FREE_NODE_FEATURES])
    
    # Compute edge feature sample to fit edge scaler
    edge_sample = []
    grouped_sub = df_train_sub.sample(n=min(5000, len(df_train_sub)), random_state=SEED).groupby('timestamp')
    for ts, group in grouped_sub:
        n = len(group)
        if n > 1:
            x_c, y_c = group['x'].values, group['y'].values
            alts, vels, heads = group['altitude'].values, group['groundspeed'].values, group['heading'].values
            for i in range(n):
                for j in range(i+1, n):
                    h_d = np.sqrt((x_c[i] - x_c[j])**2 + (y_c[i] - y_c[j])**2)
                    if h_d <= PROXIMITY_THRESHOLD_M:
                        edge_sample.append([h_d, alts[i]-alts[j], vels[i]-vels[j], (heads[i]-heads[j]+180)%360-180])
                        
    if len(edge_sample) == 0:
        edge_sample = [[50000.0, 5000.0, 100.0, 90.0]]
        
    scaler_edge = StandardScaler()
    scaler_edge.fit(edge_sample)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler_node_baseline, "models/scaler_node_baseline.joblib")
    joblib.dump(scaler_node_coordfree, "models/scaler_node_coordfree.joblib")
    joblib.dump(scaler_edge, "models/scaler_edge.joblib")
    
    # 5. Train Random Forest Baseline
    print("\n--- Training Baseline Random Forest ---")
    X_train_rf = scaler_node_baseline.transform(df_train_sub[BASELINE_NODE_FEATURES])
    y_train_rf = df_train_sub['label'].values
    
    X_val_rf = scaler_node_baseline.transform(df_val_sub[BASELINE_NODE_FEATURES])
    y_val_rf = df_val_sub['label'].values
    
    X_test_rf = scaler_node_baseline.transform(df_test[BASELINE_NODE_FEATURES])
    y_test_rf = df_test['label'].values
    
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
    clf_rf.fit(X_train_rf, y_train_rf)
    
    rf_val_probs = clf_rf.predict_proba(X_val_rf)[:, 1]
    rf_test_probs = clf_rf.predict_proba(X_test_rf)[:, 1]
    
    joblib.dump(clf_rf, "models/baseline_rf.joblib")
    print("Saved Random Forest to models/baseline_rf.joblib")
    
    # 6. Build Graphs for Original GCN and Edge-Aware GNN
    print("\nBuilding PyG Graph Snapshots...")
    train_graphs_base = build_graph_snapshots(df_train_sub, BASELINE_NODE_FEATURES, scaler_node_baseline, scaler_edge)
    val_graphs_base = build_graph_snapshots(df_val_sub, BASELINE_NODE_FEATURES, scaler_node_baseline, scaler_edge)
    test_graphs_base = build_graph_snapshots(df_test, BASELINE_NODE_FEATURES, scaler_node_baseline, scaler_edge)
    
    train_graphs_efree = build_graph_snapshots(df_train_sub, COORD_FREE_NODE_FEATURES, scaler_node_coordfree, scaler_edge)
    val_graphs_efree = build_graph_snapshots(df_val_sub, COORD_FREE_NODE_FEATURES, scaler_node_coordfree, scaler_edge)
    test_graphs_efree = build_graph_snapshots(df_test, COORD_FREE_NODE_FEATURES, scaler_node_coordfree, scaler_edge)
    
    loader_train_base = DataLoader(train_graphs_base, batch_size=32, shuffle=True)
    loader_val_base = DataLoader(val_graphs_base, batch_size=32, shuffle=False)
    loader_test_base = DataLoader(test_graphs_base, batch_size=32, shuffle=False)
    
    loader_train_efree = DataLoader(train_graphs_efree, batch_size=32, shuffle=True)
    loader_val_efree = DataLoader(val_graphs_efree, batch_size=32, shuffle=False)
    loader_test_efree = DataLoader(test_graphs_efree, batch_size=32, shuffle=False)
    
    # 7. Train Original GCN Baseline
    print("\n--- Training Original GCN Baseline (Ignoring edge_attr) ---")
    model_gcn_base = OriginalAeroGuardGCN(len(BASELINE_NODE_FEATURES))
    class_weights = torch.tensor([1.0, 10.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt_gcn = optim.Adam(model_gcn_base.parameters(), lr=0.005, weight_decay=1e-4)
    
    model_gcn_base.train()
    epochs = 8
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader_train_base:
            opt_gcn.zero_grad()
            out = model_gcn_base(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            opt_gcn.step()
            total_loss += loss.item() * batch.num_graphs
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss / len(train_graphs_base):.4f}")
        
    torch.save(model_gcn_base.state_dict(), "models/gnn_baseline.pt")
    print("Saved Original GCN to models/gnn_baseline.pt")
    
    # 8. Train Improved Edge-Aware GNN
    print("\n--- Training Improved Edge-Aware GNN (GATv2Conv with edge_dim=4) ---")
    model_gnn_edge = EdgeAwareGNN(len(COORD_FREE_NODE_FEATURES), edge_dim=len(EDGE_FEATURES))
    opt_edge = optim.Adam(model_gnn_edge.parameters(), lr=0.005, weight_decay=1e-4)
    
    model_gnn_edge.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader_train_efree:
            opt_edge.zero_grad()
            out = model_gnn_edge(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            opt_edge.step()
            total_loss += loss.item() * batch.num_graphs
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss / len(train_graphs_efree):.4f}")
        
    torch.save(model_gnn_edge.state_dict(), "models/gnn_edge_aware.pt")
    print("Saved Edge-Aware GNN to models/gnn_edge_aware.pt")
    
    # 9. Validation Inference & Threshold Calibration
    print("\n==================== THRESHOLD CALIBRATION ON VALIDATION SET ====================")
    def predict_gnn(model, loader, is_edge_aware=False):
        model.eval()
        probs_all = []
        with torch.no_grad():
            for batch in loader:
                if is_edge_aware:
                    out = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
                else:
                    out = model(batch.x, batch.edge_index)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                probs_all.extend(probs)
        return np.array(probs_all)
        
    gcn_val_probs = predict_gnn(model_gcn_base, loader_val_base, is_edge_aware=False)
    edge_val_probs = predict_gnn(model_gnn_edge, loader_val_efree, is_edge_aware=True)
    
    t_rf, f1_rf_val = calibrate_threshold(y_val_rf, rf_val_probs, "Random Forest")
    t_gcn, f1_gcn_val = calibrate_threshold(df_val_sub['label'].values, gcn_val_probs, "Original GCN")
    t_edge, f1_edge_val = calibrate_threshold(df_val_sub['label'].values, edge_val_probs, "Edge-Aware GNN")
    
    calibrated_data = {
        "timestamp": datetime.now().isoformat(),
        "random_forest": {
            "threshold": t_rf,
            "validation_f1": f1_rf_val
        },
        "original_gcn": {
            "threshold": t_gcn,
            "validation_f1": f1_gcn_val
        },
        "edge_aware_gnn": {
            "threshold": t_edge,
            "validation_f1": f1_edge_val
        }
    }
    
    with open("models/calibrated_threshold.json", "w") as f:
        json.dump(calibrated_data, f, indent=4)
    print("Calibrated thresholds saved to models/calibrated_threshold.json")
    
    # 10. OOD Test Evaluation (Jan 2 Data)
    print("\n==================== FINAL OOD EVALUATION (JAN 2 TEST SET) ====================")
    gcn_test_probs = predict_gnn(model_gcn_base, loader_test_base, is_edge_aware=False)
    edge_test_probs = predict_gnn(model_gnn_edge, loader_test_efree, is_edge_aware=True)
    
    models_eval = [
        ("Original RF (t=0.50)", y_test_rf, rf_test_probs, 0.50),
        ("Original RF (Calibrated)", y_test_rf, rf_test_probs, t_rf),
        ("Original GCN (t=0.50)", df_test['label'].values, gcn_test_probs, 0.50),
        ("Original GCN (Calibrated)", df_test['label'].values, gcn_test_probs, t_gcn),
        ("Edge-Aware GNN (t=0.50)", df_test['label'].values, edge_test_probs, 0.50),
        ("Edge-Aware GNN (Calibrated)", df_test['label'].values, edge_test_probs, t_edge),
    ]
    
    print("-" * 90)
    print(f"{'Model & Threshold':<30} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'PR-AUC':<9}")
    print("-" * 90)
    
    eval_results = []
    for name, y_true, probs, thresh in models_eval:
        preds = (probs >= thresh).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        cm = confusion_matrix(y_true, preds)
        
        print(f"{name:<30} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f} | {roc_auc:<9.4f} | {pr_auc:<9.4f}")
        eval_results.append({
            "name": name, "prec": prec, "rec": rec, "f1": f1,
            "roc_auc": roc_auc, "pr_auc": pr_auc, "cm": cm, "thresh": thresh
        })
    print("-" * 90)
    
    print("\nDetailed Confusion Matrices:")
    for res in eval_results:
        print(f"\n{res['name']} [Threshold={res['thresh']:.2f}]:")
        print(f"  TN: {res['cm'][0,0]:<6} FP: {res['cm'][0,1]:<6}")
        print(f"  FN: {res['cm'][1,0]:<6} TP: {res['cm'][1,1]:<6}")

    # 11. Save Test Predictions CSV
    df_test_out = df_test.copy()
    df_test_out['rf_prob'] = rf_test_probs
    df_test_out['rf_pred_05'] = (rf_test_probs >= 0.5).astype(int)
    df_test_out['rf_pred_calibrated'] = (rf_test_probs >= t_rf).astype(int)
    
    df_test_out['gnn_baseline_prob'] = gcn_test_probs
    df_test_out['gnn_baseline_pred_05'] = (gcn_test_probs >= 0.5).astype(int)
    df_test_out['gnn_baseline_pred_calibrated'] = (gcn_test_probs >= t_gcn).astype(int)
    
    df_test_out['gnn_edge_aware_prob'] = edge_test_probs
    df_test_out['gnn_edge_aware_pred_05'] = (edge_test_probs >= 0.5).astype(int)
    df_test_out['gnn_edge_aware_pred_calibrated'] = (edge_test_probs >= t_edge).astype(int)
    
    df_test_out['threshold_rf'] = t_rf
    df_test_out['threshold_gnn_baseline'] = t_gcn
    df_test_out['threshold_gnn_edge_aware'] = t_edge
    
    output_pred_path = "data/aero_guard_test_predictions.csv"
    df_test_out.to_csv(output_pred_path, index=False)
    print(f"\nSaved test predictions to {output_pred_path}")

if __name__ == "__main__":
    train_and_evaluate()
