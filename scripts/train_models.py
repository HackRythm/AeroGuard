import os
# Fix OpenMP duplicate linking issue in Windows python environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Feature listing
NODE_FEATURES = [
    'latitude', 'longitude', 'altitude', 'groundspeed', 'heading', 'vertical_rate',
    'acceleration', 'turn_rate', 'vertical_acceleration', 'climb_rate_change',
    'estimated_doppler_hz', 'estimated_rss_dbm', 'estimated_snr_db'
]

EDGE_FEATURES = [
    'distance', 'relative_altitude', 'relative_velocity', 'relative_heading'
]

PROXIMITY_THRESHOLD_M = 100000.0  # 100 km for edge connections

# ==============================================================================
# DYNAMIC GRAPH LOADER CONSTRUCTION
# ==============================================================================
def build_graph_snapshots(df, scaler_node, scaler_edge, is_train=True):
    """
    Groups the dataframe by timestamp, standardizes features, and constructs
    PyTorch Geometric Data snapshots G(t).
    """
    print(f"Building graph snapshots (is_train={is_train})...")
    snapshots = []
    
    # Pre-scale features globally for speed
    scaled_nodes = scaler_node.transform(df[NODE_FEATURES])
    df_scaled = df.copy()
    df_scaled[NODE_FEATURES] = scaled_nodes
    
    grouped = df_scaled.groupby('timestamp')
    
    for ts, group in grouped:
        n = len(group)
        if n == 0:
            continue
            
        group = group.reset_index(drop=True)
        node_feats = group[NODE_FEATURES].values
        labels = group['label'].values
        
        edge_index = []
        edge_attr = []
        
        # Build proximity-based edges using projected coordinates x, y
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
                        # Edge features: distance, relative_altitude, relative_velocity, relative_heading
                        alt_diff = alts[i] - alts[j]
                        vel_diff = vels[i] - vels[j]
                        head_diff = (heads[i] - heads[j] + 180.0) % 360.0 - 180.0
                        
                        raw_edge = [h_dist, alt_diff, vel_diff, head_diff]
                        edge_index.append([i, j])
                        edge_attr.append(raw_edge)
                        
        # Prepare edge tensors
        if len(edge_index) > 0:
            edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            # Standardize edge features
            edge_attr_scaled = scaler_edge.transform(edge_attr)
            edge_attr_t = torch.tensor(edge_attr_scaled, dtype=torch.float)
        else:
            edge_index_t = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_t = torch.zeros((0, len(EDGE_FEATURES)), dtype=torch.float)
            
        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            y=torch.tensor(labels, dtype=torch.long),
            # Metadata to map back to original indices
            original_indices=torch.tensor(group.index.values, dtype=torch.long)
        )
        snapshots.append(data)
        
    return snapshots

# ==============================================================================
# GNN MODEL DEFINITION (GCN)
# ==============================================================================
class AeroGuardGNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.classifier = nn.Linear(16, 2)  # Binary classification logits
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        logits = self.classifier(x)
        return logits

# ==============================================================================
# MAIN TRAINING & EVALUATION PIPELINE
# ==============================================================================
def train_and_evaluate():
    # 1. Load data
    train_path = "data/aero_guard_train.csv"
    test_path = "data/aero_guard_test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train or test CSV files not found. Generate dataset first.")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 2. Fit Scalers
    print("Fitting Scalers...")
    scaler_node = StandardScaler()
    scaler_node.fit(df_train[NODE_FEATURES])
    
    # Calculate temporary edge attributes to fit the edge scaler
    print("Calculating training edge feature parameters for Scaling...")
    edge_data_sample = []
    # Loop over sample timestamps to get realistic relative ranges
    grouped = df_train.sample(n=min(5000, len(df_train)), random_state=42).groupby('timestamp')
    for ts, group in grouped:
        n = len(group)
        if n > 1:
            x_c, y_c = group['x'].values, group['y'].values
            alts, vels, heads = group['altitude'].values, group['groundspeed'].values, group['heading'].values
            for i in range(n):
                for j in range(i+1, n):
                    h_d = np.sqrt((x_c[i] - x_c[j])**2 + (y_c[i] - y_c[j])**2)
                    if h_d <= PROXIMITY_THRESHOLD_M:
                        edge_data_sample.append([h_d, alts[i]-alts[j], vels[i]-vels[j], (heads[i]-heads[j]+180)%360-180])
                        
    if len(edge_data_sample) == 0:
        # Fallback default range
        edge_data_sample = [[50000.0, 5000.0, 100.0, 90.0]]
        
    scaler_edge = StandardScaler()
    scaler_edge.fit(edge_data_sample)
    
    # 3. Train Tabular Baseline (Random Forest)
    print("\n--- Training Tabular Baseline (Random Forest) ---")
    X_train = scaler_node.transform(df_train[NODE_FEATURES])
    y_train = df_train['label'].values
    X_test = scaler_node.transform(df_test[NODE_FEATURES])
    y_test = df_test['label'].values
    
    # Simple Random Forest Classifier (fast, lightweight, standard parameters)
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Predict probabilities and classes
    baseline_probs = clf.predict_proba(X_test)[:, 1]
    baseline_preds = clf.predict(X_test)
    
    # Save Random Forest baseline model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/baseline.joblib")
    joblib.dump(scaler_node, "models/scaler_node.joblib")
    joblib.dump(scaler_edge, "models/scaler_edge.joblib")
    print("Baseline model and scalers saved successfully in models/")
    
    # 4. Build Graphs for GNN
    train_graphs = build_graph_snapshots(df_train, scaler_node, scaler_edge, is_train=True)
    test_graphs = build_graph_snapshots(df_test, scaler_node, scaler_edge, is_train=False)
    
    # Loader
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # 5. GNN Model Training
    print("\n--- Training Dynamic GNN Prototype ---")
    num_features = len(NODE_FEATURES)
    model = AeroGuardGNN(num_features)
    
    # Set class weights to handle imbalance (Normal: 92.5%, Anomaly: 7.5%)
    # Ratio: Normal / Anomaly ~ 12.3
    class_weights = torch.tensor([1.0, 10.0]) # Emphasize anomalous nodes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # Train GNN
    model.train()
    epochs = 8 # Keep epochs low for fast demonstration prototype
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
        print(f"Epoch {epoch+1:02d} / {epochs:02d} | Loss: {total_loss / len(train_graphs):.4f}")
        
    # Save GNN weights
    torch.save(model.state_dict(), "models/gnn.pt")
    print("GNN model saved in models/gnn.pt")
    
    # 6. GNN Inference
    print("\nRunning GNN test inference...")
    model.eval()
    gnn_scores = []
    gnn_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1].numpy()
            preds = torch.argmax(out, dim=1).numpy()
            gnn_scores.extend(probs)
            gnn_preds.extend(preds)
            
    gnn_scores = np.array(gnn_scores)
    gnn_preds = np.array(gnn_preds)
    
    # 7. Model Evaluation
    print("\n==================== EVALUATION METRICS ====================")
    
    # Baseline Metrics
    base_prec = precision_score(y_test, baseline_preds, zero_division=0)
    base_rec = recall_score(y_test, baseline_preds, zero_division=0)
    base_f1 = f1_score(y_test, baseline_preds, zero_division=0)
    base_auc = roc_auc_score(y_test, baseline_probs)
    base_cm = confusion_matrix(y_test, baseline_preds)
    
    # GNN Metrics
    gnn_prec = precision_score(y_test, gnn_preds, zero_division=0)
    gnn_rec = recall_score(y_test, gnn_preds, zero_division=0)
    gnn_f1 = f1_score(y_test, gnn_preds, zero_division=0)
    gnn_auc = roc_auc_score(y_test, gnn_scores)
    gnn_cm = confusion_matrix(y_test, gnn_preds)
    
    print("\nComparison Table:")
    print("-" * 55)
    print(f"{'Model':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
    print("-" * 55)
    print(f"{'Baseline (RF)':<15} | {base_prec:<10.4f} | {base_rec:<10.4f} | {base_f1:<10.4f} | {base_auc:<10.4f}")
    print(f"{'AeroGuard GNN':<15} | {gnn_prec:<10.4f} | {gnn_rec:<10.4f} | {gnn_f1:<10.4f} | {gnn_auc:<10.4f}")
    print("-" * 55)
    
    print("\nConfusion Matrix - Baseline (RF):")
    print(base_cm)
    print("\nConfusion Matrix - AeroGuard GNN:")
    print(gnn_cm)
    
    # 8. Save test predictions CSV
    # Ensure graph inference outputs map 1-to-1 with the test dataframe rows
    # Note: when we built test_graphs, we grouped test dataset chronologically.
    # We must make sure that the length matches.
    print(f"\nOriginal test set length: {len(df_test)}, GNN predictions length: {len(gnn_scores)}")
    
    # Since our graph building iterates over groups chronologically, and resets indexes,
    # let's verify if the order was preserved.
    # Group by timestamp and sorted timestamps ensures the output maps exactly to the sorted df_test.
    # Let's double check if we sorted df_test chronologically before.
    df_test_sorted = df_test.sort_values(by=['timestamp', 'icao24']).reset_index(drop=True)
    
    # Let's rebuild the dynamic predictions matching this exact sorted order
    # To be extremely safe, we will write a direct GNN batch inference loop mapping to df_test_sorted.
    # Let's do that:
    model.eval()
    all_scores = np.zeros(len(df_test_sorted))
    all_preds = np.zeros(len(df_test_sorted))
    
    # Re-evaluate GNN on the actual test snapshots sorted chronologically
    test_graphs_sorted = build_graph_snapshots(df_test_sorted, scaler_node, scaler_edge, is_train=False)
    test_loader_sorted = DataLoader(test_graphs_sorted, batch_size=1, shuffle=False)
    
    row_counter = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader_sorted):
            out = model(batch.x, batch.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1].numpy()
            preds = torch.argmax(out, dim=1).numpy()
            
            n_nodes = len(probs)
            all_scores[row_counter : row_counter + n_nodes] = probs
            all_preds[row_counter : row_counter + n_nodes] = preds
            row_counter += n_nodes
            
    # Also evaluate Baseline Random Forest on df_test_sorted
    X_test_sorted = scaler_node.transform(df_test_sorted[NODE_FEATURES])
    baseline_probs_sorted = clf.predict_proba(X_test_sorted)[:, 1]
    baseline_preds_sorted = clf.predict(X_test_sorted)
    
    # Write columns
    df_test_sorted['baseline_score'] = baseline_probs_sorted
    df_test_sorted['baseline_pred'] = baseline_preds_sorted
    df_test_sorted['gnn_score'] = all_scores
    df_test_sorted['gnn_pred'] = all_preds
    df_test_sorted['anomaly_score'] = all_scores  # Standard anomaly score is the GNN score
    
    output_predictions_path = "data/aero_guard_test_predictions.csv"
    df_test_sorted.to_csv(output_predictions_path, index=False)
    print(f"\nPredictions output saved to {output_predictions_path}")
    
    # Print a few anomaly detections for verification
    anom_rows = df_test_sorted[df_test_sorted['label'] == 1].head(5)
    print("\nSample Anomalous Inferences:")
    print(anom_rows[['timestamp', 'icao24', 'callsign', 'anomaly_type', 'baseline_score', 'anomaly_score']])

if __name__ == "__main__":
    train_and_evaluate()
