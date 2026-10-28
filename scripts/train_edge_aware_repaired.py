"""
AeroGuard AI — Repaired Edge-Aware GNN Training & Independent Validation
========================================================================
This script provides an audited, clean, and isolated implementation of the
Edge-Aware Graph Neural Network (GATv2Conv with 4D edge attributes) addressing
the root-cause defects identified in the Week 1 evaluation pipeline:

1. Defect Fix: Edge Attribute Double-Scaling
   - Raw kinematic differences (altitude in ft, velocity in kts, heading in deg)
     are scaled once using the dedicated edge StandardScaler fitted on raw edge deltas.
   - Eliminates the double-scaling bug that compressed relative altitude and
     relative velocity differences to ~1e-4.

2. Defect Fix: Prediction-to-Label Alignment
   - Snapshot predictions are strictly aligned with node ground-truth labels and
     original dataframe indices, preventing row-scrambling caused by grouping
     by timestamp on an ICAO-sorted dataframe.

3. Geographic Domain-Shift Mitigation:
   - Uses 12-dimensional coordinate-free node representations (excluding absolute
     latitude and longitude, transforming heading to sin/cos).

4. Isolated Artifacts:
   - Preserves historical baseline models and scripts.
   - Saves repaired model to models/gnn_edge_aware_repaired.pt.
"""

import os
# Fix OpenMP duplicate linking issue in Windows python environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import joblib

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROXIMITY_THRESHOLD_M = 100000.0  # 100 km horizontal proximity threshold

COORD_FREE_NODE_FEATURES = [
    'altitude', 'groundspeed', 'heading_sin', 'heading_cos', 'vertical_rate',
    'acceleration', 'turn_rate', 'vertical_acceleration', 'climb_rate_change',
    'estimated_doppler_hz', 'estimated_rss_dbm', 'estimated_snr_db'
]

EDGE_FEATURES = [
    'distance', 'relative_altitude', 'relative_velocity', 'relative_heading'
]

# ==============================================================================
# DATA TRANSFORMATION HELPERS
# ==============================================================================
def add_coordinate_free_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds trigonometric heading representations."""
    df = df.copy()
    heading_rad = np.radians(df['heading'])
    df['heading_sin'] = np.sin(heading_rad)
    df['heading_cos'] = np.cos(heading_rad)
    return df

def build_repaired_graph_snapshots(df: pd.DataFrame, scaler_node: StandardScaler, scaler_edge: StandardScaler):
    """
    Constructs PyTorch Geometric Data snapshots grouped chronologically by timestamp.

    Guarantees:
    - Node features are scaled with scaler_node.
    - Edge attributes are computed from RAW unscaled physical values (meters, ft, kts, deg)
      and standardized once with scaler_edge.
    - Each snapshot preserves its node labels and row indices for 100% aligned evaluation.
    """
    snapshots = []

    # Scale node features
    scaled_nodes = scaler_node.transform(df[COORD_FREE_NODE_FEATURES])
    df_scaled = df.copy()
    df_scaled[COORD_FREE_NODE_FEATURES] = scaled_nodes

    grouped_scaled = df_scaled.groupby('timestamp')
    grouped_raw = df.groupby('timestamp')

    for (ts, grp_s), (_, grp_r) in zip(grouped_scaled, grouped_raw):
        n = len(grp_s)
        if n == 0:
            continue

        node_feats = grp_s[COORD_FREE_NODE_FEATURES].values
        labels = grp_s['label'].values
        orig_indices = grp_s.index.values

        edge_index = []
        raw_edges = []

        if n > 1:
            x = grp_r['x'].values
            y = grp_r['y'].values
            alts = grp_r['altitude'].values          # Raw feet
            vels = grp_r['groundspeed'].values       # Raw knots
            heads = grp_r['heading'].values          # Raw degrees

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
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

                        edge_index.append([i, j])
                        raw_edges.append([h_dist, alt_diff, vel_diff, head_diff])

        if len(edge_index) > 0:
            edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_scaled = scaler_edge.transform(raw_edges)
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
        data.orig_indices = torch.tensor(orig_indices, dtype=torch.long)
        snapshots.append(data)

    return snapshots

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class RepairedEdgeAwareGNN(nn.Module):
    """
    Edge-Aware Graph Neural Network utilizing GATv2Conv layers with 4D edge attributes.
    Suitable as the Spatial Encoder for Spatio-Temporal GNN (ST-GNN).
    """
    def __init__(self, num_features=12, edge_dim=4, hidden_dim1=32, hidden_dim2=16):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, hidden_dim1, heads=2, edge_dim=edge_dim, concat=False)
        self.conv2 = GATv2Conv(hidden_dim1, hidden_dim2, heads=2, edge_dim=edge_dim, concat=False)
        self.classifier = nn.Linear(hidden_dim2, 2)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        logits = self.classifier(x)
        return logits

    def extract_spatial_embeddings(self, x, edge_index, edge_attr):
        """Extract intermediate spatial representations for ST-GNN downstream integration."""
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        return x

# ==============================================================================
# EVALUATION & CALIBRATION HELPERS
# ==============================================================================
def evaluate_model(model: nn.Module, loader: DataLoader):
    """Performs inference over graph loader and returns aligned ground truths and probabilities."""
    model.eval()
    y_true_list = []
    probs_list = []
    indices_list = []

    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            y_true_list.extend(batch.y.cpu().numpy())
            probs_list.extend(probs)
            if hasattr(batch, 'orig_indices'):
                indices_list.extend(batch.orig_indices.cpu().numpy())

    return np.array(y_true_list), np.array(probs_list), np.array(indices_list)

def calibrate_threshold(y_true: np.ndarray, y_probs: np.ndarray):
    """Sweeps thresholds on validation set to maximize F1-score."""
    best_t = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return float(best_t), float(best_f1)

# ==============================================================================
# MAIN TRAINING & VALIDATION PIPELINE
# ==============================================================================
def main():
    print("=" * 80)
    print("AEROGUARD AI — REPAIRED EDGE-AWARE GNN VALIDATION RUN")
    print("=" * 80)

    train_path = "data/aero_guard_train.csv"
    test_path = "data/aero_guard_test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Dataset CSV files missing. Run dataset generation first.")

    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)

    df_train = add_coordinate_free_features(df_train)
    df_test = add_coordinate_free_features(df_test)

    # 1. Chronological Train/Validation Split (80% Train, 20% Val)
    unique_ts = np.sort(df_train['timestamp'].unique())
    split_idx = int(len(unique_ts) * 0.8)
    train_ts = set(unique_ts[:split_idx])
    val_ts = set(unique_ts[split_idx:])

    df_train_sub = df_train[df_train['timestamp'].isin(train_ts)].reset_index(drop=True)
    df_val_sub = df_train[df_train['timestamp'].isin(val_ts)].reset_index(drop=True)

    print(f"Train subset: {len(df_train_sub)} rows ({len(train_ts)} timestamps)")
    print(f"Val subset:   {len(df_val_sub)} rows ({len(val_ts)} timestamps)")
    print(f"Test OOD set: {len(df_test)} rows ({df_test['timestamp'].nunique()} timestamps)")

    # 2. Fit Scalers on Training Subset Only
    scaler_node = StandardScaler()
    scaler_node.fit(df_train_sub[COORD_FREE_NODE_FEATURES])

    # Fit Edge Scaler on RAW physical edges
    edge_train_samples = []
    for ts, grp in df_train_sub.groupby('timestamp'):
        n = len(grp)
        if n > 1:
            x, y = grp['x'].values, grp['y'].values
            alts = grp['altitude'].values
            vels = grp['groundspeed'].values
            heads = grp['heading'].values
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d <= PROXIMITY_THRESHOLD_M:
                        alt_d = alts[i] - alts[j]
                        vel_d = vels[i] - vels[j]
                        hd_d = (heads[i] - heads[j] + 180.0) % 360.0 - 180.0
                        edge_train_samples.append([d, alt_d, vel_d, hd_d])

    scaler_edge = StandardScaler()
    scaler_edge.fit(edge_train_samples)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler_node, "models/scaler_node_repaired.joblib")
    joblib.dump(scaler_edge, "models/scaler_edge_repaired.joblib")
    print(f"Fitted edge scaler on {len(edge_train_samples)} edges.")
    print(f"  Edge Scale Factors: distance={scaler_edge.scale_[0]:.1f}m, alt={scaler_edge.scale_[1]:.1f}ft, vel={scaler_edge.scale_[2]:.1f}kt, head={scaler_edge.scale_[3]:.1f}deg")

    # 3. Build Graph Snapshots
    print("\nBuilding PyG Graph Snapshots...")
    train_snaps = build_repaired_graph_snapshots(df_train_sub, scaler_node, scaler_edge)
    val_snaps = build_repaired_graph_snapshots(df_val_sub, scaler_node, scaler_edge)
    test_snaps = build_repaired_graph_snapshots(df_test, scaler_node, scaler_edge)

    train_loader = DataLoader(train_snaps, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_snaps, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_snaps, batch_size=32, shuffle=False)

    # 4. Train Repaired Edge-Aware GNN
    print("\nTraining Repaired Edge-Aware GNN...")
    model = RepairedEdgeAwareGNN(num_features=len(COORD_FREE_NODE_FEATURES), edge_dim=len(EDGE_FEATURES))
    class_weights = torch.tensor([1.0, 10.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    epochs = 8
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        print(f"Epoch {ep+1:02d}/{epochs:02d} | Train Loss: {total_loss / len(train_snaps):.4f}")

    torch.save(model.state_dict(), "models/gnn_edge_aware_repaired.pt")
    print("Saved repaired model to models/gnn_edge_aware_repaired.pt")

    # 5. Validation Calibration
    print("\n==================== VALIDATION & THRESHOLD CALIBRATION ====================")
    y_val, p_val, _ = evaluate_model(model, val_loader)
    val_roc = roc_auc_score(y_val, p_val)
    val_pr = average_precision_score(y_val, p_val)
    t_cal, val_f1_cal = calibrate_threshold(y_val, p_val)

    print(f"Validation ROC-AUC: {val_roc:.4f}")
    print(f"Validation PR-AUC:  {val_pr:.4f}")
    print(f"Calibrated Threshold: t={t_cal:.2f} (Val F1={val_f1_cal:.4f})")

    # 6. Independent Test Evaluation on Jan 2 OOD Set
    print("\n==================== INDEPENDENT TEST EVALUATION (JAN 2 OOD) ====================")
    y_test, p_test, test_indices = evaluate_model(model, test_loader)

    eval_configs = [
        ("Repaired Edge-Aware GNN (Default t=0.50)", 0.50),
        ("Repaired Edge-Aware GNN (Calibrated t=0.01)", t_cal)
    ]

    print("-" * 95)
    print(f"{'Configuration':<45} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'PR-AUC':<9}")
    print("-" * 95)

    for name, thresh in eval_configs:
        preds = (p_test >= thresh).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, p_test)
        pr = average_precision_score(y_test, p_test)
        cm = confusion_matrix(y_test, preds)

        print(f"{name:<45} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f} | {roc:<9.4f} | {pr:<9.4f}")
        print(f"  Confusion Matrix: TN={cm[0,0]:<6} FP={cm[0,1]:<6} FN={cm[1,0]:<6} TP={cm[1,1]:<6}")
    print("-" * 95)

    # Breakdown by anomaly type
    df_eval_aligned = df_test.iloc[test_indices].copy()
    df_eval_aligned['prob'] = p_test
    df_eval_aligned['pred_05'] = (p_test >= 0.5).astype(int)
    df_eval_aligned['pred_cal'] = (p_test >= t_cal).astype(int)

    print("\nAnomaly Type Detection Breakdown (at t=0.50):")
    for anom_type, grp in df_eval_aligned.groupby('anomaly_type'):
        n_tot = len(grp)
        n_det = grp['pred_05'].sum()
        mean_p = grp['prob'].mean()
        print(f"  - {anom_type:<25}: Detected {n_det:>4}/{n_tot:<4} ({n_det/n_tot*100:>5.1f}%) | Mean Prob = {mean_p:.4f}")

if __name__ == "__main__":
    main()
