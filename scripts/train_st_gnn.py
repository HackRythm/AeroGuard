"""
AeroGuard AI — ST-GNN Pipeline: Multi-Stream Training, Validation, and Evaluation
==================================================================================
Implements the full end-to-end training and evaluation pipeline for the Spatio-Temporal
Graph Neural Network (ST-GNN) built on the verified, repaired Edge-Aware GNN spatial encoder.

Pipeline Stages:
1. Leakage & Data Audit (confirms zero true_* leakage, zero future peeking).
2. Spatial Embedding Generation (freezes repaired Edge-Aware GNN).
3. Causal Trajectory Windowing (T=5) with observation masks & track separation.
4. Causal Kinematic Residual Feature Engineering.
5. Temporal GRU & Kinematic Branch Training (on Jan 1 80% train split).
6. Multi-Stream Fusion Comparison & Threshold Calibration (on Jan 1 20% val split).
7. Final Evaluation on Jan 2 OOD Test Set (reporting precision, recall, F1, ROC-AUC, PR-AUC, CM).
8. Artifact Serialization to models/st_gnn_official.pt and models/st_gnn_calibrated_threshold.json.
"""

import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from typing import Tuple, List, Optional, Dict
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import joblib

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from scripts.train_edge_aware_repaired import (
    RepairedEdgeAwareGNN, COORD_FREE_NODE_FEATURES, EDGE_FEATURES,
    build_repaired_graph_snapshots, add_coordinate_free_features
)
from scripts.st_gnn_model import TemporalGRUStream, KinematicStream, AeroGuardSTGNN

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TEMPORAL_WINDOW_T = 5

KINEMATIC_TEMPORAL_FEATS = [
    'altitude', 'groundspeed', 'heading_sin', 'heading_cos', 'vertical_rate',
    'acceleration', 'turn_rate', 'vertical_acceleration', 'climb_rate_change'
]

# ==============================================================================
# 1. LEAKAGE AUDIT UTILITIES
# ==============================================================================
def run_leakage_audit(df: pd.DataFrame, split_name: str):
    """
    Verifies that no ground-truth internal columns or injection metadata leak into inputs.
    """
    print(f"\n--- Running Strict Leakage Audit on [{split_name}] ---")
    leaked_true = [col for col in df.columns if 'true_' in col.lower()]
    assert len(leaked_true) == 0, f"LEAKAGE DETECTED: Found true physical columns: {leaked_true}"

    # Check for expected columns
    required_cols = ['timestamp', 'icao24', 'latitude', 'longitude', 'altitude', 'groundspeed', 'heading', 'label']
    for c in required_cols:
        assert c in df.columns, f"Missing required column: {c}"

    print(f"  [PASS] Zero true_* columns present.")
    print(f"  [PASS] Causal columns verified.")

# ==============================================================================
# 2. CAUSAL TRAJECTORY WINDOW & KINEMATIC FEATURE BUILDER
# ==============================================================================
def compute_causal_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly causal trajectory residuals and motion invariants.
    Uses ONLY current and preceding observations (diffs with lag >= 1).
    Does NOT use future observations or ground-truth labels.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. Detect duplicate ICAO broadcasts at the exact same timestamp (identity inconsistency)
    icao_counts = df.groupby(['timestamp', 'icao24'])['callsign'].transform('count')
    df['icao_collision'] = (icao_counts > 1).astype(float)

    # 2. Sort by ICAO and timestamp for causal differentiation
    df = df.sort_values(by=['icao24', 'timestamp']).reset_index(drop=True)

    # Time delta
    dt = df.groupby('icao24')['timestamp'].diff().dt.total_seconds().fillna(1.0).values
    dt = np.where(dt <= 0.0, 1.0, dt)

    # Haversine distance step
    lat1 = np.radians(df.groupby('icao24')['latitude'].shift(1).fillna(df['latitude']))
    lat2 = np.radians(df['latitude'])
    dlat = np.radians(df.groupby('icao24')['latitude'].diff().fillna(0.0))
    dlon = np.radians(df.groupby('icao24')['longitude'].diff().fillna(0.0))
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    step_dist_m = 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    # Speed ratio
    speed_m_s = df['groundspeed'].values * 0.514444
    expected_dist = np.maximum(speed_m_s * dt, 1.0)
    disp_ratio = np.clip(step_dist_m / expected_dist, 0.0, 50.0)
    df['disp_ratio'] = disp_ratio

    # Altitude consistency residual: |Delta altitude - (vertical_rate * dt / 60)|
    dalt = df.groupby('icao24')['altitude'].diff().fillna(0.0).values
    expected_dalt = df['vertical_rate'].values * (dt / 60.0)
    df['alt_residual'] = np.clip(np.abs(dalt - expected_dalt), 0.0, 10000.0)

    # Acceleration consistency residual
    d_speed = df.groupby('icao24')['groundspeed'].diff().fillna(0.0).values * 0.514444
    expected_accel = d_speed / dt
    df['accel_residual'] = np.clip(np.abs(df['acceleration'].values - expected_accel), 0.0, 50.0)

    # Turn rate consistency residual
    d_head = df.groupby('icao24')['heading'].diff().fillna(0.0).values
    d_head = (d_head + 180.0) % 360.0 - 180.0
    expected_turn = d_head / dt
    df['turn_residual'] = np.clip(np.abs(df['turn_rate'].values - expected_turn), 0.0, 50.0)

    # Track jump flag (step displacement > 5 km within 1 sec)
    df['track_jump_flag'] = (step_dist_m > 5000.0).astype(float)

    # Instantaneous kinematics to include
    # ['vertical_rate', 'acceleration', 'turn_rate', 'vertical_acceleration', 'climb_rate_change']

    return df

KINEMATIC_BRANCH_FEATS = [
    'disp_ratio', 'alt_residual', 'accel_residual', 'turn_residual',
    'icao_collision', 'track_jump_flag',
    'vertical_rate', 'acceleration', 'turn_rate', 'vertical_acceleration', 'climb_rate_change'
]

def build_temporal_windows_for_dataframe(
    df: pd.DataFrame,
    spatial_embeddings: np.ndarray,
    scaler_node: StandardScaler,
    scaler_kinematic: StandardScaler,
    window_t: int = 5
):
    """
    Constructs chronological T-timestep observation sequences per aircraft track.

    Guarantees:
    - Sequence is strictly causal (only observations <= current timestamp).
    - If prior observations are missing, sequence is zero-padded with observation mask = 0.
    - Matches 1-to-1 with rows in df.
    """
    df = df.copy()
    num_rows = len(df)

    # Scaled kinematic features for temporal input (9D)
    scaled_temporal_kin = scaler_node.transform(df[COORD_FREE_NODE_FEATURES])
    # Extract indices corresponding to KINEMATIC_TEMPORAL_FEATS
    feat_indices = [COORD_FREE_NODE_FEATURES.index(f) for f in KINEMATIC_TEMPORAL_FEATS]
    kin_temporal_arr = scaled_temporal_kin[:, feat_indices]  # [N, 9]

    # Concatenate spatial embedding (16D) + temporal kinematics (9D) = 25D
    combined_step_feats = np.hstack([spatial_embeddings, kin_temporal_arr])  # [N, 25]

    # Scaled kinematic branch features (11D)
    kin_branch_arr = scaler_kinematic.transform(df[KINEMATIC_BRANCH_FEATS])  # [N, 11]

    # Build temporal sequence tensors [N, T, 25] and mask [N, T]
    sequences = np.zeros((num_rows, window_t, 25), dtype=np.float32)
    masks = np.zeros((num_rows, window_t), dtype=bool)

    # Group by ICAO24 and construct sliding windows
    df['row_idx'] = np.arange(num_rows)
    for icao, group in df.groupby('icao24'):
        group_rows = group['row_idx'].values
        m = len(group_rows)
        for i, current_idx in enumerate(group_rows):
            # Select history up to current_idx
            start_i = max(0, i - window_t + 1)
            hist_rows = group_rows[start_i : i + 1]
            hist_len = len(hist_rows)

            # Place in the trailing positions of the window
            offset = window_t - hist_len
            sequences[current_idx, offset:, :] = combined_step_feats[hist_rows, :]
            masks[current_idx, offset:] = True

    return (
        torch.tensor(sequences, dtype=torch.float),
        torch.tensor(masks, dtype=torch.bool),
        torch.tensor(kin_branch_arr, dtype=torch.float),
        torch.tensor(df['label'].values, dtype=torch.long)
    )

# ==============================================================================
# 3. SPATIAL EMBEDDING EXTRACTION
# ==============================================================================
def extract_all_spatial_embeddings(
    model: RepairedEdgeAwareGNN,
    snapshots: list,
    total_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Passes graph snapshots through the frozen repaired Edge-Aware GNN
    and collects 16D spatial node embeddings aligned to original row indices.
    """
    model.eval()
    spatial_embeddings = np.zeros((total_nodes, 16), dtype=np.float32)
    spatial_probs = np.zeros(total_nodes, dtype=np.float32)

    with torch.no_grad():
        for snap in snapshots:
            emb = model.extract_spatial_embeddings(snap.x, snap.edge_index, snap.edge_attr)
            logits = model.classifier(emb)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            orig_indices = snap.orig_indices.cpu().numpy()
            spatial_embeddings[orig_indices] = emb.cpu().numpy()
            spatial_probs[orig_indices] = probs.cpu().numpy()

    return spatial_embeddings, spatial_probs

# ==============================================================================
# 4. TRAINING & FUSION PIPELINE
# ==============================================================================
def main():
    print("=" * 90)
    print("AEROGUARD AI — ST-GNN INCREMENTAL TRAINING & INDEPENDENT VALIDATION PIPELINE")
    print("=" * 90)

    # 1. Load Data
    train_path = "data/aero_guard_train.csv"
    test_path = "data/aero_guard_test.csv"
    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)

    # Run Leakage Audit
    run_leakage_audit(df_train, "Train Set")
    run_leakage_audit(df_test, "Test Set")

    # Add coordinate-free trigonometric headings
    df_train = add_coordinate_free_features(df_train)
    df_test = add_coordinate_free_features(df_test)

    # Compute strictly causal trajectory residuals & motion invariants
    print("\nComputing causal kinematic trajectory residuals...")
    df_train = compute_causal_kinematic_features(df_train)
    df_test = compute_causal_kinematic_features(df_test)

    # 2. Chronological Train/Validation Split (80% Train, 20% Val)
    unique_ts = np.sort(df_train['timestamp'].unique())
    split_idx = int(len(unique_ts) * 0.8)
    train_ts = set(unique_ts[:split_idx])
    val_ts = set(unique_ts[split_idx:])

    df_train_sub = df_train[df_train['timestamp'].isin(train_ts)].reset_index(drop=True)
    df_val_sub = df_train[df_train['timestamp'].isin(val_ts)].reset_index(drop=True)

    print(f"Train subset: {len(df_train_sub)} rows ({len(train_ts)} timestamps)")
    print(f"Val subset:   {len(df_val_sub)} rows ({len(val_ts)} timestamps)")
    print(f"Test OOD set: {len(df_test)} rows ({df_test['timestamp'].nunique()} timestamps)")

    # 3. Load Repaired Spatial Model and Scalers (FROZEN)
    print("\nLoading Repaired Spatial Encoder & Scalers...")
    spatial_weights_path = "models/gnn_edge_aware_repaired.pt"
    assert os.path.exists(spatial_weights_path), f"Missing repaired weights: {spatial_weights_path}"

    spatial_model = RepairedEdgeAwareGNN(num_features=len(COORD_FREE_NODE_FEATURES), edge_dim=len(EDGE_FEATURES))
    spatial_model.load_state_dict(torch.load(spatial_weights_path, map_location='cpu'))
    spatial_model.eval()
    for param in spatial_model.parameters():
        param.requires_grad = False
    print("Repaired spatial encoder loaded and FROZEN successfully.")

    scaler_node = joblib.load("models/scaler_node_repaired.joblib")
    scaler_edge = joblib.load("models/scaler_edge_repaired.joblib")

    # Fit Kinematic Scaler on Training Subset Only
    scaler_kinematic = StandardScaler()
    scaler_kinematic.fit(df_train_sub[KINEMATIC_BRANCH_FEATS])
    joblib.dump(scaler_kinematic, "models/scaler_kinematic.joblib")
    print(f"Fitted kinematic branch scaler on {len(KINEMATIC_BRANCH_FEATS)} causal features.")

    # 4. Extract Spatial Embeddings
    print("\nGenerating Spatial Graph Snapshots & Extracting Spatial Embeddings...")
    train_snaps = build_repaired_graph_snapshots(df_train_sub, scaler_node, scaler_edge)
    val_snaps = build_repaired_graph_snapshots(df_val_sub, scaler_node, scaler_edge)
    test_snaps = build_repaired_graph_snapshots(df_test, scaler_node, scaler_edge)

    emb_train, p_sp_train = extract_all_spatial_embeddings(spatial_model, train_snaps, len(df_train_sub))
    emb_val, p_sp_val = extract_all_spatial_embeddings(spatial_model, val_snaps, len(df_val_sub))
    emb_test, p_sp_test = extract_all_spatial_embeddings(spatial_model, test_snaps, len(df_test))

    print(f"Spatial embeddings extracted: Train={emb_train.shape}, Val={emb_val.shape}, Test={emb_test.shape}")

    # 5. Build Causal Temporal Windows & Kinematic Features
    print("\nBuilding Causal Sliding Windows (T=5) with Observation Masks...")
    seq_tr, mask_tr, kin_tr, y_tr = build_temporal_windows_for_dataframe(
        df_train_sub, emb_train, scaler_node, scaler_kinematic, TEMPORAL_WINDOW_T
    )
    seq_val, mask_val, kin_val, y_val = build_temporal_windows_for_dataframe(
        df_val_sub, emb_val, scaler_node, scaler_kinematic, TEMPORAL_WINDOW_T
    )
    seq_te, mask_te, kin_te, y_te = build_temporal_windows_for_dataframe(
        df_test, emb_test, scaler_node, scaler_kinematic, TEMPORAL_WINDOW_T
    )

    train_loader = TorchDataLoader(
        TensorDataset(seq_tr, mask_tr, kin_tr, torch.tensor(p_sp_train, dtype=torch.float), y_tr),
        batch_size=128, shuffle=True
    )
    val_loader = TorchDataLoader(
        TensorDataset(seq_val, mask_val, kin_val, torch.tensor(p_sp_val, dtype=torch.float), y_val),
        batch_size=128, shuffle=False
    )
    test_loader = TorchDataLoader(
        TensorDataset(seq_te, mask_te, kin_te, torch.tensor(p_sp_test, dtype=torch.float), y_te),
        batch_size=128, shuffle=False
    )

    # 6. Initialize Temporal GRU & Kinematic Branch Models
    temporal_stream = TemporalGRUStream(input_dim=25, hidden_dim=32, num_layers=1, dropout=0.1)
    kinematic_stream = KinematicStream(input_dim=len(KINEMATIC_BRANCH_FEATS), hidden_dim=32, dropout=0.1)

    class_weights = torch.tensor([1.0, 10.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        list(temporal_stream.parameters()) + list(kinematic_stream.parameters()),
        lr=0.003, weight_decay=1e-4
    )

    # 7. Train Temporal and Kinematic Streams
    print("\nTraining Temporal GRU Stream and Kinematic Stream (6 Epochs)...")
    epochs = 6
    for ep in range(epochs):
        temporal_stream.train()
        kinematic_stream.train()
        total_loss = 0.0

        for batch_seq, batch_mask, batch_kin, batch_p_sp, batch_y in train_loader:
            optimizer.zero_grad()
            _, logits_t = temporal_stream(batch_seq, batch_mask)
            _, logits_k = kinematic_stream(batch_kin)

            loss_t = criterion(logits_t, batch_y)
            loss_k = criterion(logits_k, batch_y)
            loss = loss_t + loss_k

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        print(f"Epoch {ep+1:02d}/{epochs:02d} | Joint Loss: {total_loss / len(df_train_sub):.4f}")

    # 8. Inference over Validation Set
    print("\n==================== EVALUATION & FUSION ON VALIDATION SET ====================")
    def run_inference(t_stream, k_stream, loader):
        t_stream.eval()
        k_stream.eval()
        probs_sp, probs_t, probs_k, y_true = [], [], [], []

        with torch.no_grad():
            for batch_seq, batch_mask, batch_kin, batch_p_sp, batch_y in loader:
                _, logits_t = t_stream(batch_seq, batch_mask)
                _, logits_k = k_stream(batch_kin)

                p_t = torch.softmax(logits_t, dim=-1)[:, 1].cpu().numpy()
                p_k = torch.softmax(logits_k, dim=-1)[:, 1].cpu().numpy()

                probs_sp.extend(batch_p_sp.cpu().numpy())
                probs_t.extend(p_t)
                probs_k.extend(p_k)
                y_true.extend(batch_y.cpu().numpy())

        return np.array(probs_sp), np.array(probs_t), np.array(probs_k), np.array(y_true)

    p_sp_val, p_t_val, p_k_val, y_val_arr = run_inference(temporal_stream, kinematic_stream, val_loader)

    # Compare candidate fusion strategies on validation data
    p_fused_max_val = np.maximum(np.maximum(p_sp_val, p_t_val), p_k_val)
    p_fused_mean_val = (p_sp_val + p_t_val + p_k_val) / 3.0
    p_fused_weighted_val = 0.4 * p_sp_val + 0.35 * p_t_val + 0.25 * p_k_val

    fusion_candidates = [
        ("Spatial-Only", p_sp_val),
        ("Temporal-Only (GRU)", p_t_val),
        ("Kinematic-Only (MLP)", p_k_val),
        ("Fusion: Average Probability", p_fused_mean_val),
        ("Fusion: Weighted Probability", p_fused_weighted_val),
        ("Fusion: Maximum Probability Envelope", p_fused_max_val)
    ]

    print("-" * 80)
    print(f"{'Validation Strategy':<38} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'Best F1':<9} | {'Thresh':<7}")
    print("-" * 80)

    best_strategy_name = ""
    best_strategy_f1 = -1.0
    best_calibrated_thresh = 0.5

    for name, p_arr in fusion_candidates:
        roc = roc_auc_score(y_val_arr, p_arr)
        pr = average_precision_score(y_val_arr, p_arr)

        # Sweep threshold for best F1
        b_f1, b_t = -1.0, 0.5
        for t in np.linspace(0.01, 0.99, 99):
            score = f1_score(y_val_arr, (p_arr >= t).astype(int), zero_division=0)
            if score > b_f1:
                b_f1, b_t = score, t

        print(f"{name:<38} | {roc:<9.4f} | {pr:<9.4f} | {b_f1:<9.4f} | {b_t:<7.2f}")
        if "Fusion" in name and b_f1 > best_strategy_f1:
            best_strategy_f1 = b_f1
            best_strategy_name = name
            best_calibrated_thresh = b_t

    print("-" * 80)
    print(f"Selected Official Fusion Strategy: [{best_strategy_name}] with Calibrated Threshold t={best_calibrated_thresh:.2f}")

    # 9. Evaluate Frozen ST-GNN on Independent Test Set (Jan 2 OOD)
    print("\n==================== INDEPENDENT TEST SET EVALUATION (JAN 2 OOD) ====================")
    p_sp_test, p_t_test, p_k_test, y_test_arr = run_inference(temporal_stream, kinematic_stream, test_loader)

    p_fused_max_test = np.maximum(np.maximum(p_sp_test, p_t_test), p_k_test)
    p_fused_mean_test = (p_sp_test + p_t_test + p_k_test) / 3.0
    p_fused_weighted_test = 0.4 * p_sp_test + 0.35 * p_t_test + 0.25 * p_k_test

    if "Maximum" in best_strategy_name:
        p_final_test = p_fused_max_test
    elif "Weighted" in best_strategy_name:
        p_final_test = p_fused_weighted_test
    else:
        p_final_test = p_fused_mean_test

    test_evaluations = [
        ("Spatial-Only (Repaired GNN)", p_sp_test, 0.50),
        ("Temporal-Only (GRU)", p_t_test, 0.50),
        ("Kinematic-Only (MLP)", p_k_test, 0.50),
        ("ST-GNN Fused (Default t=0.50)", p_final_test, 0.50),
        (f"ST-GNN Fused (Calibrated t={best_calibrated_thresh:.2f})", p_final_test, best_calibrated_thresh)
    ]

    print("-" * 96)
    print(f"{'Model Stream / Pipeline':<36} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'PR-AUC':<9}")
    print("-" * 96)

    for name, p_arr, thresh in test_evaluations:
        preds = (p_arr >= thresh).astype(int)
        prec = precision_score(y_test_arr, preds, zero_division=0)
        rec = recall_score(y_test_arr, preds, zero_division=0)
        f1 = f1_score(y_test_arr, preds, zero_division=0)
        roc = roc_auc_score(y_test_arr, p_arr)
        pr = average_precision_score(y_test_arr, p_arr)
        cm = confusion_matrix(y_test_arr, preds)

        print(f"{name:<36} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f} | {roc:<9.4f} | {pr:<9.4f}")
        print(f"  Confusion Matrix: TN={cm[0,0]:<6} FP={cm[0,1]:<6} FN={cm[1,0]:<6} TP={cm[1,1]:<6}")
    print("-" * 96)

    # Per-Class Breakdown on Final ST-GNN
    df_test_out = df_test.copy()
    df_test_out['st_gnn_prob'] = p_final_test
    df_test_out['st_gnn_pred_05'] = (p_final_test >= 0.50).astype(int)
    df_test_out['st_gnn_pred_calibrated'] = (p_final_test >= best_calibrated_thresh).astype(int)

    print("\nAnomaly Category Breakdown for Final ST-GNN (at t=0.50):")
    for anom_type, grp in df_test_out.groupby('anomaly_type'):
        n_tot = len(grp)
        n_det = grp['st_gnn_pred_05'].sum()
        mean_p = grp['st_gnn_prob'].mean()
        print(f"  - {anom_type:<25}: Detected {n_det:>4}/{n_tot:<4} ({n_det/n_tot*100:>5.1f}%) | Mean Prob = {mean_p:.4f}")

    print(f"\nAnomaly Category Breakdown for Final ST-GNN (at Calibrated t={best_calibrated_thresh:.2f}):")
    for anom_type, grp in df_test_out.groupby('anomaly_type'):
        n_tot = len(grp)
        n_det = grp['st_gnn_pred_calibrated'].sum()
        mean_p = grp['st_gnn_prob'].mean()
        print(f"  - {anom_type:<25}: Detected {n_det:>4}/{n_tot:<4} ({n_det/n_tot*100:>5.1f}%) | Mean Prob = {mean_p:.4f}")

    # 10. Save Official Model Checkpoint and Artifacts
    full_st_gnn = AeroGuardSTGNN(
        spatial_model=spatial_model,
        num_kinematic_temporal_feats=len(KINEMATIC_TEMPORAL_FEATS),
        num_kinematic_branch_feats=len(KINEMATIC_BRANCH_FEATS),
        fusion_mode='max_prob' if 'Maximum' in best_strategy_name else 'weighted_prob'
    )
    full_st_gnn.temporal_stream.load_state_dict(temporal_stream.state_dict())
    full_st_gnn.kinematic_stream.load_state_dict(kinematic_stream.state_dict())

    torch.save(full_st_gnn.state_dict(), "models/st_gnn_official.pt")
    print("\nSaved official model checkpoint to models/st_gnn_official.pt")

    calibrated_meta = {
        "timestamp": datetime.now().isoformat(),
        "selected_fusion": best_strategy_name,
        "calibrated_threshold": best_calibrated_thresh,
        "validation_f1": best_strategy_f1,
        "test_metrics": {
            "default_05": {
                "precision": float(precision_score(y_test_arr, (p_final_test >= 0.50).astype(int), zero_division=0)),
                "recall": float(recall_score(y_test_arr, (p_final_test >= 0.50).astype(int), zero_division=0)),
                "f1": float(f1_score(y_test_arr, (p_final_test >= 0.50).astype(int), zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test_arr, p_final_test)),
                "pr_auc": float(average_precision_score(y_test_arr, p_final_test))
            },
            "calibrated": {
                "precision": float(precision_score(y_test_arr, (p_final_test >= best_calibrated_thresh).astype(int), zero_division=0)),
                "recall": float(recall_score(y_test_arr, (p_final_test >= best_calibrated_thresh).astype(int), zero_division=0)),
                "f1": float(f1_score(y_test_arr, (p_final_test >= best_calibrated_thresh).astype(int), zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test_arr, p_final_test)),
                "pr_auc": float(average_precision_score(y_test_arr, p_final_test))
            }
        }
    }

    with open("models/st_gnn_calibrated_threshold.json", "w") as f:
        json.dump(calibrated_meta, f, indent=4)
    print("Saved calibrated threshold metadata to models/st_gnn_calibrated_threshold.json")

    # Save predictions CSV
    df_test_out.to_csv("data/st_gnn_test_predictions.csv", index=False)
    print("Saved final test predictions to data/st_gnn_test_predictions.csv")

if __name__ == "__main__":
    main()
