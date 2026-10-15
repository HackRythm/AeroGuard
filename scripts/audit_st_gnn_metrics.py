"""
AeroGuard AI — Independent ST-GNN Metrics, Causality, and Inference Audit
==========================================================================
This script performs a 100% independent audit of the ST-GNN artifacts:
1. Verifies class distribution and total record counts (32,550).
2. Recalculates confusion matrices, precision, recall, F1, specificity, FPR,
   continuous ROC-AUC, and PR-AUC at thresholds 0.50, 0.20, and JSON threshold.
3. Performs per-class confusion arithmetic checks (verifying TP summation).
4. Reproduces full inference from models/st_gnn_official.pt and compares predictions.
5. Saves reproduced predictions to data/reproduced_st_gnn_predictions.csv.
"""

import os, sys, json
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from scripts.train_edge_aware_repaired import (
    RepairedEdgeAwareGNN, COORD_FREE_NODE_FEATURES, EDGE_FEATURES,
    build_repaired_graph_snapshots, add_coordinate_free_features
)
from scripts.train_st_gnn import (
    compute_causal_kinematic_features, build_temporal_windows_for_dataframe,
    extract_all_spatial_embeddings, KINEMATIC_BRANCH_FEATS, KINEMATIC_TEMPORAL_FEATS,
    TEMPORAL_WINDOW_T
)
from scripts.st_gnn_model import AeroGuardSTGNN

def run_independent_audit():
    print("=" * 90)
    print("AEROGUARD AI — INDEPENDENT ST-GNN METRICS & INFERENCE AUDIT")
    print("=" * 90)

    # 1. Inspect Saved Predictions CSV
    csv_path = "data/st_gnn_test_predictions.csv"
    assert os.path.exists(csv_path), f"Missing prediction CSV: {csv_path}"
    df_saved = pd.read_csv(csv_path, low_memory=False)

    total_records = len(df_saved)
    print(f"Total test records in CSV: {total_records} (Expected: 32550)")
    assert total_records == 32550, f"Expected 32550 records, got {total_records}"

    y_true = df_saved['label'].values
    y_prob = df_saved['st_gnn_prob'].values

    num_normal = int(np.sum(y_true == 0))
    num_anomaly = int(np.sum(y_true == 1))
    print(f"Verified Class Distribution: Normal (0) = {num_normal}, Anomaly (1) = {num_anomaly}")
    assert num_normal == 27544, f"Expected 27544 normal, got {num_normal}"
    assert num_anomaly == 5006, f"Expected 5006 anomalies, got {num_anomaly}"

    # Check Anomaly Types
    print("\nVerified Anomaly Type Breakdown:")
    anom_counts = df_saved.groupby('anomaly_type')['label'].count()
    for anom_type, count in anom_counts.items():
        print(f"  - {anom_type:<25}: {count:>5} records")

    expected_types = {'normal': 27544, 'altitude_anomaly': 660, 'ghost_aircraft': 3600, 'position_jump': 596, 'identity_inconsistency': 150}
    for k, v in expected_types.items():
        assert anom_counts[k] == v, f"Count mismatch for {k}: expected {v}, got {anom_counts[k]}"

    # Continuous Ranking Metrics
    continuous_roc = float(roc_auc_score(y_true, y_prob))
    continuous_pr = float(average_precision_score(y_true, y_prob))
    print(f"\nContinuous Fused Ranking Metrics:")
    print(f"  ROC-AUC: {continuous_roc:.6f}")
    print(f"  PR-AUC:  {continuous_pr:.6f}")

    # 2. Check JSON Metadata
    json_path = "models/st_gnn_calibrated_threshold.json"
    with open(json_path) as f:
        meta = json.load(f)
    stored_thresh = meta['calibrated_threshold']
    print(f"\nThreshold metadata in {json_path}:")
    print(f"  Selected Fusion:      {meta['selected_fusion']}")
    print(f"  Calibrated Threshold: {stored_thresh}")
    print(f"  Validation F1:        {meta['validation_f1']:.6f}")

    # 3. Recalculate Metrics at Thresholds 0.50, 0.20, and Stored Threshold
    eval_thresholds = [0.50, 0.20]
    if stored_thresh not in eval_thresholds:
        eval_thresholds.append(stored_thresh)

    results_dict = {}
    print("\n" + "=" * 90)
    print("INDEPENDENT METRIC RECALCULATION & ARITHMETIC VERIFICATION")
    print("=" * 90)

    for t in eval_thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # Arithmetic sanity check: sum must equal 32550
        assert tn + fp + fn + tp == 32550, f"Matrix sum mismatch at t={t}: {tn+fp+fn+tp} != 32550"
        assert tn + fp == 27544, f"Actual negatives mismatch: {tn+fp} != 27544"
        assert tp + fn == 5006, f"Actual positives mismatch: {tp+fn} != 5006"

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"\n--- Decision Threshold: t = {t:.2f} ---")
        print(f"  Confusion Matrix: TN={tn:<6} FP={fp:<6} FN={fn:<6} TP={tp:<6} (Sum = {tn+fp+fn+tp})")
        print(f"  Precision:   {prec:.6f} ({tp}/{tp+fp})")
        print(f"  Recall:      {rec:.6f} ({tp}/{tp+fn})")
        print(f"  F1-Score:    {f1:.6f}")
        print(f"  Specificity: {spec:.6f} ({tn}/{tn+fp})")
        print(f"  FPR:         {fpr:.6f} ({fp}/{tn+fp})")

        # Per-class detection audit
        df_saved['temp_pred'] = preds
        print(f"  Per-Class Detections at t = {t:.2f}:")
        per_class_tp_sum = 0
        for anom in ['altitude_anomaly', 'ghost_aircraft', 'position_jump', 'identity_inconsistency']:
            sub = df_saved[df_saved['anomaly_type'] == anom]
            c_tp = int(sub['temp_pred'].sum())
            per_class_tp_sum += c_tp
            print(f"    * {anom:<25}: TP = {c_tp:>4}/{len(sub):<4} ({c_tp/len(sub)*100:>5.1f}%)")

        c_fp = int(df_saved[df_saved['anomaly_type'] == 'normal']['temp_pred'].sum())
        print(f"    * Normal (False Alarms)   : FP = {c_fp:>4}/{num_normal:<5} ({c_fp/num_normal*100:>5.2f}%)")

        # Arithmetic check: per-class TP sum MUST equal overall TP
        assert per_class_tp_sum == tp, f"Per-class TP sum mismatch: {per_class_tp_sum} != {tp}"
        assert c_fp == fp, f"Normal FP mismatch: {c_fp} != {fp}"
        print(f"  [VERIFIED] Per-class TP sum ({per_class_tp_sum}) matches overall TP ({tp}).")

        results_dict[t] = {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'prec': prec, 'rec': rec, 'f1': f1, 'spec': spec, 'fpr': fpr
        }

    # 4. Independent Reproduction from Checkpoints
    print("\n" + "=" * 90)
    print("CHECKPOINT INFERENCE REPRODUCTION AUDIT")
    print("=" * 90)

    df_test_raw = pd.read_csv("data/aero_guard_test.csv", low_memory=False)
    df_test_proc = add_coordinate_free_features(df_test_raw)
    df_test_proc = compute_causal_kinematic_features(df_test_proc)

    scaler_node = joblib.load("models/scaler_node_repaired.joblib")
    scaler_edge = joblib.load("models/scaler_edge_repaired.joblib")
    scaler_kin = joblib.load("models/scaler_kinematic.joblib")

    spatial_model = RepairedEdgeAwareGNN(len(COORD_FREE_NODE_FEATURES), len(EDGE_FEATURES))
    spatial_model.load_state_dict(torch.load("models/gnn_edge_aware_repaired.pt", map_location='cpu'))
    spatial_model.eval()

    full_st_gnn = AeroGuardSTGNN(
        spatial_model=spatial_model,
        num_kinematic_temporal_feats=len(KINEMATIC_TEMPORAL_FEATS),
        num_kinematic_branch_feats=len(KINEMATIC_BRANCH_FEATS),
        fusion_mode='weighted_prob'
    )
    full_st_gnn.load_state_dict(torch.load("models/st_gnn_official.pt", map_location='cpu'))
    full_st_gnn.eval()

    test_snaps = build_repaired_graph_snapshots(df_test_proc, scaler_node, scaler_edge)
    emb_test, p_sp_test = extract_all_spatial_embeddings(spatial_model, test_snaps, len(df_test_proc))
    seq_te, mask_te, kin_te, y_te = build_temporal_windows_for_dataframe(
        df_test_proc, emb_test, scaler_node, scaler_kin, TEMPORAL_WINDOW_T
    )

    with torch.no_grad():
        _, logits_t = full_st_gnn.temporal_stream(seq_te, mask_te)
        _, logits_k = full_st_gnn.kinematic_stream(kin_te)
        p_t = torch.softmax(logits_t, dim=-1)[:, 1].numpy()
        p_k = torch.softmax(logits_k, dim=-1)[:, 1].numpy()

    # Fused calculation using official documented weights: 0.40 spatial, 0.35 temporal, 0.25 kinematic
    p_reproduced_fused = 0.40 * p_sp_test + 0.35 * p_t + 0.25 * p_k

    abs_diff = np.abs(df_saved['st_gnn_prob'].values - p_reproduced_fused)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    is_exact = max_diff < 1e-6

    print(f"Max absolute difference between regenerated and saved st_gnn_prob: {max_diff:.8e}")
    print(f"Mean absolute difference: {mean_diff:.8e}")
    print(f"Inference Reproduction Status: {'PASSED (EXACT MATCH)' if is_exact else 'FAILED'}")
    assert is_exact, f"Inference reproduction divergence: max_diff = {max_diff}"

    # Save reproduced predictions to a new file
    reproduced_df = df_test_proc.copy()
    reproduced_df['reproduced_spatial_prob'] = p_sp_test
    reproduced_df['reproduced_temporal_prob'] = p_t
    reproduced_df['reproduced_kinematic_prob'] = p_k
    reproduced_df['reproduced_fused_prob'] = p_reproduced_fused
    reproduced_df['saved_fused_prob'] = df_saved['st_gnn_prob'].values

    out_repro_path = "data/reproduced_st_gnn_predictions.csv"
    reproduced_df.to_csv(out_repro_path, index=False)
    print(f"Saved reproduced test predictions to: {out_repro_path}")

    print("\n" + "=" * 90)
    print("AUDIT EXECUTION COMPLETE — ALL CHECKS PASSED.")
    print("=" * 90)

if __name__ == "__main__":
    run_independent_audit()
