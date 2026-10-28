"""
AeroGuard AI — Research Evaluation Benchmark Harness
===================================================
Reproducible evaluation script comparing:
1. Baseline 1: Tabular Random Forest (13D node features with coordinates).
2. Baseline 2: Original Graph Convolutional Network (GCN, 13D node features, no edge attributes).
3. Baseline 3: Historical Edge-Aware GNN (GATv2 with edge scaling defect).
4. Baseline 4: Repaired Edge-Aware GNN (Spatial Stream with coordinate-free 12D + 4D edge features).
5. Proposed: Tri-Stream Spatio-Temporal Graph Neural Network (ST-GNN) at default t=0.50 and calibrated t=0.20.

All models evaluated strictly on the identical 32,550 out-of-distribution test samples (data/aero_guard_test.csv).
Dynamically calculates all benchmark metrics from genuine prediction artifacts.
Outputs results to research/metrics/benchmark_comparison.json and research/metrics/per_class_breakdown.json.
"""

import os, sys, json
from typing import Dict, List, Tuple, Optional, Any
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes threshold-independent ranking metrics (ROC-AUC, PR-AUC) and
    threshold-dependent operational metrics (precision, recall, F1, accuracy,
    specificity, false positive rate, confusion matrix counts).

    Validation:
    - Asserts 1D matching lengths.
    - Validates probability bounds [0.0, 1.0] and absence of NaNs/Infs.
    - Safely handles single-class edge cases where ROC-AUC is undefined.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_prob)={len(y_prob)}")

    if len(y_true) == 0:
        raise ValueError("Input arrays must not be empty.")

    if np.isnan(y_prob).any() or np.isinf(y_prob).any():
        raise ValueError("Predicted probabilities contain NaN or Infinite values.")

    if (y_prob < 0.0).any() or (y_prob > 1.0).any():
        raise ValueError(
            f"Predicted probabilities must be within [0.0, 1.0]. "
            f"Observed min={y_prob.min():.4e}, max={y_prob.max():.4e}"
        )

    # Check for single-class input
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        roc_val = float('nan')
        pr_val = float('nan')
    else:
        roc_val = float(roc_auc_score(y_true, y_prob))
        pr_val = float(average_precision_score(y_true, y_prob))

    preds = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    total_samples = tn + fp + fn + tp
    acc = float((tp + tn) / total_samples) if total_samples > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "roc_auc": round(roc_val, 6) if not np.isnan(roc_val) else None,
        "pr_auc": round(pr_val, 6) if not np.isnan(pr_val) else None,
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "accuracy": round(acc, 6),
        "specificity": round(spec, 6),
        "false_positive_rate": round(fpr, 6),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

ALIGNMENT_KEY_COLUMNS = ['timestamp', 'icao24', 'latitude', 'longitude', 'altitude']

# Historical unaligned reference metrics preserved strictly for audit trail
HISTORICAL_UNALIGNED_BENCHMARK_AUDIT = [
    {
        "model_name": "Random Forest (Tabular)",
        "input_structure": "13D Node (with absolute coordinates)",
        "threshold_label": "Default",
        "threshold": 0.5,
        "roc_auc": 0.552811,
        "pr_auc": 0.202934,
        "precision": 0.574242,
        "recall": 0.075709,
        "f1_score": 0.13378,
        "accuracy": 0.849217,
        "status": "INVALID_HISTORICAL_UNALIGNED_INDEX_MISMATCH",
        "note": "Evaluated against flight-ordered labels while predictions were timestamp-ordered."
    },
    {
        "model_name": "Random Forest (Tabular)",
        "input_structure": "13D Node (with absolute coordinates)",
        "threshold_label": "Calibrated",
        "threshold": 0.15,
        "roc_auc": 0.552811,
        "pr_auc": 0.202934,
        "precision": 0.426332,
        "recall": 0.081502,
        "f1_score": 0.136844,
        "accuracy": 0.841874,
        "status": "INVALID_HISTORICAL_UNALIGNED_INDEX_MISMATCH",
        "note": "Evaluated against flight-ordered labels while predictions were timestamp-ordered."
    },
    {
        "model_name": "Original GCN Baseline",
        "input_structure": "Dynamic Graph (13D Node, unweighted edges)",
        "threshold_label": "Default",
        "threshold": 0.5,
        "roc_auc": 0.580797,
        "pr_auc": 0.267119,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "accuracy": 0.846206,
        "status": "INVALID_HISTORICAL_UNALIGNED_INDEX_MISMATCH",
        "note": "Evaluated against flight-ordered labels while predictions were timestamp-ordered."
    },
    {
        "model_name": "Edge-Aware GNN (Historical Defective)",
        "input_structure": "GATv2 (12D Node, 4D Edge with double-scaling bug)",
        "threshold_label": "Default",
        "threshold": 0.5,
        "roc_auc": 0.457826,
        "pr_auc": 0.188389,
        "precision": 0.206962,
        "recall": 0.130643,
        "f1_score": 0.160176,
        "accuracy": 0.789309,
        "status": "INVALID_HISTORICAL_UNALIGNED_INDEX_MISMATCH",
        "note": "Evaluated against flight-ordered labels while predictions were timestamp-ordered."
    }
]

def align_predictions_to_ground_truth(
    df_gt: pd.DataFrame,
    df_pred: pd.DataFrame,
    prob_col: str,
    model_name: str = "Model",
    key_cols: List[str] = ALIGNMENT_KEY_COLUMNS
) -> np.ndarray:
    """
    Performs deterministic observation-level alignment between ground truth and predictions.

    Guarantees:
    1. Validates key column presence, normalizes string & float types.
    2. Detects null values or duplicate composite keys in either dataset (raises ValueError).
    3. Merges on the exact composite key, preserving the ground-truth row index.
    4. Detects unmatched records (fails clearly if len(merged) != len(df_gt)).
    5. Detects conflicting ground-truth labels if 'label' is present in df_pred (fails clearly on conflict).
    6. Sorts back to exact ground-truth row order and returns the aligned probability array.
    """
    for col in key_cols:
        if col not in df_gt.columns:
            raise KeyError(f"Ground-truth dataset missing join key column: {col}")
        if col not in df_pred.columns:
            raise KeyError(f"Prediction dataset for {model_name} missing join key column: {col}")

    # 1. Type normalization & null check
    df_gt_k = df_gt.copy()
    df_pred_k = df_pred.copy()

    for col in ['timestamp', 'icao24']:
        if col in key_cols:
            df_gt_k[col] = df_gt_k[col].astype(str).str.strip()
            df_pred_k[col] = df_pred_k[col].astype(str).str.strip()
    for col in ['latitude', 'longitude', 'altitude']:
        if col in key_cols:
            df_gt_k[col] = df_gt_k[col].astype(float)
            df_pred_k[col] = df_pred_k[col].astype(float)

    if df_gt_k[key_cols].isnull().any().any():
        raise ValueError(f"Ground-truth dataset contains null values in join keys {key_cols}")
    if df_pred_k[key_cols].isnull().any().any():
        raise ValueError(f"Prediction dataset for {model_name} contains null values in join keys {key_cols}")

    # 2. Duplicate detection
    dup_gt = int(df_gt_k.duplicated(subset=key_cols).sum())
    if dup_gt > 0:
        raise ValueError(f"Ground truth dataset contains {dup_gt} duplicate keys on {key_cols}")
    dup_pred = int(df_pred_k.duplicated(subset=key_cols).sum())
    if dup_pred > 0:
        raise ValueError(f"Prediction dataset for {model_name} contains {dup_pred} duplicate keys on {key_cols}")

    # 3. Join on exact composite key preserving df_gt index
    df_gt_k['__gt_row_id__'] = np.arange(len(df_gt_k))
    cols_to_pull = list(key_cols) + [prob_col]
    if 'label' in df_pred_k.columns:
        cols_to_pull.append('label')

    merged = pd.merge(
        df_gt_k[['__gt_row_id__', 'label'] + key_cols],
        df_pred_k[cols_to_pull],
        on=key_cols,
        suffixes=('_gt', '_pred')
    )

    # 4. Check for unmatched records
    if len(merged) != len(df_gt_k):
        unmatched = len(df_gt_k) - len(merged)
        raise ValueError(
            f"Unmatched records for {model_name}: {unmatched} ground-truth observations "
            f"were not found in predictions (expected {len(df_gt_k)}, matched {len(merged)})."
        )

    # 5. Check for label agreement if label present in predictions
    if 'label_pred' in merged.columns:
        disagreements = int((merged['label_gt'] != merged['label_pred']).sum())
        if disagreements > 0:
            raise ValueError(
                f"Label conflict for {model_name}: {disagreements} observations have conflicting ground-truth labels."
            )

    # 6. Sort back to exact ground-truth order
    merged = merged.sort_values(by='__gt_row_id__').reset_index(drop=True)
    return merged[prob_col].to_numpy(dtype=float)

def run_research_benchmark_evaluation(
    pred_path: str = "data/aero_guard_test_predictions.csv",
    st_pred_path: str = "data/st_gnn_test_predictions.csv",
    repro_path: str = "data/reproduced_st_gnn_predictions.csv",
    output_dir: str = "research/metrics",
    save_artifacts: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes the research benchmark evaluation across all baseline and proposed models.

    Uses deterministic observation-level alignment on (timestamp, icao24, latitude, longitude, altitude)
    to guarantee every predicted probability is evaluated against its genuine ground-truth observation.
    """
    print("=" * 100)
    print("AEROGUARD AI — RESEARCH BENCHMARK EVALUATION (32,550 OOD TEST SAMPLES)")
    print("=" * 100)

    if save_artifacts and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Load Ground Truth & Predictions
    assert os.path.exists(st_pred_path), (
        f"Missing ST-GNN predictions artifact: {st_pred_path}.\n"
        f"Please run 'python scripts/audit_st_gnn_metrics.py' first."
    )
    df_st = pd.read_csv(st_pred_path, low_memory=False)
    assert len(df_st) == 32550, f"Expected 32550 test samples in df_st, got {len(df_st)}"

    df_base = None
    if os.path.exists(pred_path):
        df_base = pd.read_csv(pred_path, low_memory=False)
        assert len(df_base) == 32550, f"Expected 32550 test samples in df_base, got {len(df_base)}"
    else:
        print(f"[NOTICE] Baseline predictions file not found: {pred_path}")
        print(f"         Historical baseline models (Random Forest, GCN, Historical Edge-Aware GNN) will be marked as SKIPPED.")
        print(f"         To regenerate baseline predictions from scratch, execute: python scripts/train_models.py")

    y_true = df_st['label'].values
    num_normal = int(np.sum(y_true == 0))
    num_anomaly = int(np.sum(y_true == 1))

    print(f"Dataset Size: {len(y_true)} observations")
    print(f"Class Distribution: Normal = {num_normal} ({num_normal/len(y_true)*100:.2f}%), Anomaly = {num_anomaly} ({num_anomaly/len(y_true)*100:.2f}%)")

    # 2. Inspect Provenance and Load Genuine Repaired Spatial GNN Predictions
    df_repro = None
    if os.path.exists(repro_path):
        df_repro = pd.read_csv(repro_path, low_memory=False)
        assert len(df_repro) == len(df_st), (
            f"Row count mismatch in reproduced predictions: expected {len(df_st)}, got {len(df_repro)}"
        )
        assert 'reproduced_spatial_prob' in df_repro.columns, (
            f"Missing 'reproduced_spatial_prob' column in {repro_path}"
        )
        p_sp = df_repro['reproduced_spatial_prob'].values
        assert not np.isnan(p_sp).any(), f"NaNs detected in {repro_path} spatial probabilities"
        assert (p_sp >= 0.0).all() and (p_sp <= 1.0).all(), (
            f"Spatial probabilities in {repro_path} exceed valid [0, 1] bounds"
        )
        print(f"[PROVENANCE] Loaded genuine repaired spatial predictions from {repro_path}")
        print(f"             (Verified: {len(df_repro)} observations, probabilities in [{p_sp.min():.2e}, {p_sp.max():.2e}])")
    else:
        print(f"[WARNING] Genuine reproduced predictions file not found at: {repro_path}")
        print(f"          Execute: python scripts/audit_st_gnn_metrics.py to generate genuine artifacts.")

    # 3. Define models to benchmark
    models_config = [
        {
            "name": "Random Forest (Tabular)",
            "input_type": "13D Node (with absolute coordinates)",
            "prob_col": "rf_prob",
            "df_source": df_base,
            "source_path": pred_path,
            "generation_script": "scripts/train_models.py",
            "thresholds": [("Default", 0.50), ("Calibrated", float(df_base['threshold_rf'].iloc[0]))] if df_base is not None else [("Default", 0.50)]
        },
        {
            "name": "Original GCN Baseline",
            "input_type": "Dynamic Graph (13D Node, unweighted edges)",
            "prob_col": "gnn_baseline_prob",
            "df_source": df_base,
            "source_path": pred_path,
            "generation_script": "scripts/train_models.py",
            "thresholds": [("Default", 0.50), ("Calibrated", float(df_base['threshold_gnn_baseline'].iloc[0]))] if df_base is not None else [("Default", 0.50)]
        },
        {
            "name": "Edge-Aware GNN (Historical Defective)",
            "input_type": "GATv2 (12D Node, 4D Edge with double-scaling bug)",
            "prob_col": "gnn_edge_aware_prob",
            "df_source": df_base,
            "source_path": pred_path,
            "generation_script": "scripts/train_models.py",
            "thresholds": [("Default", 0.50), ("Calibrated", float(df_base['threshold_gnn_edge_aware'].iloc[0]))] if df_base is not None else [("Default", 0.50)]
        },
        {
            "name": "Repaired Edge-Aware GNN (Spatial Stream)",
            "input_type": "GATv2 (12D Coord-free + 4D Normalized Relational)",
            "prob_col": "reproduced_spatial_prob",
            "df_source": df_repro,
            "source_path": repro_path,
            "thresholds": [("Default", 0.50)]
        },
        {
            "name": "Tri-Stream ST-GNN (Ours)",
            "input_type": "Tri-Stream Fusion (Spatial GATv2 + Temporal GRU + Kinematic MLP)",
            "prob_col": "st_gnn_prob",
            "df_source": df_st,
            "thresholds": [("Default", 0.50), ("Calibrated (Production)", 0.20)]
        }
    ]

    benchmark_results = []
    per_class_results = {}

    for m in models_config:
        m_name = m["name"]
        input_type = m["input_type"]
        df_src = m.get("df_source")

        # Handle unavailable artifacts without substituting or inventing values
        if df_src is None or m["prob_col"] not in df_src.columns:
            gen_script = m.get("generation_script", "scripts/audit_st_gnn_metrics.py")
            print(f"[UNAVAILABLE] Predictions for '{m_name}' unavailable. Missing artifact: {m.get('source_path')} (Run: python {gen_script})")
            entry = {
                "model_name": m_name,
                "input_structure": input_type,
                "threshold_label": "Unavailable",
                "threshold": None,
                "status": "UNAVAILABLE",
                "missing_artifact": m.get("source_path"),
                "generation_script": gen_script
            }
            benchmark_results.append(entry)
            continue

        # Observation-level alignment: align probabilities to df_st row-for-row on composite key
        prob_series = align_predictions_to_ground_truth(df_st, df_src, m["prob_col"], model_name=m_name)

        for t_label, t_val in m["thresholds"]:
            metrics = compute_classification_metrics(y_true, prob_series, threshold=t_val)

            entry = {
                "model_name": m_name,
                "input_structure": input_type,
                "threshold_label": t_label,
                "threshold": round(t_val, 4),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"],
                "specificity": metrics["specificity"],
                "false_positive_rate": metrics["false_positive_rate"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "observation_alignment": "COMPOSITE_KEY_ALIGNED (timestamp, icao24, latitude, longitude, altitude)"
            }
            benchmark_results.append(entry)

            # Record per-class breakdown for key models using canonical df_st anomaly_type
            if m_name in ["Random Forest (Tabular)", "Tri-Stream ST-GNN (Ours)"]:
                key = f"{m_name} ({t_label} t={t_val:.2f})"
                per_class_results[key] = {}
                preds = (prob_series >= t_val).astype(int)
                df_temp = df_st.copy()
                df_temp['pred'] = preds
                for a_type, grp in df_temp.groupby('anomaly_type'):
                    c_tp = int(grp['pred'].sum())
                    c_tot = len(grp)
                    c_rate = c_tp / c_tot if c_tot > 0 else 0.0
                    per_class_results[key][a_type] = {
                        "detected": c_tp,
                        "total": c_tot,
                        "recall": round(c_rate, 4)
                    }

    # Print Formatted Comparison Table
    print("\n" + "=" * 125)
    print(f"{'Model Architecture':<36} | {'Threshold':<14} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'Accuracy':<9}")
    print("=" * 125)
    for r in benchmark_results:
        if r.get("status") == "UNAVAILABLE":
            print(f"{r['model_name']:<36} | {'UNAVAILABLE':<14} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9}")
            continue
        t_str = f"{r['threshold_label']} ({r['threshold']:.2f})"
        roc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] is not None else "N/A"
        pr_str = f"{r['pr_auc']:.4f}" if r['pr_auc'] is not None else "N/A"
        acc_str = f"{r.get('accuracy', 0.0):.4f}"
        print(f"{r['model_name']:<36} | {t_str:<14} | {r['precision']:<9.4f} | {r['recall']:<9.4f} | {r['f1_score']:<9.4f} | {roc_str:<9} | {pr_str:<9} | {acc_str:<9}")
    print("=" * 125)

    # Save to JSON
    if save_artifacts and output_dir:
        bench_out_path = os.path.join(output_dir, "benchmark_comparison.json")
        per_class_out_path = os.path.join(output_dir, "per_class_breakdown.json")
        hist_out_path = os.path.join(output_dir, "historical_unaligned_benchmarks.json")

        # Do not overwrite audited benchmark files with incomplete/unavailable baselines if full file exists
        if df_base is None and os.path.exists(bench_out_path):
            print(f"\n[PRESERVED] {bench_out_path} preserved intact (baseline models skipped; full audited metrics retained).")
        else:
            with open(bench_out_path, "w") as f:
                json.dump(benchmark_results, f, indent=4)
            print(f"\n[SAVED] {bench_out_path}")

        if df_base is None and os.path.exists(per_class_out_path):
            print(f"[PRESERVED] {per_class_out_path} preserved intact.")
        else:
            with open(per_class_out_path, "w") as f:
                json.dump(per_class_results, f, indent=4)
            print(f"[SAVED] {per_class_out_path}")

        hist_payload = {
            "audit_description": "Historical benchmark comparison metrics computed prior to observation-level composite key alignment. These numbers were generated by an index-misalignment artifact where timestamp-sorted baseline predictions in data/aero_guard_test_predictions.csv were evaluated against flight-sorted labels in data/st_gnn_test_predictions.csv. They are preserved strictly as an audit trail and must not be presented as valid benchmark results in scientific publications.",
            "corrected_current_reference": "research/metrics/benchmark_comparison.json",
            "historical_unaligned_metrics": HISTORICAL_UNALIGNED_BENCHMARK_AUDIT
        }
        with open(hist_out_path, "w") as f:
            json.dump(hist_payload, f, indent=4)

        print(f"\n[SUCCESS] Saved benchmark artifacts to {bench_out_path}, {per_class_out_path}, and {hist_out_path}")

    return benchmark_results, per_class_results

if __name__ == "__main__":
    run_research_benchmark_evaluation()
