"""
Tests for AeroGuard AI Benchmark Evaluation Module
=================================================
Verifies:
1. compute_classification_metrics: correctness against known ground truth.
2. compute_classification_metrics: raises ValueError on invalid probabilities (NaN, Inf, <0, >1).
3. compute_classification_metrics: raises ValueError on length mismatches.
4. compute_classification_metrics: safely handles single-class edge cases.
5. Spatial prediction provenance: row count, label match, probability ranges in reproduced_st_gnn_predictions.csv.
6. Missing prediction files: handles missing artifacts cleanly without inventing values.
7. End-to-end benchmark evaluation: dynamically computes metrics matching audited references.
"""

import os, sys, tempfile, json
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import numpy as np
import pandas as pd

from research.evaluate_benchmarks import (
    compute_classification_metrics,
    align_predictions_to_ground_truth,
    run_research_benchmark_evaluation,
    ALIGNMENT_KEY_COLUMNS
)

class TestEvaluateBenchmarks(unittest.TestCase):

    def test_compute_metrics_correctness(self):
        """Verify metric calculation against known confusion matrix values."""
        # 4 normal, 4 anomaly
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
        # At threshold 0.5:
        # y_true 0: 0.1(0), 0.2(0), 0.3(0) -> TN=3; 0.6(1) -> FP=1
        # y_true 1: 0.4(0) -> FN=1; 0.7(1), 0.8(1), 0.9(1) -> TP=3
        metrics = compute_classification_metrics(y_true, y_prob, threshold=0.50)

        self.assertEqual(metrics["tn"], 3)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tp"], 3)
        self.assertAlmostEqual(metrics["precision"], 3 / 4, places=5)
        self.assertAlmostEqual(metrics["recall"], 3 / 4, places=5)
        self.assertAlmostEqual(metrics["f1_score"], 3 / 4, places=5)
        self.assertAlmostEqual(metrics["accuracy"], 6 / 8, places=5)
        self.assertAlmostEqual(metrics["specificity"], 3 / 4, places=5)
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 4, places=5)
        self.assertIsNotNone(metrics["roc_auc"])
        self.assertIsNotNone(metrics["pr_auc"])

    def test_compute_metrics_invalid_probabilities(self):
        """Verify ValueError is raised on out-of-bound probabilities or NaNs/Infs."""
        y_true = np.array([0, 1, 0, 1])

        # Negative probability
        with self.assertRaises(ValueError):
            compute_classification_metrics(y_true, np.array([-0.1, 0.5, 0.2, 0.8]))

        # Probability > 1.0
        with self.assertRaises(ValueError):
            compute_classification_metrics(y_true, np.array([0.1, 1.05, 0.2, 0.8]))

        # NaN probability
        with self.assertRaises(ValueError):
            compute_classification_metrics(y_true, np.array([0.1, np.nan, 0.2, 0.8]))

        # Inf probability
        with self.assertRaises(ValueError):
            compute_classification_metrics(y_true, np.array([0.1, np.inf, 0.2, 0.8]))

    def test_compute_metrics_length_mismatch(self):
        """Verify ValueError is raised when y_true and y_prob lengths differ."""
        y_true = np.array([0, 1, 0])
        y_prob = np.array([0.2, 0.8])
        with self.assertRaises(ValueError):
            compute_classification_metrics(y_true, y_prob)

    def test_compute_metrics_empty_array(self):
        """Verify ValueError is raised on empty input arrays."""
        with self.assertRaises(ValueError):
            compute_classification_metrics(np.array([]), np.array([]))

    def test_compute_metrics_single_class(self):
        """Verify single-class inputs safely return None/NaN for ROC-AUC and PR-AUC."""
        y_true_all_zeros = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = compute_classification_metrics(y_true_all_zeros, y_prob, threshold=0.25)

        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["pr_auc"])
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fp"], 2)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tp"], 0)
        self.assertAlmostEqual(metrics["accuracy"], 0.5, places=5)

    @unittest.skipUnless(
        os.path.exists("data/reproduced_st_gnn_predictions.csv") and os.path.exists("data/st_gnn_test_predictions.csv"),
        "Prediction artifacts not found. Run scripts/audit_st_gnn_metrics.py first."
    )
    def test_reproduced_spatial_provenance_and_schema(self):
        """Verify genuine spatial predictions in data/reproduced_st_gnn_predictions.csv."""
        repro_path = "data/reproduced_st_gnn_predictions.csv"
        st_pred_path = "data/st_gnn_test_predictions.csv"

        self.assertTrue(os.path.exists(repro_path), f"Missing artifact: {repro_path}")
        self.assertTrue(os.path.exists(st_pred_path), f"Missing artifact: {st_pred_path}")

        df_repro = pd.read_csv(repro_path, low_memory=False)
        df_st = pd.read_csv(st_pred_path, low_memory=False)

        self.assertEqual(len(df_repro), 32550)
        self.assertEqual(len(df_st), 32550)
        self.assertIn('reproduced_spatial_prob', df_repro.columns)

        # Confirm 1-to-1 observation alignment
        self.assertTrue((df_repro['timestamp'] == df_st['timestamp']).all())
        self.assertTrue((df_repro['icao24'].astype(str) == df_st['icao24'].astype(str)).all())
        self.assertTrue((df_repro['label'] == df_st['label']).all())

        p_sp = df_repro['reproduced_spatial_prob'].values
        self.assertFalse(np.isnan(p_sp).any())
        self.assertFalse(np.isinf(p_sp).any())
        self.assertTrue((p_sp >= 0.0).all() and (p_sp <= 1.0).all())

    @unittest.skipUnless(
        os.path.exists("data/aero_guard_test_predictions.csv") and
        os.path.exists("data/st_gnn_test_predictions.csv") and
        os.path.exists("data/reproduced_st_gnn_predictions.csv"),
        "Benchmark prediction artifacts not found."
    )
    def test_evaluate_benchmarks_dynamic_repaired_spatial(self):
        """Verify dynamic calculation of Repaired Spatial GNN matches audited reference metrics."""
        results, per_class = run_research_benchmark_evaluation(save_artifacts=False)

        # Find Repaired Spatial GNN entry
        sp_entries = [r for r in results if r["model_name"] == "Repaired Edge-Aware GNN (Spatial Stream)"]
        self.assertEqual(len(sp_entries), 1)
        sp = sp_entries[0]

        # Verify reference metrics from implementation report
        self.assertAlmostEqual(sp["roc_auc"], 0.763501, places=4)
        self.assertAlmostEqual(sp["pr_auc"], 0.434653, places=4)
        self.assertAlmostEqual(sp["precision"], 0.877214, places=4)
        self.assertAlmostEqual(sp["recall"], 0.148422, places=4)
        self.assertAlmostEqual(sp["f1_score"], 0.253887, places=4)
        self.assertEqual(sp["tn"], 27440)
        self.assertEqual(sp["fp"], 104)
        self.assertEqual(sp["fn"], 4263)
        self.assertEqual(sp["tp"], 743)
        self.assertAlmostEqual(sp["specificity"], 0.996224, places=4)
        self.assertAlmostEqual(sp["false_positive_rate"], 0.003776, places=4)

    @unittest.skipUnless(
        os.path.exists("data/aero_guard_test_predictions.csv") and
        os.path.exists("data/st_gnn_test_predictions.csv"),
        "Base and ST-GNN prediction artifacts not found."
    )
    def test_evaluate_benchmarks_st_gnn_reference_metrics(self):
        """Verify dynamically computed ST-GNN metrics match audited reference values."""
        results, _ = run_research_benchmark_evaluation(save_artifacts=False)

        # ST-GNN calibrated at t=0.20
        st_cal = [
            r for r in results
            if r["model_name"] == "Tri-Stream ST-GNN (Ours)" and r["threshold_label"] == "Calibrated (Production)"
        ]
        self.assertEqual(len(st_cal), 1)
        st = st_cal[0]

        # Audited reference values:
        # ROC-AUC 0.623470, PR-AUC 0.372047, precision 0.730583, recall 0.180384, F1 0.289330
        self.assertAlmostEqual(st["roc_auc"], 0.623470, places=5)
        self.assertAlmostEqual(st["pr_auc"], 0.372047, places=5)
        self.assertAlmostEqual(st["precision"], 0.730583, places=5)
        self.assertAlmostEqual(st["recall"], 0.180384, places=5)
        self.assertAlmostEqual(st["f1_score"], 0.289330, places=5)
        self.assertEqual(st["tn"], 27211)
        self.assertEqual(st["fp"], 333)
        self.assertEqual(st["fn"], 4103)
        self.assertEqual(st["tp"], 903)

    @unittest.skipUnless(
        os.path.exists("data/aero_guard_test_predictions.csv") and
        os.path.exists("data/st_gnn_test_predictions.csv"),
        "Base and ST-GNN prediction artifacts not found."
    )
    def test_missing_spatial_file_handling(self):
        """Verify graceful handling when reproduced predictions file does not exist."""
        results, _ = run_research_benchmark_evaluation(
            repro_path="data/non_existent_predictions_file.csv",
            save_artifacts=False
        )

        sp_entries = [r for r in results if r["model_name"] == "Repaired Edge-Aware GNN (Spatial Stream)"]
        self.assertEqual(len(sp_entries), 1)
        self.assertEqual(sp_entries[0]["status"], "UNAVAILABLE")
        self.assertIn("non_existent_predictions_file.csv", sp_entries[0]["missing_artifact"])

    def test_align_predictions_shuffled_rows(self):
        """Verify that alignment function is immune to row order / shuffling."""
        # Create synthetic ground truth
        df_gt = pd.DataFrame({
            "timestamp": ["2020-01-02 10:00:00", "2020-01-02 10:00:01", "2020-01-02 10:00:02"],
            "icao24": ["a00001", "a00002", "a00003"],
            "latitude": [38.70, 38.71, 38.72],
            "longitude": [-9.10, -9.11, -9.12],
            "altitude": [1000.0, 2000.0, 3000.0],
            "label": [0, 1, 0]
        })

        # Create predictions in reverse order with distinct probabilities
        df_pred = pd.DataFrame({
            "timestamp": ["2020-01-02 10:00:02", "2020-01-02 10:00:00", "2020-01-02 10:00:01"],
            "icao24": ["a00003", "a00001", "a00002"],
            "latitude": [38.72, 38.70, 38.71],
            "longitude": [-9.12, -9.10, -9.11],
            "altitude": [3000.0, 1000.0, 2000.0],
            "label": [0, 0, 1],
            "model_prob": [0.88, 0.11, 0.55]
        })

        aligned_probs = align_predictions_to_ground_truth(df_gt, df_pred, "model_prob", "TestModel")

        # In df_gt order: row 0 (a00001) should get 0.11, row 1 (a00002) should get 0.55, row 2 (a00003) should get 0.88
        np.testing.assert_allclose(aligned_probs, np.array([0.11, 0.55, 0.88]), rtol=1e-6)

    def test_align_predictions_duplicate_key_error(self):
        """Verify that duplicate keys in predictions raise a clear ValueError."""
        df_gt = pd.DataFrame({
            "timestamp": ["2020-01-02 10:00:00", "2020-01-02 10:00:01"],
            "icao24": ["a00001", "a00002"],
            "latitude": [38.70, 38.71],
            "longitude": [-9.10, -9.11],
            "altitude": [1000.0, 2000.0],
            "label": [0, 1]
        })

        # Inject duplicate composite key
        df_pred_dup = pd.DataFrame({
            "timestamp": ["2020-01-02 10:00:00", "2020-01-02 10:00:00"],
            "icao24": ["a00001", "a00001"],
            "latitude": [38.70, 38.70],
            "longitude": [-9.10, -9.10],
            "altitude": [1000.0, 1000.0],
            "prob": [0.2, 0.3]
        })

        with self.assertRaises(ValueError) as ctx:
            align_predictions_to_ground_truth(df_gt, df_pred_dup, "prob", "DupModel")
        self.assertIn("duplicate keys", str(ctx.exception).lower())

    def test_align_predictions_missing_observation_error(self):
        """Verify that missing observations raise a clear ValueError."""
        df_gt = pd.DataFrame({
            "timestamp": ["2020-01-02 10:00:00", "2020-01-02 10:00:01"],
            "icao24": ["a00001", "a00002"],
            "latitude": [38.70, 38.71],
            "longitude": [-9.10, -9.11],
            "altitude": [1000.0, 2000.0],
            "label": [0, 1]
        })

        # Only 1 observation present
        df_pred_missing = pd.DataFrame({
            "timestamp": ["2020-01-02 10:00:00"],
            "icao24": ["a00001"],
            "latitude": [38.70],
            "longitude": [-9.10],
            "altitude": [1000.0],
            "prob": [0.2]
        })

        with self.assertRaises(ValueError) as ctx:
            align_predictions_to_ground_truth(df_gt, df_pred_missing, "prob", "MissingModel")
        self.assertIn("unmatched records", str(ctx.exception).lower())

    def test_align_predictions_conflicting_label_error(self):
        """Verify that conflicting ground-truth labels raise a clear ValueError."""
        df_gt = pd.DataFrame({
            "timestamp": ["2020-01-02 10:00:00"],
            "icao24": ["a00001"],
            "latitude": [38.70],
            "longitude": [-9.10],
            "altitude": [1000.0],
            "label": [0]
        })

        # Conflicting label 1 in predictions
        df_pred_conflict = pd.DataFrame({
            "timestamp": ["2020-01-02 10:00:00"],
            "icao24": ["a00001"],
            "latitude": [38.70],
            "longitude": [-9.10],
            "altitude": [1000.0],
            "label": [1],
            "prob": [0.2]
        })

        with self.assertRaises(ValueError) as ctx:
            align_predictions_to_ground_truth(df_gt, df_pred_conflict, "prob", "ConflictModel")
        self.assertIn("label conflict", str(ctx.exception).lower())

    @unittest.skipUnless(
        os.path.exists("data/aero_guard_test_predictions.csv") and
        os.path.exists("data/st_gnn_test_predictions.csv"),
        "Prediction artifacts not found."
    )
    def test_aligned_random_forest_metrics(self):
        """Verify that observation-aligned Random Forest achieves genuine verified metrics."""
        results, per_class = run_research_benchmark_evaluation(save_artifacts=False)

        rf_entries = [
            r for r in results
            if r["model_name"] == "Random Forest (Tabular)" and r["threshold_label"] == "Default"
        ]
        self.assertEqual(len(rf_entries), 1)
        rf = rf_entries[0]

        # Aligned verification: Precision 1.0, Recall 0.131842, F1 0.232969, ROC-AUC 0.804715, PR-AUC 0.410722
        self.assertAlmostEqual(rf["precision"], 1.0000, places=4)
        self.assertAlmostEqual(rf["recall"], 0.131842, places=4)
        self.assertAlmostEqual(rf["f1_score"], 0.232969, places=4)
        self.assertAlmostEqual(rf["roc_auc"], 0.804715, places=4)
        self.assertAlmostEqual(rf["pr_auc"], 0.410722, places=4)
        self.assertEqual(rf["tn"], 27544)
        self.assertEqual(rf["fp"], 0)
        self.assertEqual(rf["fn"], 4346)
        self.assertEqual(rf["tp"], 660)

        # Check per-class breakdown: RF detected all 660 altitude anomalies and 0 others
        rf_pc = per_class["Random Forest (Tabular) (Default t=0.50)"]
        self.assertEqual(rf_pc["altitude_anomaly"]["detected"], 660)
        self.assertEqual(rf_pc["altitude_anomaly"]["recall"], 1.0)
        self.assertEqual(rf_pc["ghost_aircraft"]["detected"], 0)
        self.assertEqual(rf_pc["identity_inconsistency"]["detected"], 0)
        self.assertEqual(rf_pc["position_jump"]["detected"], 0)

if __name__ == "__main__":
    unittest.main()
