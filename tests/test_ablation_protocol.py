"""
Tests for AeroGuard AI Dynamic Multi-Stream Ablation Protocol
=============================================================
Verifies:
1. Fusion formulas: Average, Max-Envelope, and Official Weighted (0.40/0.35/0.25).
2. Input validation: missing stream columns, empty DataFrames, invalid probabilities (NaN/Inf/<0/>1).
3. Ground-truth label validation: non-binary labels.
4. Alignment verification: test CSV vs prediction CSV row counts and label alignment.
5. Per-attack metrics: standard calculation, sample counts, and single-class edge cases (returning None).
6. Missing anomaly_type column handling.
7. Genuine prediction artifact validation (32,550 rows, valid bounds, label match).
8. End-to-end ablation protocol execution and metric verification against audited benchmarks.
"""

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

from research.ablation_protocol import (
    validate_ablation_inputs,
    compute_fusion_probabilities,
    compute_per_attack_metrics,
    run_ablation_audit,
    HISTORICAL_ABLATION_REFERENCE,
    ANOMALY_CATEGORIES
)


class TestAblationProtocol(unittest.TestCase):

    def test_fusion_formulas(self):
        """Verify mathematical correctness of all fusion strategies."""
        ps = np.array([0.2, 0.6, 0.9])
        pt = np.array([0.4, 0.5, 0.1])
        pk = np.array([0.6, 0.1, 0.5])

        fused = compute_fusion_probabilities(ps, pt, pk)

        # 1. Unweighted Average: (ps + pt + pk) / 3
        expected_avg = np.array([(0.2 + 0.4 + 0.6) / 3.0, (0.6 + 0.5 + 0.1) / 3.0, (0.9 + 0.1 + 0.5) / 3.0])
        np.testing.assert_allclose(fused["average_fusion"], expected_avg, rtol=1e-5)

        # 2. Max Envelope: max(ps, pt, pk)
        expected_max = np.array([0.6, 0.6, 0.9])
        np.testing.assert_allclose(fused["max_envelope_fusion"], expected_max, rtol=1e-5)

        # 3. Official Weighted ST-GNN: 0.40 * ps + 0.35 * pt + 0.25 * pk
        expected_weighted = 0.40 * ps + 0.35 * pt + 0.25 * pk
        np.testing.assert_allclose(fused["official_weighted_fusion"], expected_weighted, rtol=1e-5)

        # 4. Stream identities
        np.testing.assert_allclose(fused["spatial_only"], ps)
        np.testing.assert_allclose(fused["temporal_only"], pt)
        np.testing.assert_allclose(fused["kinematic_only"], pk)

    def test_fusion_length_mismatch(self):
        """Verify ValueError is raised if input stream lengths mismatch."""
        ps = np.array([0.2, 0.6])
        pt = np.array([0.4, 0.5, 0.1])
        pk = np.array([0.6, 0.1])
        with self.assertRaises(ValueError):
            compute_fusion_probabilities(ps, pt, pk)

    def test_validate_missing_columns(self):
        """Verify ValueError is raised if any stream column is missing."""
        df = pd.DataFrame({
            "reproduced_spatial_prob": [0.5, 0.2],
            "reproduced_temporal_prob": [0.4, 0.3],
            "label": [0, 1]
            # missing reproduced_kinematic_prob
        })
        with self.assertRaises(ValueError) as ctx:
            validate_ablation_inputs(df)
        self.assertIn("reproduced_kinematic_prob", str(ctx.exception))

    def test_validate_empty_dataframe(self):
        """Verify ValueError is raised on empty DataFrame."""
        df = pd.DataFrame(columns=[
            "reproduced_spatial_prob", "reproduced_temporal_prob",
            "reproduced_kinematic_prob", "label"
        ])
        with self.assertRaises(ValueError):
            validate_ablation_inputs(df)

    def test_validate_invalid_probabilities(self):
        """Verify ValueError on NaN, Inf, or out-of-range probabilities."""
        base_df = {
            "reproduced_spatial_prob": [0.5, 0.2],
            "reproduced_temporal_prob": [0.4, 0.3],
            "reproduced_kinematic_prob": [0.1, 0.8],
            "label": [0, 1]
        }

        # Value > 1.0
        df_high = pd.DataFrame(base_df)
        df_high.loc[0, "reproduced_spatial_prob"] = 1.05
        with self.assertRaises(ValueError):
            validate_ablation_inputs(df_high)

        # Value < 0.0
        df_low = pd.DataFrame(base_df)
        df_low.loc[0, "reproduced_temporal_prob"] = -0.01
        with self.assertRaises(ValueError):
            validate_ablation_inputs(df_low)

        # NaN
        df_nan = pd.DataFrame(base_df)
        df_nan.loc[0, "reproduced_kinematic_prob"] = np.nan
        with self.assertRaises(ValueError):
            validate_ablation_inputs(df_nan)

        # Inf
        df_inf = pd.DataFrame(base_df)
        df_inf.loc[0, "reproduced_spatial_prob"] = np.inf
        with self.assertRaises(ValueError):
            validate_ablation_inputs(df_inf)

    def test_validate_invalid_labels(self):
        """Verify ValueError on non-binary labels."""
        df = pd.DataFrame({
            "reproduced_spatial_prob": [0.5, 0.2],
            "reproduced_temporal_prob": [0.4, 0.3],
            "reproduced_kinematic_prob": [0.1, 0.8],
            "label": [0, 2]  # 2 is invalid
        })
        with self.assertRaises(ValueError):
            validate_ablation_inputs(df)

    def test_label_and_row_misalignment(self):
        """Verify run_ablation_audit detects row count or label misalignment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_file = os.path.join(tmpdir, "preds.csv")
            test_file = os.path.join(tmpdir, "test.csv")

            # Mismatched lengths
            df_pred = pd.DataFrame({
                "reproduced_spatial_prob": [0.1, 0.2, 0.8],
                "reproduced_temporal_prob": [0.1, 0.2, 0.7],
                "reproduced_kinematic_prob": [0.1, 0.2, 0.9],
                "label": [0, 0, 1]
            })
            df_test_mismatch = pd.DataFrame({
                "label": [0, 1]  # 2 rows vs 3 rows
            })

            df_pred.to_csv(pred_file, index=False)
            df_test_mismatch.to_csv(test_file, index=False)

            with self.assertRaises(ValueError) as ctx:
                run_ablation_audit(repro_path=pred_file, aero_test_path=test_file, save_artifacts=False)
            self.assertIn("Row count mismatch", str(ctx.exception))

            # Mismatched labels with same length
            df_test_labels = pd.DataFrame({
                "label": [1, 0, 0]  # different labels
            })
            df_test_labels.to_csv(test_file, index=False)
            with self.assertRaises(ValueError) as ctx:
                run_ablation_audit(repro_path=pred_file, aero_test_path=test_file, save_artifacts=False)
            self.assertIn("Ground-truth label misalignment", str(ctx.exception))

    def test_per_attack_metrics_standard(self):
        """Verify per-attack ROC-AUC and sample counts on synthetic data."""
        df = pd.DataFrame({
            "anomaly_type": ["normal", "normal", "altitude_anomaly", "altitude_anomaly", "ghost_aircraft"],
            "label": [0, 0, 1, 1, 1]
        })
        # Perfect discrimination for altitude_anomaly vs normal
        # normal: [0.1, 0.2], altitude: [0.8, 0.9] -> ROC-AUC = 1.0
        # ghost_aircraft: [0.15] -> normal: [0.1, 0.2], ghost: [0.15] -> ROC-AUC = 0.5
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15])

        per_class_roc, counts = compute_per_attack_metrics(df, y_prob, categories=["altitude_anomaly", "ghost_aircraft"])

        self.assertEqual(counts["altitude_anomaly"], 2)
        self.assertEqual(counts["ghost_aircraft"], 1)
        self.assertEqual(per_class_roc["altitude_anomaly"], 1.0)
        self.assertEqual(per_class_roc["ghost_aircraft"], 0.5)

    def test_per_attack_metrics_single_class_edge_cases(self):
        """Verify categories with 0 positive or 0 negative samples return None, not 0.0."""
        df = pd.DataFrame({
            "anomaly_type": ["normal", "normal", "normal"],
            "label": [0, 0, 0]
        })
        y_prob = np.array([0.1, 0.2, 0.3])

        per_class_roc, counts = compute_per_attack_metrics(df, y_prob, categories=["position_jump"])
        self.assertEqual(counts["position_jump"], 0)
        self.assertIsNone(per_class_roc["position_jump"])

        # Missing anomaly_type column returns None for all categories
        df_no_type = pd.DataFrame({"label": [0, 1]})
        rocs, cnts = compute_per_attack_metrics(df_no_type, np.array([0.2, 0.8]))
        for cat in ANOMALY_CATEGORIES:
            self.assertIsNone(rocs[cat])
            self.assertEqual(cnts[cat], 0)

    @unittest.skipUnless(
        os.path.exists("data/reproduced_st_gnn_predictions.csv"),
        "Prediction artifact data/reproduced_st_gnn_predictions.csv not found. Run scripts/audit_st_gnn_metrics.py first."
    )
    def test_genuine_prediction_artifact(self):
        """Verify genuine prediction artifact data/reproduced_st_gnn_predictions.csv."""
        pred_path = "data/reproduced_st_gnn_predictions.csv"
        self.assertTrue(os.path.exists(pred_path), f"Missing artifact: {pred_path}")

        df = pd.read_csv(pred_path, low_memory=False)
        self.assertEqual(len(df), 32550)

        # Check required columns
        for col in ["reproduced_spatial_prob", "reproduced_temporal_prob", "reproduced_kinematic_prob", "label"]:
            self.assertIn(col, df.columns)

        # Check ranges
        for col in ["reproduced_spatial_prob", "reproduced_temporal_prob", "reproduced_kinematic_prob"]:
            vals = df[col].to_numpy(dtype=float)
            self.assertFalse(np.isnan(vals).any())
            self.assertFalse(np.isinf(vals).any())
            self.assertGreaterEqual(vals.min(), 0.0)
            self.assertLessEqual(vals.max(), 1.0)

        # Check class distribution: 27,544 normal, 5,006 anomaly
        self.assertEqual(int((df["label"] == 0).sum()), 27544)
        self.assertEqual(int((df["label"] == 1).sum()), 5006)

    @unittest.skipUnless(
        os.path.exists("data/reproduced_st_gnn_predictions.csv") and os.path.exists("data/aero_guard_test.csv"),
        "Prediction artifact data/reproduced_st_gnn_predictions.csv or test dataset not found."
    )
    def test_run_ablation_audit_end_to_end(self):
        """Execute run_ablation_audit end-to-end and check audited values."""
        results = run_ablation_audit(
            repro_path="data/reproduced_st_gnn_predictions.csv",
            aero_test_path="data/aero_guard_test.csv",
            save_artifacts=False
        )

        self.assertEqual(len(results), 6)
        strat_map = {r["strategy_id"]: r for r in results}

        # 1. Spatial-Only: threshold 0.50, ROC ~ 0.7635, PR ~ 0.4347
        sp = strat_map["spatial_only"]
        self.assertEqual(sp["threshold"], 0.50)
        self.assertAlmostEqual(sp["precision"], 0.8772, places=3)
        self.assertAlmostEqual(sp["recall"], 0.1484, places=3)
        self.assertAlmostEqual(sp["f1_score"], 0.2539, places=3)
        self.assertAlmostEqual(sp["roc_auc"], 0.7635, places=3)
        self.assertAlmostEqual(sp["pr_auc"], 0.4347, places=3)
        self.assertAlmostEqual(sp["per_class_roc"]["altitude_anomaly"], 1.0000, places=3)
        self.assertAlmostEqual(sp["per_class_roc"]["ghost_aircraft"], 0.8422, places=3)

        # 2. Temporal-Only: threshold 0.50, ROC ~ 0.3333
        temp = strat_map["temporal_only"]
        self.assertEqual(temp["threshold"], 0.50)
        self.assertAlmostEqual(temp["roc_auc"], 0.3333, places=3)
        self.assertAlmostEqual(temp["f1_score"], 0.2527, places=3)

        # 3. Kinematic-Only: threshold 0.50, ROC in documented [0.51, 0.56] range due to scaler sensitivity
        kin = strat_map["kinematic_only"]
        self.assertEqual(kin["threshold"], 0.50)
        self.assertGreaterEqual(kin["roc_auc"], 0.510)
        self.assertLessEqual(kin["roc_auc"], 0.560)
        self.assertGreaterEqual(kin["f1_score"], 0.120)
        self.assertLessEqual(kin["f1_score"], 0.170)
        self.assertGreater(kin["per_class_roc"]["position_jump"], 0.950)
        self.assertGreater(kin["per_class_roc"]["identity_inconsistency"], 0.980)

        # 4. Official Weighted ST-GNN: threshold 0.20, F1 ~ 0.289, Precision ~ 0.73-0.75
        off = strat_map["official_weighted_fusion"]
        self.assertEqual(off["threshold"], 0.20)
        self.assertAlmostEqual(off["f1_score"], 0.289, places=2)
        self.assertGreaterEqual(off["precision"], 0.720)
        self.assertLessEqual(off["precision"], 0.760)
        self.assertAlmostEqual(off["recall"], 0.180, places=2)
        self.assertAlmostEqual(off["accuracy"], 0.864, places=2)
        self.assertGreater(off["roc_auc"], 0.580)

        # Verify historical reference and confusion matrix presence
        for r in results:
            self.assertIn("historical_reference", r)
            self.assertIn("confusion_matrix", r)
            self.assertIn("per_class_roc", r)
            self.assertIn("per_class_sample_counts", r)

    def test_kinematic_scaler_near_zero_variance_audit(self):
        """
        Audit models/scaler_kinematic.joblib to detect near-zero-variance features.

        Confirms that accel_residual and turn_residual have scale_ < 1e-13 in the
        official frozen scaler, which explains the empirical ~0.027 ROC variation.
        """
        import joblib
        scaler_path = "models/scaler_kinematic.joblib"
        self.assertTrue(os.path.exists(scaler_path), f"Missing {scaler_path}")
        scaler = joblib.load(scaler_path)

        from scripts.train_st_gnn import KINEMATIC_BRANCH_FEATS
        idx_accel = KINEMATIC_BRANCH_FEATS.index("accel_residual")
        idx_turn = KINEMATIC_BRANCH_FEATS.index("turn_residual")

        # Verify confirmed root cause: scales are near machine epsilon (< 1e-13)
        self.assertLess(scaler.scale_[idx_accel], 1e-13)
        self.assertLess(scaler.scale_[idx_turn], 1e-13)

        # Other primary features must have valid, healthy non-zero scale
        self.assertGreater(scaler.scale_[KINEMATIC_BRANCH_FEATS.index("disp_ratio")], 1.0)
        self.assertGreater(scaler.scale_[KINEMATIC_BRANCH_FEATS.index("alt_residual")], 1.0)

    def test_kinematic_inference_deterministic_repeatability(self):
        """Verify that kinematic stream inference is 100% deterministic on identical input."""
        import torch
        from scripts.st_gnn_model import AeroGuardSTGNN
        from scripts.train_edge_aware_repaired import RepairedEdgeAwareGNN

        spatial_model = RepairedEdgeAwareGNN(12, 4)
        st_gnn = AeroGuardSTGNN(spatial_model=spatial_model, num_kinematic_temporal_feats=9, num_kinematic_branch_feats=11)
        st_gnn.eval()

        x = torch.randn(20, 11)
        with torch.no_grad():
            _, out1 = st_gnn.kinematic_stream(x)
            _, out2 = st_gnn.kinematic_stream(x)

        diff = (out1 - out2).abs().max().item()
        self.assertEqual(diff, 0.0)


if __name__ == "__main__":
    unittest.main()
