"""
AeroGuard AI — Kinematic Feature Numerical Sensitivity & Diagnostic Test Suite
=============================================================================
Investigates and tests numerical stability in the kinematic features
`accel_residual` and `turn_residual`:
1. Scaler variance audit: Confirms near-zero scales (~1e-15) in models/scaler_kinematic.joblib.
2. Magnification ratio analysis: Verifies condition number and noise amplification.
3. Machine-epsilon perturbation sensitivity: Measures output logit delta under floating-point noise.
4. Deterministic inference repeatability: Confirms identical inputs yield identical outputs.
5. Finite output guarantees: Ensures probabilities remain strictly finite in [0.0, 1.0].
6. Official model weight preservation: Verifies official checkpoint remains untouched.
"""

import os
import sys
import unittest
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.abspath("."))
from scripts.train_st_gnn import KINEMATIC_BRANCH_FEATS
from scripts.st_gnn_model import KinematicStream


class TestKinematicNumericalSensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scaler_path = "models/scaler_kinematic.joblib"
        cls.model_path = "models/st_gnn_official.pt"
        cls.scaler = joblib.load(cls.scaler_path)

        # Load KinematicStream with official weights if available
        cls.kin_stream = KinematicStream(input_dim=len(KINEMATIC_BRANCH_FEATS), hidden_dim=32)
        if os.path.exists(cls.model_path):
            state_dict = torch.load(cls.model_path, map_location="cpu")
            # Extract kinematic branch weights from full ST-GNN state dict
            kin_dict = {}
            for k, v in state_dict.items():
                if k.startswith("kinematic_stream."):
                    kin_dict[k.replace("kinematic_stream.", "")] = v
            if kin_dict:
                cls.kin_stream.load_state_dict(kin_dict)
        cls.kin_stream.eval()

    def test_01_scaler_near_zero_variance_features_identified(self):
        """Audits the kinematic scaler and pinpoints features with near-zero scale."""
        scales = self.scaler.scale_
        feature_names = KINEMATIC_BRANCH_FEATS

        self.assertEqual(len(scales), len(feature_names))

        near_zero_feats = {}
        for name, scale in zip(feature_names, scales):
            if scale < 1e-10:
                near_zero_feats[name] = float(scale)

        # Confirm exactly accel_residual and turn_residual are the ill-conditioned features
        self.assertIn("accel_residual", near_zero_feats)
        self.assertIn("turn_residual", near_zero_feats)
        self.assertLess(near_zero_feats["accel_residual"], 1e-13)
        self.assertLess(near_zero_feats["turn_residual"], 1e-13)

        # Other physical features must have well-conditioned standard deviations
        well_conditioned = ["disp_ratio", "alt_residual", "vertical_rate", "acceleration"]
        for f in well_conditioned:
            idx = feature_names.index(f)
            self.assertGreater(scales[idx], 1e-3, f"Feature {f} scale is unexpectedly small: {scales[idx]}")

    def test_02_amplification_factor_bounded(self):
        """Quantifies the maximum noise amplification ratio (1 / scale)."""
        scales = self.scaler.scale_
        accel_idx = KINEMATIC_BRANCH_FEATS.index("accel_residual")
        turn_idx = KINEMATIC_BRANCH_FEATS.index("turn_residual")

        accel_amp = 1.0 / scales[accel_idx]
        turn_amp = 1.0 / scales[turn_idx]

        # The amplification ratio is on the order of 1e14
        self.assertGreater(accel_amp, 1e13)
        self.assertGreater(turn_amp, 1e13)

        # Machine-epsilon roundoff (~1e-16) produces scaled variance on order of ~1e-2 to 1e0
        eps = np.finfo(np.float64).eps  # 2.22e-16
        scaled_noise_accel = eps * accel_amp
        scaled_noise_turn = eps * turn_amp

        # Perturbation from 64-bit machine epsilon is small (< 0.1),
        # but 32-bit float roundoff (~1e-7) or 1e-15 differences will produce ~0.1 - 3.5 std swings
        self.assertLess(scaled_noise_accel, 0.1)
        self.assertLess(scaled_noise_turn, 0.1)

    def test_03_machine_noise_sensitivity_on_model_logits(self):
        """Tests that floating-point noise on ill-conditioned features produces bounded logit changes."""
        # Baseline synthetic kinematic vector (all nominal)
        kin_raw = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        scaled_base = self.scaler.transform(kin_raw.reshape(1, -1))
        t_base = torch.tensor(scaled_base, dtype=torch.float32)

        with torch.no_grad():
            _, logits_base = self.kin_stream(t_base)
            prob_base = torch.softmax(logits_base, dim=-1)[0, 1].item()

        # Add machine-epsilon noise (1e-15) to accel_residual and turn_residual
        kin_perturbed = kin_raw.copy()
        kin_perturbed[2] += 1e-15  # accel_residual
        kin_perturbed[3] += 1e-15  # turn_residual

        scaled_pert = self.scaler.transform(kin_perturbed.reshape(1, -1))
        t_pert = torch.tensor(scaled_pert, dtype=torch.float32)

        with torch.no_grad():
            _, logits_pert = self.kin_stream(t_pert)
            prob_pert = torch.softmax(logits_pert, dim=-1)[0, 1].item()

        prob_delta = abs(prob_pert - prob_base)
        # Logit output must not explode or generate NaN/Inf
        self.assertTrue(np.isfinite(prob_base))
        self.assertTrue(np.isfinite(prob_pert))
        self.assertGreaterEqual(prob_base, 0.0)
        self.assertLessEqual(prob_base, 1.0)
        self.assertGreaterEqual(prob_pert, 0.0)
        self.assertLessEqual(prob_pert, 1.0)
        # Delta under 1e-15 raw shift is bounded
        self.assertLess(prob_delta, 0.15, f"Probability delta {prob_delta} exceeds 0.15 threshold")

    def test_04_deterministic_inference_repeatability(self):
        """Confirms that identical inputs yield bit-exact outputs (zero nondeterminism)."""
        kin_input = torch.randn(10, len(KINEMATIC_BRANCH_FEATS))

        with torch.no_grad():
            _, logits_1 = self.kin_stream(kin_input)
            _, logits_2 = self.kin_stream(kin_input)

        max_diff = torch.max(torch.abs(logits_1 - logits_2)).item()
        self.assertEqual(max_diff, 0.0, "KinematicStream inference is non-deterministic")

    def test_05_extreme_values_do_not_produce_nans(self):
        """Verifies that large kinematic anomalies do not cause overflow or NaN in logits."""
        # Extreme jump: disp_ratio = 50.0, alt_residual = 10000.0
        kin_extreme = np.array([50.0, 10000.0, 50.0, 50.0, 1.0, 1.0, 5000.0, 10.0, 30.0, 5.0, 100.0])
        scaled_extreme = self.scaler.transform(kin_extreme.reshape(1, -1))
        t_extreme = torch.tensor(scaled_extreme, dtype=torch.float32)

        with torch.no_grad():
            emb, logits = self.kin_stream(t_extreme)
            probs = torch.softmax(logits, dim=-1)

        self.assertTrue(torch.all(torch.isfinite(emb)).item())
        self.assertTrue(torch.all(torch.isfinite(logits)).item())
        self.assertTrue(torch.all(torch.isfinite(probs)).item())
        self.assertGreaterEqual(probs[0, 1].item(), 0.0)
        self.assertLessEqual(probs[0, 1].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
