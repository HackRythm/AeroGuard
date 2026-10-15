"""
Unit Tests for AeroGuard Explainable AI Engine
==============================================
Tests:
1. SpatialExplainer: Node attribute & edge mask generation using GNNExplainer.
2. SpatialExplainer Isolated Node: Graceful handling of aircraft with 0 incident edges.
3. SpatialExplainer Metadata: Verifies neighbor ICAO24 and callsign mapping.
4. TemporalExplainer: Gradient-based saliency over 5-step window.
5. TemporalExplainer Masking: Verifies padded historical timesteps receive strictly 0 importance.
6. KinematicExplainer: Residual sensitivity & diagnostic violation triggers.
7. KinematicExplainer Separation: Verifies model-derived evidence is separated from rule heuristics.
8. UnifiedExplainer: Full explanation schema generation.
9. UnifiedExplainer Case Categories: Verifies TP, FP, FN, TN classification, confidence, and disclaimers.
"""

import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import torch
import pandas as pd
from torch_geometric.data import Data

from scripts.train_edge_aware_repaired import RepairedEdgeAwareGNN
from scripts.st_gnn_model import AeroGuardSTGNN, TemporalGRUStream, KinematicStream
from scripts.aeroguard_explainer import (
    SpatialExplainer, TemporalExplainer, KinematicExplainer, AeroGuardUnifiedExplainer
)

class TestXAIExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spatial_model = RepairedEdgeAwareGNN(12, 4)
        cls.full_model = AeroGuardSTGNN(cls.spatial_model, 9, 11)
        cls.full_model.eval()

    def test_spatial_explainer(self):
        explainer = SpatialExplainer(self.spatial_model, epochs=10)
        x = torch.randn(4, 12)
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 3]], dtype=torch.long)
        edge_attr = torch.randn(4, 4)

        res = explainer.explain_node(x, edge_index, edge_attr, target_node_idx=0)

        self.assertIn('top_node_features', res)
        self.assertIn('critical_edges', res)
        self.assertEqual(len(res['top_node_features']), 5)
        self.assertTrue(len(res['critical_edges']) > 0)
        self.assertAlmostEqual(sum([f['importance'] for f in res['all_node_features']]), 1.0, places=3)

    def test_spatial_explainer_isolated_node(self):
        """Verifies that an aircraft with 0 neighbors does not crash and returns graceful fallback."""
        explainer = SpatialExplainer(self.spatial_model, epochs=5)
        x = torch.randn(2, 12)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)

        res = explainer.explain_node(x, edge_index, edge_attr, target_node_idx=0)

        self.assertTrue(res['isolated_aircraft'])
        self.assertEqual(len(res['critical_edges']), 0)
        self.assertEqual(len(res['top_node_features']), 5)
        self.assertAlmostEqual(sum([f['importance'] for f in res['all_node_features']]), 1.0, places=3)

    def test_spatial_explainer_neighbor_metadata(self):
        """Verifies that neighbor ICAO and callsigns are attached to critical edges."""
        explainer = SpatialExplainer(self.spatial_model, epochs=5)
        x = torch.randn(3, 12)
        edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
        edge_attr = torch.randn(2, 4)
        identities = [
            {'icao24': '400001', 'callsign': 'TGT01'},
            {'icao24': '400002', 'callsign': 'NBR02'},
            {'icao24': '400003', 'callsign': 'NBR03'}
        ]

        res = explainer.explain_node(x, edge_index, edge_attr, target_node_idx=0, node_identities=identities)

        self.assertTrue(len(res['critical_edges']) > 0)
        first_edge = res['critical_edges'][0]
        self.assertIn('neighbor_icao24', first_edge)
        self.assertIn('neighbor_callsign', first_edge)
        self.assertIn(first_edge['neighbor_icao24'], ['400002', '400003'])

    def test_temporal_explainer(self):
        explainer = TemporalExplainer(self.full_model.temporal_stream)
        seq = torch.randn(5, 25)
        mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)

        res = explainer.explain_sequence(seq, mask)

        self.assertIn('peak_timestep', res)
        self.assertIn('temporal_step_importance', res)
        self.assertEqual(len(res['temporal_step_importance']), 5)
        self.assertIn('top_temporal_features', res)
        self.assertAlmostEqual(sum([t['importance'] for t in res['temporal_step_importance']]), 1.0, places=3)

    def test_temporal_explainer_padded_mask(self):
        """Verifies that padded historical timesteps receive strictly 0 importance."""
        explainer = TemporalExplainer(self.full_model.temporal_stream)
        seq = torch.randn(5, 25)
        mask = torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool)

        res = explainer.explain_sequence(seq, mask)

        step_imps = res['temporal_step_importance']
        # The first two steps (t-4 and t-3) are padded and must have 0.0 importance
        self.assertFalse(step_imps[0]['is_valid_observation'])
        self.assertFalse(step_imps[1]['is_valid_observation'])
        self.assertEqual(step_imps[0]['importance'], 0.0)
        self.assertEqual(step_imps[1]['importance'], 0.0)
        # Valid steps must sum to 1.0
        valid_sum = step_imps[2]['importance'] + step_imps[3]['importance'] + step_imps[4]['importance']
        self.assertAlmostEqual(valid_sum, 1.0, places=3)
        self.assertEqual(res['valid_timesteps_count'], 3)

    def test_kinematic_explainer(self):
        explainer = KinematicExplainer(self.full_model.kinematic_stream)
        record = pd.Series({
            'disp_ratio': 25.0,
            'alt_residual': 6500.0,
            'icao_collision': 1.0,
            'track_jump_flag': 1.0
        })
        k_tensor = torch.randn(11)

        res = explainer.explain_kinematics(record, k_tensor)

        self.assertIn('top_kinematic_residuals', res)
        self.assertIn('diagnostic_flags', res)
        self.assertTrue(len(res['diagnostic_flags']) >= 2)

    def test_kinematic_explainer_heuristic_separation(self):
        """Verifies rule warnings are separated from model sensitivities and disclaimed."""
        explainer = KinematicExplainer(self.full_model.kinematic_stream)
        record = pd.Series({
            'disp_ratio': 12.0,
            'alt_residual': 1500.0,
            'icao_collision': 0.0,
            'track_jump_flag': 0.0
        })
        k_tensor = torch.randn(11)

        res = explainer.explain_kinematics(record, k_tensor)

        self.assertIn('independent_rule_warnings', res)
        self.assertIn('disclaimer', res)
        self.assertTrue(len(res['independent_rule_warnings']) >= 2)
        self.assertIn("must not be interpreted as definitive proof", res['disclaimer'])

    def test_unified_explainer(self):
        unified = AeroGuardUnifiedExplainer(self.full_model)
        record = pd.Series({
            'timestamp': '2020-01-02 10:15:30',
            'icao24': '500001',
            'callsign': 'RYR201',
            'disp_ratio': 1.0,
            'alt_residual': 7500.0,
            'icao_collision': 0.0,
            'track_jump_flag': 0.0
        })
        data = Data(
            x=torch.randn(3, 12),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 4)
        )
        seq = torch.randn(5, 25)
        mask = torch.ones(5, dtype=torch.bool)
        kin_tensor = torch.randn(11)

        exp = unified.generate_full_explanation(
            record=record,
            spatial_graph=data,
            target_node_idx=0,
            temporal_seq=seq,
            temporal_mask=mask,
            scaled_kin_tensor=kin_tensor,
            fused_prob=0.88,
            threshold=0.20,
            ground_truth_label=1
        )

        self.assertEqual(exp['threat_level'], 'CRITICAL')
        self.assertEqual(exp['icao24'], '500001')
        self.assertEqual(exp['case_category'], 'TRUE_POSITIVE')
        self.assertEqual(exp['predicted_status'], 'ANOMALY')
        self.assertEqual(exp['attribution_confidence'], 'HIGH')
        self.assertIn('spatial_explanation', exp)
        self.assertIn('temporal_explanation', exp)
        self.assertIn('kinematic_explanation', exp)
        self.assertIn('limitations_and_uncertainty', exp)

    def test_unified_explainer_case_categories(self):
        """Verifies correct categorization and limitation text across TP, FP, FN, TN."""
        unified = AeroGuardUnifiedExplainer(self.full_model)
        record = pd.Series({'timestamp': '2020-01-02 10:00:00', 'icao24': '500002', 'callsign': 'TAP10'})
        data = Data(x=torch.randn(2, 12), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 4)))
        seq = torch.randn(5, 25)
        mask = torch.ones(5, dtype=torch.bool)
        kin_tensor = torch.randn(11)

        # False Positive: label=0, prob=0.45 >= threshold(0.20)
        exp_fp = unified.generate_full_explanation(
            record, data, 0, seq, mask, kin_tensor, fused_prob=0.45, threshold=0.20, ground_truth_label=0
        )
        self.assertEqual(exp_fp['case_category'], 'FALSE_POSITIVE')
        self.assertIn("False Alert", exp_fp['limitations_and_uncertainty'])

        # False Negative: label=1, prob=0.10 < threshold(0.20)
        exp_fn = unified.generate_full_explanation(
            record, data, 0, seq, mask, kin_tensor, fused_prob=0.10, threshold=0.20, ground_truth_label=1
        )
        self.assertEqual(exp_fn['case_category'], 'FALSE_NEGATIVE')
        self.assertIn("Undetected Anomaly", exp_fn['limitations_and_uncertainty'])

        # Borderline Inconclusive score: prob=0.21, threshold=0.20
        exp_border = unified.generate_full_explanation(
            record, data, 0, seq, mask, kin_tensor, fused_prob=0.21, threshold=0.20
        )
        self.assertIn("INCONCLUSIVE", exp_border['attribution_confidence'])

if __name__ == "__main__":
    unittest.main()
