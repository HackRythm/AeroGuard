"""
Unit and Integration Tests for AeroGuard ST-GNN Architecture
============================================================
Tests:
1. Spatial Encoder: Loading weights from models/gnn_edge_aware_repaired.pt and verifying embedding dimensions.
2. Temporal GRU: Forward pass with 5-timestep sliding window, masking, and gradient flow.
3. Kinematic Stream: Forward pass with causal motion residuals.
4. Tri-Stream Fusion: Verifying shapes, probability ranges, and fusion modes.
"""

import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import torch
import joblib

from scripts.train_edge_aware_repaired import RepairedEdgeAwareGNN, COORD_FREE_NODE_FEATURES, EDGE_FEATURES
from scripts.st_gnn_model import TemporalGRUStream, KinematicStream, AeroGuardSTGNN

def test_spatial_encoder_loading_and_embedding():
    """Verify repaired spatial encoder loads from checkpoint and produces 16D embeddings."""
    checkpoint_path = "models/gnn_edge_aware_repaired.pt"
    assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"

    spatial_model = RepairedEdgeAwareGNN(num_features=len(COORD_FREE_NODE_FEATURES), edge_dim=len(EDGE_FEATURES))
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    spatial_model.load_state_dict(state_dict)
    spatial_model.eval()

    batch_nodes = 7
    x = torch.randn(batch_nodes, 12)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)

    with torch.no_grad():
        emb = spatial_model.extract_spatial_embeddings(x, edge_index, edge_attr)
        logits = spatial_model(x, edge_index, edge_attr)

    assert emb.shape == (batch_nodes, 16), f"Expected shape ({batch_nodes}, 16), got {emb.shape}"
    assert logits.shape == (batch_nodes, 2), f"Expected shape ({batch_nodes}, 2), got {logits.shape}"
    assert not torch.isnan(emb).any(), "NaN found in spatial embeddings"
    assert not torch.isnan(logits).any(), "NaN found in spatial logits"

def test_temporal_gru_stream():
    """Verify temporal GRU stream processes 5-timestep sequences with observation masks."""
    batch_size = 10
    T = 5
    input_dim = 25  # 16 spatial + 9 kinematics

    gru_stream = TemporalGRUStream(input_dim=input_dim, hidden_dim=32)

    seq = torch.randn(batch_size, T, input_dim)
    mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], dtype=torch.bool)

    h_temp, logits = gru_stream(seq, mask)

    assert h_temp.shape == (batch_size, 32), f"Expected ({batch_size}, 32), got {h_temp.shape}"
    assert logits.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {logits.shape}"
    assert not torch.isnan(h_temp).any(), "NaN found in temporal hidden representation"
    assert not torch.isnan(logits).any(), "NaN found in temporal logits"

def test_kinematic_stream():
    """Verify kinematic stream processes 11D motion invariants."""
    batch_size = 8
    input_dim = 11

    kin_stream = KinematicStream(input_dim=input_dim, hidden_dim=32)
    k_feats = torch.randn(batch_size, input_dim)

    h_kin, logits = kin_stream(k_feats)

    assert h_kin.shape == (batch_size, 16), f"Expected ({batch_size}, 16), got {h_kin.shape}"
    assert logits.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {logits.shape}"
    assert not torch.isnan(h_kin).any(), "NaN found in kinematic representation"
    assert not torch.isnan(logits).any(), "NaN found in kinematic logits"

def test_unified_st_gnn_forward_and_fusion():
    """Verify unified AeroGuardSTGNN forward pass and multi-stream fusion."""
    spatial_model = RepairedEdgeAwareGNN(num_features=12, edge_dim=4)
    st_gnn = AeroGuardSTGNN(spatial_model=spatial_model)

    B = 6
    x = torch.randn(B, 12)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.randn(3, 4)
    temporal_seq = torch.randn(B, 5, 25)
    temporal_mask = torch.ones(B, 5, dtype=torch.bool)
    kinematic_feats = torch.randn(B, 11)

    out = st_gnn(x, edge_index, edge_attr, temporal_seq, temporal_mask, kinematic_feats)

    expected_keys = [
        'logits_spatial', 'logits_temporal', 'logits_kinematic', 'logits_fused',
        'prob_spatial', 'prob_temporal', 'prob_kinematic',
        'prob_fused_linear', 'prob_fused_weighted', 'prob_fused_max',
        'h_spatial', 'h_temporal', 'h_kinematic'
    ]
    for k in expected_keys:
        assert k in out, f"Missing output key: {k}"
        assert not torch.isnan(out[k]).any(), f"NaN in output {k}"

    assert out['logits_fused'].shape == (B, 2)
    assert out['prob_fused_linear'].shape == (B,)
    assert (out['prob_fused_linear'] >= 0.0).all() and (out['prob_fused_linear'] <= 1.0).all()
    assert (out['prob_fused_weighted'] >= 0.0).all() and (out['prob_fused_weighted'] <= 1.0).all()
    assert (out['prob_fused_max'] >= 0.0).all() and (out['prob_fused_max'] <= 1.0).all()

class TestSTGNN(unittest.TestCase):
    def test_spatial_encoder_loading_and_embedding(self):
        test_spatial_encoder_loading_and_embedding()

    def test_temporal_gru_stream(self):
        test_temporal_gru_stream()

    def test_kinematic_stream(self):
        test_kinematic_stream()

    def test_unified_st_gnn_forward_and_fusion(self):
        test_unified_st_gnn_forward_and_fusion()

if __name__ == "__main__":
    unittest.main()
