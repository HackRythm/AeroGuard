"""
AeroGuard AI — Spatio-Temporal Graph Neural Network (ST-GNN) Model Architecture
================================================================================
Implements the multi-stream ST-GNN architecture composed of:
1. Spatial Stream:
   - Frozen Repaired Edge-Aware GNN (GATv2Conv with 4D edge attributes).
   - Outputs 16D spatial node embeddings h_spatial(t) and spatial logits.
2. Temporal Stream:
   - Recurrent GRU network processing chronological 5-timestep sliding windows
     of concatenated [h_spatial(tau) || k_kinematics(tau)] (dim=25).
   - Explicit boolean observation mask handles gaps, start-of-track, and reacquisition.
3. Kinematic Stream:
   - Causal trajectory residual MLP evaluating physical motion consistency:
     horizontal displacement speed ratio, altitude residual, acceleration mismatch,
     turn rate mismatch, and transponder ICAO collision indicators.
4. Tri-Stream Fusion:
   - Combines spatial, temporal, and kinematic streams into calibrated final anomaly predictions.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from scripts.train_edge_aware_repaired import RepairedEdgeAwareGNN

# ==============================================================================
# STREAM 1: SPATIAL STREAM (FROZEN REPAIRED GNN)
# ==============================================================================
# RepairedEdgeAwareGNN from train_edge_aware_repaired.py is utilized directly:
# - extract_spatial_embeddings(x, edge_index, edge_attr) -> [num_nodes, 16]
# - classifier(h_spatial) -> [num_nodes, 2]

# ==============================================================================
# STREAM 2: TEMPORAL STREAM (GRU OVER SLIDING WINDOWS)
# ==============================================================================
class TemporalGRUStream(nn.Module):
    """
    Temporal Recurrent Stream (GRU) that processes chronological observation sequences.

    Inputs:
    - seq: Tensor of shape [batch_size, window_size, input_dim]
           where input_dim = 16 (spatial embedding) + 9 (kinematics) = 25.
    - mask: Tensor of shape [batch_size, window_size] (True for valid observations).

    Outputs:
    - h_temp: Tensor of shape [batch_size, hidden_dim]
    - logits: Tensor of shape [batch_size, 2]
    """
    def __init__(self, input_dim: int = 25, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = 5

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)
        )

    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes temporal sequences. Zero-pads invalid masked positions.
        Extracts the final valid hidden state per sequence.
        """
        batch_size, T, D = seq.shape

        # Zero out invalid positions according to mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
            seq_masked = seq * mask_expanded
        else:
            seq_masked = seq

        out, h_n = self.gru(seq_masked)  # out: [B, T, hidden_dim], h_n: [1, B, hidden_dim]

        if mask is not None:
            # Gather last valid timestep hidden representation
            # mask has shape [B, T] with 1 for valid, 0 for missing/padded
            lengths = mask.sum(dim=1).long()  # [B]
            lengths = torch.clamp(lengths - 1, min=0)  # 0-indexed last valid index
            idx = lengths.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)  # [B, 1, hidden_dim]
            last_h = out.gather(1, idx).squeeze(1)  # [B, hidden_dim]
        else:
            last_h = out[:, -1, :]  # [B, hidden_dim]

        last_h = self.dropout(last_h)
        logits = self.head(last_h)  # [B, 2]
        return last_h, logits

# ==============================================================================
# STREAM 3: KINEMATIC STREAM (CAUSAL MOTION INVARIANTS MLP)
# ==============================================================================
class KinematicStream(nn.Module):
    """
    Dedicated kinematic branch processing causal motion residuals, discrete derivatives,
    and physical anomaly invariants.

    Inputs:
    - k_features: Tensor of shape [batch_size, num_kinematic_features]

    Outputs:
    - h_kin: Tensor of shape [batch_size, 16]
    - logits: Tensor of shape [batch_size, 2]
    """
    def __init__(self, input_dim: int = 11, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, k_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_kin = self.feature_extractor(k_features)  # [B, 16]
        logits = self.classifier(h_kin)             # [B, 2]
        return h_kin, logits

# ==============================================================================
# MULTI-STREAM FUSION & UNIFIED ST-GNN MODEL
# ==============================================================================
class AeroGuardSTGNN(nn.Module):
    """
    Unified Spatio-Temporal Graph Neural Network combining:
    1. Spatial Stream (Repaired Edge-Aware GNN - frozen during temporal training)
    2. Temporal Stream (GRU over 5-timestep sliding windows)
    3. Kinematic Stream (Causal physical residual MLP)

    Supports validation-selected fusion mechanisms:
    - 'learned_linear': Linear layer combining concatenated stream logits.
    - 'weighted_prob': Convex combination of stream probabilities (w_s, w_t, w_k).
    - 'max_prob': Maximum probability envelope max(p_s, p_t, p_k).
    """
    def __init__(
        self,
        spatial_model: RepairedEdgeAwareGNN,
        num_kinematic_temporal_feats: int = 9,
        num_kinematic_branch_feats: int = 11,
        temporal_hidden_dim: int = 32,
        fusion_mode: str = 'learned_linear'
    ):
        super().__init__()
        self.spatial_stream = spatial_model

        # Temporal stream takes: spatial_dim (16) + temporal_kinematics (9) = 25
        temporal_input_dim = 16 + num_kinematic_temporal_feats
        self.temporal_stream = TemporalGRUStream(
            input_dim=temporal_input_dim,
            hidden_dim=temporal_hidden_dim
        )

        # Kinematic stream
        self.kinematic_stream = KinematicStream(
            input_dim=num_kinematic_branch_feats
        )

        # Fusion configuration
        self.fusion_mode = fusion_mode
        self.fusion_linear = nn.Linear(6, 2)  # 2 logits per stream * 3 streams = 6 inputs

        # Parametric probability weights for weighted_prob mode
        self.prob_weights = nn.Parameter(torch.tensor([0.33, 0.34, 0.33], dtype=torch.float))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        temporal_seq: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        kinematic_feats: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing representations and logits for all streams and final fusion.
        """
        # 1. Spatial stream
        h_spatial = self.spatial_stream.extract_spatial_embeddings(x, edge_index, edge_attr)
        logits_spatial = self.spatial_stream.classifier(h_spatial)

        # 2. Temporal stream
        h_temporal, logits_temporal = self.temporal_stream(temporal_seq, temporal_mask)

        # 3. Kinematic stream
        h_kinematic, logits_kinematic = self.kinematic_stream(kinematic_feats)

        # 4. Fusion
        concat_logits = torch.cat([logits_spatial, logits_temporal, logits_kinematic], dim=-1)
        logits_fused = self.fusion_linear(concat_logits)

        # Probabilities
        prob_spatial = torch.softmax(logits_spatial, dim=-1)[:, 1]
        prob_temporal = torch.softmax(logits_temporal, dim=-1)[:, 1]
        prob_kinematic = torch.softmax(logits_kinematic, dim=-1)[:, 1]
        prob_fused_linear = torch.softmax(logits_fused, dim=-1)[:, 1]

        # Weighted prob
        w = torch.softmax(self.prob_weights, dim=0)
        prob_weighted = w[0] * prob_spatial + w[1] * prob_temporal + w[2] * prob_kinematic

        # Max prob
        prob_max = torch.maximum(torch.maximum(prob_spatial, prob_temporal), prob_kinematic)

        return {
            'logits_spatial': logits_spatial,
            'logits_temporal': logits_temporal,
            'logits_kinematic': logits_kinematic,
            'logits_fused': logits_fused,
            'prob_spatial': prob_spatial,
            'prob_temporal': prob_temporal,
            'prob_kinematic': prob_kinematic,
            'prob_fused_linear': prob_fused_linear,
            'prob_fused_weighted': prob_weighted,
            'prob_fused_max': prob_max,
            'h_spatial': h_spatial,
            'h_temporal': h_temporal,
            'h_kinematic': h_kinematic
        }
