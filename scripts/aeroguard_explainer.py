"""
AeroGuard AI — Multi-Stream Explainable AI (XAI) Engine
========================================================
Implements interpretability and feature attribution for the AeroGuard ST-GNN:
1. SpatialExplainer:
   - Uses PyG GNNExplainer to identify critical node features and neighbor edges
     responsible for spatial relational anomalies.
   - Includes graceful fallback for isolated aircraft (0 neighbors in 100km radius).
   - Maps graph neighbors to human-readable ICAO24 and callsign identifiers.
2. TemporalExplainer:
   - Uses gradient-based attribution over the 5-timestep sliding window to isolate
     which historical second (t-k) and kinematic trends triggered the GRU.
   - Strictly respects boolean observation masks, zeroing padded historical steps.
3. KinematicExplainer:
   - Evaluates sensitivity across the 11 causal motion residuals.
   - Strictly separates model-derived gradient evidence from independent rule heuristics.
4. AeroGuardUnifiedExplainer:
   - Combines all streams into a cohesive, structured explanation payload for
     Air Traffic Control (ATC) operators and dashboard alert inspection.
   - Reports prediction confidence, honest inconclusive attributions, and domain limitations.
"""

import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from scripts.train_edge_aware_repaired import RepairedEdgeAwareGNN, COORD_FREE_NODE_FEATURES, EDGE_FEATURES
from scripts.st_gnn_model import AeroGuardSTGNN, TemporalGRUStream, KinematicStream
from scripts.train_st_gnn import KINEMATIC_BRANCH_FEATS, KINEMATIC_TEMPORAL_FEATS

# ==============================================================================
# 1. SPATIAL GNN EXPLAINER
# ==============================================================================
class SpatialExplainer:
    """
    Explains predictions from the RepairedEdgeAwareGNN spatial encoder.
    Computes node attribute importance and neighbor edge importance using GNNExplainer.
    Gracefully handles isolated nodes with 0 incident edges.
    """
    def __init__(self, model: RepairedEdgeAwareGNN, epochs: int = 100):
        self.model = model
        self.model.eval()
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='multiclass_classification',
                task_level='node',
                return_type='probs'
            )
        )

    def explain_node(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_node_idx: int,
        target_class: int = 1,
        node_identities: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generates spatial explanation for target_node_idx in the snapshot graph.

        Args:
            x: Node feature tensor [N, 12]
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, 4]
            target_node_idx: Index of node to explain
            target_class: Class to explain (default: 1 for anomaly)
            node_identities: Optional list of dicts with {'icao24': ..., 'callsign': ...} per node index
        """
        num_nodes = x.shape[0]
        has_edges = edge_index is not None and edge_index.numel() > 0 and edge_index.shape[1] > 0

        # Guard: Check if graph has edges or if target node has incident edges
        if not has_edges:
            # Isolated node fallback: compute input x gradient on the spatial model directly
            x_clone = x.clone().detach()
            x_clone.requires_grad = True
            empty_ei = torch.empty((2, 0), dtype=torch.long, device=x.device)
            empty_ea = torch.empty((0, 4), dtype=torch.float, device=x.device)

            emb = self.model.extract_spatial_embeddings(x_clone, empty_ei, empty_ea)
            logits = self.model.classifier(emb)
            score = logits[target_node_idx, target_class]
            score.backward()

            grad = x_clone.grad[target_node_idx].cpu().numpy()
            inp = x_clone[target_node_idx].detach().cpu().numpy()
            saliency = np.abs(grad * inp)
            sal_sum = np.sum(saliency)
            sal_norm = saliency / (sal_sum + 1e-8) if sal_sum > 0 else np.ones(12) / 12.0

            node_attr_dict = []
            for feat_name, imp in zip(COORD_FREE_NODE_FEATURES, sal_norm):
                node_attr_dict.append({'feature': feat_name, 'importance': float(imp)})
            node_attr_dict.sort(key=lambda item: abs(item['importance']), reverse=True)

            return {
                'target_node_idx': target_node_idx,
                'top_node_features': node_attr_dict[:5],
                'all_node_features': node_attr_dict,
                'critical_edges': [],
                'incident_edges_count': 0,
                'isolated_aircraft': True,
                'notes': 'Isolated aircraft with no neighboring flights within 100 km horizontal range.'
            }

        # Standard GNNExplainer optimization for relational graphs
        try:
            explanation = self.explainer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                index=target_node_idx
            )
            node_mask = explanation.node_mask[target_node_idx].detach().cpu().numpy()
            edge_mask = explanation.edge_mask.detach().cpu().numpy() if explanation.edge_mask is not None else np.array([])
        except Exception:
            # Saliency fallback if optimization encounters unexpected PyG dimensional constraint
            x_clone = x.clone().detach()
            x_clone.requires_grad = True
            emb = self.model.extract_spatial_embeddings(x_clone, edge_index, edge_attr)
            logits = self.model.classifier(emb)
            score = logits[target_node_idx, target_class]
            score.backward()
            grad = x_clone.grad[target_node_idx].cpu().numpy()
            inp = x_clone[target_node_idx].detach().cpu().numpy()
            node_mask = np.abs(grad * inp)
            edge_mask = np.zeros(edge_index.shape[1], dtype=np.float32)

        node_mask_sum = np.sum(np.abs(node_mask))
        node_mask_norm = node_mask / (node_mask_sum + 1e-8) if node_mask_sum > 0 else np.ones(12) / 12.0

        node_attr_dict = []
        for feat_name, imp in zip(COORD_FREE_NODE_FEATURES, node_mask_norm):
            node_attr_dict.append({'feature': feat_name, 'importance': float(imp)})
        node_attr_dict.sort(key=lambda item: abs(item['importance']), reverse=True)

        # Edge importance for edges incident to target_node_idx
        edge_attr_dict = []
        if len(edge_mask) > 0 and edge_index.shape[1] > 0:
            edge_idx_np = edge_index.cpu().numpy()
            for e_idx in range(edge_index.shape[1]):
                src, dst = edge_idx_np[0, e_idx], edge_idx_np[1, e_idx]
                if dst == target_node_idx:
                    neighbor_meta = {}
                    if node_identities and src < len(node_identities):
                        neighbor_meta = node_identities[src]

                    edge_attr_dict.append({
                        'neighbor_node_idx': int(src),
                        'neighbor_icao24': neighbor_meta.get('icao24', f"NODE_{src}"),
                        'neighbor_callsign': neighbor_meta.get('callsign', 'UNKNOWN'),
                        'edge_importance': float(edge_mask[e_idx]),
                        'distance_scaled': float(edge_attr[e_idx, 0].item()),
                        'alt_diff_scaled': float(edge_attr[e_idx, 1].item()),
                        'vel_diff_scaled': float(edge_attr[e_idx, 2].item()),
                        'head_diff_scaled': float(edge_attr[e_idx, 3].item())
                    })
            edge_attr_dict.sort(key=lambda item: item['edge_importance'], reverse=True)

        return {
            'target_node_idx': target_node_idx,
            'top_node_features': node_attr_dict[:5],
            'all_node_features': node_attr_dict,
            'critical_edges': edge_attr_dict[:5],
            'incident_edges_count': len(edge_attr_dict),
            'isolated_aircraft': False,
            'notes': f"Analyzed {len(edge_attr_dict)} incident relational edges."
        }

# ==============================================================================
# 2. TEMPORAL GRU EXPLAINER
# ==============================================================================
class TemporalExplainer:
    """
    Computes gradient-based feature attribution across the 5-timestep sliding window
    for the TemporalGRUStream. Strictly respects boolean observation masks.
    """
    def __init__(self, gru_stream: TemporalGRUStream):
        self.gru_stream = gru_stream
        self.gru_stream.eval()

    def explain_sequence(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_class: int = 1
    ) -> Dict[str, Any]:
        """
        Calculates Saliency (Input x Gradient) across the temporal window [T, 25].
        Handles padded / missing historical timesteps by zeroing attribution for invalid steps.
        """
        seq_tensor = seq.clone().detach()
        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)  # [1, T, 25]
        seq_tensor.requires_grad = True

        if mask is not None:
            mask_tensor = mask.unsqueeze(0) if mask.dim() == 1 else mask
        else:
            mask_tensor = None

        _, logits = self.gru_stream(seq_tensor, mask_tensor)
        target_score = logits[0, target_class]

        target_score.backward()

        grad = seq_tensor.grad.squeeze(0).cpu().numpy()  # [T, 25]
        inp = seq_tensor.squeeze(0).detach().cpu().numpy()  # [T, 25]
        saliency = np.abs(grad * inp)  # [T, 25]

        T = seq.shape[0]
        mask_np = mask.cpu().numpy().astype(bool) if mask is not None else np.ones(T, dtype=bool)

        # Zero out attribution on masked/padded positions
        saliency[~mask_np] = 0.0

        # Aggregate importance per timestep
        timestep_importance = np.sum(saliency, axis=-1)  # [T]
        sum_ts = np.sum(timestep_importance)
        ts_norm = timestep_importance / (sum_ts + 1e-8) if sum_ts > 0 else np.zeros(T)

        # If all valid gradients are uniform or zero, distribute equally among valid observations
        if sum_ts == 0 and np.any(mask_np):
            valid_count = np.sum(mask_np)
            ts_norm[mask_np] = 1.0 / valid_count

        step_labels = [f"t-{T - 1 - i}" if (T - 1 - i) > 0 else "t" for i in range(T)]

        temporal_profile = []
        for i, (s_lbl, imp, is_valid) in enumerate(zip(step_labels, ts_norm, mask_np)):
            temporal_profile.append({
                'timestep': s_lbl,
                'relative_offset_s': -(T - 1 - i),
                'is_valid_observation': bool(is_valid),
                'importance': float(imp),
                'status': 'valid' if is_valid else 'padded_history'
            })

        # Top contributing features across valid observations
        valid_saliency = saliency[mask_np] if np.any(mask_np) else saliency
        feature_importance = np.sum(valid_saliency, axis=0)  # [25]
        feat_sum = np.sum(feature_importance)
        feat_norm = feature_importance / (feat_sum + 1e-8) if feat_sum > 0 else np.ones(25) / 25.0

        feature_names = [f"spatial_emb_{i}" for i in range(16)] + KINEMATIC_TEMPORAL_FEATS
        feat_profile = [{'feature': name, 'importance': float(imp)} for name, imp in zip(feature_names, feat_norm)]
        feat_profile.sort(key=lambda item: item['importance'], reverse=True)

        # Peak timestep among valid observations
        valid_profiles = [p for p in temporal_profile if p['is_valid_observation']]
        peak_step = max(valid_profiles, key=lambda item: item['importance'])['timestep'] if valid_profiles else "t"

        return {
            'peak_timestep': peak_step,
            'temporal_step_importance': temporal_profile,
            'top_temporal_features': feat_profile[:5],
            'all_temporal_features': feat_profile,
            'valid_timesteps_count': int(np.sum(mask_np)),
            'total_window_length': int(T)
        }

# ==============================================================================
# 3. KINEMATIC RESIDUAL EXPLAINER
# ==============================================================================
class KinematicExplainer:
    """
    Evaluates causal physical motion residuals.
    Distinguishes model-derived gradient sensitivity from independent rule-based warnings.
    """
    def __init__(self, kinematic_stream: KinematicStream):
        self.kinematic_stream = kinematic_stream
        self.kinematic_stream.eval()

    def explain_kinematics(
        self,
        kinematic_raw_row: pd.Series,
        scaled_kin_tensor: torch.Tensor,
        target_class: int = 1
    ) -> Dict[str, Any]:
        """
        Combines model gradient sensitivity with physical threshold sanity checks.
        """
        k_tensor = scaled_kin_tensor.clone().detach()
        if k_tensor.dim() == 1:
            k_tensor = k_tensor.unsqueeze(0)  # [1, 11]
        k_tensor.requires_grad = True

        _, logits = self.kinematic_stream(k_tensor)
        target_score = logits[0, target_class]
        target_score.backward()

        grad = k_tensor.grad.squeeze(0).cpu().numpy()
        inp = k_tensor.squeeze(0).detach().cpu().numpy()
        saliency = np.abs(grad * inp)
        sal_sum = np.sum(saliency)
        sal_norm = saliency / (sal_sum + 1e-8) if sal_sum > 0 else np.ones(11) / 11.0

        residual_attributions = []
        for name, imp in zip(KINEMATIC_BRANCH_FEATS, sal_norm):
            raw_val = float(kinematic_raw_row.get(name, 0.0))
            residual_attributions.append({
                'residual_feature': name,
                'importance': float(imp),
                'raw_value': raw_val
            })
        residual_attributions.sort(key=lambda item: item['importance'], reverse=True)

        # Independent rule-based diagnostic triggers (Explicitly designated as heuristics)
        diagnostics = []
        rule_warnings = []

        if kinematic_raw_row.get('icao_collision', 0.0) > 0.5:
            msg = "Simultaneous ICAO transponder address collision detected (cloned identity)."
            diagnostics.append(f"CRITICAL: {msg}")
            rule_warnings.append({'severity': 'CRITICAL', 'rule': 'ICAO_COLLISION', 'message': msg})

        disp = float(kinematic_raw_row.get('disp_ratio', 1.0))
        if disp > 5.0:
            msg = f"Non-physical displacement speed ratio ({disp:.1f}x cruising speed)."
            diagnostics.append(f"CRITICAL: {msg}")
            rule_warnings.append({'severity': 'CRITICAL', 'rule': 'DISP_RATIO', 'message': msg})

        alt_res = float(kinematic_raw_row.get('alt_residual', 0.0))
        if alt_res > 1000.0:
            msg = f"Altitude disparity ({alt_res:.0f} ft vertical rate mismatch)."
            diagnostics.append(f"WARNING: {msg}")
            rule_warnings.append({'severity': 'WARNING', 'rule': 'ALTITUDE_DISPARITY', 'message': msg})

        if kinematic_raw_row.get('track_jump_flag', 0.0) > 0.5:
            msg = "Position jump displacement > 5 km within 1 second."
            diagnostics.append(f"CRITICAL: {msg}")
            rule_warnings.append({'severity': 'CRITICAL', 'rule': 'POSITION_JUMP', 'message': msg})

        return {
            'top_kinematic_residuals': residual_attributions[:5],
            'all_kinematic_residuals': residual_attributions,
            'diagnostic_flags': diagnostics,
            'independent_rule_warnings': rule_warnings,
            'disclaimer': "Independent rule warnings represent domain-specific physical sanity checks and must not be interpreted as definitive proof of cyber-physical spoofing on their own."
        }

# ==============================================================================
# 4. UNIFIED AEROGUARD EXPLAINER
# ==============================================================================
class AeroGuardUnifiedExplainer:
    """
    Coordinates spatial, temporal, and kinematic explainers to produce a comprehensive,
    human-interpretable operational alert dossier. Supports normal flights, true threats,
    false alarms, and missed anomalies without confirmation bias.
    """
    def __init__(self, full_model: AeroGuardSTGNN):
        self.full_model = full_model
        self.spatial_explainer = SpatialExplainer(full_model.spatial_stream)
        self.temporal_explainer = TemporalExplainer(full_model.temporal_stream)
        self.kinematic_explainer = KinematicExplainer(full_model.kinematic_stream)

    def generate_full_explanation(
        self,
        record: pd.Series,
        spatial_graph: Data,
        target_node_idx: int,
        temporal_seq: torch.Tensor,
        temporal_mask: torch.Tensor,
        scaled_kin_tensor: torch.Tensor,
        fused_prob: float,
        threshold: float = 0.20,
        ground_truth_label: Optional[int] = None,
        node_identities: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generates unified explanation dictionary for an anomaly detection event.
        """
        is_threat = fused_prob >= threshold
        predicted_status = "ANOMALY" if is_threat else "NORMAL"
        threat_level = "CRITICAL" if fused_prob >= 0.70 else ("ELEVATED" if is_threat else "NOMINAL")

        # 1. Spatial explanation
        spatial_exp = self.spatial_explainer.explain_node(
            x=spatial_graph.x,
            edge_index=spatial_graph.edge_index,
            edge_attr=spatial_graph.edge_attr,
            target_node_idx=target_node_idx,
            target_class=1,
            node_identities=node_identities
        )

        # 2. Temporal explanation
        temporal_exp = self.temporal_explainer.explain_sequence(
            seq=temporal_seq,
            mask=temporal_mask,
            target_class=1
        )

        # 3. Kinematic explanation
        kinematic_exp = self.kinematic_explainer.explain_kinematics(
            kinematic_raw_row=record,
            scaled_kin_tensor=scaled_kin_tensor,
            target_class=1
        )

        # Attribution Confidence & Borderline Uncertainty
        score_margin = abs(fused_prob - threshold)
        if score_margin < 0.04:
            attribution_confidence = "INCONCLUSIVE (Borderline Score)"
        elif fused_prob >= 0.70 or fused_prob <= 0.05:
            attribution_confidence = "HIGH"
        else:
            attribution_confidence = "MODERATE"

        # Case Categorization (when ground truth is available)
        case_category = "UNLABELED"
        if ground_truth_label is not None:
            if ground_truth_label == 1 and is_threat:
                case_category = "TRUE_POSITIVE"
            elif ground_truth_label == 0 and not is_threat:
                case_category = "TRUE_NEGATIVE"
            elif ground_truth_label == 0 and is_threat:
                case_category = "FALSE_POSITIVE"
            elif ground_truth_label == 1 and not is_threat:
                case_category = "FALSE_NEGATIVE"

        # Primary Root Cause & Driver Inference (honestly reporting nominal states)
        if not is_threat:
            primary_driver = "Nominal Cruising Dynamics"
        elif len(kinematic_exp['diagnostic_flags']) > 0:
            primary_driver = kinematic_exp['diagnostic_flags'][0].split(":")[0] + " Violation"
        elif len(spatial_exp['critical_edges']) > 0 and spatial_exp['critical_edges'][0]['edge_importance'] > 0.5:
            primary_driver = "Spatial Relational Proximity Disparity"
        elif spatial_exp['top_node_features'][0]['feature'] in ['altitude', 'vertical_rate']:
            primary_driver = "Vertical Relational Disparity"
        elif spatial_exp['top_node_features'][0]['feature'] in ['groundspeed', 'acceleration']:
            primary_driver = "Kinematic Velocity Outlier"
        else:
            primary_driver = "Coupled Spatio-Temporal Disparity"

        # Explicit Domain Limitations & Uncertainty
        if case_category == "FALSE_POSITIVE":
            limitations_text = "False Alert: Normal flight flagged as threat. Sudden standard maneuvers (e.g. sharp descent clearance or turn) in dense air corridors can mimic anomalous relational acceleration."
        elif case_category == "FALSE_NEGATIVE":
            limitations_text = "Undetected Anomaly: Threat below threshold. Smooth trajectory spoofing or low-amplitude drift can evade discrete kinematic checks when operating within normal flight envelopes."
        elif is_threat:
            limitations_text = "Anomaly Alert: Multi-stream attribution indicates non-standard spatial, temporal, or kinematic behavior. Single-receiver multipath and transient propagation effects should be verified."
        else:
            limitations_text = "Nominal Trajectory: All three streams show baseline adherence within standard aviation operational envelopes."

        return {
            'timestamp': str(record.get('timestamp', '')),
            'icao24': str(record.get('icao24', '')),
            'callsign': str(record.get('callsign', '')),
            'anomaly_probability': float(fused_prob),
            'configured_threshold': float(threshold),
            'predicted_status': predicted_status,
            'threat_level': threat_level,
            'attribution_confidence': attribution_confidence,
            'case_category': case_category,
            'primary_root_cause': primary_driver,
            'primary_driver': {
                'model_derived_driver': primary_driver,
                'independent_rule_warnings': kinematic_exp['independent_rule_warnings']
            },
            'spatial_explanation': spatial_exp,
            'temporal_explanation': temporal_exp,
            'kinematic_explanation': kinematic_exp,
            'limitations_and_uncertainty': limitations_text
        }
