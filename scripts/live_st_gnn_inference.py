"""
AeroGuard AI — Live ST-GNN Inference & ADS-B Track Pipeline
=============================================================
Connects decoded software-defined ADS-B messages and airspace telemetry to the
trained Spatio-Temporal Graph Neural Network (ST-GNN) and Explainable AI (XAI) engine.

Key Subsystems:
1. ADSBTrack & ADSBTrackBuffer:
   - Aggregates incoming DecodedADSBMessage instances or telemetry frames by ICAO24.
   - Calculates discrete kinematic derivatives (dt, acceleration, turn_rate, vertical_accel, climb_rate_change).
   - Computes causal physical residuals (disp_ratio, alt_residual, accel_residual, turn_residual, track_jump_flag).
   - Maintains chronological sliding observation window (T=5) with boolean observation masks.
   - Honestly manages RF feature observation (measured SNR/power vs unobserved neutral standardized zero).

2. LiveSTGNNPipeline:
   - Loads frozen spatial GNN, official ST-GNN, pre-fitted scalers, and calibrated threshold (t=0.20).
   - Builds dynamic 100 km relational edge graphs between active aircraft tracks.
   - Executes multi-stream forward pass (Spatial, Temporal, Kinematic, Tri-Stream Fusion).
   - Generates real-time anomaly scores and triggers AeroGuardUnifiedExplainer for full dossiers.
"""

import os, sys, time, math, json
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import joblib
from torch_geometric.data import Data

# Import existing model definitions & preprocessor feature definitions
from scripts.train_edge_aware_repaired import (
    RepairedEdgeAwareGNN, COORD_FREE_NODE_FEATURES, EDGE_FEATURES, PROXIMITY_THRESHOLD_M
)
from scripts.st_gnn_model import AeroGuardSTGNN
from scripts.train_st_gnn import (
    KINEMATIC_BRANCH_FEATS, KINEMATIC_TEMPORAL_FEATS, TEMPORAL_WINDOW_T
)
from scripts.aeroguard_explainer import AeroGuardUnifiedExplainer
from scripts.rtl_sdr_adapter import DecodedADSBMessage

# Unit conversion constants
KTS_TO_MS = 0.514444
FPM_TO_MS = 0.00508
EARTH_RADIUS_M = 6371000.0

def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle horizontal distance between two points in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(max(0.0, a)), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_M * c

# ==============================================================================
# 1. AIRCRAFT TRACK & DISCRETE KINEMATICS
# ==============================================================================
@dataclass
class TrackObservation:
    """Single chronological snapshot of an aircraft's reported state & derivatives."""
    timestamp: float
    latitude: float
    longitude: float
    altitude: float         # feet
    groundspeed: float      # knots
    heading: float          # degrees [0, 360)
    vertical_rate: float    # feet per minute

    # Kinematic derivatives
    dt: float = 1.0
    acceleration: float = 0.0          # m/s^2
    turn_rate: float = 0.0             # deg/s
    vertical_acceleration: float = 0.0 # m/s^2
    climb_rate_change: float = 0.0     # fpm/s

    # Causal motion invariants & residuals
    disp_ratio: float = 1.0
    alt_residual: float = 0.0
    accel_residual: float = 0.0
    turn_residual: float = 0.0
    track_jump_flag: float = 0.0
    icao_collision: float = 0.0

    # RF Physical Layer Observation
    rf_observed: bool = False
    estimated_doppler_hz: Optional[float] = None
    estimated_rss_dbm: Optional[float] = None
    estimated_snr_db: Optional[float] = None

class ADSBTrack:
    """
    Maintains the state and chronological observation history (T=5 window)
    for a single aircraft identified by ICAO24 transponder address.
    """
    def __init__(self, icao24: str, callsign: str = "UNKNOWN"):
        self.icao24 = icao24.upper()
        self.callsign = callsign.strip().upper()

        # Current state values
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.altitude: float = 30000.0
        self.groundspeed: float = 400.0
        self.heading: float = 0.0
        self.vertical_rate: float = 0.0
        self.last_timestamp: float = 0.0

        # RF parameters
        self.rf_observed: bool = False
        self.estimated_doppler_hz: Optional[float] = None
        self.estimated_rss_dbm: Optional[float] = None
        self.estimated_snr_db: Optional[float] = None

        # Chronological observation history (stores up to 10 snapshots)
        self.history: deque = deque(maxlen=10)

    def update_telemetry(
        self,
        timestamp: float,
        latitude: float,
        longitude: float,
        altitude: float,
        groundspeed: float,
        heading: float,
        vertical_rate: float,
        callsign: Optional[str] = None,
        rf_snr_db: Optional[float] = None,
        rf_rss_dbm: Optional[float] = None,
        rf_doppler_hz: Optional[float] = None,
        icao_collision_detected: bool = False
    ):
        """Updates track with a new telemetry frame and computes causal discrete derivatives."""
        if callsign and callsign.strip():
            self.callsign = callsign.strip().upper()

        prev_obs = self.history[-1] if len(self.history) > 0 else None

        # Calculate time delta
        if prev_obs is not None:
            dt = max(0.1, timestamp - prev_obs.timestamp)
        else:
            dt = 1.0

        # Kinematic derivatives
        if prev_obs is not None:
            # 1. Acceleration (m/s^2)
            d_speed_ms = (groundspeed - prev_obs.groundspeed) * KTS_TO_MS
            acceleration = d_speed_ms / dt

            # 2. Vertical Acceleration (m/s^2)
            d_vr_ms = (vertical_rate - prev_obs.vertical_rate) * FPM_TO_MS
            vertical_acceleration = d_vr_ms / dt

            # 3. Turn Rate (deg/s) with shortest angular distance unwrapping
            d_hdg = (heading - prev_obs.heading + 180.0) % 360.0 - 180.0
            turn_rate = d_hdg / dt

            # 4. Climb Rate Change (fpm/s)
            climb_rate_change = (vertical_rate - prev_obs.vertical_rate) / dt

            # 5. Horizontal displacement & speed ratio
            step_dist_m = haversine_distance_m(prev_obs.latitude, prev_obs.longitude, latitude, longitude)
            expected_dist_m = max(groundspeed * KTS_TO_MS * dt, 1.0)
            disp_ratio = float(np.clip(step_dist_m / expected_dist_m, 0.0, 50.0))

            # 6. Altitude consistency residual: |Delta alt - (vertical_rate * dt / 60)|
            d_alt = altitude - prev_obs.altitude
            expected_dalt = vertical_rate * (dt / 60.0)
            alt_residual = float(np.clip(abs(d_alt - expected_dalt), 0.0, 10000.0))

            # 7. Acceleration residual: |accel - expected_accel|
            accel_residual = float(np.clip(abs(acceleration - (d_speed_ms / dt)), 0.0, 50.0))

            # 8. Turn residual
            turn_residual = float(np.clip(abs(turn_rate - (d_hdg / dt)), 0.0, 50.0))

            # 9. Track jump flag (step displacement > 5 km within 1 sec)
            track_jump_flag = 1.0 if (step_dist_m > 5000.0 and dt <= 1.0) else 0.0
        else:
            # Initial track observation baseline
            dt = 1.0
            acceleration = 0.0
            vertical_acceleration = 0.0
            turn_rate = 0.0
            climb_rate_change = 0.0
            disp_ratio = 1.0
            alt_residual = 0.0
            accel_residual = 0.0
            turn_residual = 0.0
            track_jump_flag = 0.0

        icao_col_val = 1.0 if icao_collision_detected else 0.0

        # RF Physical Layer values
        rf_observed = (rf_snr_db is not None or rf_rss_dbm is not None or rf_doppler_hz is not None)

        obs = TrackObservation(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            groundspeed=groundspeed,
            heading=heading,
            vertical_rate=vertical_rate,
            dt=dt,
            acceleration=acceleration,
            turn_rate=turn_rate,
            vertical_acceleration=vertical_acceleration,
            climb_rate_change=climb_rate_change,
            disp_ratio=disp_ratio,
            alt_residual=alt_residual,
            accel_residual=accel_residual,
            turn_residual=turn_residual,
            track_jump_flag=track_jump_flag,
            icao_collision=icao_col_val,
            rf_observed=rf_observed,
            estimated_doppler_hz=rf_doppler_hz,
            estimated_rss_dbm=rf_rss_dbm,
            estimated_snr_db=rf_snr_db
        )

        # Update current track state
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.groundspeed = groundspeed
        self.heading = heading
        self.vertical_rate = vertical_rate
        self.last_timestamp = timestamp
        self.rf_observed = rf_observed
        self.estimated_doppler_hz = rf_doppler_hz
        self.estimated_rss_dbm = rf_rss_dbm
        self.estimated_snr_db = rf_snr_db

        self.history.append(obs)

    def update_from_message(self, msg: DecodedADSBMessage, current_time: Optional[float] = None):
        """Updates track from a decoded Mode-S message."""
        now = current_time if current_time is not None else time.time()

        # Inherit latest known kinematics
        lat = self.latitude
        lon = self.longitude
        alt = self.altitude
        spd = self.groundspeed
        hdg = self.heading
        vr = self.vertical_rate
        cs = self.callsign

        if msg.callsign:
            cs = msg.callsign
        if msg.groundspeed_kts is not None:
            spd = msg.groundspeed_kts
        if msg.heading_deg is not None:
            hdg = msg.heading_deg
        if msg.vertical_rate_fpm is not None:
            vr = msg.vertical_rate_fpm
        if msg.altitude_ft is not None:
            alt = msg.altitude_ft

        self.update_telemetry(
            timestamp=now,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            groundspeed=spd,
            heading=hdg,
            vertical_rate=vr,
            callsign=cs,
            rf_snr_db=msg.signal_snr_db,
            rf_rss_dbm=None,
            rf_doppler_hz=None
        )

# ==============================================================================
# 2. ADSB TRACK BUFFER
# ==============================================================================
class ADSBTrackBuffer:
    """
    Manages active aircraft tracks in the monitored airspace.
    Performs track aggregation, deduplication, and stale track pruning.
    """
    def __init__(self, max_stale_age_s: float = 120.0):
        self.tracks: Dict[str, ADSBTrack] = {}
        self.max_stale_age_s = max_stale_age_s
        self.observed_icaos_at_timestamp: Dict[float, set] = {}

    def get_or_create_track(self, icao24: str, callsign: str = "UNKNOWN") -> ADSBTrack:
        icao_clean = icao24.upper().strip()
        if icao_clean not in self.tracks:
            self.tracks[icao_clean] = ADSBTrack(icao_clean, callsign=callsign)
        return self.tracks[icao_clean]

    def add_message(self, msg: DecodedADSBMessage, current_time: Optional[float] = None):
        """Processes a single decoded ADS-B message."""
        track = self.get_or_create_track(msg.icao24, msg.callsign or "UNKNOWN")
        track.update_from_message(msg, current_time=current_time)

    def ingest_telemetry_frame(self, record: Dict[str, Any]):
        """Ingests a telemetry dictionary into the track buffer."""
        icao24 = str(record.get('icao24', '')).upper().strip()
        if not icao24:
            return

        track = self.get_or_create_track(icao24, record.get('callsign', 'UNKNOWN'))

        # Extract timestamp
        raw_ts = record.get('timestamp_epoch', record.get('timestamp'))
        if isinstance(raw_ts, (int, float)):
            ts = float(raw_ts)
        else:
            ts = time.time()

        if ts not in self.observed_icaos_at_timestamp:
            self.observed_icaos_at_timestamp[ts] = set()

        collision = (icao24 in self.observed_icaos_at_timestamp[ts])
        self.observed_icaos_at_timestamp[ts].add(icao24)

        track.update_telemetry(
            timestamp=ts,
            latitude=float(record.get('latitude', track.latitude)),
            longitude=float(record.get('longitude', track.longitude)),
            altitude=float(record.get('altitude', track.altitude)),
            groundspeed=float(record.get('groundspeed', track.groundspeed)),
            heading=float(record.get('heading', track.heading)),
            vertical_rate=float(record.get('vertical_rate', track.vertical_rate)),
            callsign=record.get('callsign', track.callsign),
            rf_snr_db=record.get('estimated_snr_db'),
            rf_rss_dbm=record.get('estimated_rss_dbm'),
            rf_doppler_hz=record.get('estimated_doppler_hz'),
            icao_collision_detected=collision
        )

    def get_active_tracks(self, current_time: Optional[float] = None) -> List[ADSBTrack]:
        """Returns list of currently active aircraft tracks."""
        now = current_time if current_time is not None else time.time()
        active = []
        for track in self.tracks.values():
            if len(track.history) > 0:
                age = now - track.last_timestamp
                # If timestamp is synthetic/replay (very different from now), include all tracks with history
                if age <= self.max_stale_age_s or abs(age) > 86400.0:
                    active.append(track)
        return active

    def prune_stale_tracks(self, current_time: Optional[float] = None):
        """Removes tracks that have ceased broadcasting."""
        now = current_time if current_time is not None else time.time()
        stale_keys = []
        for icao, track in self.tracks.items():
            age = now - track.last_timestamp
            if age > self.max_stale_age_s:
                stale_keys.append(icao)
        for k in stale_keys:
            del self.tracks[k]

    def clear(self):
        self.tracks.clear()
        self.observed_icaos_at_timestamp.clear()

# ==============================================================================
# 3. LIVE ST-GNN INFERENCE PIPELINE
# ==============================================================================
class LiveSTGNNPipeline:
    """
    Coordinates live data formatting, spatial graph creation, temporal sliding window
    assembly, ST-GNN evaluation, and explainability on live aircraft tracks.
    """
    def __init__(
        self,
        st_gnn_model_path: str = "models/st_gnn_official.pt",
        spatial_model_path: str = "models/gnn_edge_aware_repaired.pt",
        threshold_path: str = "models/st_gnn_calibrated_threshold.json",
        scaler_node_path: str = "models/scaler_node_repaired.joblib",
        scaler_edge_path: str = "models/scaler_edge_repaired.joblib",
        scaler_kin_path: str = "models/scaler_kinematic.joblib"
    ):
        self.calibrated_threshold = 0.20
        if os.path.exists(threshold_path):
            try:
                with open(threshold_path, "r") as f:
                    meta = json.load(f)
                    self.calibrated_threshold = float(meta.get("calibrated_threshold", 0.20))
            except Exception:
                pass

        # Load scalers
        self.scaler_node = joblib.load(scaler_node_path)
        self.scaler_edge = joblib.load(scaler_edge_path)
        self.scaler_kin = joblib.load(scaler_kin_path)

        # Load PyG Spatial GNN
        self.spatial_model = RepairedEdgeAwareGNN(len(COORD_FREE_NODE_FEATURES), len(EDGE_FEATURES))
        self.spatial_model.load_state_dict(torch.load(spatial_model_path, map_location='cpu'))
        self.spatial_model.eval()

        # Load Unified Multi-Stream ST-GNN
        self.full_st_gnn = AeroGuardSTGNN(
            spatial_model=self.spatial_model,
            num_kinematic_temporal_feats=len(KINEMATIC_TEMPORAL_FEATS),
            num_kinematic_branch_feats=len(KINEMATIC_BRANCH_FEATS),
            fusion_mode='weighted_prob'
        )
        self.full_st_gnn.load_state_dict(torch.load(st_gnn_model_path, map_location='cpu'))
        self.full_st_gnn.eval()

        # Initialize XAI Explainer
        self.explainer = AeroGuardUnifiedExplainer(self.full_st_gnn)

    def build_graph_snapshot(
        self,
        active_tracks: List[ADSBTrack]
    ) -> Tuple[Data, List[Dict[str, str]], List[Dict[str, float]]]:
        """
        Constructs PyG Data graph snapshot from active aircraft tracks:
        1. 12D Coordinate-free node features.
        2. 4D Relational edge attributes for aircraft pairs within 100 km.
        3. Scaled with pre-fitted official scalers.
        """
        n = len(active_tracks)
        node_identities = []
        raw_node_rows = []

        # 1. Build Node Features
        node_features_raw = []
        for track in active_tracks:
            node_identities.append({
                'icao24': track.icao24,
                'callsign': track.callsign
            })

            cur_obs = track.history[-1] if len(track.history) > 0 else None
            hdg_rad = math.radians(track.heading)

            # Map RF features honestly:
            # If observed from IQ envelope/power, use observed; if unobserved, use pre-fitted scaler mean
            # so that (val - mean) / scale = 0.0 (neutral baseline).
            if track.rf_observed and track.estimated_snr_db is not None:
                snr = track.estimated_snr_db
                rss = track.estimated_rss_dbm if track.estimated_rss_dbm is not None else -86.0
                dop = track.estimated_doppler_hz if track.estimated_doppler_hz is not None else -771.4
            else:
                # Scaler training means for neutral RF imputation
                snr = float(self.scaler_node.mean_[COORD_FREE_NODE_FEATURES.index('estimated_snr_db')])
                rss = float(self.scaler_node.mean_[COORD_FREE_NODE_FEATURES.index('estimated_rss_dbm')])
                dop = float(self.scaler_node.mean_[COORD_FREE_NODE_FEATURES.index('estimated_doppler_hz')])

            raw_row = {
                'altitude': track.altitude,
                'groundspeed': track.groundspeed,
                'heading_sin': math.sin(hdg_rad),
                'heading_cos': math.cos(hdg_rad),
                'vertical_rate': track.vertical_rate,
                'acceleration': cur_obs.acceleration if cur_obs else 0.0,
                'turn_rate': cur_obs.turn_rate if cur_obs else 0.0,
                'vertical_acceleration': cur_obs.vertical_acceleration if cur_obs else 0.0,
                'climb_rate_change': cur_obs.climb_rate_change if cur_obs else 0.0,
                'estimated_doppler_hz': dop,
                'estimated_rss_dbm': rss,
                'estimated_snr_db': snr
            }
            raw_node_rows.append(raw_row)
            node_features_raw.append([raw_row[feat] for feat in COORD_FREE_NODE_FEATURES])

        if n > 0:
            node_features_scaled = self.scaler_node.transform(node_features_raw)
            x_tensor = torch.tensor(node_features_scaled, dtype=torch.float)
        else:
            x_tensor = torch.zeros((0, len(COORD_FREE_NODE_FEATURES)), dtype=torch.float)

        # 2. Build Relational Edges (100 km threshold)
        edge_index = []
        raw_edges = []

        if n > 1:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    t_i = active_tracks[i]
                    t_j = active_tracks[j]

                    dist_m = haversine_distance_m(t_i.latitude, t_i.longitude, t_j.latitude, t_j.longitude)
                    if dist_m <= PROXIMITY_THRESHOLD_M:
                        alt_diff = t_i.altitude - t_j.altitude
                        vel_diff = t_i.groundspeed - t_j.groundspeed
                        hdg_diff = (t_i.heading - t_j.heading + 180.0) % 360.0 - 180.0

                        edge_index.append([i, j])
                        raw_edges.append([dist_m, alt_diff, vel_diff, hdg_diff])

        if len(edge_index) > 0:
            edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_scaled = self.scaler_edge.transform(raw_edges)
            edge_attr_t = torch.tensor(edge_attr_scaled, dtype=torch.float)
        else:
            edge_index_t = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_t = torch.zeros((0, len(EDGE_FEATURES)), dtype=torch.float)

        data = Data(x=x_tensor, edge_index=edge_index_t, edge_attr=edge_attr_t)
        return data, node_identities, raw_node_rows

    def build_temporal_and_kinematic_tensors(
        self,
        active_tracks: List[ADSBTrack],
        spatial_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
        """
        Constructs causal sliding observation sequences (T=5) and kinematic branch inputs:
        - temporal_seq: [N, 5, 25] where 25 = 16 (spatial embedding) + 9 (temporal kinematics).
        - temporal_mask: [N, 5] boolean mask (True for valid historical observations).
        - kinematic_branch_feats: [N, 11] scaled physical residuals.
        """
        n = len(active_tracks)
        w_size = TEMPORAL_WINDOW_T

        seq_tensor = np.zeros((n, w_size, 25), dtype=np.float32)
        mask_tensor = np.zeros((n, w_size), dtype=bool)
        raw_kin_rows = []
        kin_branch_raw_list = []

        spatial_emb_np = spatial_embeddings.detach().cpu().numpy()

        for idx, track in enumerate(active_tracks):
            hist = list(track.history)
            hist_len = min(len(hist), w_size)
            offset = w_size - hist_len

            # Trailing alignment for causal window
            cur_emb = spatial_emb_np[idx] if idx < len(spatial_emb_np) else np.zeros(16, dtype=np.float32)

            for h_i, obs in enumerate(hist[-hist_len:]):
                slot = offset + h_i
                hdg_rad = math.radians(obs.heading)

                # Standardize 9D temporal kinematics using scaler_node
                raw_9d = [
                    obs.altitude,
                    obs.groundspeed,
                    math.sin(hdg_rad),
                    math.cos(hdg_rad),
                    obs.vertical_rate,
                    obs.acceleration,
                    obs.turn_rate,
                    obs.vertical_acceleration,
                    obs.climb_rate_change,
                    # dummy RF features to match scaler_node dimensionality
                    float(self.scaler_node.mean_[9]),
                    float(self.scaler_node.mean_[10]),
                    float(self.scaler_node.mean_[11])
                ]
                scaled_12d = self.scaler_node.transform([raw_9d])[0]
                scaled_9d = scaled_12d[:9] # First 9 are KINEMATIC_TEMPORAL_FEATS

                # Combine spatial embedding (16D) + temporal kinematics (9D) = 25D
                seq_tensor[idx, slot, :16] = cur_emb
                seq_tensor[idx, slot, 16:] = scaled_9d
                mask_tensor[idx, slot] = True

            # Current snapshot kinematic residual features (11D)
            latest_obs = hist[-1] if len(hist) > 0 else TrackObservation(
                timestamp=0, latitude=0, longitude=0, altitude=30000,
                groundspeed=400, heading=0, vertical_rate=0
            )

            raw_kin = {
                'disp_ratio': latest_obs.disp_ratio,
                'alt_residual': latest_obs.alt_residual,
                'accel_residual': latest_obs.accel_residual,
                'turn_residual': latest_obs.turn_residual,
                'icao_collision': latest_obs.icao_collision,
                'track_jump_flag': latest_obs.track_jump_flag,
                'vertical_rate': latest_obs.vertical_rate,
                'acceleration': latest_obs.acceleration,
                'turn_rate': latest_obs.turn_rate,
                'vertical_acceleration': latest_obs.vertical_acceleration,
                'climb_rate_change': latest_obs.climb_rate_change
            }
            raw_kin_rows.append(raw_kin)
            kin_branch_raw_list.append([raw_kin[feat] for feat in KINEMATIC_BRANCH_FEATS])

        if n > 0:
            kin_branch_scaled = self.scaler_kin.transform(kin_branch_raw_list)
            kin_tensor = torch.tensor(kin_branch_scaled, dtype=torch.float)
        else:
            kin_tensor = torch.zeros((0, len(KINEMATIC_BRANCH_FEATS)), dtype=torch.float)

        return (
            torch.tensor(seq_tensor, dtype=torch.float),
            torch.tensor(mask_tensor, dtype=torch.bool),
            kin_tensor,
            raw_kin_rows
        )

    def predict_tracks(self, active_tracks: List[ADSBTrack]) -> List[Dict[str, Any]]:
        """
        Executes full forward pass of ST-GNN on active airspace tracks.
        Returns per-aircraft predictions with stream probabilities and threat assessment.
        """
        n = len(active_tracks)
        if n == 0:
            return []

        # 1. Spatial graph snapshot
        graph_snap, identities, raw_nodes = self.build_graph_snapshot(active_tracks)

        with torch.no_grad():
            # 2. Extract spatial node embeddings [N, 16]
            h_spatial = self.spatial_model.extract_spatial_embeddings(
                graph_snap.x, graph_snap.edge_index, graph_snap.edge_attr
            )

            # 3. Assemble temporal sliding windows & kinematic branch
            seq_t, mask_t, kin_t, raw_kins = self.build_temporal_and_kinematic_tensors(active_tracks, h_spatial)

            # 4. Multi-stream ST-GNN forward pass
            preds = self.full_st_gnn(
                x=graph_snap.x,
                edge_index=graph_snap.edge_index,
                edge_attr=graph_snap.edge_attr,
                temporal_seq=seq_t,
                temporal_mask=mask_t,
                kinematic_feats=kin_t
            )

        fused_probs = preds['prob_fused_weighted'].cpu().numpy()
        spatial_probs = preds['prob_spatial'].cpu().numpy()
        temporal_probs = preds['prob_temporal'].cpu().numpy()
        kinematic_probs = preds['prob_kinematic'].cpu().numpy()

        results = []
        for i in range(n):
            track = active_tracks[i]
            p_fused = float(fused_probs[i])
            is_anomaly = bool(p_fused >= self.calibrated_threshold)

            threat_level = "CRITICAL" if p_fused >= 0.70 else ("ELEVATED" if is_anomaly else "NOMINAL")

            results.append({
                'id': track.icao24,
                'icao24': track.icao24,
                'callsign': track.callsign,
                'lat': track.latitude,
                'lon': track.longitude,
                'latitude': track.latitude,
                'longitude': track.longitude,
                'altitude': track.altitude,
                'groundspeed': track.groundspeed,
                'heading': track.heading,
                'vertical_rate': track.vertical_rate,
                'verticalRate': track.vertical_rate,
                'st_gnn_prob': round(p_fused, 4),
                'prob_fused': round(p_fused, 4),
                'prob_spatial': round(float(spatial_probs[i]), 4),
                'prob_temporal': round(float(temporal_probs[i]), 4),
                'prob_kinematic': round(float(kinematic_probs[i]), 4),
                'is_anomaly': is_anomaly,
                'isAnomaly': is_anomaly,
                'threat_level': threat_level,
                'threshold': self.calibrated_threshold,
                'rf_observed': track.rf_observed,
                'signal_source': 'SYNTHETIC_SDR' if track.rf_observed else 'LIVE_BUFFER',
                'history_length': len(track.history)
            })

        return results

    def explain_track(self, icao24: str, active_tracks: List[ADSBTrack]) -> Optional[Dict[str, Any]]:
        """
        Generates unified 4-stream Explainable AI dossier for target active track.
        """
        target_idx = -1
        for idx, track in enumerate(active_tracks):
            if track.icao24.upper() == icao24.upper():
                target_idx = idx
                break

        if target_idx == -1:
            return None

        graph_snap, identities, raw_nodes = self.build_graph_snapshot(active_tracks)

        with torch.no_grad():
            h_spatial = self.spatial_model.extract_spatial_embeddings(
                graph_snap.x, graph_snap.edge_index, graph_snap.edge_attr
            )
            seq_t, mask_t, kin_t, raw_kins = self.build_temporal_and_kinematic_tensors(active_tracks, h_spatial)
            preds = self.full_st_gnn(
                x=graph_snap.x,
                edge_index=graph_snap.edge_index,
                edge_attr=graph_snap.edge_attr,
                temporal_seq=seq_t,
                temporal_mask=mask_t,
                kinematic_feats=kin_t
            )
            fused_p = float(preds['prob_fused_weighted'][target_idx].item())

        # Target record dict
        record = pd.Series(raw_kins[target_idx])
        record['icao24'] = active_tracks[target_idx].icao24
        record['callsign'] = active_tracks[target_idx].callsign

        # Sub-slice single-item tensors for explanation (explainer expects [T, 25] and [T])
        target_seq = seq_t[target_idx]
        target_mask = mask_t[target_idx]
        target_kin = kin_t[target_idx:target_idx+1]

        dossier = self.explainer.generate_full_explanation(
            record=record,
            spatial_graph=graph_snap,
            target_node_idx=target_idx,
            temporal_seq=target_seq,
            temporal_mask=target_mask,
            scaled_kin_tensor=target_kin,
            fused_prob=fused_p,
            threshold=self.calibrated_threshold,
            ground_truth_label=None,
            node_identities=identities
        )
        return dossier

# Global singleton live inference pipeline instance
live_pipeline = LiveSTGNNPipeline()
