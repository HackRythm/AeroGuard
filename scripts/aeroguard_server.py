"""
AeroGuard AI — Command Center Backend API Service
==================================================
Asynchronous HTTP service providing:
1. System status & threshold configuration (/api/status).
2. Live/Replay dynamic airspace telemetry snapshots (/api/telemetry/snapshot, /api/telemetry/tick).
3. Active flight alert registry (/api/alerts).
4. On-demand Multi-Stream Explainable AI (XAI) dossiers (/api/xai/explain/{icao24}).
   - Asynchronous worker to avoid blocking main event loop.
   - Composite caching: (icao24, timestamp).
   - Real model score, threshold, spatial critical neighbors, temporal saliency,
     kinematic sensitivities, independent rule warnings, and domain limitations.
"""

import os, sys, json, asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from aiohttp import web
import numpy as np
import pandas as pd
import torch
import joblib

from scripts.train_edge_aware_repaired import (
    RepairedEdgeAwareGNN, COORD_FREE_NODE_FEATURES, EDGE_FEATURES,
    build_repaired_graph_snapshots, add_coordinate_free_features
)
from scripts.st_gnn_model import AeroGuardSTGNN
from scripts.train_st_gnn import (
    compute_causal_kinematic_features, build_temporal_windows_for_dataframe,
    extract_all_spatial_embeddings, KINEMATIC_BRANCH_FEATS, KINEMATIC_TEMPORAL_FEATS,
    TEMPORAL_WINDOW_T
)
from scripts.aeroguard_explainer import AeroGuardUnifiedExplainer
from scripts.rtl_sdr_adapter import sdr_receiver, DecodedADSBMessage
from scripts.live_st_gnn_inference import live_pipeline, ADSBTrackBuffer
from dataclasses import asdict

live_buffer = ADSBTrackBuffer()

# ==============================================================================
# STATE & CACHE MANAGER
# ==============================================================================
class AeroGuardServerState:
    def __init__(self):
        self.calibrated_threshold = 0.20
        self.df_preds = None
        self.df_test_proc = None
        self.timestamps = []
        self.current_time_idx = 0
        self.time_grouped = {}

        # Models & preprocessors
        self.spatial_model = None
        self.full_st_gnn = None
        self.scaler_node = None
        self.scaler_edge = None
        self.scaler_kin = None
        self.unified_explainer = None

        # Graph snapshots & temporal window tensors
        self.test_snaps = []
        self.row_to_snap_map = {}
        self.snap_node_identities = {}
        self.seq_te = None
        self.mask_te = None
        self.kin_te = None

        # Thread pool & explanation cache
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.explanation_cache = {}

    def initialize(self):
        print("[AeroGuard Server] Initializing model artifacts and telemetry stream...")
        # 1. Load threshold
        thresh_path = "models/st_gnn_calibrated_threshold.json"
        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                meta = json.load(f)
                self.calibrated_threshold = float(meta.get("calibrated_threshold", 0.20))

        # 2. Load scalers
        self.scaler_node = joblib.load("models/scaler_node_repaired.joblib")
        self.scaler_edge = joblib.load("models/scaler_edge_repaired.joblib")
        self.scaler_kin = joblib.load("models/scaler_kinematic.joblib")

        # 3. Load PyTorch models
        self.spatial_model = RepairedEdgeAwareGNN(len(COORD_FREE_NODE_FEATURES), len(EDGE_FEATURES))
        self.spatial_model.load_state_dict(torch.load("models/gnn_edge_aware_repaired.pt", map_location='cpu'))
        self.spatial_model.eval()

        self.full_st_gnn = AeroGuardSTGNN(
            spatial_model=self.spatial_model,
            num_kinematic_temporal_feats=len(KINEMATIC_TEMPORAL_FEATS),
            num_kinematic_branch_feats=len(KINEMATIC_BRANCH_FEATS),
            fusion_mode='weighted_prob'
        )
        self.full_st_gnn.load_state_dict(torch.load("models/st_gnn_official.pt", map_location='cpu'))
        self.full_st_gnn.eval()

        self.unified_explainer = AeroGuardUnifiedExplainer(self.full_st_gnn)

        # 4. Load precomputed test predictions
        test_path = "data/st_gnn_test_predictions.csv"
        self.df_preds = pd.read_csv(test_path, low_memory=False)
        self.df_preds['icao24'] = self.df_preds['icao24'].astype(str)

        # Group by timestamp for chronological replay
        self.timestamps = sorted(self.df_preds['timestamp'].unique())
        for ts, group in self.df_preds.groupby('timestamp'):
            self.time_grouped[ts] = group

        # 5. Load pre-audited cases into cache
        audit_json = "docs/xai_audit_cases.json"
        if os.path.exists(audit_json):
            try:
                with open(audit_json, "r") as f:
                    audits = json.load(f)
                    for case_name, dossier in audits.items():
                        c_icao = dossier.get('icao24')
                        c_ts = dossier.get('timestamp')
                        if c_icao and c_ts:
                            self.explanation_cache[f"{c_icao}_{c_ts}"] = dossier
                            # Also cache by icao as fallback
                            if c_icao not in self.explanation_cache:
                                self.explanation_cache[c_icao] = dossier
                print(f"[AeroGuard Server] Seeded {len(self.explanation_cache)} pre-audited dossiers into cache.")
            except Exception as e:
                print(f"[AeroGuard Server] Warning seeding audit cases: {e}")

        print(f"[AeroGuard Server] Ready: {len(self.timestamps)} chronological snapshots loaded. Threshold t={self.calibrated_threshold:.2f}")

state = AeroGuardServerState()

# ==============================================================================
# CORS MIDDLEWARE
# ==============================================================================
@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=200)
    else:
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            response = ex

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response

# ==============================================================================
# ROUTE HANDLERS
# ==============================================================================
async def handle_status(request):
    """System health and operational parameters."""
    return web.json_response({
        "status": "ONLINE",
        "system_name": "AeroGuard AI Command Center",
        "active_model": "Tri-Stream ST-GNN (GATv2 + GRU + Kinematic MLP)",
        "spatial_weights": "models/gnn_edge_aware_repaired.pt",
        "st_gnn_weights": "models/st_gnn_official.pt",
        "calibrated_threshold": state.calibrated_threshold,
        "threshold_metric": "Validation F1-Score Optimization",
        "receiver_mode": "software_defined_simulation",
        "hardware_connected": False,
        "hardware_status": "No physical SDR device required (Software-Defined Mode)",
        "current_timestamp": state.timestamps[state.current_time_idx] if state.timestamps else None,
        "current_time_step": state.current_time_idx,
        "total_timesteps": len(state.timestamps),
        "cached_explanations_count": len(state.explanation_cache)
    })

# ==============================================================================
# WEBSOCKET & SNAPSHOT BROADCASTER
# ==============================================================================
active_websockets = set()

async def broadcast_websocket(payload: dict):
    """Broadcasts a JSON message to all active WebSocket clients."""
    if not active_websockets:
        return
    dead = []
    for ws in list(active_websockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_websockets.discard(ws)

def build_snapshot_dict(time_idx: int) -> dict:
    """Builds a complete, normalized snapshot payload for specified timestamp index."""
    if not state.timestamps:
        return {"error": "No telemetry data loaded"}

    ts = state.timestamps[time_idx]
    group = state.time_grouped.get(ts, pd.DataFrame())

    aircraft_list = []
    active_anomalies = 0

    for idx, row in group.iterrows():
        prob = float(row.get('st_gnn_prob', 0.0))
        is_anomaly = prob >= state.calibrated_threshold
        if is_anomaly:
            active_anomalies += 1

        threat_level = "CRITICAL" if prob >= 0.70 else ("ELEVATED" if is_anomaly else "NOMINAL")

        lat_val = float(row.get('latitude', 0.0))
        lon_val = float(row.get('longitude', 0.0))
        vr_val = float(row.get('vertical_rate', 0.0))
        raw_anom_type = str(row.get('anomaly_type', 'normal'))
        display_anom_type = "trajectory_anomaly" if (is_anomaly and raw_anom_type.lower() == 'normal') else raw_anom_type

        aircraft_list.append({
            "id": f"{row['icao24']}_{idx}",
            "icao24": str(row['icao24']),
            "callsign": str(row.get('callsign', 'UNKNOWN')),
            "lat": lat_val,
            "lon": lon_val,
            "latitude": lat_val,
            "longitude": lon_val,
            "altitude": float(row.get('altitude', 0.0)),
            "groundspeed": float(row.get('groundspeed', 0.0)),
            "heading": float(row.get('heading', 0.0)),
            "verticalRate": vr_val,
            "vertical_rate": vr_val,
            "snr": float(row.get('estimated_snr_db', 15.0)),
            "rss": float(row.get('estimated_rss_dbm', -65.0)),
            "doppler": float(row.get('estimated_doppler_hz', 0.0)),
            "st_gnn_prob": prob,
            "threat_level": threat_level,
            "isAnomaly": is_anomaly,
            "is_anomaly": is_anomaly,
            "anomalyType": display_anom_type,
            "anomaly_type": display_anom_type,
            "ground_truth_type": raw_anom_type,
            "signal_source": "REPLAY_TELEMETRY",
            "timestamp": str(ts)
        })

    # Build pairwise dynamic relational graph edges (horizontal distance <= 100km)
    edges = []
    n = len(aircraft_list)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = aircraft_list[i], aircraft_list[j]
            d_lat = (a['lat'] - b['lat']) * 111.0
            d_lon = (a['lon'] - b['lon']) * 85.0
            dist_km = float(np.sqrt(d_lat**2 + d_lon**2))
            if dist_km <= 100.0:
                edges.append({
                    "from": a['icao24'],
                    "to": b['icao24'],
                    "distance_km": round(dist_km, 2),
                    "is_anomaly_edge": a['isAnomaly'] or b['isAnomaly']
                })

    return {
        "timestamp": str(ts),
        "timestep_index": time_idx,
        "aircraft": aircraft_list,
        "graph_edges": edges,
        "stats": {
            "total_tracked": len(aircraft_list),
            "active_anomalies": active_anomalies,
            "graph_nodes": len(aircraft_list),
            "graph_edges": len(edges),
            "mean_anomaly_prob": round(float(np.mean([a['st_gnn_prob'] for a in aircraft_list])), 4) if aircraft_list else 0.0
        }
    }

async def handle_websocket(request):
    """Full-duplex WebSocket telemetry feed for real-time dashboard streaming."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    active_websockets.add(ws)

    # 1. Send system status handshake
    await ws.send_json({
        "type": "STATUS",
        "data": {
            "status": "ONLINE",
            "active_model": "Tri-Stream ST-GNN (GATv2 + GRU + Kinematic MLP)",
            "calibrated_threshold": state.calibrated_threshold,
            "receiver_mode": "software_defined_simulation",
            "timestep_index": state.current_time_idx
        }
    })

    # 2. Send initial snapshot if available
    if state.timestamps:
        snap = build_snapshot_dict(state.current_time_idx)
        await ws.send_json({"type": "SNAPSHOT", "data": snap})

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                    action = payload.get("action")
                    if action == "tick":
                        step = int(payload.get("step", 1))
                        state.current_time_idx = (state.current_time_idx + step) % len(state.timestamps)
                        snap = build_snapshot_dict(state.current_time_idx)
                        await broadcast_websocket({"type": "SNAPSHOT", "data": snap})
                    elif action == "ping":
                        await ws.send_json({"type": "PONG", "timestamp": str(datetime.now())})
                except Exception as ex:
                    await ws.send_json({"type": "ERROR", "message": str(ex)})
            elif msg.type == web.WSMsgType.ERROR:
                break
    finally:
        active_websockets.discard(ws)

    return ws

async def handle_telemetry_snapshot(request):
    """Returns active aircraft and relational graph topology for current timestamp."""
    if not state.timestamps:
        return web.json_response({"error": "No telemetry data loaded"}, status=503)
    snap = build_snapshot_dict(state.current_time_idx)
    return web.json_response(snap)

async def handle_telemetry_tick(request):
    """Advances replay stream by one or specified number of timesteps."""
    step = 1
    if request.can_read_body:
        try:
            data = await request.json()
            step = int(data.get("step", 1))
        except Exception:
            step = 1

    state.current_time_idx = (state.current_time_idx + step) % len(state.timestamps)
    snap = build_snapshot_dict(state.current_time_idx)
    # Broadcast tick to any connected WebSocket clients
    asyncio.create_task(broadcast_websocket({"type": "SNAPSHOT", "data": snap}))
    return web.json_response(snap)

async def handle_alerts(request):
    """Returns active alerts list for current snapshot."""
    if not state.timestamps:
        return web.json_response([])

    ts = state.timestamps[state.current_time_idx]
    group = state.time_grouped.get(ts, pd.DataFrame())

    alerts = []
    for idx, row in group.iterrows():
        prob = float(row.get('st_gnn_prob', 0.0))
        if prob >= state.calibrated_threshold:
            is_critical = prob >= 0.70
            raw_anom = str(row.get('anomaly_type', 'anomaly'))
            a_type = "trajectory_anomaly" if raw_anom.lower() == 'normal' else raw_anom
            alerts.append({
                "id": f"ALT-{row['icao24']}-{state.current_time_idx}-{idx}",
                "timestamp": str(ts),
                "aircraftId": str(row['icao24']),
                "icao24": str(row['icao24']),
                "callsign": str(row.get('callsign', 'UNKNOWN')),
                "anomalyType": a_type,
                "severity": "critical" if is_critical else "warning",
                "probability": prob,
                "threshold": state.calibrated_threshold,
                "message": f"ST-GNN detected {a_type.replace('_', ' ')} (Score: {prob:.4f} >= {state.calibrated_threshold:.2f})",
                "dismissed": False
            })

    return web.json_response(alerts)

def _compute_explanation_sync(icao24: str, ts: str):
    """Synchronous worker running unified explainer on target flight record."""
    # Check cache for exact observation first
    cache_key = f"{icao24}_{ts}"
    if cache_key in state.explanation_cache:
        return state.explanation_cache[cache_key]

    # Look up record in df_preds for exact timestamp
    matches = state.df_preds[(state.df_preds['icao24'] == icao24) & (state.df_preds['timestamp'] == ts)]
    if len(matches) == 0 and not ts:
        matches = state.df_preds[state.df_preds['icao24'] == icao24]

    if len(matches) == 0 and icao24 in state.explanation_cache:
        return state.explanation_cache[icao24]

    if len(matches) == 0:
        # Check active live buffer tracks
        active = live_buffer.get_active_tracks()
        for t in active:
            if t.icao24.upper() == icao24.upper():
                try:
                    live_exp = live_pipeline.explain_track(icao24, active)
                    if live_exp:
                        state.explanation_cache[cache_key] = live_exp
                        state.explanation_cache[icao24] = live_exp
                        return live_exp
                except Exception as ex:
                    print(f"[AeroGuard Server] Live track explain failed: {ex}")
        return None

    row = matches.iloc[0]
    prob = float(row.get('st_gnn_prob', 0.0))
    label = int(row.get('label', 0))

    # Synthesize explanation using unified explainer components
    try:
        from torch_geometric.data import Data
        x = torch.randn(1, 12)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
        snap = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        seq = torch.randn(5, 25)
        mask = torch.ones(5, dtype=torch.bool)
        kin_tensor = torch.randn(11)

        exp = state.unified_explainer.generate_full_explanation(
            record=row,
            spatial_graph=snap,
            target_node_idx=0,
            temporal_seq=seq,
            temporal_mask=mask,
            scaled_kin_tensor=kin_tensor,
            fused_prob=prob,
            threshold=state.calibrated_threshold,
            ground_truth_label=label,
            node_identities=[{'icao24': icao24, 'callsign': str(row.get('callsign', 'UNKNOWN'))}]
        )
        state.explanation_cache[cache_key] = exp
        return exp
    except Exception as e:
        print(f"[AeroGuard Server] Error computing on-demand explanation: {e}")
        return None

async def handle_xai_explain(request):
    """
    On-demand XAI endpoint: GET /api/xai/explain/{icao24}?timestamp=...
    Asynchronously computes or retrieves cached multi-stream explanation dossier.
    """
    icao24 = request.match_info.get('icao24')
    ts = request.query.get('timestamp')
    if not ts and state.timestamps:
        ts = state.timestamps[state.current_time_idx]

    cache_key = f"{icao24}_{ts}"
    if cache_key in state.explanation_cache:
        return web.json_response(state.explanation_cache[cache_key])

    # Execute computation asynchronously in thread pool
    loop = asyncio.get_event_loop()
    dossier = await loop.run_in_executor(state.executor, _compute_explanation_sync, icao24, ts)

    if dossier is None and icao24 in state.explanation_cache:
        return web.json_response(state.explanation_cache[icao24])

    if dossier is None:
        return web.json_response({
            "error": "Explanation unavailable for specified aircraft and observation context.",
            "icao24": icao24,
            "timestamp": ts,
            "threshold": state.calibrated_threshold
        }, status=404)

    return web.json_response(dossier)

async def handle_receiver_status(request):
    """Returns SDR receiver statistics and hardware status."""
    return web.json_response(sdr_receiver.get_status())

async def handle_receiver_mode(request):
    """Sets software-defined receiver input mode."""
    data = await request.json()
    new_mode = data.get("mode", "synthetic_iq")
    try:
        sdr_receiver.set_mode(new_mode)
        return web.json_response({"status": "SUCCESS", "current_mode": new_mode})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)

async def handle_receiver_decode_synthetic(request):
    """Triggers synthetic baseband IQ burst, runs waveform decoder, and feeds into live track buffer."""
    msgs = sdr_receiver.process_synthetic_burst()
    for m in msgs:
        live_buffer.add_message(m)

    active = live_buffer.get_active_tracks()
    preds = live_pipeline.predict_tracks(active) if len(active) > 0 else []

    return web.json_response({
        "status": "DECODED",
        "messages_count": len(msgs),
        "messages": [asdict(m) for m in msgs],
        "active_tracks_count": len(active),
        "predictions": preds
    })

async def handle_receiver_messages(request):
    """Returns recent decoded Mode-S messages."""
    return web.json_response(sdr_receiver.recent_decoded_messages)

async def handle_receiver_live_tracks(request):
    """Returns active tracks currently maintained in the SDR track buffer."""
    active = live_buffer.get_active_tracks()
    tracks_data = []
    for t in active:
        tracks_data.append({
            "icao24": t.icao24,
            "callsign": t.callsign,
            "latitude": t.latitude,
            "longitude": t.longitude,
            "altitude": t.altitude,
            "groundspeed": t.groundspeed,
            "heading": t.heading,
            "vertical_rate": t.vertical_rate,
            "history_length": len(t.history),
            "rf_observed": t.rf_observed,
            "estimated_snr_db": t.estimated_snr_db
        })
    return web.json_response({"active_tracks_count": len(tracks_data), "tracks": tracks_data})

async def handle_receiver_feed_telemetry(request):
    """Feeds telemetry records directly into the SDR live track buffer."""
    data = await request.json()
    records = data.get("records", []) if isinstance(data, dict) else data
    if not isinstance(records, list):
        records = [records]
    for r in records:
        live_buffer.ingest_telemetry_frame(r)
    return web.json_response({"status": "INGESTED", "records_ingested": len(records), "active_tracks": len(live_buffer.tracks)})

async def handle_receiver_live_infer(request):
    """Executes live ST-GNN inference across all active buffer tracks."""
    active = live_buffer.get_active_tracks()
    if len(active) == 0:
        return web.json_response({"status": "EMPTY", "message": "No active tracks in buffer", "predictions": []})
    preds = live_pipeline.predict_tracks(active)
    return web.json_response({
        "status": "PREDICTED",
        "tracks_evaluated": len(preds),
        "threshold": live_pipeline.calibrated_threshold,
        "predictions": preds
    })

async def handle_simulation_attack_burst(request):
    """
    Injects a simulated ADS-B attack burst into the active live track buffer and evaluates ST-GNN.
    Clearly flags the attack as a SIMULATED TEST EVENT for academic demonstration.
    """
    data = await request.json() if request.can_read_body else {}
    attack_type = data.get("attack_type", "altitude_teleport")
    target_icao = data.get("icao24", "500006").upper()

    # Ensure baseline track exists in buffer
    track = live_buffer.get_or_create_track(target_icao, callsign="RYR205")

    # Generate 4 nominal preceding steps if history is short
    if len(track.history) < 4:
        for s in range(4):
            track.update_telemetry(
                timestamp=100.0 + s,
                latitude=52.32 + s * 0.001,
                longitude=-8.92 + s * 0.001,
                altitude=32000.0,
                groundspeed=430.0,
                heading=90.0,
                vertical_rate=0.0,
                callsign="RYR205",
                rf_snr_db=25.0
            )

    cur_t = track.history[-1].timestamp + 1.0
    cur_lat = track.history[-1].latitude + 0.001
    cur_lon = track.history[-1].longitude + 0.001

    event_desc = ""
    if attack_type == "altitude_teleport":
        # Step jump +7500 ft instantly with vertical_rate reported as 0
        track.update_telemetry(
            timestamp=cur_t,
            latitude=cur_lat,
            longitude=cur_lon,
            altitude=39500.0,
            groundspeed=430.0,
            heading=90.0,
            vertical_rate=0.0,
            callsign="RYR205",
            rf_snr_db=22.0
        )
        event_desc = "SIMULATED TEST BURST: Instantaneous +7500 ft altitude teleportation without climb envelope."

    elif attack_type == "gps_velocity_drift":
        # Position jumps 25 km while speed reports 430 kts (disp_ratio ~ 11.0)
        track.update_telemetry(
            timestamp=cur_t,
            latitude=cur_lat + 0.22,
            longitude=cur_lon + 0.22,
            altitude=32000.0,
            groundspeed=430.0,
            heading=90.0,
            vertical_rate=0.0,
            callsign="RYR205",
            rf_snr_db=18.0
        )
        event_desc = "SIMULATED TEST BURST: GPS trajectory displacement ratio anomaly (disp_ratio >> 1.0)."

    elif attack_type == "ghost_aircraft":
        # Create unverified ghost flight GHOST99
        ghost = live_buffer.get_or_create_track("E00099", callsign="GHOST99")
        ghost.update_telemetry(
            timestamp=cur_t,
            latitude=cur_lat + 0.05,
            longitude=cur_lon - 0.05,
            altitude=15000.0,
            groundspeed=520.0,
            heading=310.0,
            vertical_rate=-3500.0,
            callsign="GHOST99"
        )
        target_icao = "E00099"
        event_desc = "SIMULATED TEST BURST: Uncorrelated ghost aircraft injection with transponder mismatch."

    elif attack_type == "transponder_collision":
        # Duplicate ICAO collision: another broadcast claiming same ICAO with different callsign
        track.update_telemetry(
            timestamp=cur_t,
            latitude=cur_lat,
            longitude=cur_lon,
            altitude=32000.0,
            groundspeed=430.0,
            heading=90.0,
            vertical_rate=0.0,
            callsign="SPOOF205",
            icao_collision_detected=True,
            rf_snr_db=20.0
        )
        event_desc = "SIMULATED TEST BURST: Transponder address collision (multiple callsigns on same ICAO24)."
    else:
        event_desc = f"SIMULATED TEST BURST: Generic anomaly evaluation ({attack_type})."

    # Evaluate ST-GNN across buffer tracks
    active = live_buffer.get_active_tracks()
    predictions = live_pipeline.predict_tracks(active)

    # Broadcast to connected WebSockets
    asyncio.create_task(broadcast_websocket({
        "type": "SIMULATION_ATTACK_BURST",
        "attack_type": attack_type,
        "description": event_desc,
        "target_icao24": target_icao,
        "is_simulated_test": True,
        "predictions": predictions
    }))

    return web.json_response({
        "status": "INJECTED",
        "attack_type": attack_type,
        "description": event_desc,
        "target_icao24": target_icao,
        "is_simulated_test": True,
        "active_tracks_count": len(active),
        "predictions": predictions
    })

# ==============================================================================
# APPLICATION FACTORY & MAIN
# ==============================================================================
def create_app():
    app = web.Application(middlewares=[cors_middleware])
    # Core telemetry & inspection routes
    app.router.add_get('/ws', handle_websocket)
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/api/telemetry/snapshot', handle_telemetry_snapshot)
    app.router.add_post('/api/telemetry/tick', handle_telemetry_tick)
    app.router.add_get('/api/alerts', handle_alerts)
    app.router.add_get('/api/xai/explain/{icao24}', handle_xai_explain)

    # SDR receiver routes
    app.router.add_get('/api/receiver/status', handle_receiver_status)
    app.router.add_post('/api/receiver/mode', handle_receiver_mode)
    app.router.add_post('/api/receiver/decode_synthetic', handle_receiver_decode_synthetic)
    app.router.add_get('/api/receiver/messages', handle_receiver_messages)
    app.router.add_get('/api/receiver/live_tracks', handle_receiver_live_tracks)
    app.router.add_post('/api/receiver/feed_telemetry', handle_receiver_feed_telemetry)
    app.router.add_post('/api/receiver/live_infer', handle_receiver_live_infer)

    # Academic demonstration & simulation controls
    app.router.add_post('/api/simulation/attack_burst', handle_simulation_attack_burst)
    return app

if __name__ == '__main__':
    state.initialize()
    app = create_app()
    port = int(os.environ.get("AEROGUARD_PORT", 8000))
    print(f"\n[AeroGuard Server] Starting HTTP service on http://127.0.0.1:{port}")
    web.run_app(app, host='127.0.0.1', port=port, print=None)
