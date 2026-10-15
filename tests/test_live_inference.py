"""
AeroGuard AI — Unit Tests for Live ST-GNN Inference & Track Buffer
==================================================================
Tests:
1. Discrete kinematic derivative calculations (acceleration, turn_rate, etc.).
2. Causal kinematic residual invariants (disp_ratio, alt_residual, track_jump_flag).
3. RF physical layer observation handling (observed vs honest scaler-mean neutral baseline).
4. Track buffer aggregation, sliding history, and stale track pruning.
5. Relational graph edge formation with 100 km proximity filtering.
6. Chronological sliding window (T=5) causal masking.
7. Multi-stream ST-GNN forward pass and anomaly probability bounds.
8. Live Explainable AI (XAI) explanation dossier generation.
"""

import unittest
import numpy as np
import torch
import math

from scripts.live_st_gnn_inference import (
    ADSBTrack, ADSBTrackBuffer, TrackObservation, LiveSTGNNPipeline,
    haversine_distance_m, KTS_TO_MS, FPM_TO_MS
)
from scripts.rtl_sdr_adapter import DecodedADSBMessage
from scripts.train_edge_aware_repaired import COORD_FREE_NODE_FEATURES

class TestLiveSTGNNInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = LiveSTGNNPipeline()

    def test_haversine_distance(self):
        # Distance between Lisbon (38.7742, -9.1342) and Sintra (38.8029, -9.3817) ~21.5 km
        dist = haversine_distance_m(38.7742, -9.1342, 38.8029, -9.3817)
        self.assertGreater(dist, 20000.0)
        self.assertLess(dist, 23000.0)

    def test_track_kinematics_derivatives(self):
        track = ADSBTrack(icao24="4840D6", callsign="KLM1023")

        # Observation 1: baseline
        track.update_telemetry(
            timestamp=100.0,
            latitude=38.70,
            longitude=-9.10,
            altitude=30000.0,
            groundspeed=400.0,
            heading=90.0,
            vertical_rate=0.0
        )
        self.assertEqual(len(track.history), 1)
        obs1 = track.history[0]
        self.assertEqual(obs1.acceleration, 0.0)
        self.assertEqual(obs1.turn_rate, 0.0)
        self.assertEqual(obs1.disp_ratio, 1.0)

        # Observation 2: dt = 2.0 s, speed +20 kts, heading +10 deg, vert_rate +600 fpm
        track.update_telemetry(
            timestamp=102.0,
            latitude=38.70,
            longitude=-9.095,
            altitude=30020.0,
            groundspeed=420.0,
            heading=100.0,
            vertical_rate=600.0
        )
        self.assertEqual(len(track.history), 2)
        obs2 = track.history[1]
        self.assertAlmostEqual(obs2.dt, 2.0, places=2)
        # Acceleration = (20 * 0.514444) / 2.0 = 5.14444 m/s^2
        self.assertAlmostEqual(obs2.acceleration, (20.0 * KTS_TO_MS) / 2.0, places=3)
        # Turn rate = (100 - 90) / 2.0 = 5.0 deg/s
        self.assertAlmostEqual(obs2.turn_rate, 5.0, places=2)
        # Vertical acceleration = (600 * 0.00508) / 2.0 = 1.524 m/s^2
        self.assertAlmostEqual(obs2.vertical_acceleration, (600.0 * FPM_TO_MS) / 2.0, places=3)
        # Climb rate change = (600 - 0) / 2.0 = 300 fpm/s
        self.assertAlmostEqual(obs2.climb_rate_change, 300.0, places=1)

    def test_causal_residual_invariants(self):
        track = ADSBTrack(icao24="SPOOF1", callsign="GHOST01")
        track.update_telemetry(
            timestamp=100.0,
            latitude=38.70,
            longitude=-9.10,
            altitude=30000.0,
            groundspeed=400.0,
            heading=90.0,
            vertical_rate=0.0
        )
        # Sudden 10 km position teleportation within 1.0s (cyber-physical spoofing jump)
        track.update_telemetry(
            timestamp=101.0,
            latitude=38.79, # ~10 km displacement
            longitude=-9.10,
            altitude=30000.0,
            groundspeed=400.0,
            heading=90.0,
            vertical_rate=0.0
        )
        obs = track.history[-1]
        self.assertEqual(obs.track_jump_flag, 1.0)
        self.assertGreater(obs.disp_ratio, 10.0) # > 10x expected displacement

    def test_rf_feature_handling_honest(self):
        # Case A: Track with unobserved RF characteristics (pure telemetry)
        track_unobs = ADSBTrack(icao24="UNOBS1")
        track_unobs.update_telemetry(
            timestamp=100.0, latitude=38.7, longitude=-9.1,
            altitude=30000, groundspeed=400, heading=90, vertical_rate=0
        )
        self.assertFalse(track_unobs.rf_observed)

        # Build snapshot and check that scaled node features for RF columns are 0.0
        snap, _, _ = self.pipeline.build_graph_snapshot([track_unobs])
        snr_idx = COORD_FREE_NODE_FEATURES.index('estimated_snr_db')
        rss_idx = COORD_FREE_NODE_FEATURES.index('estimated_rss_dbm')
        dop_idx = COORD_FREE_NODE_FEATURES.index('estimated_doppler_hz')

        # When unobserved, standardized value must be 0.0 (scaler mean)
        self.assertAlmostEqual(float(snap.x[0, snr_idx].item()), 0.0, places=4)
        self.assertAlmostEqual(float(snap.x[0, rss_idx].item()), 0.0, places=4)
        self.assertAlmostEqual(float(snap.x[0, dop_idx].item()), 0.0, places=4)

        # Case B: Track with observed RF characteristics (from IQ baseband)
        track_obs = ADSBTrack(icao24="OBS001")
        track_obs.update_telemetry(
            timestamp=100.0, latitude=38.7, longitude=-9.1,
            altitude=30000, groundspeed=400, heading=90, vertical_rate=0,
            rf_snr_db=32.0, rf_rss_dbm=-75.0, rf_doppler_hz=-500.0
        )
        self.assertTrue(track_obs.rf_observed)
        snap_b, _, _ = self.pipeline.build_graph_snapshot([track_obs])
        # Standardized value should differ from 0.0
        self.assertNotEqual(round(float(snap_b.x[0, snr_idx].item()), 2), 0.0)

    def test_track_buffer_and_pruning(self):
        buffer = ADSBTrackBuffer(max_stale_age_s=60.0)

        # Add message
        msg = DecodedADSBMessage(
            raw_hex="8D4840D6202CC371C32CE0576098",
            timestamp="2026-10-14 12:00:00",
            df=17,
            icao24="4840D6",
            tc=4,
            message_type="AIRCRAFT_IDENTIFICATION",
            callsign="KLM1023",
            crc_valid=True
        )
        buffer.add_message(msg, current_time=100.0)
        self.assertIn("4840D6", buffer.tracks)
        self.assertEqual(buffer.tracks["4840D6"].callsign, "KLM1023")

        # Ingest telemetry for a second track at t=100.0
        buffer.ingest_telemetry_frame({
            'icao24': '500006',
            'callsign': 'RYR205',
            'latitude': 38.75,
            'longitude': -9.15,
            'altitude': 32000.0,
            'groundspeed': 450.0,
            'heading': 180.0,
            'vertical_rate': 0.0,
            'timestamp': 100.0
        })
        self.assertEqual(len(buffer.tracks), 2)

        # Pruning check at t=200.0 (age = 100s > 60s max stale)
        buffer.prune_stale_tracks(current_time=200.0)
        self.assertEqual(len(buffer.tracks), 0)

    def test_relational_graph_edge_formation(self):
        # Aircraft 1 & 2 are 25 km apart (<= 100 km threshold)
        # Aircraft 3 is 250 km away (> 100 km threshold)
        t1 = ADSBTrack(icao24="AC001")
        t1.update_telemetry(100.0, 38.70, -9.10, 30000, 400, 90, 0)

        t2 = ADSBTrack(icao24="AC002")
        t2.update_telemetry(100.0, 38.90, -9.10, 31000, 420, 95, 200) # ~22 km from AC001

        t3 = ADSBTrack(icao24="AC003")
        t3.update_telemetry(100.0, 41.00, -9.10, 35000, 480, 180, 0) # ~255 km from AC001

        snap, ids, _ = self.pipeline.build_graph_snapshot([t1, t2, t3])

        # Exactly 2 directed edges: (0 -> 1) and (1 -> 0). Node 2 has 0 edges.
        self.assertEqual(snap.edge_index.shape[1], 2)
        edge_pairs = snap.edge_index.t().tolist()
        self.assertIn([0, 1], edge_pairs)
        self.assertIn([1, 0], edge_pairs)
        self.assertEqual(snap.edge_attr.shape[0], 2)
        self.assertEqual(snap.edge_attr.shape[1], 4)

    def test_temporal_sliding_window_masking(self):
        track_short = ADSBTrack(icao24="SHORT1")
        # Only 2 observations
        track_short.update_telemetry(100.0, 38.7, -9.1, 30000, 400, 90, 0)
        track_short.update_telemetry(101.0, 38.7, -9.1, 30000, 400, 90, 0)

        snap, _, _ = self.pipeline.build_graph_snapshot([track_short])
        h_spatial = self.pipeline.spatial_model.extract_spatial_embeddings(
            snap.x, snap.edge_index, snap.edge_attr
        )
        seq, mask, kin, _ = self.pipeline.build_temporal_and_kinematic_tensors([track_short], h_spatial)

        # T=5 window: first 3 False (padded), last 2 True
        self.assertEqual(seq.shape, (1, 5, 25))
        self.assertEqual(mask.shape, (1, 5))
        self.assertListEqual(mask[0].tolist(), [False, False, False, True, True])

    def test_live_st_gnn_prediction(self):
        t1 = ADSBTrack(icao24="NORM01", callsign="TAP101")
        t2 = ADSBTrack(icao24="ANOM01", callsign="GHOST02")

        # Populate history
        for i in range(5):
            t1.update_telemetry(100.0 + i, 38.70 + i*0.001, -9.10, 30000, 400, 90, 0)

        # Anomaly track with violent trajectory jump
        for i in range(4):
            t2.update_telemetry(100.0 + i, 38.75 + i*0.001, -9.15, 30000, 400, 90, 0)
        t2.update_telemetry(104.0, 38.95, -9.15, 30000, 400, 90, 0) # sudden jump

        preds = self.pipeline.predict_tracks([t1, t2])
        self.assertEqual(len(preds), 2)

        for p in preds:
            self.assertIn('prob_fused', p)
            self.assertIn('prob_spatial', p)
            self.assertIn('prob_temporal', p)
            self.assertIn('prob_kinematic', p)
            self.assertGreaterEqual(p['prob_fused'], 0.0)
            self.assertLessEqual(p['prob_fused'], 1.0)
            self.assertEqual(p['threshold'], 0.20)
            self.assertIn(p['threat_level'], ['NOMINAL', 'ELEVATED', 'CRITICAL'])

    def test_live_xai_explanation(self):
        t = ADSBTrack(icao24="EXPLAIN1", callsign="TESTXAI")
        for i in range(5):
            t.update_telemetry(100.0 + i, 38.70, -9.10, 30000, 400, 90, 0)

        dossier = self.pipeline.explain_track("EXPLAIN1", [t])
        self.assertIsNotNone(dossier)
        self.assertEqual(dossier['icao24'], "EXPLAIN1")
        self.assertIn('spatial_explanation', dossier)
        self.assertIn('temporal_explanation', dossier)
        self.assertIn('kinematic_explanation', dossier)
        self.assertIn('limitations_and_uncertainty', dossier)
        self.assertIn('disclaimer', dossier['kinematic_explanation'])

if __name__ == '__main__':
    unittest.main()
