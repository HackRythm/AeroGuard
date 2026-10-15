"""
AeroGuard AI — End-to-End Software-Defined SDR & ST-GNN Integration Test
========================================================================
Validates the complete end-to-end software-defined chain without physical hardware:
1. Synthetic ADS-B Baseband IQ Generation (1090 MHz, 2 MSPS, PPM, AWGN).
2. Mode-S Baseband Decoding & CRC-24 Verification (G(x) = 0x1FFF409).
3. Aircraft Track Ingestion & Discrete Kinematic Differentiation.
4. Relational Graph Construction & 100 km Edge Formation.
5. Chronological Sliding Window Assembly (T=5) with Observation Masking.
6. Multi-Stream ST-GNN Forward Pass & Calibrated Threshold Anomaly Detection (t=0.20).
7. Unified Multi-Stream Explainable AI (XAI) Attribution Generation.
"""

import unittest
import numpy as np
import torch
from dataclasses import asdict

from scripts.rtl_sdr_adapter import (
    ADSBWaveformModulator, ADSBWaveformDecoder, SyntheticIQSource,
    compute_modes_crc24, verify_modes_crc24
)
from scripts.live_st_gnn_inference import (
    ADSBTrack, ADSBTrackBuffer, LiveSTGNNPipeline
)

class TestEndToEndSDRToSTGNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_rate = 2000000.0 # 2 MSPS standard RTL-SDR sample rate
        cls.modulator = ADSBWaveformModulator()
        cls.decoder = ADSBWaveformDecoder(sample_rate=cls.sample_rate)
        cls.pipeline = LiveSTGNNPipeline()

    def test_full_software_defined_chain(self):
        # ------------------------------------------------------------------
        # Step 1: Synthesize Physically Accurate Mode-S Baseband Waveforms
        # ------------------------------------------------------------------
        # Aircraft 1: Normal Flight RYR205 (ICAO: 500006)
        f1_id = self.modulator.encode_callsign_frame("500006", "RYR205")
        f1_vel = self.modulator.encode_airborne_velocity_frame("500006", speed_kts=420.0, heading_deg=180.0, vert_rate_fpm=0.0)

        # Aircraft 2: Normal Flight KLM1023 (ICAO: 4840D6)
        f2_id = self.modulator.encode_callsign_frame("4840D6", "KLM1023")
        f2_vel = self.modulator.encode_airborne_velocity_frame("4840D6", speed_kts=440.0, heading_deg=90.0, vert_rate_fpm=0.0)

        # Modulate frames into complex IQ samples with 25 dB SNR
        iq1 = self.modulator.synthesize_ppm_baseband(f1_id, sample_rate=self.sample_rate, snr_db=30.0)
        iq2 = self.modulator.synthesize_ppm_baseband(f1_vel, sample_rate=self.sample_rate, snr_db=28.0)
        gap = np.zeros(int(self.sample_rate * 0.001), dtype=np.complex64) # 1 ms silence
        iq3 = self.modulator.synthesize_ppm_baseband(f2_id, sample_rate=self.sample_rate, snr_db=30.0)
        iq4 = self.modulator.synthesize_ppm_baseband(f2_vel, sample_rate=self.sample_rate, snr_db=25.0)

        full_iq_stream = np.concatenate([iq1, gap, iq2, gap, iq3, gap, iq4])
        self.assertIsInstance(full_iq_stream, np.ndarray)
        self.assertEqual(full_iq_stream.dtype, np.complex64)
        self.assertGreater(len(full_iq_stream), 4000)

        # ------------------------------------------------------------------
        # Step 2: Baseband Envelope Detection, Preamble Correlation & CRC
        # ------------------------------------------------------------------
        decoded_messages = self.decoder.decode_iq(full_iq_stream, source_label="SYNTHETIC_SDR")
        self.assertGreaterEqual(len(decoded_messages), 3, "Expected at least 3 decoded Mode-S messages from IQ burst")

        decoded_icaos = {m.icao24 for m in decoded_messages}
        self.assertIn("500006", decoded_icaos)
        self.assertIn("4840D6", decoded_icaos)

        for msg in decoded_messages:
            self.assertTrue(msg.crc_valid, f"CRC check failed for {msg.raw_hex}")

        # ------------------------------------------------------------------
        # Step 3: Aircraft Track Ingestion & Discrete Kinematic History
        # ------------------------------------------------------------------
        buffer = ADSBTrackBuffer()

        # Feed decoded messages
        for msg in decoded_messages:
            buffer.add_message(msg, current_time=100.0)

        # Add spatial coordinate context for the two aircraft within 40 km
        t1 = buffer.get_or_create_track("500006", "RYR205")
        t2 = buffer.get_or_create_track("4840D6", "KLM1023")

        # Build 5 chronological observations for each aircraft
        for step in range(5):
            t1.update_telemetry(
                timestamp=100.0 + step,
                latitude=38.70 + step * 0.001,
                longitude=-9.10,
                altitude=32000.0,
                groundspeed=420.0,
                heading=180.0,
                vertical_rate=0.0,
                callsign="RYR205"
            )
            t2.update_telemetry(
                timestamp=100.0 + step,
                latitude=38.90 + step * 0.001, # ~22 km away
                longitude=-9.10,
                altitude=31000.0,
                groundspeed=440.0,
                heading=90.0,
                vertical_rate=0.0,
                callsign="KLM1023"
            )

        active_tracks = buffer.get_active_tracks(current_time=104.0)
        self.assertEqual(len(active_tracks), 2)
        self.assertGreaterEqual(len(t1.history), 5)
        self.assertGreaterEqual(len(t2.history), 5)

        # ------------------------------------------------------------------
        # Step 4: Relational Graph Construction & 100 km Edge Formation
        # ------------------------------------------------------------------
        graph_snap, identities, raw_nodes = self.pipeline.build_graph_snapshot(active_tracks)
        self.assertEqual(graph_snap.x.shape[0], 2)
        self.assertEqual(graph_snap.x.shape[1], 12) # 12 node features
        # Mutual edges within 22 km (< 100 km threshold)
        self.assertEqual(graph_snap.edge_index.shape[1], 2)
        self.assertEqual(graph_snap.edge_attr.shape[1], 4) # 4 edge attributes

        # ------------------------------------------------------------------
        # Step 5: Multi-Stream ST-GNN Forward Pass
        # ------------------------------------------------------------------
        predictions = self.pipeline.predict_tracks(active_tracks)
        self.assertEqual(len(predictions), 2)

        for pred in predictions:
            self.assertIn(pred['icao24'], ["500006", "4840D6"])
            self.assertGreaterEqual(pred['prob_fused'], 0.0)
            self.assertLessEqual(pred['prob_fused'], 1.0)
            self.assertEqual(pred['threshold'], 0.20)
            self.assertIn(pred['threat_level'], ['NOMINAL', 'ELEVATED', 'CRITICAL'])

        # ------------------------------------------------------------------
        # Step 6: Multi-Stream Explainable AI (XAI) Attribution
        # ------------------------------------------------------------------
        dossier = self.pipeline.explain_track("500006", active_tracks)
        self.assertIsNotNone(dossier)
        self.assertEqual(dossier['icao24'], "500006")
        self.assertEqual(dossier['callsign'], "RYR205")

        # Verify all 4 explanation pillars exist
        self.assertIn('spatial_explanation', dossier)
        self.assertIn('temporal_explanation', dossier)
        self.assertIn('kinematic_explanation', dossier)
        self.assertIn('limitations_and_uncertainty', dossier)

        # Spatial attribution check
        self.assertIn('top_node_features', dossier['spatial_explanation'])
        self.assertIn('critical_edges', dossier['spatial_explanation'])

        # Temporal attribution check
        self.assertIn('temporal_step_importance', dossier['temporal_explanation'])
        self.assertEqual(len(dossier['temporal_explanation']['temporal_step_importance']), 5)

        # Kinematic attribution check
        self.assertIn('top_kinematic_residuals', dossier['kinematic_explanation'])
        self.assertIn('disclaimer', dossier['kinematic_explanation'])

if __name__ == '__main__':
    unittest.main()
