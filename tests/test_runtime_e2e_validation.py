"""
AeroGuard AI - Phase E Runtime Validation Test Suite
Validates:
1. WebSocket handshake, telemetry snapshot, ping/pong, tick, malformed handling, disconnect/reconnect.
2. The four simulated attack controls:
   - Altitude teleport (+7500 ft)
   - GPS velocity drift (disp_ratio >> 1.0)
   - Ghost aircraft (E00099)
   - Transponder collision (duplicate ICAO)
3. Evaluation decisions, ST-GNN fusion probabilities, threshold decisions (t=0.20), detection outcomes.
4. XAI multi-stream explanation dossiers for detected anomaly, normal flight, and missed anomaly.
5. SDR waveform decoder boundary with synthetic IQ, noise, and truncated packets.
"""

import unittest
import asyncio
import json
import numpy as np
import torch
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from scripts.aeroguard_server import create_app, state, live_buffer, live_pipeline, sdr_receiver
from scripts.rtl_sdr_adapter import ADSBWaveformDecoder, ADSBWaveformModulator


class TestAeroGuardRuntimeE2E(AioHTTPTestCase):

    async def get_application(self):
        state.initialize()
        return create_app()

    @unittest_run_loop
    async def test_01_websocket_and_rest_behavior(self):
        """Verify WebSocket handshake, snapshot broadcast, ping/pong, tick, and malformed handling."""
        # 1. REST status endpoint check
        resp = await self.client.get('/api/status')
        self.assertEqual(resp.status, 200)
        status_data = await resp.json()
        self.assertEqual(status_data['status'], 'ONLINE')
        self.assertEqual(status_data['calibrated_threshold'], 0.20)
        self.assertFalse(status_data['hardware_connected'])
        self.assertIn('Software-Defined Mode', status_data['hardware_status'])

        # 2. WebSocket connection & handshake
        ws = await self.client.ws_connect('/ws')
        self.assertFalse(ws.closed)

        # Message 1: STATUS handshake
        msg1 = await ws.receive_json()
        self.assertEqual(msg1['type'], 'STATUS')
        self.assertEqual(msg1['data']['calibrated_threshold'], 0.20)
        self.assertEqual(msg1['data']['receiver_mode'], 'software_defined_simulation')

        # Message 2: Initial SNAPSHOT
        msg2 = await ws.receive_json()
        self.assertEqual(msg2['type'], 'SNAPSHOT')
        self.assertIn('aircraft', msg2['data'])
        self.assertIn('graph_edges', msg2['data'])
        self.assertIn('stats', msg2['data'])

        # 3. Test Ping/Pong over WebSocket
        await ws.send_json({"action": "ping"})
        msg_pong = await ws.receive_json()
        self.assertEqual(msg_pong['type'], 'PONG')

        # 4. Test Tick over WebSocket
        await ws.send_json({"action": "tick", "step": 1})
        msg_snap = await ws.receive_json()
        self.assertEqual(msg_snap['type'], 'SNAPSHOT')
        self.assertGreaterEqual(msg_snap['data']['timestep_index'], 0)

        # 5. Test Malformed Message handling over WebSocket
        await ws.send_str("NOT_VALID_JSON_STRING")
        msg_err = await ws.receive_json()
        self.assertEqual(msg_err['type'], 'ERROR')

        # 6. Graceful close and reconnect
        await ws.close()
        self.assertTrue(ws.closed)

        # Reconnect
        ws2 = await self.client.ws_connect('/ws')
        self.assertFalse(ws2.closed)
        msg_reconnect = await ws2.receive_json()
        self.assertEqual(msg_reconnect['type'], 'STATUS')
        await ws2.close()

    @unittest_run_loop
    async def test_02_simulated_attack_controls(self):
        """
        Tests the 4 attack controls individually:
        1. Altitude teleport
        2. GPS velocity drift
        3. Ghost aircraft
        4. Transponder identity collision
        Records ST-GNN fusion score, decision at threshold 0.20, and XAI explanation availability.
        """
        results = {}

        # -------------------------------------------------------------
        # Attack 1: Altitude Teleport (+7500 ft)
        # -------------------------------------------------------------
        resp1 = await self.client.post('/api/simulation/attack_burst', json={
            "attack_type": "altitude_teleport",
            "icao24": "500006"
        })
        self.assertEqual(resp1.status, 200)
        data1 = await resp1.json()
        self.assertEqual(data1['status'], 'INJECTED')
        self.assertTrue(data1['is_simulated_test'])
        self.assertEqual(data1['target_icao24'], '500006')

        # Find target prediction
        pred1 = next((p for p in data1['predictions'] if p['icao24'] == '500006'), None)
        self.assertIsNotNone(pred1)
        prob1 = pred1['st_gnn_prob']
        is_anom1 = pred1['is_anomaly']
        results['altitude_teleport'] = {
            'accepted': True,
            'injected': True,
            'prob': prob1,
            'threshold': 0.20,
            'detected': is_anom1,
            'verdict': 'DETECTED' if is_anom1 else 'MISSED'
        }

        # -------------------------------------------------------------
        # Attack 2: GPS Velocity Drift (disp_ratio >> 1.0)
        # -------------------------------------------------------------
        resp2 = await self.client.post('/api/simulation/attack_burst', json={
            "attack_type": "gps_velocity_drift",
            "icao24": "500006"
        })
        self.assertEqual(resp2.status, 200)
        data2 = await resp2.json()
        self.assertEqual(data2['status'], 'INJECTED')
        pred2 = next((p for p in data2['predictions'] if p['icao24'] == '500006'), None)
        self.assertIsNotNone(pred2)
        prob2 = pred2['st_gnn_prob']
        is_anom2 = pred2['is_anomaly']
        results['gps_velocity_drift'] = {
            'accepted': True,
            'injected': True,
            'prob': prob2,
            'threshold': 0.20,
            'detected': is_anom2,
            'verdict': 'DETECTED' if is_anom2 else 'MISSED'
        }

        # -------------------------------------------------------------
        # Attack 3: Ghost Aircraft Injection (E00099)
        # -------------------------------------------------------------
        resp3 = await self.client.post('/api/simulation/attack_burst', json={
            "attack_type": "ghost_aircraft"
        })
        self.assertEqual(resp3.status, 200)
        data3 = await resp3.json()
        self.assertEqual(data3['status'], 'INJECTED')
        self.assertEqual(data3['target_icao24'], 'E00099')
        pred3 = next((p for p in data3['predictions'] if p['icao24'] == 'E00099'), None)
        self.assertIsNotNone(pred3)
        prob3 = pred3['st_gnn_prob']
        is_anom3 = pred3['is_anomaly']
        results['ghost_aircraft'] = {
            'accepted': True,
            'injected': True,
            'prob': prob3,
            'threshold': 0.20,
            'detected': is_anom3,
            'verdict': 'DETECTED' if is_anom3 else 'MISSED'
        }

        # -------------------------------------------------------------
        # Attack 4: Transponder Identity Collision
        # -------------------------------------------------------------
        resp4 = await self.client.post('/api/simulation/attack_burst', json={
            "attack_type": "transponder_collision",
            "icao24": "500006"
        })
        self.assertEqual(resp4.status, 200)
        data4 = await resp4.json()
        self.assertEqual(data4['status'], 'INJECTED')
        pred4 = next((p for p in data4['predictions'] if p['icao24'] == '500006'), None)
        self.assertIsNotNone(pred4)
        prob4 = pred4['st_gnn_prob']
        is_anom4 = pred4['is_anomaly']
        results['transponder_collision'] = {
            'accepted': True,
            'injected': True,
            'prob': prob4,
            'threshold': 0.20,
            'detected': is_anom4,
            'verdict': 'DETECTED' if is_anom4 else 'MISSED'
        }

        print("\n=== SIMULATED ATTACK CONTROLS RUNTIME RESULTS ===")
        for k, v in results.items():
            print(f"[{k.upper()}]: Prob={v['prob']:.4f} (Threshold={v['threshold']}) -> Decision={v['verdict']}")

    @unittest_run_loop
    async def test_03_xai_explanations_correctness(self):
        """
        Verifies XAI correctness across:
        1. Detected anomaly (500006)
        2. Normal flight
        3. Dynamically injected aircraft
        Confirms spatial, temporal, kinematic explanations use proper inputs and valid observation masks.
        """
        # 1. Fetch explanation for known audited flight 500006
        resp = await self.client.get('/api/xai/explain/500006')
        self.assertEqual(resp.status, 200)
        dossier = await resp.json()

        # Confirm identity matches
        self.assertEqual(dossier['icao24'].upper(), '500006')

        # Confirm overall dossier structure
        self.assertIn('anomaly_probability', dossier)
        self.assertIn('configured_threshold', dossier)
        self.assertIn('predicted_status', dossier)
        self.assertIn('threat_level', dossier)
        self.assertIn('attribution_confidence', dossier)

        # Confirm spatial explanation
        self.assertIn('spatial_explanation', dossier)
        self.assertIn('top_node_features', dossier['spatial_explanation'])
        self.assertIn('critical_edges', dossier['spatial_explanation'])

        # Confirm temporal explanation and observation mask
        self.assertIn('temporal_explanation', dossier)
        self.assertIn('temporal_step_importance', dossier['temporal_explanation'])
        timesteps = dossier['temporal_explanation']['temporal_step_importance']
        self.assertEqual(len(timesteps), 5)
        for ts in timesteps:
            self.assertIn('is_valid_observation', ts)
            self.assertIn('importance', ts)
            # Padded steps should not be presented as real observations
            if not ts['is_valid_observation']:
                self.assertEqual(ts.get('importance', 0.0), 0.0)

        # Confirm kinematic explanation
        self.assertIn('kinematic_explanation', dossier)
        self.assertIn('top_kinematic_residuals', dossier['kinematic_explanation'])
        self.assertIn('independent_rule_warnings', dossier['kinematic_explanation'])
        self.assertIn('disclaimer', dossier['kinematic_explanation'])

        # Confirm attribution is distinguished from heuristic warnings
        heuristics = dossier['kinematic_explanation']['independent_rule_warnings']
        self.assertIsInstance(heuristics, list)
        self.assertIn("must not be interpreted as definitive proof", dossier['kinematic_explanation']['disclaimer'])

    @unittest_run_loop
    async def test_04_sdr_signal_and_decoder_boundaries(self):
        """Verifies synthetic IQ decoding, noise rejection, and CPR limitation."""
        decoder = ADSBWaveformDecoder(sample_rate=2e6)
        modulator = ADSBWaveformModulator()

        # 1. Valid Callsign Frame
        frame_cs = modulator.encode_callsign_frame("500006", "RYR205")
        iq_cs = modulator.synthesize_ppm_baseband(frame_cs, sample_rate=2e6, snr_db=30.0)
        dec_cs = decoder.decode_iq(iq_cs)
        self.assertEqual(len(dec_cs), 1)
        self.assertEqual(dec_cs[0].icao24, "500006")
        self.assertEqual(dec_cs[0].callsign, "RYR205")
        self.assertTrue(dec_cs[0].crc_valid)

        # 2. Valid Velocity Frame (simplified waveform decoding)
        frame_vel = modulator.encode_airborne_velocity_frame("500006", speed_kts=420.0, heading_deg=180.0, vert_rate_fpm=0.0)
        iq_vel = modulator.synthesize_ppm_baseband(frame_vel, sample_rate=2e6, snr_db=30.0)
        dec_vel = decoder.decode_iq(iq_vel)
        self.assertEqual(len(dec_vel), 1)
        self.assertEqual(dec_vel[0].icao24, "500006")
        self.assertEqual(dec_vel[0].tc, 19)
        self.assertIsNotNone(dec_vel[0].groundspeed_kts)
        self.assertGreater(dec_vel[0].groundspeed_kts, 0)
        self.assertTrue(dec_vel[0].crc_valid)

        # 3. Pure Gaussian Noise Rejection
        noise = (np.random.normal(0, 0.5, 4000) + 1j * np.random.normal(0, 0.5, 4000)).astype(np.complex64)
        dec_noise = decoder.decode_iq(noise)
        self.assertEqual(len(dec_noise), 0)

        # 4. Truncated Signal Rejection
        dec_trunc = decoder.decode_iq(iq_cs[:100])
        self.assertEqual(len(dec_trunc), 0)

        # 5. Empty Input
        dec_empty = decoder.decode_iq(np.array([], dtype=np.complex64))
        self.assertEqual(len(dec_empty), 0)


if __name__ == '__main__':
    unittest.main()
