"""
Unit Tests for AeroGuard Command Center Backend Server
=======================================================
Tests:
1. System Status endpoint (/api/status).
2. Telemetry snapshot endpoint (/api/telemetry/snapshot).
3. Telemetry tick forward (/api/telemetry/tick).
4. Flight alerts registry (/api/alerts).
5. On-demand Explainable AI dossier (/api/xai/explain/{icao24}).
"""

import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from scripts.aeroguard_server import create_app, state

class TestAeroGuardServer(AioHTTPTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        state.initialize()

    async def get_application(self):
        return create_app()

    @unittest_run_loop
    async def test_status_endpoint(self):
        resp = await self.client.request("GET", "/api/status")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertAlmostEqual(data["calibrated_threshold"], 0.20, places=2)
        self.assertIn("software_defined", data["receiver_mode"])

    @unittest_run_loop
    async def test_telemetry_snapshot(self):
        resp = await self.client.request("GET", "/api/telemetry/snapshot")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("aircraft", data)
        self.assertIn("graph_edges", data)
        self.assertTrue(len(data["aircraft"]) >= 5)
        # Check first aircraft fields
        first_ac = data["aircraft"][0]
        self.assertIn("icao24", first_ac)
        self.assertIn("st_gnn_prob", first_ac)
        self.assertIn("threat_level", first_ac)

    @unittest_run_loop
    async def test_telemetry_tick(self):
        resp = await self.client.request("POST", "/api/telemetry/tick", json={"step": 1})
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertTrue(data["timestep_index"] >= 1)

    @unittest_run_loop
    async def test_alerts_endpoint(self):
        resp = await self.client.request("GET", "/api/alerts")
        self.assertEqual(resp.status, 200)
        alerts = await resp.json()
        self.assertIsInstance(alerts, list)

    @unittest_run_loop
    async def test_xai_explain_endpoint(self):
        resp = await self.client.request("GET", "/api/telemetry/snapshot")
        data = await resp.json()
        target_icao = data["aircraft"][0]["icao24"]

        resp = await self.client.request("GET", f"/api/xai/explain/{target_icao}")
        self.assertEqual(resp.status, 200)
        exp = await resp.json()
        self.assertEqual(exp["icao24"], target_icao)
        self.assertIn("anomaly_probability", exp)
        self.assertIn("threat_level", exp)
        self.assertIn("spatial_explanation", exp)
        self.assertIn("temporal_explanation", exp)
        self.assertIn("kinematic_explanation", exp)
        self.assertIn("limitations_and_uncertainty", exp)

    @unittest_run_loop
    async def test_receiver_status_endpoint(self):
        resp = await self.client.request("GET", "/api/receiver/status")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("receiver_mode", data)
        self.assertIn("sample_rate_msps", data)
        self.assertIn("hardware_status", data)

    @unittest_run_loop
    async def test_receiver_decode_synthetic(self):
        resp = await self.client.request("POST", "/api/receiver/decode_synthetic")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["status"], "DECODED")
        self.assertTrue(data["messages_count"] >= 2)
        # Verify first message has CRC valid and ICAO
        first_m = data["messages"][0]
        self.assertTrue(first_m["crc_valid"])
        self.assertIn("icao24", first_m)

    @unittest_run_loop
    async def test_receiver_mode_toggle(self):
        resp = await self.client.request("POST", "/api/receiver/mode", json={"mode": "telemetry_replay"})
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["current_mode"], "telemetry_replay")

    @unittest_run_loop
    async def test_receiver_live_tracks_and_infer(self):
        # 1. Feed telemetry frames
        frames = [
            {
                "icao24": "LIVE01",
                "callsign": "TEST01",
                "latitude": 38.70,
                "longitude": -9.10,
                "altitude": 30000.0,
                "groundspeed": 400.0,
                "heading": 90.0,
                "vertical_rate": 0.0,
                "timestamp": 1000.0
            },
            {
                "icao24": "LIVE02",
                "callsign": "TEST02",
                "latitude": 38.80,
                "longitude": -9.10,
                "altitude": 31000.0,
                "groundspeed": 420.0,
                "heading": 95.0,
                "vertical_rate": 0.0,
                "timestamp": 1000.0
            }
        ]
        resp_feed = await self.client.request("POST", "/api/receiver/feed_telemetry", json=frames)
        self.assertEqual(resp_feed.status, 200)
        feed_data = await resp_feed.json()
        self.assertEqual(feed_data["status"], "INGESTED")
        self.assertEqual(feed_data["records_ingested"], 2)

        # 2. Query live tracks
        resp_tracks = await self.client.request("GET", "/api/receiver/live_tracks")
        self.assertEqual(resp_tracks.status, 200)
        tracks_data = await resp_tracks.json()
        self.assertGreaterEqual(tracks_data["active_tracks_count"], 2)

        # 3. Trigger live inference
        resp_infer = await self.client.request("POST", "/api/receiver/live_infer")
        self.assertEqual(resp_infer.status, 200)
        infer_data = await resp_infer.json()
        self.assertEqual(infer_data["status"], "PREDICTED")
        self.assertGreaterEqual(infer_data["tracks_evaluated"], 2)
        first_pred = infer_data["predictions"][0]
        self.assertIn("prob_fused", first_pred)
        self.assertIn("is_anomaly", first_pred)

    @unittest_run_loop
    async def test_simulation_attack_burst(self):
        resp = await self.client.request(
            "POST",
            "/api/simulation/attack_burst",
            json={"attack_type": "altitude_teleport", "icao24": "500006"}
        )
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["status"], "INJECTED")
        self.assertTrue(data["is_simulated_test"])
        self.assertEqual(data["attack_type"], "altitude_teleport")
        self.assertGreaterEqual(len(data["predictions"]), 1)

    @unittest_run_loop
    async def test_websocket_handshake(self):
        ws = await self.client.ws_connect("/ws")
        msg = await ws.receive_json()
        self.assertEqual(msg["type"], "STATUS")
        self.assertEqual(msg["data"]["status"], "ONLINE")
        await ws.close()

if __name__ == "__main__":
    unittest.main()
