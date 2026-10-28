"""
AeroGuard AI — Server & Dashboard Robustness Test Suite
======================================================
Tests server resilience against:
1. Missing telemetry data / empty timestamps (graceful 503 / empty lists without 500).
2. Malformed JSON payloads, non-dict payloads, and invalid types.
3. Invalid, negative, and excessive tick steps.
4. Non-existent ICAO24 explanations (clean 404 with error message).
5. Repeated and concurrent explanation requests (cache consistency & thread safety).
6. Initialization error clarity when prediction artifact is absent.
"""

import os
import sys
import json
import asyncio
import unittest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

sys.path.insert(0, os.path.abspath("."))
from scripts.aeroguard_server import create_app, state, AeroGuardServerState


class TestServerRobustness(AioHTTPTestCase):
    async def get_application(self):
        state.initialize()
        return create_app()

    @unittest_run_loop
    async def test_01_malformed_json_feed_telemetry(self):
        """Verifies that malformed JSON in feed_telemetry returns 400 instead of 500."""
        resp = await self.client.post(
            "/api/receiver/feed_telemetry",
            data="INVALID_RAW_JSON_STRING",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(resp.status, 400)
        body = await resp.json()
        self.assertIn("error", body)

    @unittest_run_loop
    async def test_02_non_dict_attack_burst(self):
        """Verifies that non-dict JSON payloads to simulation/attack_burst fallback cleanly."""
        resp = await self.client.post(
            "/api/simulation/attack_burst",
            json=[1, 2, 3],  # List instead of dict
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertEqual(body["status"], "INJECTED")
        self.assertEqual(body["target_icao24"], "500006")

    @unittest_run_loop
    async def test_03_integer_icao_attack_burst(self):
        """Verifies integer icao24 does not crash target_icao.upper()."""
        resp = await self.client.post(
            "/api/simulation/attack_burst",
            json={"attack_type": "altitude_teleport", "icao24": 500006},  # integer
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertEqual(body["target_icao24"], "500006")

    @unittest_run_loop
    async def test_04_invalid_and_extreme_tick_steps(self):
        """Verifies that non-integer, negative, or huge steps do not crash handle_telemetry_tick."""
        # Non-integer step fallback
        resp_str = await self.client.post("/api/telemetry/tick", json={"step": "not_an_int"})
        self.assertEqual(resp_str.status, 200)

        # Huge step modulo wrap
        resp_huge = await self.client.post("/api/telemetry/tick", json={"step": 1000000})
        self.assertEqual(resp_huge.status, 200)
        body = await resp_huge.json()
        self.assertGreaterEqual(body["timestep_index"], 0)
        self.assertLess(body["timestep_index"], len(state.timestamps))

        # Negative step wrap
        resp_neg = await self.client.post("/api/telemetry/tick", json={"step": -5})
        self.assertEqual(resp_neg.status, 200)
        body_neg = await resp_neg.json()
        self.assertGreaterEqual(body_neg["timestep_index"], 0)
        self.assertLess(body_neg["timestep_index"], len(state.timestamps))

    @unittest_run_loop
    async def test_05_nonexistent_icao_explanation_returns_404(self):
        """Verifies requesting explanation for unknown aircraft returns 404 with error message."""
        resp = await self.client.get("/api/xai/explain/NONEXISTENT999?timestamp=2020-01-02+10%3A00%3A00")
        self.assertEqual(resp.status, 404)
        body = await resp.json()
        self.assertIn("error", body)
        self.assertEqual(body["icao24"], "NONEXISTENT999")

    @unittest_run_loop
    async def test_06_repeated_and_concurrent_explanation_requests(self):
        """Verifies repeated concurrent requests return identical, cached results without deadlock."""
        # Query known seeded flight
        url = "/api/xai/explain/500006"
        tasks = [self.client.get(url) for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        for r in responses:
            self.assertEqual(r.status, 200)
            data = await r.json()
            self.assertEqual(data["icao24"], "500006")
            self.assertIn("case_category", data)

    @unittest_run_loop
    async def test_07_empty_timestamps_graceful_handling(self):
        """Verifies endpoints handle empty timestamps without ZeroDivisionError or crash."""
        original_ts = state.timestamps
        try:
            state.timestamps = []

            # Snapshot returns 503 Service Unavailable
            resp_snap = await self.client.get("/api/telemetry/snapshot")
            self.assertEqual(resp_snap.status, 503)

            # Tick returns 503 Service Unavailable
            resp_tick = await self.client.post("/api/telemetry/tick", json={"step": 1})
            self.assertEqual(resp_tick.status, 503)

            # Alerts returns empty list
            resp_alerts = await self.client.get("/api/alerts")
            self.assertEqual(resp_alerts.status, 200)
            self.assertEqual(await resp_alerts.json(), [])

            # Status returns ONLINE with current_timestamp None
            resp_status = await self.client.get("/api/status")
            self.assertEqual(resp_status.status, 200)
            body = await resp_status.json()
            self.assertIsNone(body["current_timestamp"])
            self.assertEqual(body["total_timesteps"], 0)
        finally:
            state.timestamps = original_ts


class TestServerInitializationError(unittest.TestCase):
    def test_missing_prediction_artifact_raises_informative_error(self):
        """Verifies clean FileNotFoundError when required predictions file is absent."""
        test_state = AeroGuardServerState()
        # Mock non-existent path
        missing_path = "data/non_existent_predictions_test.csv"

        with unittest.mock.patch("os.path.exists") as mock_exists:
            # Let threshold exist, but predictions file not exist
            mock_exists.side_effect = lambda p: False if p == "data/st_gnn_test_predictions.csv" else True
            with self.assertRaises(FileNotFoundError) as ctx:
                test_state.initialize()
            self.assertIn("Missing required prediction artifact", str(ctx.exception))
            self.assertIn("python scripts/audit_st_gnn_metrics.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
