"""
AeroGuard AI - Dashboard & Backend Consistency Audit Tests
==========================================================
Verifies:
1. Snapshot at t=0 arithmetic: exactly 9 tracked, 9 anomalies, all with prob >= 0.20.
2. Anomaly semantic labeling: no anomaly is labeled "normal" in snapshot aircraft or alerts.
3. Transponder collision handling: duplicate ICAO broadcasts receive unique observation IDs and unique alert IDs.
4. XAI timestamp fidelity: explanation queries for an aircraft at a specific timestamp return the exact observation
   and do not get overridden by stale cached dossiers from other timestamps.
"""

import os, sys, unittest
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from scripts.aeroguard_server import create_app, state, build_snapshot_dict

class TestDashboardConsistency(AioHTTPTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        state.initialize()

    async def get_application(self):
        return create_app()

    def test_initial_snapshot_arithmetic(self):
        """Verifies why initial snapshot shows 9 tracked and 9 active anomalies at threshold 0.20."""
        snap = build_snapshot_dict(0)
        aircraft = snap["aircraft"]
        stats = snap["stats"]

        # 1. Total tracked count verification
        self.assertEqual(len(aircraft), 9)
        self.assertEqual(stats["total_tracked"], 9)

        # 2. Verify all 9 aircraft exceed calibrated threshold 0.20 at cold start t=0
        for ac in aircraft:
            self.assertGreaterEqual(ac["st_gnn_prob"], 0.20)
            self.assertTrue(ac["isAnomaly"])
            self.assertTrue(ac["is_anomaly"])

        # 3. Active anomalies matches count
        self.assertEqual(stats["active_anomalies"], 9)
        self.assertEqual(stats["graph_nodes"], 9)

    def test_anomaly_label_semantic_consistency(self):
        """Verifies that no anomaly is labeled 'normal' in aircraft or alerts."""
        snap = build_snapshot_dict(0)
        for ac in snap["aircraft"]:
            if ac["isAnomaly"]:
                self.assertNotEqual(ac["anomalyType"].lower(), "normal")
                self.assertIn(ac["anomalyType"], ["trajectory_anomaly", "ghost_aircraft", "altitude_anomaly", "position_jump", "identity_inconsistency"])

    @unittest_run_loop
    async def test_alerts_semantic_consistency(self):
        """Verifies that alerts endpoint does not emit alerts claiming 'ST-GNN detected normal'."""
        state.current_time_idx = 0
        resp = await self.client.get("/api/alerts")
        self.assertEqual(resp.status, 200)
        alerts = await resp.json()

        self.assertEqual(len(alerts), 9)
        alert_ids = set()
        for a in alerts:
            self.assertNotEqual(a["anomalyType"].lower(), "normal")
            self.assertNotIn("detected normal", a["message"].lower())
            alert_ids.add(a["id"])

        # Ensure all alert IDs are distinct
        self.assertEqual(len(alert_ids), len(alerts))

    def test_transponder_collision_unique_ids(self):
        """Verifies that duplicate ICAO broadcasts during transponder collision receive unique IDs."""
        # Find a timestamp with duplicate ICAO (e.g. identity_inconsistency on 500002)
        dup_ts = None
        for ts, group in state.time_grouped.items():
            if len(group[group['icao24'] == '500002']) > 1:
                dup_ts = ts
                break

        self.assertIsNotNone(dup_ts, "Expected to find transponder collision timestamp")
        time_idx = state.timestamps.index(dup_ts)
        snap = build_snapshot_dict(time_idx)

        # Ensure every aircraft has a unique 'id' even if 'icao24' is shared
        node_ids = [ac["id"] for ac in snap["aircraft"]]
        self.assertEqual(len(node_ids), len(set(node_ids)), "Aircraft IDs in snapshot must be unique")

    @unittest_run_loop
    async def test_xai_timestamp_fidelity(self):
        """Verifies that querying XAI for 500006 at 10:00:00 returns the 10:00:00 observation."""
        state.current_time_idx = 0
        target_ts = state.timestamps[0] # "2020-01-02 10:00:00"

        resp = await self.client.get(f"/api/xai/explain/500006?timestamp={target_ts}")
        self.assertEqual(resp.status, 200)
        exp = await resp.json()

        self.assertEqual(exp["icao24"], "500006")
        self.assertEqual(exp["timestamp"], target_ts)
        # At 10:00:00, 500006 has st_gnn_prob ~0.406, NOT 0.999 (which belongs to 10:48:39)
        self.assertAlmostEqual(exp["anomaly_probability"], 0.4060, places=3)

if __name__ == "__main__":
    unittest.main()
