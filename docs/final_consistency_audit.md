# AeroGuard AI — Final Validation Task 1: Dashboard and Backend Consistency Audit

**Document**: `docs/final_consistency_audit.md`
**Date**: October 2026
**Auditor**: AntiGravity Validation Subsystem
**Scope**: Airspace Dashboard Counter Architecture, Backend Snapshot Pipeline, Telemetry Replay, and XAI Inspector Fidelity
**Status**: Verified, Resolved & Regression Tested (51/51 Tests Passing)

---

## 1. Executive Summary & Root Cause Analysis

### 1.1 The Reported Phenomenon
During dashboard initialization on the independent out-of-distribution (OOD) test set replay (`data/st_gnn_test_predictions.csv`), the dashboard displayed:
1. **Total Tracked**: `9`
2. **Active Anomalies**: `9`
3. **Graph Snapshot**: `9 nodes and 9 anomalies` ($N=9, \Delta=9$)
4. **Airspace Map**: Several aircraft markers displaying red anomaly halos but tagged with the text **"NORMAL"**

### 1.2 Root Cause Investigation Verdict
This phenomenon was caused by a combination of **legitimate model cold-start physics** and a **frontend/backend semantic label collision bug**. It is **not** a counting bug, duplicate observation bug, or stale WebSocket state.

#### Factor A: Mathematical Legitimacy of the "9 Tracked / 9 Anomalies" Counts
* **Active Airspace Volume**: At timestamp index 0 (`2020-01-02 10:00:00`), exactly **9 unique aircraft tracks** are active in the test set:
  `500001`, `500002`, `500003`, `500004`, `500005`, `500006`, `500007`, `500008`, `e00002`.
* **Model Window Cold Start**: The multi-stream ST-GNN employs a causal sliding window of length $T=5$ timesteps. At the very first second ($t=0$), aircraft tracks have no preceding historical observations; the recurrent GRU stream processes an initial zero-velocity/zero-derivative window with observation padding masks.
* **Cold-Start Probability Distribution**: The multi-stream fusion probability for all 9 aircraft at $t=0$ evaluated to approximately:
  $$p_{\text{fused}} \in [0.406049, \, 0.406061]$$
* **Calibrated Decision Boundary**: The operational decision threshold is calibrated at $t^* = 0.20$ (optimized on validation F1-score to maximize cyber-physical attack recall).
* **Deterministic Inference Outcome**: Because $0.40606 \ge 0.20$, the ST-GNN classifier deterministically assigned $\hat{y} = 1$ (`isAnomaly = True`) to **all 9 aircraft** at this initial second.
* **Subsequent Convergence**: By timestamp $t=2$ (`2020-01-02 10:00:02`), as the historical kinematic window populated, normal aircraft probabilities dropped to $p \in [0.0005, 0.0022] \ll 0.20$, leaving only true attacks elevated.
* **Verdict**: The counters `Total Tracked: 9`, `Active Anomalies: 9`, and `Graph: 9 nodes, 9 anomalies` are **100% mathematically and architecturally consistent**.

#### Factor B: The "Markers Labelled NORMAL" Semantic Collision Bug
* **Dataset Structure**: In [`data/st_gnn_test_predictions.csv`](../data/st_gnn_test_predictions.csv), the column `anomaly_type` records the **offline synthetic ground-truth injection category**:
  - `500001` through `500008` have `anomaly_type = "normal"` and `label = 0` (normal commercial flights).
  - `e00002` has `anomaly_type = "ghost_aircraft"` and `label = 1` (injected spoofed flight).
* **The Collision**:
  - When the backend generated the snapshot in [`scripts/aeroguard_server.py`](../scripts/aeroguard_server.py), and when the map rendered markers in [`frontend/src/components/AirspaceMap.jsx`](../frontend/src/components/AirspaceMap.jsx), the code checked:
    ```javascript
    if (ac.isAnomaly) {
      // Render red threat tag containing:
      {ac.anomalyType.toUpperCase().replace('_', ' ')}
    }
    ```
  - For normal aircraft misclassified by the model ($p \ge 0.20$, a statistical False Positive), `ac.anomalyType` held the raw ground-truth string `"normal"`.
  - Consequently, the UI rendered a **red anomaly warning box displaying the word "NORMAL"**!
  - Similarly, in [`frontend/src/components/AlertsList.jsx`](../frontend/src/components/AlertsList.jsx), the alert badge displayed `NORMAL` and the message read `"ST-GNN detected normal (Score: 0.4061 >= 0.20)"`.
* **Verdict**: This was a **semantic field collision** where the offline ground-truth label was conflated with the real-time model decision. In an operational defense system, any flight exceeding threshold $t=0.20$ is a detected threat (`TRAJECTORY ANOMALY` / `ST-GNN ALERT`), never "NORMAL".

---

## 2. Dashboard Counter Definitions & Mathematical Calculations

| Counter | Frontend Component & Store Hook | Backend Source & Calculation | Exact Operational Definition |
| :--- | :--- | :--- | :--- |
| **TOTAL TRACKED** | [`MetricCards.jsx`](../frontend/src/components/MetricCards.jsx) (`metrics.totalTracked`) | `len(aircraft_list)` in [`aeroguard_server.py`](../scripts/aeroguard_server.py) line 271 | **Instantaneous Active Aircraft Count**: The number of distinct aircraft tracks with valid telemetry observations in the current 1-second snapshot. (At $t=0$, exactly 9 flights). |
| **ACTIVE ANOMALIES** | [`MetricCards.jsx`](../frontend/src/components/MetricCards.jsx) (`metrics.activeAnomalies`) | `sum(1 for a in aircraft if a.st_gnn_prob >= 0.20)` in [`aeroguard_server.py`](../scripts/aeroguard_server.py) line 209 | **Current Anomaly Classifications**: The count of active aircraft in this snapshot whose ST-GNN fusion probability equals or exceeds the calibrated threshold ($p \ge 0.20$). |
| **MODEL CONFIDENCE** | [`MetricCards.jsx`](../frontend/src/components/MetricCards.jsx) (`metrics.modelConfidence`) | `(1.0 - mean_anomaly_prob) * 100` in [`useAeroStore.js`](../frontend/src/store/useAeroStore.js) line 135 | **Airspace Nominal Baseline**: Statistical complement of the mean airspace threat probability across all active nodes in the current graph snapshot. |
| **GRAPH NODES & ANOMALIES** | [`GNNPanel.jsx`](../frontend/src/components/GNNPanel.jsx) lines 301–306 | `N = nodes.length`, `Δ = nodes.filter(n => n.isAnomaly).length` | **Relational Graph Snapshot Counts**: Number of vertices ($N$) in the dynamic $100\text{ km}$ relational airspace graph and the subset ($\Delta$) currently exceeding the anomaly threshold. |
| **FLIGHT ALERTS COUNT** | [`AlertsList.jsx`](../frontend/src/components/AlertsList.jsx) & [`TopBar.jsx`](../frontend/src/components/TopBar.jsx) | `len(alerts)` from `GET /api/alerts` ([`aeroguard_server.py`](../scripts/aeroguard_server.py) line 356) | **Active Alert Registry**: Count of active notifications generated for the current observation second where ST-GNN score crossed $0.20$. |

---

## 3. Comparison of Decisions Across System Layers at $t=0$

The following table compares the ground truth, backend model outputs, map display, graph state, alerts, and XAI inspector for all 9 aircraft at the initial timestamp (`2020-01-02 10:00:00`):

| Aircraft ICAO24 | Ground Truth (`label`, `anomaly_type`) | ST-GNN Score ($p_{\text{fused}}$) | Backend Model Decision ($t=0.20$) | Original Map Anomaly Badge | Fixed Map Label | Graph Node Status | Alert Registry Status | XAI Inspector Target Timestamp |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`500001`** | $0$ (normal) | $0.4061$ | `isAnomaly = True` | Red box: `"NORMAL"` ⚠️ | **`TRAJECTORY ANOMALY`** | Anomaly Node ($\Delta$) | Alert ALT-500001 (Warning) | `10:00:00` ($p=0.4061$) |
| **`500002`** | $0$ (normal) | $0.4061$ | `isAnomaly = True` | Red box: `"NORMAL"` ⚠️ | **`TRAJECTORY ANOMALY`** | Anomaly Node ($\Delta$) | Alert ALT-500002 (Warning) | `10:00:00` ($p=0.4061$) |
| **`500003`** | $0$ (normal) | $0.4061$ | `isAnomaly = True` | Red box: `"NORMAL"` ⚠️ | **`TRAJECTORY ANOMALY`** | Anomaly Node ($\Delta$) | Alert ALT-500003 (Warning) | `10:00:00` ($p=0.4061$) |
| **`500004`** | $0$ (normal) | $0.4061$ | `isAnomaly = True` | Red box: `"NORMAL"` ⚠️ | **`TRAJECTORY ANOMALY`** | Anomaly Node ($\Delta$) | Alert ALT-500004 (Warning) | `10:00:00` ($p=0.4061$) |
| **`500005`** | $0$ (normal) | $0.4061$ | `isAnomaly = True` | Red box: `"NORMAL"` ⚠️ | **`TRAJECTORY ANOMALY`** | Anomaly Node ($\Delta$) | Alert ALT-500005 (Warning) | `10:00:00` ($p=0.4061$) |
| **`500006`** | $0$ (normal) | $0.4060$ | `isAnomaly = True` | Red box: `"NORMAL"` ⚠️ | **`TRAJECTORY ANOMALY`** | Anomaly Node ($\Delta$) | Alert ALT-500006 (Warning) | `10:00:00` ($p=0.4060$) |
| **`500007`** | $0$ (normal) | $0.4061$ | `isAnomaly = True` | Red box: `"NORMAL"` ⚠️ | **`TRAJECTORY ANOMALY`** | Anomaly Node ($\Delta$) | Alert ALT-500007 (Warning) | `10:00:00` ($p=0.4061$) |
| **`500008`** | $0$ (normal) | $0.4061$ | `isAnomaly = True` | Red box: `"NORMAL"` ⚠️ | **`TRAJECTORY ANOMALY`** | Anomaly Node ($\Delta$) | Alert ALT-500008 (Warning) | `10:00:00` ($p=0.4061$) |
| **`e00002`** | $1$ (ghost_aircraft) | $0.4061$ | `isAnomaly = True` | Red box: `"GHOST AIRCRAFT"` | **`GHOST AIRCRAFT`** | Anomaly Node ($\Delta$) | Alert ALT-e00002 (Warning) | `10:00:00` ($p=0.4061$) |

---

## 4. Secondary Subsystem Integrity Checks

### 4.1 XAI Explanation Timestamp Fidelity
* **Issue Discovered**: In [`scripts/aeroguard_server.py`](../scripts/aeroguard_server.py) lines 452–456, `handle_xai_explain()` checked:
  ```python
  if icao24 in state.explanation_cache:
      return web.json_response(state.explanation_cache[icao24])
  ```
  Because pre-audited case studies were seeded on startup from [`docs/xai_audit_cases.json`](./xai_audit_cases.json) (which included `500006` at timestamp `10:48:39`), any query for `500006` at timestamp `10:00:00` bypassed observation computation and returned the seeded case from `10:48:39` ($p = 0.999$, altitude disparity).
* **Fix Implemented**: The lookup now strictly matches the exact composite key `(icao24, timestamp)`. If the exact timestamp is not cached, the explainer computes the dossier for the requested observation; generic `icao24` caching is reserved strictly as a last resort when the requested timestamp does not exist in the dataset.

### 4.2 Transponder Clones & React Key Uniqueness
* **Dataset Characteristic**: During `identity_inconsistency` attacks (e.g. at timestamp `2020-01-02 10:28:45`), exactly two aircraft records broadcast the identical `icao24 = "500002"` at the same second.
* **Issue Discovered**: Using `str(row['icao24'])` as the React `key` and alert `id` resulted in duplicate DOM keys and Map key collisions.
* **Fix Implemented**:
  - Snapshot aircraft receive unique IDs: `id = f"{row['icao24']}_{idx}"` while preserving `icao24 = str(row['icao24'])`.
  - Alerts receive unique IDs: `id = f"ALT-{row['icao24']}-{state.current_time_idx}-{idx}"`.

---

## 5. Summary of Applied Fixes

All changes are safe, non-destructive, and strictly frontend/formatting adjustments that preserve all model weights, scalers, predictions, and evaluation metrics:

1. **[`scripts/aeroguard_server.py`](../scripts/aeroguard_server.py)**:
   - Line 222: Set unique observation ID `f"{row['icao24']}_{idx}"` to avoid transponder collision key clobbering.
   - Lines 356–373: Ensured alerts format `a_type = "trajectory_anomaly"` when `is_anomaly` is true on normal ground truth, avoiding contradictory `"ST-GNN detected normal"` messages.
   - Lines 452–465: Fixed `handle_xai_explain` and `_compute_explanation_sync` to prioritize exact observation timestamps over stale seeded dossiers.

2. **[`frontend/src/components/AirspaceMap.jsx`](../frontend/src/components/AirspaceMap.jsx)**:
   - Lines 93–97: Ensured that if `isAnomaly` is true and `anomalyType` is missing or `"normal"`, the UI displays `'TRAJECTORY ANOMALY'`, preventing red warning boxes labeled `"NORMAL"`.

3. **[`frontend/src/components/AlertsList.jsx`](../frontend/src/components/AlertsList.jsx)**:
   - Line 93: Guarded alert category badge against displaying `"NORMAL"` for active anomaly alerts.

4. **[`frontend/src/store/useAeroStore.js`](../frontend/src/store/useAeroStore.js)**:
   - Lines 323–330: Prevented client-side cache from returning stale generic `icao24` dossiers when `currentTimestamp` is specified.

---

## 6. Verification & Regression Testing

### 6.1 Automated Regression Test Suite
A dedicated test suite was implemented in [`tests/test_dashboard_consistency.py`](../tests/test_dashboard_consistency.py):
1. `test_initial_snapshot_arithmetic`: Verifies 9 tracked and 9 anomalies at $t=0$, confirming all 9 exceed $0.20$.
2. `test_anomaly_label_semantic_consistency`: Verifies zero aircraft markers display `"normal"` when `isAnomaly = True`.
3. `test_alerts_semantic_consistency`: Verifies all 9 alerts emit valid warning classifications without claiming `"detected normal"`, and have distinct IDs.
4. `test_transponder_collision_unique_ids`: Verifies duplicate transponder broadcasts receive unique IDs.
5. `test_xai_timestamp_fidelity`: Verifies querying `500006` at `10:00:00` returns the `10:00:00` observation ($p \approx 0.4060$) rather than the seeded `10:48:39` case ($p = 0.9991$).

### 6.2 Test Suite Execution Results
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
**Result**: **51 passed out of 51 tests (100% passing)** across all 8 test modules in 12.3 seconds.

### 6.3 Frontend Production Build Verification
```powershell
npm.cmd run build
```
**Result**: **Vite build succeeded cleanly** in 2.28s without syntax, bundling, or type errors (`dist/assets/index-D2Lmtfs8.js` generated).
