# AeroGuard AI: Final Viva Demonstration Checklist & Runbook

This runbook provides the verified, reproducible step-by-step procedure for demonstrating **AeroGuard AI** (Flight Trajectory Anomaly & ADS-B Spoofing Detection using Spatio-Temporal Graph Neural Networks) during the Semester 3 Software Defined Communication Systems (SDC) viva.

---

## 1. Pre-Demo Dependency & Port Checklist

Before launching the demonstration, ensure the following prerequisites are met:

- [x] **Operating System**: Windows (PowerShell 5.1 / 7+)
- [x] **Python Environment**: Located at `.venv\Scripts\python.exe` (Python 3.11 with `torch==2.6.0+cpu`, `torch_geometric`, `aiohttp`, `scikit-learn`, `joblib`, `pandas`, `numpy`).
- [x] **Node Environment**: Node.js & `npm.cmd` installed (PowerShell blocks `npm.ps1`; always use `npm.cmd`).
- [x] **Port 8000**: Free for AeroGuard backend (FastAPI/AioHTTP REST & `/ws` WebSocket).
- [x] **Port 5173**: Free for Vite frontend development server.
- [x] **Models & Artifacts**:
  - `models/st_gnn_official.pt` (Multi-Stream ST-GNN Checkpoint, verified)
  - `models/st_gnn_scaler.joblib` (Feature normalizer)
  - `data/st_gnn_test_predictions.csv` (Audited Jan 2 OOD test set, 32,550 flights)
- [x] **Software Mode**: Hardware RTL-SDR is **not** connected or required. Dashboard displays `HW: UNAVAILABLE (SOFTWARE MODE)`.

### Verification Command
Run this command to confirm that ports 8000 and 5173 are currently free:
```powershell
Get-NetTCPConnection -LocalPort 8000,5173 -ErrorAction SilentlyContinue
```
*(If no output or an error is returned, both ports are free.)*

---

## 2. Exact Launch & Teardown Commands

Open two separate Windows PowerShell terminals:

### Terminal 1 — Start the Python AeroGuard Backend
```powershell
cd "d:\SEM_3\SDC\project code\AeroGuard"
.\.venv\Scripts\Activate.ps1
python scripts/aeroguard_server.py --port 8000
```
**Expected Terminal Output:**
```text
[AeroGuard Server] Initializing model artifacts and telemetry stream...
[AeroGuard Server] Seeded 12 pre-audited dossiers into cache.
[AeroGuard Server] Ready: 3600 chronological snapshots loaded. Threshold t=0.20
[AeroGuard Server] Starting HTTP service on http://127.0.0.1:8000
```

### Terminal 2 — Start the React Dashboard
```powershell
cd "d:\SEM_3\SDC\project code\AeroGuard\frontend"
npm.cmd run dev
```
**Expected Terminal Output:**
```text
  VITE v6.4.3  ready in 1124 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Teardown Command
To stop either service cleanly, press `Ctrl + C` in the respective terminal window.

---

## 3. Five-Minute Viva Presentation Sequence

| Minute | Phase | Demonstrated Action & Visual Evidence | Viva Explanation for Examiner |
| :---: | :--- | :--- | :--- |
| **0:00 - 1:00** | **System & SDC Scope** | Open `http://localhost:5173/` in the browser. Point to the top bar: `AeroGuard AI`, telemetry status `ONLINE`, and `HW: UNAVAILABLE (SOFTWARE MODE)`. | *"AeroGuard AI is an SDR communication pipeline designed to detect cyber-physical ADS-B spoofing without hardware dependency. Notice that our UI honestly reports SOFTWARE MODE—we do not claim physical RF reception or antenna hardware."* |
| **1:00 - 2:00** | **SDR Demodulation & Map** | Observe aircraft tracks moving on the canvas. Click **REPLAY CONTROLS** or **TICK** to advance snapshots. | *"Baseband PPM pulses are correlated against an 8 μs Mode-S preamble and sliced into 112 bits, then validated with a 24-bit CRC polynomial (G(x) = 0x1FFF409). Callsign and airborne velocity are extracted from TC=1..4 and TC=19. Aircraft are dynamically normalized onto the airspace map."* |
| **2:00 - 3:00** | **ST-GNN Architecture** | Select an aircraft (e.g., `500006`). In the right panel, toggle to **ARCHITECTURE VIEW**. | *"Our architecture combines three synchronized streams: (1) Spatial GATv2 with k=8 dynamic relational edges (distance, azimuth, Δv, Δalt); (2) Temporal causal GRU processing 5 timesteps with observation validity masks; and (3) Kinematic MLP processing 11 physical residuals."* |
| **3:00 - 4:00** | **Simulated Attack Bursts** | Click **SIMULATE ATTACK** in the top bar. Select **Altitude Teleport (+7500ft)**. | *"We inject an instantaneous +7500 ft altitude step without climb time. Within one second, the fusion probability spikes to 0.9978, well above our calibrated threshold of t = 0.20. The UI immediately flags the aircraft with a prominent SIMULATED TEST BURST badge."* |
| **4:00 - 5:00** | **XAI Attribution & Limitations** | In the XAI panel, inspect the top kinematic residual: `alt_residual` accounts for >99% attribution. Then summarize audited performance. | *"Our explainer isolates the exact root cause: alt_residual = 9905 ft vertical mismatch. In our audited out-of-distribution test set, AeroGuard achieved 73.06% precision with a 1.21% false alarm rate, detecting 100% of altitude and transponder anomalies."* |

---

## 4. Results Table for the Four Attack Simulations

Each simulated attack was executed against the running backend and evaluated by the official Multi-Stream ST-GNN checkpoint (`models/st_gnn_official.pt`) at threshold $t = 0.20$:

| Attack Simulation | Target ICAO | Injected Physical Violation | ST-GNN Score | Decision ($t=0.20$) | Outcome | Primary Root Cause / XAI Finding |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **1. Altitude Teleport** | `500006` | Instantaneous $+7,500\,\text{ft}$ step with $v_r = 0\,\text{fpm}$ | **$0.9978$** | `ANOMALY` | **DETECTED** | `alt_residual` ($9,905\,\text{ft}$ mismatch) accounts for $>99.2\%$ of kinematic attribution. |
| **2. GPS Velocity Drift** | `500006` | Spatial jump of $+0.22^\circ$ ($25\,\text{km}$) with constant speed | **$0.3322$** | `ANOMALY` | **DETECTED** | `disp_ratio` residual exceeds envelope ($11.0 \gg 1.0$), triggering fusion threshold. |
| **3. Ghost Aircraft** | `E00099` | Injected unverified flight with transponder mismatch | **$0.5568$** | `ANOMALY` | **DETECTED** | High descent rate ($-3,500\,\text{fpm}$) and transponder mismatch trigger elevated temporal risk. |
| **4. Transponder Collision** | `500006` | Duplicate broadcast claiming same ICAO with different callsign | **$0.2070$** | `ANOMALY` | **DETECTED** | `icao_collision` binary residual triggers elevated threat score crossing $0.20$. |

---

## 5. Verified Test Suite Results

The complete automated test suite was executed against the Python virtual environment:
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
**Actual Result**: **46 passed out of 46 tests (100% passing)** across:
1. `tests/test_adsb_waveform_decoder.py` (Preamble matched filter, Manchester decoding, CRC-24, bit flips, noise rejection)
2. `tests/test_st_gnn_model.py` (GATv2 spatial stream, causal GRU temporal stream, 11D kinematic MLP, fusion head)
3. `tests/test_live_inference.py` (Track buffer update, $T=5$ temporal sliding window, inference format)
4. `tests/test_server.py` (FastAPI REST endpoints, `/api/simulation/attack_burst`, WebSocket handshake)
5. `tests/test_runtime_e2e_validation.py` (WebSocket full-duplex protocol, 4 simulated attacks, XAI schema, SDR decoding)

**Frontend Production Build**:
```powershell
cd frontend; npm.cmd run build
```
**Actual Result**: Exited with code 0 (`built in 623ms`, `dist/assets/index-3uGf3v4E.js` 339 kB, 0 errors).

---

## 6. Official Evaluation Metrics (OOD Test Partition)

The authoritative evaluation numbers audited directly from `data/st_gnn_test_predictions.csv` (Jan 2, 2024 Out-Of-Distribution test set; $N = 32,550$ flight observations, 27,544 Normal, 5,006 Anomaly) at threshold $t = 0.20$:

- **Precision**: **$73.06\%$** ($\text{TP}=903$, $\text{FP}=333$)
- **Recall**: **$18.04\%$** ($\text{TP}=903$, $\text{FN}=4,103$)
- **$F_1$ Score**: **$0.2893$**
- **False Alarm Rate (FAR)**: **$1.21\%$** ($\text{TN}=27,211$, $\text{FP}=333$)
- **ROC-AUC**: **$0.6235$**
- **PR-AUC**: **$0.3720$**

*Note on Earlier Reports*: Earlier internal reports quoted $F_1 \approx 0.51 - 0.52$. That figure was computed on the **in-distribution validation set** (Jan 1). The authoritative generalizability score on unseen flights is $F_1 = 0.2893$.

---

## 7. Project Limitations & SDC Boundary Statement

1. **Software-Only Emulation**: No physical RTL-SDR dongle, RF tuner, or antenna is connected. Decoded messages originate from synthetic baseband I/Q arrays or recorded `.cu8` sample files.
2. **Simplified Waveform & Decoding Scope**: The SDR decoder extracts callsign (TC=1..4) and velocity/heading (TC=19). Compact Position Reporting (CPR, TC=9..18) coordinate reconstruction is not implemented in the baseband bit slicer; spatial coordinates enter via telemetry stream metadata.
3. **Out-of-Distribution Sensitivity**: On unseen airspace sectors, recall is $18.04\%$. Kinematic violations (`altitude_teleport`, `transponder_collision`) are detected at $100\%$, while subtle ghost flights and single-step position jumps are frequently missed without physical RF TDoA authentication.
4. **Attribution vs. Proof**: XAI feature attributions reflect mathematical model sensitivity and physical residual magnitudes; they should not be treated as legal or causal proof of cyber-physical attack.

---

## 8. Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **Port 8000 / 5173 Already in Use** | A previous Python or Node process is still running. | Run `Get-NetTCPConnection -LocalPort 8000,5173` to find PID, then run `Stop-Process -Id <PID> -Force`. |
| **`npm` command not recognized or script execution blocked** | PowerShell execution policy blocks `npm.ps1`. | Always invoke Node commands using `npm.cmd` (e.g., `npm.cmd run dev` or `npm.cmd run build`). |
| **Dashboard shows `BACKEND OFFLINE`** | Backend server on port 8000 is not running. | Verify Terminal 1 has `scripts/aeroguard_server.py` running and that `/ws` is accessible. |
| **Browser automation driver failure** | Playwright CDN returned 404 on Azure driver download. | Open `http://localhost:5173/` directly in Google Chrome, Microsoft Edge, or Firefox. |
