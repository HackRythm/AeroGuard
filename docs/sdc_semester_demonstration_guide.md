# AeroGuard AI — Software-Defined Communication (SDC) Demonstration & Academic Guide

**Project Title**: AeroGuard AI: Flight Trajectory Anomaly & ADS-B Spoofing Detection Using Graph Neural Networks
**Academic Subject**: Software Defined Communication Systems (SDC)
**Academic Level**: Semester 3 Final Deliverable & Demonstration Package

---

## 1. Executive Summary & SDC Principles

AeroGuard AI bridges physical-layer Software-Defined Radio (SDR) communication principles with deep graph learning and Explainable AI (XAI) to detect ADS-B cyber-physical threats (spoofing, trajectory modification, and ghost aircraft injection).

```
                     ┌────────────────────────────────────────────────────────┐
                     │   Software-Defined Communication (SDC) Subsystem       │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
          Mode 1: Recorded IQ File (.cu8)       │  Mode 2: Synthetic 1090 MHz Baseband
          [uint8 interleaved, 2 MSPS, 1090 MHz] │  [PPM modulation, AWGN, 8us preamble]
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │          ADSBWaveformDecoder (Baseband DSP)            │
                     │  1. Envelope Detection: |s(t)| = sqrt(I^2 + Q^2)       │
                     │  2. Matched Filter Preamble Correlation (8 us)         │
                     │  3. Pulse Position Modulation (PPM) Bit Slicing        │
                     │  4. 24-bit Mode-S CRC Verification (G(x) = 0x1FFF409)  │
                     └──────────────────────────┬─────────────────────────────┘
                                                │ DecodedADSBMessage
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │           ADSBTrackBuffer & Kinematic Engine           │
                     │  - Chronological observation history (T=5 window)      │
                     │  - Discrete derivatives (accel, turn_rate, climb_rate) │
                     │  - Causal physical residuals (disp_ratio, alt_res)     │
                     │  - Neutral RF feature imputation for unobserved frames │
                     └──────────────────────────┬─────────────────────────────┘
                                                │ Active Airspace Tracks
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │          Multi-Stream ST-GNN Classifier                │
                     │  Stream 1: Spatial GNN (Edge-Aware GATv2Conv, 100 km)  │
                     │  Stream 2: Temporal Recurrent GRU (5-timestep history) │
                     │  Stream 3: Causal Kinematic Residual MLP               │
                     │  Tri-Stream Weighted Fusion (Threshold t = 0.20)       │
                     └──────────────────────────┬─────────────────────────────┘
                                                │ Anomaly Score >= 0.20
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │         Unified Multi-Stream Explainable AI            │
                     │  - Spatial: PyG GNNExplainer critical neighbor edges   │
                     │  - Temporal: GRU Input x Gradient saliency profile     │
                     │  - Kinematic: Physical residual gradient attributions  │
                     │  - Sanity Checks: Independent physical rule warnings   │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │        AeroGuard Command Center (React + Aiohttp)      │
                     │  Live Radar Display | Multi-Stream XAI | SDR Test Bursts│
                     └────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Hardware Transparency Disclosure**:
> In compliance with strict academic integrity standards, **no physical RTL-SDR radio hardware dongle was purchased, claimed, or required**. The SDR receiver subsystem operates entirely in **Software-Defined Simulation Mode** via physical baseband modulation (`ADSBWaveformModulator`), recorded complex IQ file ingestion (`RecordedIQSource`), and synthetic baseband signal processing (`SyntheticIQSource`), verified against the official Mode-S DO-260B specification.

---

## 2. Mathematical & Algorithmic Formulations

### A. RF Baseband Modulation & Demodulation
- **Carrier Frequency**: $f_c = 1090\,\text{MHz}$
- **Sampling Rate**: $f_s = 2.0\,\text{MSPS}$ (each microsecond corresponds to 2 complex baseband samples)
- **Modulation**: 1 Mbit/s Pulse Position Modulation (PPM). A binary `1` transmits a pulse in the first $0.5\,\mu\text{s}$ chip; a binary `0` transmits in the second $0.5\,\mu\text{s}$ chip.
- **Preamble**: $8.0\,\mu\text{s}$ sequence of 16 half-microsecond chips with pulse positions at indices $0, 2, 7, 9$:
  $$P = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]$$
- **Envelope Detection**:
  $$|s(t)| = \sqrt{I^2(t) + Q^2(t)}$$
- **Matched Filter Correlation Trigger**: Triggered when pulse energy ratio over quiet valley energy satisfies:
  $$\frac{E_{\text{pulse}}}{E_{\text{valley}} + \epsilon} > 1.8 \quad \text{and} \quad E_{\text{pulse}} > 0.15$$

### B. 24-Bit Mode-S CRC Polynomial
The ICAO Mode-S generator polynomial is 25 bits ($0x1FFF409$):
$$G(x) = x^{24} + x^{23} + x^{22} + x^{21} + x^{20} + x^{19} + x^{18} + x^{17} + x^{16} + x^{15} + x^{14} + x^{13} + x^{12} + x^{10} + x^3 + 1$$
Every valid 112-bit message satisfies $\sum \text{remainder}[88:112] \equiv 0 \pmod{G(x)}$.

### C. Tri-Stream Spatio-Temporal GNN (ST-GNN)
1. **Spatial Encoder ($h_{\text{spatial}}$)**: 2-layer Edge-Aware `GATv2Conv` processing 12D coordinate-free node features and 4D edge attributes within $100\,\text{km}$ proximity:
   $$e_{ij} = [d_{ij}, \Delta \text{alt}, \Delta v, \Delta \theta]$$
2. **Temporal Recurrent Stream ($h_{\text{temporal}}$)**: Single-layer GRU processing sliding windows $T=5$ of concatenated $[h_{\text{spatial}}(\tau) \parallel k_{9\text{D}}(\tau)]$ with explicit boolean observation mask:
   $$\text{mask}[i, \tau] = \begin{cases} 1 & \text{if observation exists} \\ 0 & \text{if padded / missing} \end{cases}$$
3. **Kinematic Residual Stream ($h_{\text{kinematic}}$)**: 2-layer MLP evaluating causal physical invariants:
   $$\text{disp\_ratio} = \frac{\Delta s_{\text{haversine}}}{\max(v \cdot \Delta t, 1.0)}, \quad \text{alt\_residual} = \left| \Delta \text{alt} - v_z \cdot \frac{\Delta t}{60} \right|$$
4. **Calibrated Decision Threshold**:
   $$\text{Predicted Status} = \begin{cases} \text{ANOMALY} & \text{if } P_{\text{fused}} \ge 0.20 \\ \text{NOMINAL} & \text{if } P_{\text{fused}} < 0.20 \end{cases}$$

---

## 3. Verified Empirical Benchmarks

All models were evaluated on the out-of-distribution (OOD) Jan 2, 2020 test set (3,600 graph snapshots, 5,007 spoofed anomaly observations):

| Model Architecture | Threshold ($t$) | Precision | Recall | F1-Score | Accuracy | Specificity | FPR | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tabular Random Forest Baseline | $0.50$ (Default) | $0.5742$ | $0.0757$ | $0.1338$ | $0.8492$ | $0.9898$ | $0.0102$ | $0.5528$ | $0.2029$ |
| Original GCN Baseline | $0.50$ (Default) | $0.0000$ | $0.0000$ | $0.0000$ | $0.8462$ | $1.0000$ | $0.0000$ | $0.5808$ | $0.2671$ |
| Repaired Edge-Aware GNN (Spatial) | $0.50$ (Default) | $0.8772$ | $0.1484$ | $0.2539$ | $0.8658$ | $0.9962$ | $0.0038$ | $0.7635$ | $0.4347$ |
| **Tri-Stream ST-GNN (Ours)** | **0.20 (Calibrated)** | **0.7306** | **0.1804** | **0.2893** | **0.8637** | **0.9879** | **0.0121** | **0.6235** | **0.3720** |

---

## 4. Step-by-Step Demonstration Instructions

### Step 1: Execute Complete Automated Test Suite (94 Tests)
Run the complete unit and integration test suite:
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```
*Expected Output*: `Ran 94 tests in ~35s, OK`. Covers baseband DSP, CRC-24, ST-GNN inference, XAI attribution, WebSocket streaming, benchmark evaluation, ablation protocol, server robustness, and attack burst simulations.

### Step 2: Test End-to-End Software SDR Pipeline
Run the dedicated end-to-end integration test:
```powershell
.venv\Scripts\python.exe -m unittest tests.test_end_to_end_sdr_st_gnn
```
*Expected Output*: `Ran 1 test, OK`. Validates waveform generation $\to$ PPM matched filter $\to$ CRC check $\to$ track buffer $\to$ ST-GNN inference $\to$ XAI dossier.

### Step 3: Start Command Center Backend API & WebSocket Service
Launch the async HTTP & WebSocket server on port 8000:
```powershell
.venv\Scripts\python.exe scripts\aeroguard_server.py
```
*Verification*: Open `http://127.0.0.1:8000/api/status` in browser or run:
```powershell
curl http://127.0.0.1:8000/api/status
```

### Step 4: Launch Vite Frontend GUI
In a separate terminal:
```powershell
cd frontend
npm.cmd run dev
```
Open `http://localhost:5173` to explore:
- **Live Radar Canvas**: Dynamic coordinate bounding box projection supporting real-time tracking with range rings and compass rose.
- **Top Bar**: SDR receiver status badge (`SDC 1090 MHz | SYNTHETIC IQ`), "TEST SDR BURST" button, and "SIMULATE ATTACK BURST" menu.
- **GNN Panel**: Switch between the live 100 km relational graph mesh and the verified 3-Stream ST-GNN architecture topology.
- **Active Flight Alerts & XAI**: Click any flight to open the 4-tab Multi-Stream XAI Inspector (Spatial, Temporal, Kinematics, Limitations).

### Step 5: Test Simulated Attack Bursts via Command Line or GUI
Trigger an academic test injection (e.g. altitude teleportation):
```powershell
curl -X POST http://127.0.0.1:8000/api/simulation/attack_burst -H "Content-Type: application/json" -d "{\"attack_type\": \"altitude_teleport\", \"icao24\": \"500006\"}"
```
*Response*:
```json
{
  "status": "INJECTED",
  "attack_type": "altitude_teleport",
  "description": "SIMULATED TEST BURST: Instantaneous +7500 ft altitude teleportation without climb envelope.",
  "target_icao24": "500006",
  "is_simulated_test": true,
  "predictions": [...]
}
```

---

## 5. Summary of SDC Course Deliverables

| Deliverable | Location | Description |
|---|---|---|
| SDR Receiver Subsystem | `scripts/rtl_sdr_adapter.py` | Software-defined receiver with envelope detector, PPM matched filtering & CRC-24. |
| Live Inference Pipeline | `scripts/live_st_gnn_inference.py` | Track buffer, discrete kinematics, sliding window ($T=5$), ST-GNN. |
| Multi-Stream XAI Engine | `scripts/aeroguard_explainer.py` | PyG GNNExplainer, temporal saliency, kinematic residuals. |
| Full Automated Suite | `tests/` (94 tests) | 100% passing test discovery suite covering DSP, ML, WebSocket, benchmarks, ablation, server robustness, and APIs. |
| Interactive Web Dashboard | `frontend/` (React + Vite) | Command center radar, architecture inspector, and 4-tab XAI panel. |
| Audited XAI Case Studies | `docs/xai_audit_cases.json` | Audited dossiers (true anomalies, normal, borderline). |
| Technical Reports | `docs/` | Comprehensive reports covering GNN repair, ST-GNN evaluation, and XAI. |
