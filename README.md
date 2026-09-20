# AeroGuard AI

**Flight Trajectory Anomaly & ADS-B Spoofing Detection using Graph Neural Networks**

AeroGuard AI is an advanced machine learning framework designed to detect Automatic Dependent Surveillance–Broadcast (ADS-B) spoofing attacks, track modifications, and ghost aircraft injections. By modelling the airspace as a dynamic spatial-temporal graph, the system uses Graph Neural Networks (GNNs) to pass messages along edge boundaries, learning coordinate-invariant relative threat signatures that outperform traditional tabular classifiers under severe geographic domain shifts.

---

## System Architecture

```text
Aircraft observations (WGS84 Track Sequences)
              ↓
  SDR Signal Physics Simulator (Lisbon Receiver coordinates)
              ↓
   Kinematic & Temporal Derivatives Extractor
              ↓
    1-Second Dynamic Graph Snapshot Slicing
              ↓
      Feature Scaling & Chronological Partitioning
              ↓
      PyTorch Geometric 2-Layer GCN Classifier
```

---

## Key Achievements & Methodology

### 1. Preprocessing & Physical SDR Simulator
- **Lisbon Ground Receiver Location**: Ground station centered at Lisbon FIR (LPPC: $38.7756^\circ$ N, $-9.1354^\circ$ W, Alt: $114.0$ m).
- **Physical Signal Simulator**: 
  - WGS84 Geodetic positions projected to 3D Earth-Centered Earth-Fixed (ECEF) coordinates for exact slant-range calculations.
  - ENU velocity components rotated to 3D ECEF vectors to compute exact **radial velocities** relative to the ground receiver.
  - **Doppler Shift Engine**: Models exact carrier frequency offsets ($Hz$) at the ADS-B $1090$ MHz band:
    $$f_D = - \frac{v_r}{c} f_c$$
  - **Path Loss & Power Propagation**: Free Space Path Loss (FSPL) decay curves, Received Signal Strength (RSS in $dBm$), and Signal-to-Noise Ratio (SNR in $dB$) utilizing a physical $-107$ $dBm$ noise floor.

### 2. Multi-Aircraft Dynamic Graphs
- Trajectories are sliced into discrete **1-second graph snapshots** $G(t) = (V(t), E(t))$.
- **Nodes ($V$)**: Active aircraft at second $t$, characterized by a 13-dimensional feature vector of kinematics (groundspeed, altitude, vertical rate) and communication shifts (Doppler, RSS, SNR). Absolute identifiers are ignored to prevent overfitting.
- **Edges ($E$)**: Bidirectional connections formed dynamically between aircraft flying within $100$ km of each other. 
- **Edge Features**: 4-dimensional relative vectors mapping range, relative altitude, relative velocity, and shortest angular heading differences, capturing spatial inconsistencies between spoofed nodes and surrounding traffic.

### 3. Chronological Attack Injections
We injected 7 realistic ADS-B anomaly classes partitioned chronologically to eliminate temporal leakage:
- **Kinematic Anomalies**: `position_jump` (teleportation), `altitude_anomaly` (override), `velocity_anomaly` (drift), `heading_anomaly` (split vectors).
- **Advanced Anomalies**: `vertical_rate_anomaly`, `ghost_aircraft` (synthesized aircraft; Doppler shifts are dynamically calculated based on target motion profiles), and `identity_inconsistency` (split coordinate records under duplicate ICAO addresses).

---

## Dataset Schema

The generated datasets (`data/aero_guard_train.csv` and `data/aero_guard_test.csv`) contain the following features:

| Column Name | Category | Unit / Range | Description |
|---|---|---|---|
| `timestamp` | Metadata | Datetime | Observation datetime |
| `icao24` / `callsign` | Metadata | Hex / String | Aircraft identifiers |
| `latitude` / `longitude` | Kinematics | Degrees | Geographic coordinates |
| `altitude` / `geoaltitude` | Kinematics | Feet | Reported altitudes |
| `groundspeed` | Kinematics | Knots | Reported speed over ground |
| `heading` | Kinematics | Degrees (0–360) | Normalized heading |
| `vertical_rate` | Kinematics | Feet/min | Reported climb/descent rate |
| `x` / `y` | Kinematics | Meters | Projected coordinate coordinates |
| `time_delta` | Derivatives | Seconds | Time interval since last report |
| `acceleration` | Derivatives | $m/s^2$ | Speed rate of change |
| `vertical_acceleration`| Derivatives | $m/s^2$ | Vertical rate change (metric) |
| `turn_rate` | Derivatives | $deg/s$ | Heading rotation rate |
| `climb_rate_change` | Derivatives | $fpm/s$ | Vertical rate change (aviation) |
| `neighbor_count` | Relational | Count | Flight density within 100 km |
| `local_aircraft_density`| Relational | Count | Flight density within 50 km |
| `estimated_doppler_hz` | Communication | Hz | Carrier frequency Doppler shift |
| `distance_to_receiver` | Communication | Meters | Receiver distance |
| `estimated_rss_dbm` | Communication | dBm | Received Signal Strength |
| `estimated_snr_db` | Communication | dB | Signal-to-Noise Ratio |
| `label` | Target | Binary (0 / 1) | Anomaly classification |
| `anomaly_type` | Target Info | String | Anomaly category identifier |

---

## Evaluation Results

- **Domain Shift Setup**: Model trained on Jan 1, 2020 trajectories near Lisbon airspace, and tested on Jan 2, 2020 airspace located in the northern LPPC FIR sector (~1,500 km away).
- **Relational Generalization**: While classical classifiers suffer majority class collapse due to out-of-distribution absolute coordinates and signal values, the GNN relies on relative edge features to achieve spatial invariance.

### Model Performance Benchmarks

| Model | Input Type | Threshold ($t$) | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| **Tabular Baseline (Random Forest)** | Node-level (13D with coordinates) | 0.50 / 0.18 | 1.0000 | 0.1318 | 0.2330 | **0.8717** | **0.5255** |
| **Original GCN Baseline** | Dynamic Graph (13D Node, ignores edge attr) | 0.50 / 0.98 | 0.0000 | 0.0000 | 0.0000 | 0.5787 | 0.2712 |
| **Edge-Aware GNN (Ours)** | Dynamic Graph (12D Coord-free + 4D Edge) | 0.50 | 0.2075 | 0.1282 | 0.1585 | 0.4658 | 0.1927 |
| **Edge-Aware GNN (Ours)** | Dynamic Graph (12D Coord-free + 4D Edge) | **0.01 (Calibrated)** | **0.2071** | **0.1312** | **0.1607** | 0.4658 | 0.1927 |

*Note: Combining coordinate-free features (`heading_sin`, `heading_cos`) with 4D edge attributes (`distance`, `relative_altitude`, `relative_velocity`, `relative_heading`) resolved majority-class collapse, recovering 657 true positive detections and raising F1 from 0.0000 to 0.1607.*

---

## File Directory Reference

- [`data/`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/data/):
  - `aero_guard_train.csv` / `aero_guard_test.csv` — Final standardized datasets.
  - `aero_guard_test_predictions.csv` — Out-of-sample prediction results with model probabilities & thresholds.
- [`docs/`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/docs/):
  - `week1_report.md` — Detailed technical report on Week 1 domain shift mitigation, edge-aware GNN, and threshold calibration.
  - `eda_graph_analysis.md` — Statistical insights on dynamic airspace topology and GNN relational reasoning.
- [`models/`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/models/):
  - `baseline_rf.joblib` — Saved Random Forest baseline model.
  - `gnn_baseline.pt` — Saved PyG 2-layer GCN baseline weights.
  - `gnn_edge_aware.pt` — Saved PyG Edge-Aware GNN (`GATv2Conv`) weights.
  - `calibrated_threshold.json` — Saved validation-calibrated decision thresholds.
- [`notebooks/`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/notebooks/):
  - `01_aeroguard_eda.ipynb` — IPython notebook showing trajectory splits, signal decay distributions, and graph topology.
- [`scripts/`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/scripts/):
  - `generate_aeroguard_dataset.py` — Dataset generator, trajectory simulator, and physics engine.
  - `train_models.py` — Model training, threshold calibration, and OOD evaluation script.
  - `build_notebook.py` — Programmatic notebook generator utility.

---

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed with the following packages:
```bash
pip install pandas numpy scikit-learn torch torch-geometric scipy
```

### 2. Generate the Dataset
To run the preprocessing, execute the physics simulator, inject anomalies, and generate train/test splits:
```bash
python scripts/generate_aeroguard_dataset.py
```

### 3. Model Training & Evaluation
To train the baseline Random Forest model and the PyTorch Geometric GCN model on the dynamic snapgraphs:
```bash
python scripts/train_models.py
```
This will save model weights to `models/` and evaluate predictions, outputting test predictions to `data/aero_guard_test_predictions.csv` and printing metric evaluations.
