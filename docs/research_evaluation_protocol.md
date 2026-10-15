# AeroGuard AI — Research Evaluation Protocol for Scholarly Publication

**Document**: `docs/research_evaluation_protocol.md`
**Date**: October 2026
**Target Submission**: IEEE / ACM Conference on Cyber-Physical Systems, Avionics, or Wireless Security (e.g., IEEE DASC, ACM WiSec)
**Status**: Verified & Reproducible Evaluation Protocol

---

## 1. Introduction & Research Scope

AeroGuard AI is a cyber-physical surveillance framework designed to detect Automatic Dependent Surveillance–Broadcast (ADS-B) spoofing and trajectory anomalies without relying on encrypted transponders or external radar infrastructure. Operating on the unauthenticated $1090\text{ MHz}$ line-of-sight RF channel, ADS-B is vulnerable to low-cost Software Defined Radio (SDR) replay, teleportation, velocity manipulation, and phantom aircraft injection attacks.

This protocol establishes the **strict, reproducible evaluation procedure** for assessing the proposed **Tri-Stream Spatio-Temporal Graph Neural Network (ST-GNN)** against classical tabular and spatial graph baselines under geographical domain shift.

---

## 2. Dataset Partitioning, Geography, & Domain Shift Setup

### 2.1 Geographic Domain Shift
To assess realistic generalization across air traffic control sectors, the evaluation pipeline enforces an out-of-distribution (OOD) geographic domain shift between training and test sets:
* **Training & Validation Partition (Jan 1, 2020)**:
  - **Center Coordinate**: Lisbon Portela Airport FIR Center (LPPC: $38.7756^\circ\text{ N}$, $-9.1354^\circ\text{ W}$, Elevation: $114.0\text{ m}$).
  - **Airspace Coverage**: Iberian Peninsula / Atlantic approach corridors ($36.5^\circ\text{ N} - 41.5^\circ\text{ N}$, $-12.0^\circ\text{ W} - -6.0^\circ\text{ W}$).
  - **Total Samples**: $97,350$ observation records across $27$ unique commercial aircraft.
  - **Split Separation**:
    - Chronological Training Split (first 80%, $t = 0\text{ s} - 2880\text{ s}$): $77,910$ rows.
    - Chronological Validation Split (last 20%, $t = 2881\text{ s} - 3600\text{ s}$): $19,440$ rows.
* **Independent Test Partition (Jan 2, 2020 OOD)**:
  - **Airspace Coverage**: Northern European corridor located $\approx 1,500\text{ km}$ northeast of the training sector ($50.5^\circ\text{ N} - 54.0^\circ\text{ N}$, $-11.5^\circ\text{ W} - -6.5^\circ\text{ W}$).
  - **Total Samples**: $32,550$ observation records across $8$ unique commercial aircraft and $1$ injected phantom identity (`e00002`).
  - **Isolation**: $100\%$ held out until all model weights, scalers, and decision thresholds were frozen.

### 2.2 Class Distribution & Attack Injections
Across the $32,550$ test observations:
* **Normal Traffic (Class 0)**: $27,544$ records ($84.62\%$).
* **Anomalous Traffic (Class 1)**: $5,006$ records ($15.38\%$), partitioned across four mutually exclusive synthetic attack scenarios:
  1. **`altitude_anomaly`**: $660$ records ($13.18\%$ of anomalies). Instantaneous $+6,000$ to $+12,000\text{ ft}$ vertical overrides without climb transition.
  2. **`ghost_aircraft`**: $3,600$ records ($71.91\%$ of anomalies). Synthesized phantom flight (`e00002` / `GHOST02`) cruising in a parallel corridor ($0.12^\circ$ offset) with kinematically plausible cruise speed and zero vertical rate.
  3. **`position_jump`**: $596$ records ($11.91\%$ of anomalies). Coordinate teleportation ($0.08^\circ - 0.15^\circ$ lat/lon displacement) maintaining subsequent smooth cruising track.
  4. **`identity_inconsistency`**: $150$ records ($3.00\%$ of anomalies). Transponder cloning injecting concurrent duplicate broadcasts under the same 24-bit Mode-S address (`500002`) in different spatial coordinates.

---

## 3. Data Leakage Controls & Preprocessing Pipeline

To ensure scientific integrity and eliminate synthetic artifacts, the evaluation pipeline enforces four strict leakage barriers:

```
Raw Telemetry Observation ──► [Leakage Audit Filter] ──► [Coordinate-Free Transform] ──► Scalers (Fitted on Train Only)
(Strictly Drops true_*)        (Asserts zero future)     (lat/lon dropped; sin/cos hdg)   (scaler_node, scaler_edge, scaler_kin)
```

1. **Zero Internal Ground-Truth Leakage**:
   - The dataset generation script ([`scripts/generate_aeroguard_dataset.py`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/scripts/generate_aeroguard_dataset.py)) uses internal columns (`true_latitude`, `true_longitude`, `true_altitude`, `true_groundspeed`, `true_heading`, `true_vertical_rate`) during RF Doppler/RSS simulation.
   - All `true_*` columns are strictly stripped prior to dataset serialization.
   - The leakage audit in [`scripts/train_st_gnn.py`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/scripts/train_st_gnn.py) asserts $\text{count}(\text{true\_*}) = 0$.
2. **Zero Temporal Peeking (Causal Consistency)**:
   - Numerical kinematic residuals ($\text{disp\_ratio}$, $\text{alt\_residual}$, $\text{accel\_residual}$, $\text{turn\_residual}$) and sliding history sequences ($T=5$) use backward lags only ($\tau \le t$).
   - Future observations ($\tau > t$) are inaccessible to both training and test passes.
3. **Strict Split Isolation**:
   - `StandardScaler` instances ([`models/scaler_node_repaired.joblib`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/models/scaler_node_repaired.joblib), [`models/scaler_edge_repaired.joblib`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/models/scaler_edge_repaired.joblib), [`models/scaler_kinematic.joblib`](file:///d:/SEM_3/SDC/project%20code/AeroGuard/models/scaler_kinematic.joblib)) were fitted strictly on the $80\%$ training split of Jan 1.
   - The operational decision threshold ($t^* = 0.20$) was calibrated strictly on the $20\%$ validation split of Jan 1 by maximizing validation F1-score.
4. **Coordinate-Free Node Representation**:
   - To prevent spatial over-fitting to specific geographic runways or corridors, absolute `latitude` and `longitude` are omitted from the GNN node feature vector.
   - Heading ($\theta$) is decomposed into continuous unit-circle components ($\sin\theta, \cos\theta$) to eliminate $0^\circ \leftrightarrow 360^\circ$ angular discontinuities.

---

## 4. Evaluated Model Architectures & Baselines

All models are evaluated on the exact same $32,550$ out-of-distribution test observations:

### 4.1 Baseline 1: Tabular Random Forest
* **Input Structure**: 13-dimensional node features including absolute latitude and longitude, velocity, altitude, heading, and physical signal estimates.
* **Architecture**: 100 decision trees, max depth 12, balanced class weighting.
* **Hypothesis**: Tabular models relying on absolute coordinates will experience severe performance degradation when tested on distant airspace sectors.

### 4.2 Baseline 2: Original Graph Convolutional Network (GCN)
* **Input Structure**: Dynamic 1-second airspace graph snapshots ($100\text{ km}$ proximity edges); 13D node features.
* **Message Passing**: 2 layers of standard `GCNConv` (Kipf & Welling) with symmetric normalized adjacency; edge attributes are ignored.
* **Hypothesis**: Relational neighborhood aggregation provides partial spatial smoothing, but ignoring edge distances and relative velocity vectors causes majority class collapse.

### 4.3 Baseline 3: Repaired Edge-Aware GNN (Spatial Stream)
* **Input Structure**: Dynamic graph snapshots with 12D coordinate-free node features and 4D relative edge attributes:
  $$\mathbf{e}_{ij} = [d_h(i, j), \, h_i - h_j, \, v_i - v_j, \, (\theta_i - \theta_j + 180^\circ)\bmod 360^\circ - 180^\circ]$$
* **Message Passing**: 2 layers of `GATv2Conv` with multi-head attention conditioned directly on normalized 4D edge attributes.
* **Output**: 16-dimensional spatial relational embeddings $h_i^{\text{spatial}}(t)$ and spatial anomaly probability $p_{\text{spatial}}$.

### 4.4 Proposed Model: Tri-Stream ST-GNN
* **Architecture**: Modular tri-stream architecture fusing spatial, temporal, and kinematic dynamics:
  1. *Spatial Stream*: Frozen Repaired Edge-Aware GATv2Conv generating $16\text{D}$ spatial relational representations.
  2. *Temporal Stream*: Recurrent Gated Recurrent Unit (GRU, hidden dimension $32$) processing causal sliding sequences of length $T=5$ ($25\text{D}$ concatenated spatial and kinematic states) with boolean observation validity masks.
  3. *Kinematic Stream*: Multi-Layer Perceptron (MLP, $11 \to 32 \to 16 \to 2$) evaluating causal motion invariants and transponder collision indicators.
  4. *Fusion Head*: Weighted probability combination:
     $$p_{\text{fused}} = 0.40 \cdot p_{\text{spatial}} + 0.35 \cdot p_{\text{temporal}} + 0.25 \cdot p_{\text{kinematic}}$$
* **Operational Threshold**: $t^* = 0.20$ (calibrated on validation split).

---

## 5. Evaluation Metrics & Statistical Formulas

Because civil airspace anomaly detection exhibits severe natural class imbalance ($15.38\%$ positive anomalies), raw accuracy is evaluated alongside threshold-independent ranking metrics and aviation alarm metrics:

1. **Overall Accuracy**:
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
2. **Precision & Recall (Sensitivity)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
3. **F1-Score**:
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
4. **Specificity (True Negative Rate) & False Alarm Rate (FPR)**:
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}, \quad \text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}} = 1 - \text{Specificity}$$
5. **Continuous Ranking Metrics**:
   - **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve evaluated on continuous model probability outputs.
   - **PR-AUC**: Area Under the Precision-Recall Curve (Average Precision score), robust to class imbalance.

---

## 6. Execution Commands for Independent Replication

All evaluation scripts are fully automated and require no external dependencies beyond the environment virtual environment:

```powershell
# Step 1: Activate Environment
cd "d:\SEM_3\SDC\project code\AeroGuard"
.\.venv\Scripts\Activate.ps1

# Step 2: Execute Independent Research Benchmark Harness
python research/evaluate_benchmarks.py

# Step 3: Execute Stream Ablation Protocol (Preservation Mode)
python research/ablation_protocol.py

# Step 4: Execute Scientific XAI Attribution Audit
python research/audit_xai_scientific.py

# Step 5: Execute Automated Verification Test Suite (51 Tests)
python -m unittest discover -s tests -p "test_*.py"
```
