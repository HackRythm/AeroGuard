# AeroGuard AI — Scholarly Conference Paper Outline & Manuscript Plan

**Document**: `docs/conference_paper_outline.md`
**Date**: October 2026
**Target Venues**:
- **IEEE DASC** (Digital Avionics Systems Conference)
- **ACM WiSec** (Conference on Security and Privacy in Wireless and Mobile Networks)
- **IEEE / ACM IoTDI** (Conference on Internet of Things Design and Implementation)

---

## Paper Metadata

### Proposed Title
**"AeroGuard AI: Tri-Stream Spatio-Temporal Graph Neural Networks for ADS-B Spoofing Detection and Explainable Airspace Surveillance"**

### Authors
*Software Defined Communication Systems (SDC) Research Group, Department of Electronics & Communication Engineering*

### Abstract Plan (Structured Abstract, ~250 words)
* **Background**: Automatic Dependent Surveillance–Broadcast (ADS-B) is the global backbone of modern Air Traffic Control (ATC), yet operates on an unauthenticated, unencrypted $1090\text{ MHz}$ line-of-sight RF channel. Low-cost Software Defined Radios (SDRs) enable adversaries to broadcast fabricated transponder signals, inject phantom aircraft, and hijack flight paths.
* **Limitations of Existing Work**: Classical tabular anomaly detectors evaluate aircraft in isolation and rely on absolute geographic coordinates, suffering severe majority-class collapse when evaluated on unseen airspace sectors under geographic domain shifts.
* **Proposed Approach**: We propose **AeroGuard AI**, an end-to-end cyber-physical surveillance architecture based on a **Tri-Stream Spatio-Temporal Graph Neural Network (ST-GNN)**. The architecture couples: (1) an edge-aware spatial Graph Attention Network (`GATv2Conv`) operating on coordinate-free dynamic airspace topologies with 4D relative edge attributes (range, relative altitude, velocity difference, heading separation); (2) a causal recurrent Gated Recurrent Unit (GRU) stream processing sliding observation sequences ($T=5$) with explicit observation masks; and (3) a kinematic residual Multi-Layer Perceptron (MLP) branch enforcing physical motion invariants and transponder collision indicators.
* **Evaluation & Results**: Evaluated on an out-of-distribution (OOD) test set of $32,550$ flight observations located $\approx 1,500\text{ km}$ away from training traffic, AeroGuard AI achieves **$86.37\%$ overall accuracy**, **$73.06\%$ precision**, and an F1-score of **$0.2893$** (outperforming tabular Random Forest by $+0.153$ F1), while maintaining a low false alarm rate of **$1.21\%$**. The pipeline resolves transponder cloning vulnerability (achieving **$100.0\%$ detection** on duplicate ICAO collisions and altitude overrides).
* **Explainability & Trust**: A tri-stream Explainable AI (XAI) framework provides multi-quadrant incident dossiers that isolate physical and relational threat drivers while explicitly distinguishing model attributions from physical causal proof.

### Keywords
ADS-B Security, Spatio-Temporal Graph Neural Networks, Cyber-Physical Systems, Software-Defined Radio, Explainable AI, Avionics Surveillance.

---

## Detailed Section-by-Section Manuscript Outline

```
                                  CONFERENCE PAPER STRUCTURE
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Introduction: Unauthenticated 1090 MHz ADS-B & The Threat of Low-Cost SDR Spoofing │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Related Work: Tabular ML vs. Sequence Models vs. Relational Graph Networks in ATC  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Threat Model & Signal Physics Modeling: Receiver Geometry, Doppler & FSPL Decay     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Tri-Stream ST-GNN Architecture: Spatial GATv2 + Temporal GRU + Kinematic Residuals  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. Explainable AI (XAI) Framework: GNNExplainer, Saliency & Attribution vs. Causal Proof│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 6. Experimental Evaluation: 32,550 OOD Samples, Baseline Comparisons & Ablation Study │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 7. Operational Discussion & Physical Boundaries: Ghost Mimicry, CPR & TDoA Limits     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 8. Conclusion & Future Directions                                                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Section I: Introduction
1. **The Modern Airspace Vulnerability**:
   - Transition from primary/secondary radar to ADS-B (DO-260B standard).
   - The fundamental architectural flaw: lack of cryptographic signatures, origin authentication, or replay protection on the $1090\text{ MHz}$ pulse position modulated (PPM) extended squitter channel.
2. **The Democratization of RF Attackers**:
   - Accessibility of consumer SDRs (RTL-SDR, HackRF, BladeRF) capable of transmitting arbitrary Mode-S packets.
   - Attack taxonomy: altitude step teleportation, trajectory drifting, cloned transponder replay (`identity_inconsistency`), and coordinated ghost aircraft injections.
3. **Shortcomings of Conventional Detection Methods**:
   - Tabular machine learning classifiers evaluate single aircraft in isolation, failing to detect relational conflicts (e.g., neighbor altitude disparities or identical transponder hex addresses in neighboring sectors).
   - Overfitting to localized geographic coordinates: when models trained on Mediterranean airspace are deployed to Northern Europe, classifiers collapse due to absolute coordinate shifts.
4. **Key Contributions of AeroGuard AI**:
   - *Coordinate-Free Relational Formulation*: Dynamic 1-second graph snapshots connecting aircraft within $100\text{ km}$ using 4D relative edge attributes (distance, $\Delta h$, $\Delta v$, $\Delta \theta$).
   - *Tri-Stream ST-GNN Architecture*: Fusing spatial relational attention, causal temporal recurrent dynamics ($T=5$), and physical motion residuals.
   - *Zero-Leakage OOD Generalization*: Rigorous evaluation under geographic domain shift ($1,500\text{ km}$ separation).
   - *Multi-Stream XAI Framework*: Attribution dossiers providing transparent root-cause analysis for ATC decision support.

---

### Section II: Related Work
1. **Classical & Tabular Anomaly Detection in ADS-B**:
   - Overview of One-Class SVM, Isolation Forest, Random Forest, and autoencoders applied to ADS-B telemetry.
   - Limitation: Evaluation within identical geographic sectors, ignoring cross-aircraft relational topology.
2. **Deep Sequence Modeling**:
   - Recurrent Neural Networks (LSTMs, GRUs) and Transformers modeling single-aircraft track histories.
   - Limitation: Inability to capture multi-aircraft spatial geometry or simultaneous transponder cloning collisions.
3. **Graph Neural Networks (GNNs) in Air Traffic Management**:
   - Recent applications of GCNs/GATs to flight delay propagation and airspace sector congestion.
   - Research gap: Lack of edge-conditioned attention models designed for real-time cyber-physical spoofing detection under geographic domain shift.
4. **Physical-Layer RF Fingerprinting vs. Telemetry Verification**:
   - Hardware-level I/Q transient analysis vs. message-level kinematic verification.
   - Trade-offs: Hardware fingerprinting requires expensive high-bandwidth SDR arrays; message-level ST-GNN operates on standard Mode-S demodulators.

---

### Section III: Threat Model & Signal Physics Simulator
1. **Attacker Capabilities & Assumptions**:
   - Adversary with transmit-capable SDR operating within line-of-sight of air traffic.
   - Ability to inject, modify, or replay Mode-S DF17 messages.
2. **Virtual Ground Receiver Geometry (Lisbon LPPC FIR)**:
   - Centered at Lisbon Airport Portela ($38.7756^\circ\text{ N}, -9.1354^\circ\text{ W}$, Elevation: $114.0\text{ m}$).
   - WGS84 Geodetic to 3D Earth-Centered Earth-Fixed (ECEF) Cartesian coordinate transformations.
3. **Analytical Signal Physics Engine**:
   - *3D Slant-Range Vector*: $\vec{R} = \vec{P}_{ac} - \vec{P}_{rx}$.
   - *3D ECEF Velocity Rotation*: Local East-North-Up (ENU) vectors rotated to ECEF frame via geodetic rotation matrix $R^T$.
   - *Doppler Shift Modeling*: $f_D = - \frac{v_r}{c} f_c$ on the $1090\text{ MHz}$ carrier.
   - *Signal Power Propagation*: Free Space Path Loss (FSPL), Received Signal Strength ($\text{RSS} = P_{tx} - \text{FSPL}$), and Signal-to-Noise Ratio ($\text{SNR} = \text{RSS} - N_0$).

---

### Section IV: The Tri-Stream ST-GNN Architecture
1. **Dynamic Airspace Graph Formulation**:
   - Snapshot slicing at $1\text{ Hz}$ temporal resolution: $G(t) = (V(t), E(t))$.
   - Proximity-based edge formulation ($d_h(i, j) \le 100\text{ km}$).
   - Coordinate-free node representations: 12D behavioral vector omitting absolute lat/lon; unit-circle heading decomposition ($\sin\theta, \cos\theta$).
   - 4D relative edge attributes: $\mathbf{e}_{ij} = [d_h, \Delta h, \Delta v, \Delta \theta]$.
2. **Stream 1: Spatial Relational GATv2**:
   - Edge-aware attention message passing:
     $$\alpha_{ij} = \frac{\exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}_v \mathbf{h}_i + \mathbf{W}_u \mathbf{h}_j + \mathbf{W}_e \mathbf{e}_{ij}\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}_v \mathbf{h}_i + \mathbf{W}_u \mathbf{h}_k + \mathbf{W}_e \mathbf{e}_{ik}\right)\right)}$$
   - Extraction of 16-dimensional spatial relational embeddings $h_i^{\text{spatial}}(t)$.
3. **Stream 2: Temporal Causal GRU**:
   - Sliding historical track window: $T=5$ timesteps $[t-4, t-3, t-2, t-1, t]$.
   - Input vector at step $\tau$: Concatenation of spatial embedding (16D) and normalized kinematics (9D) $\to \mathbf{z}_i(\tau) \in \mathbb{R}^{25}$.
   - Boolean observation validity masking: zeroing padded pre-flight intervals to eliminate ghost transitions.
4. **Stream 3: Motion Invariant Kinematic MLP**:
   - Causal motion residuals:
     - Haversine step displacement ratio: $\text{disp\_ratio} = \frac{\Delta s / \Delta t}{\max(v \cdot 0.5144, 1.0)}$.
     - Barometric altitude consistency: $\text{alt\_residual} = \left| \Delta h - v_z \cdot \frac{\Delta t}{60} \right|$.
     - Acceleration & turn-rate discrete residuals.
     - Concurrent ICAO address collision indicator: $\text{icao\_collision} = \mathbb{I}(\text{count of ICAO24 broadcasts at } t > 1)$.
5. **Multi-Stream Fusion & Decision Boundary Calibration**:
   - Weighted probability combination: $p_{\text{fused}} = 0.40 \cdot p_{\text{spatial}} + 0.35 \cdot p_{\text{temporal}} + 0.25 \cdot p_{\text{kinematic}}$.
   - Operational threshold calibrated at $t^* = 0.20$ via validation F1-score optimization.

---

### Section V: Multi-Stream Explainable AI (XAI)
1. **The Need for Attribution in Safety-Critical ATC**:
   - Why black-box scores cannot be operationalized in civil aviation.
2. **Modular Attribution Mechanics**:
   - *Spatial Attribution*: `GNNExplainer` mutual information optimization identifying top node features and critical neighbor flights.
   - *Temporal Saliency*: Input $\times$ Gradient backpropagation across the 5-step window identifying peak divergence timesteps.
   - *Kinematic Sensitivity*: Partial derivative sensitivity across the 11 physical motion residuals.
3. **Attribution vs. Causal Proof (A Crucial Distinction)**:
   - Differentiating mathematical model attributions (explaining neural network logits) from physical causal proof (which requires multi-station TDoA or independent radar validation).

---

### Section VI: Experimental Evaluation & Benchmark Results
1. **Experimental Setup & Split Isolation**:
   - Jan 1 training ($77,910$ rows) and validation ($19,440$ rows) vs. Jan 2 OOD test ($32,550$ rows).
2. **Benchmark Comparisons on Identical 32,550 Test Samples**:
   - Inclusion of Table comparing Random Forest, Original GCN, Historical Edge-Aware GNN, Repaired Spatial GNN, and Tri-Stream ST-GNN.
3. **Multi-Stream Ablation Study**:
   - Independent evaluation of Spatial-only, Temporal-only, Kinematic-only, and Fusion strategies.
   - Stream specialization table: Spatial excels on altitude ($1.0000$), Kinematics excels on transponder cloning ($0.9994$) and position jumps ($0.9697$).
4. **Per-Attack Detection Rates**:
   - Altitude override ($100.0\%$), Transponder cloning ($100.0\%$), Normal specificity ($98.79\%$), Ghost aircraft ($2.6\%$), Position jumps ($0.2\%$).
5. **Discrepancy Reconciliation & Historical Debugging Analysis**:
   - Documentation of the Week 1 edge double-scaling bug and row-ordering mismatch, demonstrating how empirical auditing restored spatial GNN ROC-AUC from $0.4578$ to $0.7635$.

---

### Section VII: Operational Discussion & Physical Boundaries
1. **The Ghost Aircraft Cruise Challenge**:
   - Analysis of why kinematically smooth synthetic ghost flights achieve low single-sensor recall ($2.6\%$).
   - Requirement for external airline schedule cross-referencing or distributed RF TDoA.
2. **The Position Jump Discontinuity Boundary**:
   - Only the initial teleportation boundary second triggers kinematic step ratios; subsequent parallel cruise mimics steady flight.
3. **Software-Defined Mode & Protocol Decoding Limitations**:
   - Software simulation boundary (absence of hardware USB SDR).
   - Type Codes 1–4 and 19 decoded; Compact Position Reporting (CPR, Type Codes 9–18) bypassed via telemetry stream.
   - Absence of physical-layer TDoA multilateration.

---

### Section VIII: Conclusion & Future Directions
1. **Summary of Findings**:
   - Coordinate-free relational spatial-temporal graphs effectively prevent majority-class collapse under geographic domain shift.
   - Tri-stream fusion resolves transponder cloning and kinematic step violations with a low $1.21\%$ false alarm rate.
2. **Future Research Directions**:
   - Distributed multi-receiver Time Difference of Arrival (TDoA) graph edge attributes.
   - Real-world flight plan schedule integration via Eurocontrol B2B web services.
   - Full 112-bit CPR odd/even ambiguity resolution on live hardware RTL-SDR testbeds.
