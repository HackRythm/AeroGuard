# AeroGuard AI: Flight Trajectory Anomaly and ADS-B Spoofing Detection Using Graph Neural Networks

**Authors:** Anonymous Authors (Submitted for Blind Peer Review)<br>
**Conference Track:** Avionics Cyber-Physical Security / Machine Learning for Air Traffic Management<br>
**Manuscript Status:** Full Conference Draft (Conference-Neutral Format)<br>
**Evaluated Artifact Baseline:** AeroGuard AI Research Evaluation Release v1.0.0

---

## Abstract

Automatic Dependent Surveillance–Broadcast (ADS-B) constitutes the unencrypted, unauthenticated surveillance standard for contemporary civil aviation and Air Traffic Control (ATC). The absence of cryptographic message authentication on the $1090\,\text{MHz}$ Mode-S Extended Squitter downlink permits adversaries using low-cost Software-Defined Radios (SDRs) to inject falsified aircraft trajectories, override barometric altitudes, and clone transponder addresses. Conventional machine learning approaches evaluate aircraft tracks as isolated scalar sequences using tabular classifiers or recurrent models. When evaluated under spatial domain shifts, isolated approaches struggle to generalize and remain incapable of capturing cooperative relational dependencies between adjacent aircraft.

In this paper, we evaluate **AeroGuard AI**, a cyber-physical surveillance framework based on a Tri-Stream Spatio-Temporal Graph Neural Network (ST-GNN). The architecture couples: (1) an absolute-coordinate-free spatial Graph Attention Network ($\text{GATv2}$) operating on dynamic $100\,\text{km}$ horizontal proximity graphs parameterized by 4D relative edge attributes (range, relative altitude, velocity disparity, and heading separation); (2) a causal Gated Recurrent Unit (GRU) stream processing $T=5$ sliding observation sequences with input validity masking; and (3) a kinematic Multi-Layer Perceptron (MLP) stream evaluating 11 physical motion residuals and transponder collision indicators.

We evaluate the framework on an independent, synthetically generated Out-of-Distribution (OOD) test dataset of $32,550$ flight observations simulating a geographic domain shift of $\approx 1,500\,\text{km}$ from training traffic. Operating the official weighted ST-GNN at its validation-calibrated operational threshold ($t^*=0.20$), the model achieves an accuracy of $86.37\%$, precision of $73.06\%$, recall of $18.04\%$, and an F1-score of $0.2893$, while maintaining an operational false alarm rate of $1.21\%$ (specificity $98.79\%$). In ablation comparisons, an alternative maximum-probability fusion strategy achieves higher F1 ($0.3741$) and higher recall ($30.76\%$), but at an elevated false alarm rate of $6.12\%$ ($1,686$ false alarms), underscoring the operational trade-off required for air traffic surveillance. In isolated threat categories, the framework achieves $100.0\%$ detection on barometric altitude overrides ($660/660$) and $100.0\%$ detection on transponder address collisions ($150/150$), with the latter driven by deterministic kinematic collision indicators. However, our evaluation establishes clear physical and operational boundaries: aerodynamically compliant ghost aircraft mimicking nominal cruising flight achieve an operational recall of only $2.56\%$ ($92/3,600$), while sustained position jumps achieve only $0.17\%$ operational recall ($1/596$) at threshold $0.20$ despite elevated continuous kinematic discrimination ($\text{ROC-AUC}=0.9697$). We accompany the detector with an Explainable AI (XAI) engine providing multi-quadrant attribution dossiers, while critically distinguishing mathematical model attributions from physical causal proof. We document reproducible data generation, model weights, and alignment-verified evaluation protocols.

---

## Keywords

ADS-B Security, Cyber-Physical Systems, Spatio-Temporal Graph Neural Networks, Air Traffic Control, Explainable AI, Anomaly Detection.

---

## 1. Introduction & Contributions

Civil aviation safety relies on cooperative surveillance systems to maintain safe separation minima between aircraft in controlled airspace. Under the mandates of the International Civil Aviation Organization (ICAO) and the Federal Aviation Administration (FAA), Automatic Dependent Surveillance–Broadcast (ADS-B) has largely superseded secondary surveillance radar (SSR) as the primary surveillance infrastructure [1, 2]. Under the RTCA DO-260B standard [3], aircraft determine their position via Global Navigation Satellite System (GNSS) receivers and broadcast periodic Mode-S Extended Squitter packets (Downlink Format 17) at $1090\,\text{MHz}$.

Despite its operational efficiency, ADS-B is fundamentally unauthenticated and unencrypted [4]. Broadcasts transmit in plain text across an open wireless channel without digital signatures, cryptographic time-stamps, or physical-layer authentication. Over the past decade, the availability of commercial off-the-shelf (COTS) Software-Defined Radios (SDRs)—such as the HackRF One, BladeRF, and USRP—has lowered the barrier to broadcasting synthetic ADS-B transmissions. Modeled threat vectors include injecting ghost flights, altering flight telemetry, modifying coordinate reports, and cloning valid transponder hex identifiers (`icao24`), directly misleading ground-based air traffic management systems and airborne collision avoidance algorithms [4, 5].

```
                     CONVENTIONAL VS. AEROGUARD AI FORMULATION
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Conventional Tabular / Recurrent Approach (Isolated Trajectory)        │
  │   Aircraft i: [lat, lon, alt, speed] ──► Isolated Classifier ──► Score │
  │   * Sensitive to coordinate shifts; blind to relational airspace context│
  ├────────────────────────────────────────────────────────────────────────┤
  │ AeroGuard AI Tri-Stream Relational Formulation (Cooperative Airspace)  │
  │   Aircraft i ◄─── 4D Relative Edge ───► Aircraft j                    │
  │   ┌──────────────────┐ ┌───────────────────┐ ┌─────────────────────┐   │
  │   │ Spatial GATv2    │ │ Causal GRU (T=5)  │ │ Kinematic Residuals │   │
  │   │(Absolute-Coord-  │ │(Input Masked,     │ │(Physical Residuals, │   │
  │   │ Free 16D Embed)  │ │ Sequence-Gathered)│ │ ICAO Collision Flag)│   │
  │   └────────┬─────────┘ └─────────┬─────────┘ └──────────┬──────────┘   │
  │            └───────────────► Weighted Fusion ◄──────────┘              │
  │                                    │                                   │
  │                        Decision: Anomaly (t=0.20)                      │
  └────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Limitations of Prior Work

Existing machine learning approaches to ADS-B anomaly detection typically exhibit several methodological constraints:

1. **Isolated Sequence Modeling**: Many prior works train classifiers—such as Random Forests, Support Vector Machines, or isolated LSTMs—on individual aircraft track histories [6, 7, 8]. These architectures evaluate each flight in isolation, omitting the spatial relational geometry between neighboring aircraft. Consequently, they cannot detect relational conflicts, such as transponder cloning where two distinct aircraft broadcast the same 24-bit identifier in the same surveillance volume.
2. **Sensitivity to Absolute Geographic Coordinates**: Prior models frequently ingest absolute latitude and longitude coordinates directly [8]. While effective when evaluated on identically distributed splits within the same terminal area, models conditioned directly on absolute coordinates can degrade when tested in distant sectors where spatial coordinate baselines shift by thousands of kilometers.
3. **Metric Reporting Granularity**: Machine learning literature frequently emphasizes continuous ranking metrics (such as ROC-AUC) without reporting threshold-dependent operational metrics (precision, recall, and false alarm rate) on realistic class-imbalanced splits. High overall accuracy or specificity can easily mask near-zero recall on subtle attack categories, such as aerodynamically compliant ghost aircraft or sustained trajectory offsets.

### 1.2 Scope & Contributions of This Work

To address these modeling challenges and establish clear empirical boundaries, we evaluate **AeroGuard AI**, an open-source framework for flight anomaly detection. Our primary contributions are:

- **Absolute-Coordinate-Free Relational Formulation**: We model the airspace as a dynamic graph $G(t)=(V(t), E(t))$ where edges are established dynamically based on a $100\,\text{km}$ horizontal proximity threshold. By eliminating absolute geographic coordinates (latitude and longitude) from node states and conditioning spatial attention on 4D relative edge attributes (range, relative altitude, velocity disparity, and heading separation), we improve model transferability across distant simulated airspace sectors.
- **Tri-Stream Spatio-Temporal Graph Architecture**: We formulate a tri-stream architecture that decomposes flight telemetry into three complementary branches: (1) an edge-aware spatial Graph Attention Network ($\text{GATv2}$); (2) a causal Gated Recurrent Unit (GRU) sequence encoder with input validity masking; and (3) a kinematic Multi-Layer Perceptron (MLP) stream evaluating 11 physical motion residuals and transponder collision indicators.
- **Rigorous Out-of-Distribution (OOD) Benchmark**: We benchmark our model against tabular and graph baselines on an independent synthetic test dataset of $32,550$ flight observations displaced geographically by $\approx 1,500\,\text{km}$ from training traffic. We employ an observation-level join on a 5-tuple composite primary key to ensure deterministic label alignment.
- **Transparent Operational Boundary Analysis**: Rather than presenting the model as universally robust, we analyze its empirical limitations. We show that while the tri-stream ST-GNN reliably detects altitude overrides ($100.0\%$) and transponder cloning ($100.0\%$), kinematically smooth ghost aircraft ($2.56\%$ recall) and sustained position jumps ($0.17\%$ recall) present substantial detection hurdles for single-receiver message-level surveillance.
- **Explainable AI with Causal Boundaries**: We provide a multi-quadrant attribution engine coupling $\text{GNNExplainer}$, gradient saliency, and kinematic sensitivity. We explicitly formalize the distinction between mathematical model attributions (explaining neural network logits) and physical causal proof (which requires multi-lateration or independent primary sensor verification).

---

## 2. Background & Related Work

### 2.1 The ADS-B Broadcast Protocol & Physical Layer

The Automatic Dependent Surveillance–Broadcast system operates on the $1090\,\text{MHz}$ Mode-S Extended Squitter channel [3]. Transmissions utilize Pulse Position Modulation (PPM) at a nominal data rate of $1\,\text{Mbit/s}$. A standard Extended Squitter packet (Downlink Format 17) comprises an $8\,\mu\text{s}$ preamble followed by a 112-bit data block containing:
- A 5-bit Downlink Format identifier ($\text{DF}=17$).
- A 3-bit Capability field ($\text{CA}$).
- A 24-bit ICAO aircraft address (`icao24`).
- A 56-bit payload field containing airborne position, velocity, identification, or status data.
- A 24-bit Cyclic Redundancy Check (CRC) parity field ($G(x) = \text{0x1FFF409}$).

The 24-bit parity field is designed strictly to detect random bit errors caused by channel noise and multi-path fading; it provides **zero cryptographic protection** against intentional fabrication [4]. An attacker possessing a software-defined transceiver can calculate valid CRC polynomial remainders for arbitrary message contents. In civil aviation standards (DO-260B), aircraft positions are encoded into 17-bit Compact Position Reporting (CPR) formats requiring alternating even and odd frames to resolve absolute coordinates globally. Because receivers decode these packets open-loop, any injected signal matching the preamble structure and CRC polynomial is ingested by standard surveillance decoders.

### 2.2 Machine Learning for ADS-B Anomaly Detection

Prior research in ADS-B anomaly detection falls into three broad methodological categories:

1. **Tabular Machine Learning**: Early investigations applied classical tabular methods—such as One-Class SVMs, Isolation Forests, and Random Forests—to extracted ADS-B telemetry fields [6, 7]. While computationally efficient, tabular classifiers treat each aircraft state as an independent observation. When deployed across distinct geographical sectors, tabular models trained with absolute coordinates can suffer from severe distributional shift.
2. **Deep Sequence Models**: Recurrent Neural Networks (LSTMs, GRUs) and temporal autoencoders have been deployed to evaluate trajectory continuity [8, 9]. In particular, Habler and Shabtai [8] demonstrated that LSTM encoder-decoders can detect anomalous message sequences by monitoring reconstruction errors. However, single-track sequence models process each flight independently, leaving them blind to multi-aircraft spatial geometry or simultaneous transponder cloning collisions.
3. **Graph Neural Networks (GNNs) in Aviation**: Recently, GNNs have been applied to aviation systems for air traffic flow management and sector congestion forecasting [10]. For instance, Cai et al. [10] utilized time-evolving graph convolutional networks across airport networks to model flight delay propagation. Standard Graph Convolutional Networks (GCNs) [11], however, typically aggregate unweighted neighbor features, which can lead to over-smoothing across dominant normal traffic classes and lack conditioning on continuous pairwise edge attributes.

### 2.3 Physical-Layer vs. Message-Level Verification

To counter spoofing, physical-layer verification techniques—such as Time Difference of Arrival (TDoA) multi-lateration, Doppler shift tracking, and RF fingerprinting—have been extensively investigated [4, 12]. TDoA systems correlate signal arrival times across multiple geographically distributed sensor stations to triangulate transmitter physical coordinates. While physically robust, multi-lateration requires tightly synchronized clocks across distributed receiver infrastructure. In contrast, AeroGuard AI operates at the **message-level and relational cyber-physical layer**, serving as a software-based analytical monitor designed to run alongside standard single-station ADS-B receiver data streams.

---

## 3. Problem Formulation & Threat Model

### 3.1 Modeled Threat Scenario

We model an active cyber-physical threat scenario within line-of-sight of an air traffic surveillance sector. The modeled adversary is assumed to possess a transmit-capable Software-Defined Radio (e.g., HackRF One, USRP) capable of broadcasting Mode-S Extended Squitter packets at $1090\,\text{MHz}$ with sufficient power to reach a local receiving station. We emphasize that this represents a **modeled threat scenario evaluated via synthetic data generation**, not a physical over-the-air radio experiment. We investigate four specific attack categories:

1. **Barometric Altitude Override (`altitude_anomaly`)**: The attacker broadcasts falsified altitude fields, injecting abrupt step displacements ($+6,000\,\text{ft}$ to $+12,000\,\text{ft}$) while reporting nominal vertical rates. This attack models an attempt to trigger false collision advisories or mask unauthorized altitude deviations.
2. **Trajectory Position Jump (`position_jump`)**: The attacker overrides GNSS latitude and longitude reports, introducing instantaneous coordinate displacements ($0.08^\circ$ to $0.15^\circ$, approximately $9\text{--}16\,\text{km}$) while maintaining normal cruising speed and heading reports.
3. **Transponder Address Collision / Cloning (`identity_inconsistency`)**: The attacker replays or duplicates valid Extended Squitter messages from an active legitimate flight, alters the reported coordinates, and broadcasts the signal simultaneously with the target flight's real transponder address. This creates two distinct tracks sharing the identical 24-bit hex address in the same surveillance volume.
4. **Coordinated Ghost Aircraft Injection (`ghost_aircraft`)**: The attacker synthesizes an entirely fabricated flight trajectory (assigned unassigned transponder addresses, e.g., `e00002`) flying parallel to legitimate air corridors. The synthetic flight adheres to nominal cruising speeds, climb rates, and turn dynamics to mimic legitimate commercial air traffic.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        THREAT TAXONOMY MAPPING                         │
├────────────────────────┬─────────────────────┬─────────────────────────┤
│ Attack Class           │ Kinematic Profile   │ Injected Mechanism      │
├────────────────────────┼─────────────────────┼─────────────────────────┤
│ `altitude_anomaly`     │ Discontinuous Step  │ Barometric override     │
│ `position_jump`        │ Discrete Jump       │ Lat/Lon teleportation   │
│ `identity_inconsistency`│ Duplicate Hex       │ Transponder replay      │
│ `ghost_aircraft`       │ Aerodynamic Cruise  │ Full track synthesis    │
└────────────────────────┴─────────────────────┴─────────────────────────┘
```

### 3.2 Airspace as a Dynamic Relational Graph

We model the monitored airspace at discrete 1-second epochs $t \in \{1, 2, \dots, T\}$ as an attributed dynamic graph:
$$G(t) = (V(t), E(t), \mathbf{X}(t), \mathbf{E}(t))$$
where:
- $V(t) = \{v_1, v_2, \dots, v_{N(t)}\}$ is the set of $N(t)$ aircraft actively broadcasting telemetry within the surveillance volume during epoch $t$.
- $E(t) \subseteq V(t) \times V(t)$ is the set of undirected relational edges. An edge $e_{ij} = (v_i, v_j)$ exists if and only if the horizontal distance $d_h(i, j)$ between aircraft $i$ and aircraft $j$ is within the relational interaction radius $R_{\text{prox}} = 100\,\text{km}$:
  $$E(t) = \{(v_i, v_j) \mid d_h(v_i, v_j) \le 100\,\text{km}, \, i \neq j\}$$
- $\mathbf{X}(t) \in \mathbb{R}^{N(t) \times D_v}$ is the node feature matrix ($D_v = 12$).
- $\mathbf{E}(t) \in \mathbb{R}^{|E(t)| \times D_e}$ is the edge feature matrix containing continuous 4D relational attributes ($D_e = 4$).

---

## 4. Dataset Construction & Signal Physics Modeling

### 4.1 Synthetic Trajectory Synthesis & Receiver Modeling

To evaluate the proposed architecture, we construct a synthetic trajectory simulation pipeline (`scripts/generate_aeroguard_dataset.py`). The dataset is synthesized by initializing commercial jet kinematic parameters (cruising speeds $400\text{--}480\,\text{kts}$, flight levels $28,000\text{--}38,000\,\text{ft}$) and stepping trajectories forward at $1\,\text{Hz}$ using kinematic dead-reckoning equations with minor Gaussian perturbations. We emphasize that **the dataset consists of synthetically generated telemetry records structured to match standard ADS-B state parameters**, rather than raw over-the-air RF captures from an operational ATC facility.

The simulation pipeline incorporates a modeled ground receiver located at Lisbon Portela Airport (LPPC FIR center: $38.7756^\circ\,\text{N}$, $-9.1354^\circ\,\text{W}$, elevation $114.0\,\text{m}$). For each simulated observation, the engine computes deterministic propagation metrics:

1. **ECEF Transformation**: Geodetic coordinates $(\phi, \lambda, h)$ are converted to WGS84 Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates $(X, Y, Z)$ via:
   $$X = (N(\phi) + h) \cos\phi \cos\lambda, \quad Y = (N(\phi) + h) \cos\phi \sin\lambda, \quad Z = \left(N(\phi)(1 - e^2) + h\right) \sin\phi$$
   where $N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$, with semi-major axis $a = 6,378,137.0\,\text{m}$ and eccentricity squared $e^2 = 0.00669437999014$.
2. **Slant Range**: The 3D range vector $\vec{R} = \vec{P}_{\text{ac}} - \vec{P}_{\text{rx}}$ yields slant distance $d = \|\vec{R}\|_2$.
3. **Doppler Shift Modeling**: Velocity components $(v_g, \theta, v_z)$ are mapped to local East-North-Up (ENU) coordinates and rotated into ECEF velocity $\vec{v}_{\text{ecef}}$. Radial velocity is computed via unit projection $v_r = \vec{v}_{\text{ecef}} \cdot (\vec{R}/\|\vec{R}\|)$, yielding nominal Doppler shift:
   $$f_D = - \frac{v_r}{c} f_c$$
   where $f_c = 1090\,\text{MHz}$ and $c \approx 299,792,458\,\text{m/s}$.
4. **Signal Attenuation (FSPL & SNR)**: Free-space path loss is modeled as:
   $$\text{FSPL }(\text{dB}) = 20 \log_{10}(d) + 20 \log_{10}(f_c) - 147.56$$
   Received Signal Strength ($\text{RSS} = P_{\text{tx}} - \text{FSPL}$) is evaluated assuming nominal transmitter power $P_{\text{tx}} = 54.0\,\text{dBm}$ ($250\,\text{W}$) with receiver thermal noise floor $N_0 = -107.0\,\text{dBm}$, yielding $\text{SNR} = \text{RSS} - N_0$.

> **Simulation Consistency Note**: In internal verification checks, correlations between radial velocity and Doppler shift ($-1.0000$) and distance and RSS ($-0.9611$) represent **closed-form numerical consistency checks of the simulation generator**, rather than empirical validation of an over-the-air physical radio channel. No physical RTL-SDR hardware was deployed, and no over-the-air spoofing transmissions were conducted.

### 4.2 Data Partitioning & Synthetic Geographic Domain Shift

To evaluate model transferability under spatial displacement, the simulation pipeline synthesizes geographically separated sectors:

- **Training Partition (Jan 1, 2020)**: $77,910$ observations ($t = 0\text{--}2880\,\text{s}$) across 26 commercial aircraft simulated in the Lisbon FIR sector.
- **Validation Partition (Jan 1, 2020)**: $19,440$ observations ($t = 2881\text{--}3600\,\text{s}$) across the same 26 aircraft. Used strictly for model selection and threshold calibration ($t^* = 0.20$); isolated from test data.
- **Out-of-Distribution (OOD) Test Partition (Jan 2, 2020)**: $32,550$ observations across 8 unseen commercial aircraft and 1 synthetic ghost identity (`e00002`). The base coordinates are displaced by $\approx 1,500\,\text{km}$ to a northern European airspace sector ($52.0^\circ\,\text{N}$, $-9.0^\circ\,\text{W}$).

```
┌────────────────────────────────────────────────────────────────────────┐
│               CANONICAL OOD TEST SET TAXONOMY (32,550 SAMPLES)         │
├────────────────────────┬─────────────┬──────────────┬──────────────────┤
│ Category Label         │ Target ICAO │ Sample Count │ % of Anomalies   │
├────────────────────────┼─────────────┼──────────────┼──────────────────┤
│ `normal`               │ 8 Aircraft  │ 27,544       │ — (Nominal)      │
│ `ghost_aircraft`       │ `e00002`    │ 3,600        │ 71.91%           │
│ `altitude_anomaly`     │ `500006`    │ 660          │ 13.18%           │
│ `position_jump`        │ `500002`    │ 596          │ 11.91%           │
│ `identity_inconsistency`│ `500002`   │ 150          │ 3.00%            │
├────────────────────────┼─────────────┼──────────────┼──────────────────┤
│ Total Test Space       │ —           │ 32,550       │ 100.0% (5,006 A) │
└────────────────────────┴─────────────┴──────────────┴──────────────────┘
```

The four anomaly categories are **strictly mutually exclusive**: each observation contains exactly one ground-truth label and one anomaly category ($3,600 + 660 + 596 + 150 = 5,006$ anomalies).

---

## 5. The Tri-Stream ST-GNN Architecture

AeroGuard AI processes flight telemetry snapshots through three branches capturing spatial, temporal, and physical aspects of the surveillance data.

```
                            TRI-STREAM ST-GNN PIPELINE
   Raw ADS-B Telemetry Snapshots G(t) = (V(t), E(t))
   │
   ├──► [Stream 1: Spatial Relational GATv2]
   │    Node: 12D Absolute-Coord-Free | Edge: 4D Relational [dh, d_alt, d_vel, d_hdg]
   │    2x GATv2Conv Layers (16 Hidden) ──► Spatial Embedding h_sp (16D) ──► p_sp
   │
   ├──► [Stream 2: Temporal Causal Recurrent GRU]
   │    Sliding Window T=5 [t-4 ... t] + Input Zero-Masking M(tau)
   │    Causal GRU (32 Hidden) + Last-Valid Gathering ──► Temporal Embedding h_temp ──► p_temp
   │
   └──► [Stream 3: Physical Invariant Kinematic MLP]
        11 Motion Residuals: [disp_ratio, alt_residual, icao_collision, ...]
        2-Layer MLP (32 -> 16) ──► Kinematic Embedding h_kin ──► p_kin
   │
   ▼
   Weighted Score Fusion: p_fused = 0.40*p_sp + 0.35*p_temp + 0.25*p_kin
   Decision Boundary: If p_fused >= 0.20 ──► FLAG ANOMALY (Otherwise NORMAL)
```

### 5.1 Stream 1: Spatial Relational GATv2

To improve transferability across geographic sectors, absolute latitude, longitude, and hex identifiers are omitted from node features. Each node $v_i$ is parameterized by a 12-dimensional vector $\mathbf{x}_i \in \mathbb{R}^{12}$:
$$\mathbf{x}_i = [h, v_g, \sin\theta, \cos\theta, v_z, a, \omega, a_z, \Delta v_z, f_D, \text{RSS}, \text{SNR}]$$

> **Coordinate Independence Clarification**: We describe this representation as **absolute-coordinate-free** because direct latitude and longitude coordinates are excluded. However, we note that features such as $f_D$ (Doppler shift), $\text{RSS}$, and $\text{SNR}$ are physically coupled to the aircraft's position and velocity relative to the receiver location. Thus, while direct coordinate dependence is removed, receiver-relative spatial geometry is partially retained.

Every edge $e_{ij} \in E(t)$ is assigned a 4D continuous relative edge vector:
$$\mathbf{e}_{ij} = [d_h(i, j), \, |h_i - h_j|, \, |v_i - v_j|, \, \Delta \theta_{ij}]$$
where $\Delta \theta_{ij} = \arccos(\cos(\theta_i - \theta_j))$ represents angular heading separation.

We deploy a two-layer Graph Attention Network with edge conditioning ($\text{GATv2Conv}$) [13]. For node $i$, the attention coefficient $\alpha_{ij}$ to neighbor $j \in \mathcal{N}(i) \cup \{i\}$ is computed dynamically:
$$\alpha_{ij} = \frac{\exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}_v \mathbf{h}_i + \mathbf{W}_u \mathbf{h}_j + \mathbf{W}_e \mathbf{e}_{ij}\right)\right)}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}_v \mathbf{h}_i + \mathbf{W}_u \mathbf{h}_k + \mathbf{W}_e \mathbf{e}_{ik}\right)\right)}$$
The updated node representation aggregates neighbor embeddings weighted by $\alpha_{ij}$:
$$\mathbf{h}_i^{(1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} \mathbf{W}_v \mathbf{h}_j\right)$$
Layer 1 projects the 12D inputs to 32 dimensions; Layer 2 extracts a 16-dimensional spatial relational embedding $\mathbf{h}_i^{\text{spatial}}$ passed to a linear classification head yielding spatial anomaly probability $p_{\text{spatial}}$.

### 5.2 Stream 2: Temporal Causal GRU

The temporal stream captures short-term trajectory dynamics using a sliding window of $T=5$ seconds: $\tau \in [t-4, t-3, t-2, t-1, t]$. At each step $\tau$, the input vector concatenates the spatial embedding with normalized kinematics: $\mathbf{z}_i(\tau) \in \mathbb{R}^{25}$. The sliding window operates strictly on past and present observations, ensuring no future data leakage.

In the PyTorch implementation (`scripts/st_gnn_model.py`), track padding (e.g., at track initiation where fewer than 5 historical steps exist) is handled via a boolean observation mask $\mathbf{M} \in \{0, 1\}^{B \times 5}$. Input features at padded timesteps are zeroed:
$$\tilde{\mathbf{z}}_i(\tau) = \mathbf{z}_i(\tau) \cdot M_i(\tau)$$
The zeroed sequence passes through the standard GRU recurrence:
$$\mathbf{h}_i^{\text{temporal}}(\tau) = \text{GRU}\left(\tilde{\mathbf{z}}_i(\tau), \, \mathbf{h}_i^{\text{temporal}}(\tau-1)\right)$$
We note that zeroing input features does not freeze recurrent state transitions, as recurrent weights and bias terms remain active. To extract the relevant representation, the implementation explicitly indexes the hidden state at the last valid timestep $L_i = \max(0, \sum_\tau M_i(\tau) - 1)$:
$$\mathbf{h}_i^{\text{valid}} = \mathbf{h}_i^{\text{temporal}}(L_i)$$
This gathered hidden representation is processed through dropout ($p=0.1$) and a linear head to produce temporal probability $p_{\text{temporal}}$.

### 5.3 Stream 3: Physical Invariant Kinematic MLP

The kinematic branch computes 11 physical motion residuals derived from aeronautical motion constraints:
1. **Haversine Step Displacement Ratio**:
   $$\text{disp\_ratio} = \frac{\Delta s / \Delta t}{\max(v_g \cdot 0.5144, \, 1.0)}$$
   where $\Delta s$ is the Haversine distance between consecutive positions. For nominal flight, $\text{disp\_ratio} \approx 1.0$; abrupt coordinate jumps cause $\text{disp\_ratio} \gg 1.0$.
2. **Barometric Altitude Residual**:
   $$\text{alt\_residual} = \left| \Delta h - v_z \cdot \frac{\Delta t}{60} \right|$$
   Measures mismatch between geometric altitude change and integrated barometric climb rate.
3. **Concurrent Transponder Address Collision Indicator**:
   $$\text{icao\_collision} = \mathbb{I}\left(\sum_{j \in V(t)} \mathbb{I}(\text{icao24}_j == \text{icao24}_i) > 1\right)$$
   A deterministic binary indicator flagging whether multiple simultaneous messages broadcast the identical 24-bit ICAO address at epoch $t$.
4. **Additional Motion Derivatives**: Acceleration residuals, turn rate derivatives, vertical acceleration, and climb rate variance.

The 11D residual vector is fed to a 2-layer MLP ($11 \to 32 \to 16 \to 1$) with ReLU activations and Sigmoid output, producing kinematic anomaly probability $p_{\text{kinematic}}$.

### 5.4 Multi-Stream Score Fusion

The official deployed decision score is a weighted linear combination of the three probabilities:
$$p_{\text{fused}} = 0.40 \cdot p_{\text{spatial}} + 0.35 \cdot p_{\text{temporal}} + 0.25 \cdot p_{\text{kinematic}}$$

The operational decision threshold was calibrated on the independent validation split by evaluating the F1-score across thresholds $t \in [0.05, 0.95]$. The optimal validation threshold was derived at $t^* = 0.20$ (yielding validation F1 of $0.5271$). The test partition was not exposed during threshold selection. An alert is flagged whenever $p_{\text{fused}} \ge 0.20$.

---

## 6. Experimental Evaluation & Benchmark Results

### 6.1 Baseline Alignment Methodology & Key Integrity

To guarantee evaluation integrity, all baseline predictions must be aligned observation-for-observation against ground-truth labels. While ground-truth test data is ordered chronologically by timestamp ($T=3,600$ snapshots), historical training scripts grouped observations by flight track (`icao24`). Furthermore, naive joins on `(timestamp, icao24)` fail because transponder collision attacks (`identity_inconsistency`) intentionally introduce simultaneous duplicate hex broadcasts.

We establish deterministic alignment using an inner join on the 5-tuple composite primary key:
$$\text{Key} = (\text{timestamp}, \, \text{icao24}, \, \text{latitude}, \, \text{longitude}, \, \text{altitude})$$
Across all $32,550$ observations in all baseline and prediction files, this key exhibits **zero nulls, zero duplicates, and 100.0% unique alignment** ($32,550/32,550$ matched rows with $100.0\%$ ground-truth label agreement). Historical unaligned evaluation numbers are preserved in an isolated audit trail (`research/metrics/historical_unaligned_benchmarks.json`).

### 6.2 Overall Benchmark Comparison on Identical Test Observations

Table 1 presents the performance of the evaluated architectures on the $32,550$ OOD test observations ($27,544$ normal flights, $5,006$ anomalies).

```
Table 1: Benchmark evaluation on 32,550 Out-of-Distribution test observations.
All baseline predictions aligned deterministically via 5-tuple composite primary key.
```

| Model Architecture | Input Features & Structure | Decision Threshold ($t$) | Precision | Recall | F1-Score | Accuracy | Specificity | FPR | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tabular)** | Node-level 13D (with coordinates) | Default ($0.50$) | **$1.0000$** | $0.1318$ | $0.2330$ | **$0.8665$** | **$1.0000$** | **$0.0000$** | **$0.8047$** | $0.4107$ |
| **Random Forest (Tabular)** | Node-level 13D (with coordinates) | Calibrated ($0.15$) | $0.6897$ | $0.1318$ | $0.2214$ | $0.8574$ | $0.9892$ | $0.0108$ | **$0.8047$** | $0.4107$ |
| **Original GCN Baseline** | Dynamic Graph (13D, unweighted) | Default ($0.50$) | $0.0000$ | $0.0000$ | $0.0000$ | $0.8462$ | **$1.0000$** | **$0.0000$** | $0.4778$ | $0.1372$ |
| **Original GCN Baseline** | Dynamic Graph (13D, unweighted) | Calibrated ($0.12$) | $0.0000$ | $0.0000$ | $0.0000$ | $0.8462$ | **$1.0000$** | **$0.0000$** | $0.4778$ | $0.1372$ |
| **Historical Edge GNN (Defective)** | GATv2 (Double-scaled edge attributes) | Default ($0.50$) | $0.2089$ | $0.1318$ | $0.1616$ | $0.7897$ | $0.9092$ | $0.0908$ | $0.6552$ | $0.3086$ |
| **Historical Edge GNN (Defective)** | GATv2 (Double-scaled edge attributes) | Calibrated ($0.99$) | $0.2544$ | $0.1318$ | $0.1737$ | $0.8071$ | $0.9298$ | $0.0702$ | $0.6552$ | $0.3086$ |
| **Repaired Spatial GNN** | GATv2 (12D Coord-free + 4D Relational) | Default ($0.50$) | $0.8772$ | $0.1484$ | $0.2539$ | $0.8658$ | $0.9962$ | $0.0038$ | $0.7635$ | **$0.4347$** |
| **Tri-Stream ST-GNN (Ours)** | Spatial + Temporal + Kinematic Fusion | Default ($0.50$) | $0.8825$ | $0.1470$ | $0.2521$ | $0.8658$ | $0.9964$ | $0.0036$ | $0.6235$ | $0.3720$ |
| **Tri-Stream ST-GNN (Ours)** | Spatial + Temporal + Kinematic Fusion | **Calibrated ($0.20$)** | $0.7306$ | **$0.1804$** | **$0.2893$** | $0.8637$ | $0.9879$ | $0.0121$ | $0.6235$ | $0.3720$ |

### 6.3 Detailed Analysis of Benchmark Results

The benchmark results demonstrate clear architectural trade-offs:

1. **Threshold Derivation Procedures**:
   - For Random Forest, the default threshold is $0.50$; its validation-tuned threshold ($t=0.15$) was selected by maximizing validation F1.
   - For Original GCN, probabilities collapse below $0.12$, yielding zero positive predictions at both default ($0.50$) and validation-calibrated ($0.12$) thresholds.
   - For the Tri-Stream ST-GNN, the operational threshold ($t^*=0.20$) was calibrated on the validation partition, boosting recall from $14.70\%$ to $18.04\%$ and F1 from $0.2521$ to $0.2893$.
2. **Majority-Class Collapse in Unweighted GCN**: The unweighted GCN baseline predicts zero anomalies ($0.0000$ precision, recall, and F1). Rather than demonstrating over-smoothing conclusively (which would require layer-depth ablation), we observe that without continuous edge conditioning, the model collapses to the majority normal class ($84.62\%$).
3. **Tabular Random Forest Behavior**: At $t=0.50$, Random Forest achieves $1.0000$ precision and higher ROC-AUC ($0.8047$) than the fused ST-GNN ($0.6235$). However, this performance is driven exclusively by detecting altitude anomalies ($660/660$). Random Forest detects zero ghost aircraft, zero position jumps, and zero transponder collisions, resulting in a low recall of $13.18\%$.
4. **Tri-Stream ST-GNN Trade-Off**: The fused ST-GNN does not outperform every baseline on every metric. While Random Forest achieves higher precision and ROC-AUC due to its narrow specialization on altitude jumps, the Tri-Stream ST-GNN achieves broader detection capability across attack categories, yielding an F1-score of **$0.2893$** at an operational false alarm rate of $1.21\%$.

```
                CONFUSION MATRIX ARITHMETIC VERIFICATION (t = 0.20)
                       Predicted Normal (0)   Predicted Anomaly (1)      Total
       True Normal (0)        27,211                   333               27,544
       True Anomaly (1)        4,103                   903                5,006
       Total                  31,314                 1,236               32,550
```

Verified confusion matrix counts:
$$\text{Accuracy} = \frac{27,211 + 903}{32,550} = \mathbf{0.863717} \quad (86.37\%)$$
$$\text{Precision} = \frac{903}{903 + 333} = \frac{903}{1,236} = \mathbf{0.730583} \quad (73.06\%)$$
$$\text{Recall} = \frac{903}{903 + 4,103} = \frac{903}{5,006} = \mathbf{0.180384} \quad (18.04\%)$$
$$\text{F1-Score} = \frac{2 \times 0.730583 \times 0.180384}{0.730583 + 0.180384} = \mathbf{0.289330}$$

---

## 7. Multi-Stream Ablation Study & Per-Class Analysis

### 7.1 Architectural Stream Ablation

Table 2 evaluates each constituent stream and alternative score fusion strategies on the identical test split.

```
Table 2: Dynamic Multi-Stream Ablation Study on 32,550 test observations.
Artifact source: research/metrics/ablation_results.json
```

| Stream / Fusion Strategy | Threshold | Precision | Recall | F1-Score | Accuracy | Specificity | FPR | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spatial-Only (Repaired GATv2)** | $0.50$ | $0.8772$ | $0.1484$ | $0.2539$ | $0.8658$ | $0.9962$ | $0.0038$ | **$0.7635$** | **$0.4347$** |
| **Temporal-Only (Causal GRU, T=5)** | $0.50$ | $0.8777$ | $0.1476$ | $0.2527$ | $0.8657$ | $0.9963$ | $0.0037$ | $0.3333$ | $0.2499$ |
| **Kinematic-Only (Residual MLP)** | $0.50$ | $0.1388$ | $0.1686$ | $0.1523$ | $0.7113$ | $0.8099$ | $0.1901$ | $0.5468$ | $0.2077$ |
| **Fusion: Average Probability** | $0.27$ | **$0.7379$** | $0.1794$ | $0.2886$ | $0.8640$ | $0.9884$ | $0.0116$ | $0.6228$ | $0.3702$ |
| **Fusion: Maximum Probability** | $0.73$ | $0.4774$ | **$0.3076$** | **$0.3741$** | $0.8417$ | $0.9388$ | $0.0612$ | $0.6203$ | $0.3747$ |
| **Official Weighted ST-GNN** | **$0.20$** | $0.7306$ | $0.1804$ | $0.2893$ | $0.8637$ | $0.9879$ | $0.0121$ | $0.6235$ | $0.3720$ |

> **Ablation Performance Acknowledgment**: We explicitly note that the **Maximum Probability Fusion strategy achieves the highest F1-score ($0.3741$) and recall ($30.76\%$)** among all evaluated configurations at threshold $t=0.73$. However, this occurs at a substantially elevated False Positive Rate of **$6.12\%$** ($1,686$ false alarms, compared to $333$ for the weighted model). In operational air traffic surveillance, a $6\%$ false alarm rate imposes severe controller workload. The official weighted model was chosen because it achieves a balanced F1 ($0.2893$) and precision ($73.06\%$) while maintaining a low false alarm rate of $1.21\%$.

### 7.2 Diagnostic Discrimination by Anomaly Type (Isolated ROC-AUC)

To evaluate continuous ranking capability without interference from other attack classes, we compute the **one-vs-normal ROC-AUC** for each anomaly category. Specifically, for each category, the evaluation subset contains exclusively the positive observations of that attack class evaluated against all $27,544$ normal flight observations, excluding observations from the remaining three attack types.

```
Table 3: Isolated ROC-AUC by Anomaly Category (Evaluated against 27,544 Normal Flights).
```

| Evaluated Stream / Strategy | `altitude_anomaly` (660) | `ghost_aircraft` (3,600) | `position_jump` (596) | `identity_inconsistency` (150) |
| :--- | :---: | :---: | :---: | :---: |
| **Spatial-Only (Repaired GATv2)** | **$1.0000$** | **$0.8422$** | $0.1351$ | $0.3300$ |
| **Temporal-Only (Causal GRU)** | **$1.0000$** | $0.2399$ | $0.1827$ | $0.2385$ |
| **Kinematic-Only (Residual MLP)** | $0.4525$ | $0.4752$ | **$0.9697$** | **$0.9994$** |
| **Fusion: Average Probability** | $0.9993$ | $0.4814$ | $0.9658$ | $0.9955$ |
| **Fusion: Maximum Probability** | $0.9999$ | $0.4778$ | $0.9660$ | $0.9958$ |
| **Tri-Stream Fused ST-GNN** | **$0.9994$** | $0.4824$ | **$0.9654$** | **$0.9954$** |

### 7.3 Operational Per-Class Detection Breakdown

Table 4 reports the operational detections across categories for the official model at default ($t=0.50$) and operational ($t=0.20$) thresholds.

```
Table 4: Per-Anomaly Category Operational Detections (Tri-Stream ST-GNN).
Artifact source: research/metrics/per_class_breakdown.json
```

| Anomaly Category | Ground Truth ($N$) | Detections ($t=0.50$) | Recall ($t=0.50$) | Detections ($t=0.20$) | Recall ($t=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`altitude_anomaly`** | 660 | 660 | **$100.0\%$** | 660 | **$100.0\%$** |
| **`identity_inconsistency`** | 150 | 0 | $0.0\%$ | 150 | **$100.0\%$** |
| **`ghost_aircraft`** | 3,600 | 76 | $2.11\%$ | 92 | **$2.56\%$** |
| **`position_jump`** | 596 | 0 | $0.0\%$ | 1 | **$0.17\%$** |
| **Total Test Anomalies** | **5,006** | **736** | **$14.70\%$** | **903** | **$18.04\%$** |
| **Normal Traffic (False Alarms)** | 27,544 | 98 | $0.36\%$ (FPR) | 333 | $1.21\%$ (FPR) |

### 7.4 Detailed Empirical Insights & Operational Nuances

1. **Resolution of Transponder Cloning (`identity_inconsistency`)**:
   - In the standalone spatial encoder, transponder cloning yielded **$0.0\%$ recall** and poor ROC-AUC ($0.3300$). Because an instantaneous snapshot depicts two aircraft separated in space, the spatial GNN treats both aircraft as nominal neighbors.
   - The kinematic stream achieves an isolated ROC-AUC of **$0.9994$**, enabling **$100.0\%$ operational detection ($150/150$)** in the fused model.
   - *Deterministic Feature Attribution*: This detection is driven primarily by the deterministic `icao_collision` indicator rather than learned graph representations. When multiple broadcasts claim the same 24-bit Mode-S address in the same second, the indicator flags the conflict directly.
2. **Analysis of Position-Jump Predictions (ROC-AUC vs. Operational Recall)**:
   - For `position_jump`, the fused model achieves a high one-vs-normal ROC-AUC of **$0.9654$**, yet detects only **1 out of 596 observations ($0.17\%$ recall)** at threshold $t=0.20$.
   - *Statistical Explanation*: Inspection of prediction distributions shows that position-jump anomaly scores are consistently elevated across the attack trajectory: scores range from $0.1757$ to $0.2476$ with a mean of **$0.1908$** (median $0.1906$). Compared to normal traffic (median score **$0.0090$**, 75th percentile $0.0975$), the model clearly ranks position-jump observations higher than nominal flights, explaining the high ROC-AUC ($0.9654$). However, because the global operational threshold was calibrated at $t^*=0.20$, the bulk of position-jump scores sit narrowly just below the boundary ($0.191 < 0.20$), resulting in only the single peak transition observation ($0.2476$) crossing the operational decision threshold.
3. **The Ghost Aircraft Cruise Mimicry Challenge**:
   - `ghost_aircraft` represents $71.91\%$ of test anomalies ($3,600/5,006$).
   - The operational recall on ghost aircraft is only **$2.56\%$ ($92/3,600$)** at $t=0.20$.
   - *Physical Explanation*: The synthetic ghost flight (`e00002`) was generated as a physically consistent flight path flying parallel to commercial routes. Because its speeds, climb rates, and turning dynamics adhere strictly to civil aviation flight envelopes, single-receiver kinematic models cannot differentiate it from legitimate flights without multi-station multi-lateration or cryptographic credentials.
4. **Temporal Stream Inversion ($0.3333$ ROC-AUC)**:
   - The standalone GRU stream yields an overall ROC-AUC of $0.3333$.
   - In the test dataset, the synthetic ghost aircraft trajectory exhibits mathematical smoothness without the minor turbulence perturbations present in baseline flights. The GRU ranks these smooth synthetic trajectories as *more regular* than nominal flights. Fusing spatial edge attention ($0.8422$ ROC) and kinematic residuals compensates for this temporal inversion.

---

## 8. Multi-Stream Explainable AI (XAI) Framework

In civil Air Traffic Control, black-box predictions cannot be operationalized without interpretable rationale.

### 8.1 Modular Attribution Architecture

AeroGuard AI incorporates a multi-quadrant Explainable AI (XAI) engine (`scripts/aeroguard_explainer.py`):
1. **Spatial Attribution ($\text{GNNExplainer}$)**: Solves a mutual information optimization problem over edge and node masks to identify the most influential neighboring flights and node features within the $100\,\text{km}$ interaction radius [13].
2. **Temporal Saliency (Input $\times$ Gradient)**: Computes the gradient of the temporal logit with respect to the 5-step sequence $\mathbf{z}(\tau)$, isolating the second of maximum kinematic divergence.
3. **Kinematic Sensitivity**: Evaluates first-order partial derivatives across the 11 physical motion residuals, identifying specific aerodynamic violations.

```
[FIGURE PLACEHOLDER: fig_xai_attribution_quadrant.png]
Label: Figure 1 (Placeholder) — Multi-Quadrant Explainable AI Attribution Dossier for Injected Altitude Override.
Description: Quadrant 1 depicts the local relational graph showing neighboring aircraft identified by GNNExplainer.
Quadrant 2 illustrates temporal gradient saliency across the 5-step sequence window.
Quadrant 3 displays kinematic motion residual attributions (alt_residual reporting 9,905 ft vertical mismatch).
Quadrant 4 shows rule-based validation warnings corroborating neural gradient attributions.
```

### 8.2 Quantitative Attribution Audit

In the audited true positive case for `altitude_anomaly` on flight `500006` (timestamp `2020-01-02 10:48:39`, model anomaly probability $0.9992$), artifact `research/metrics/xai_scientific_audit.json` records:
- **Spatial Stream**: `GNNExplainer` identified `altitude` as the top node feature with an attribution importance of **$0.167$**.
- **Temporal Stream**: Input-gradient saliency concentrated **$88.4\%$** of temporal attribution on the current second $t$, isolating the injection point.
- **Kinematic Stream**: Top residual was `alt_residual` with a raw vertical disparity of **$9,905.0\,\text{ft}$**, triggering the rule warning `ALTITUDE_DISPARITY: 9905 ft vertical rate mismatch`.

### 8.3 Scientific Boundary: Attribution vs. Causal Proof

A vital scientific distinction must be maintained between **model attributions** and **causal proof**:

$$\text{Model Prediction} \longrightarrow \text{Gradient Saliency / Attributions} \neq \text{Physical Causal Proof}$$

- **Model Attribution**: Explains *why the neural network generated its prediction* based on mathematical gradients and feature weights. For example, high attribution on `alt_residual` proves that the model flagged flight `500006` because its altitude report climbed $9,905\,\text{ft}$ in 1 second.
- **Physical Causal Proof**: Demonstrates the *underlying real-world physical reality*. A mathematical attribution cannot definitively prove whether an anomaly was caused by an adversarial SDR spoofer, a malfunctioning pitot-static tube, a barometric sensor freeze, or an unannounced emergency maneuver. Definitive physical verification requires independent multi-sensor cross-validation (e.g., primary radar or TDoA multi-lateration).

---

## 9. Discussion & Physical Deployment Boundaries

### 9.1 The Operational Boundary of Single-Station ADS-B Defense

Our empirical findings establish clear operational boundaries for single-receiver message-level surveillance:
1. **Kinematic Discontinuities are Detectable**: Cyber-attacks that violate physical motion laws (altitude steps, vertical rate mismatches, impossible displacement ratios) are reliably detected ($100.0\%$ recall on altitude overrides).
2. **Protocol Collisions are Readily Flagged**: Duplicate transponder addresses can be detected with near-zero false alarms using deterministic collision indicators ($100.0\%$ recall).
3. **Aerodynamically Compliant Ghost Flights Bypass Message-Level Filters**: When an attacker injects a trajectory that adheres to realistic aircraft dynamics and nominal cruising corridors, single-receiver kinematic analysis cannot reliably differentiate forged messages from valid traffic ($2.56\%$ operational recall). Detecting such attacks requires multi-lateration TDoA infrastructure or physical-layer RF fingerprinting.

### 9.2 The High Specificity vs. Low Recall Trade-Off in Aviation

In air traffic management, false alarms impose severe operational costs: spurious alerts degrade controller trust and induce unnecessary go-arounds. The official AeroGuard AI model maintains a high specificity of **$98.79\%$** (FPR of $1.21\%$). However, high specificity cannot be viewed as a standalone indicator of operational readiness when overall recall is **$18.04\%$**. Future work must bridge this gap by incorporating multi-station spatial consensus.

---

## 10. Reproducibility & Artifact Availability

To enable independent verification, all artifacts, scripts, and checkpoints are structured for end-to-end reproducibility:

```
Table 5: Repository Artifact Provenance and Verification Directory.
```

| Component | Path in Repository | Git Status | Description |
| :--- | :--- | :--- | :--- |
| **Official Model Checkpoint** | `models/st_gnn_official.pt` | Tracked in Git | Pre-trained weighted Tri-Stream ST-GNN weights. |
| **Calibrated Threshold** | `models/st_gnn_calibrated_threshold.json` | Tracked in Git | Validation-derived operational decision boundary ($t^* = 0.20$). |
| **Data Generation Script** | `scripts/generate_aeroguard_dataset.py` | Tracked in Git | Deterministic synthetic dataset generator (`SEED = 42`). |
| **Benchmark Comparison** | `research/metrics/benchmark_comparison.json` | Tracked in Git | Aligned metrics for all evaluated architectures. |
| **Multi-Stream Ablation** | `research/metrics/ablation_results.json` | Tracked in Git | Stream-by-stream isolated metrics and ROC-AUCs. |
| **Per-Class Breakdown** | `research/metrics/per_class_breakdown.json` | Tracked in Git | Detections across all 4 canonical anomaly classes. |
| **Evaluation Suite** | `research/evaluate_benchmarks.py` | Tracked in Git | Observation-level composite key evaluation harness. |
| **Ablation Protocol** | `research/ablation_protocol.py` | Tracked in Git | Multi-stream ablation evaluation script. |
| **Test Dataset (OOD)** | `data/aero_guard_test.csv` | Git-Ignored | Regenerated locally via `generate_aeroguard_dataset.py`. |

> **Fresh-Clone Data Regeneration Protocol**: In adherence to repository hygiene, large CSV datasets (`data/*.csv`) are excluded from Git tracking via `.gitignore`. A newly cloned repository contains the complete dataset generation code, pre-trained model weights, and evaluation harness. To reproduce the evaluation pipeline from a fresh clone:
> ```bash
> # 1. Regenerate synthetic training and OOD test datasets (SEED=42)
> python scripts/generate_aeroguard_dataset.py
>
> # 2. Run ST-GNN inference audit to produce prediction CSVs
> python scripts/audit_st_gnn_metrics.py
>
> # 3. Execute observation-aligned benchmark and ablation evaluations
> python research/evaluate_benchmarks.py
> python research/ablation_protocol.py
> ```
> All unit tests pass cleanly:
> ```bash
> python -m unittest discover -s tests -p "test_*.py"
> ```

---

## 11. Conclusion & Future Directions

In this work, we evaluated AeroGuard AI, a cyber-physical surveillance framework utilizing a Tri-Stream Spatio-Temporal Graph Neural Network to detect ADS-B trajectory anomalies. By modeling the airspace as a dynamic graph conditioned on 4D relative edge attributes and combining spatial attention, causal temporal recurrence, and kinematic residuals, the architecture demonstrates transferability across a simulated $1,500\,\text{km}$ geographic domain shift.

Evaluated on an independent synthetic test set of $32,550$ observations, the official model achieved $86.37\%$ accuracy, $73.06\%$ precision, and an F1-score of $0.2893$ at a false alarm rate of $1.21\%$. The framework reliably identified barometric altitude overrides ($100.0\%$) and transponder identity collisions ($100.0\%$). However, our evaluation highlighted fundamental limits of single-station message-level surveillance: aerodynamically compliant ghost aircraft ($2.56\%$ recall) and sustained position jumps ($0.17\%$ recall at $t=0.20$) evade single-receiver kinematic filtering. Future research will explore multi-station distributed graph architectures that ingest raw I/Q samples to correlate message-level kinematics with physical Time Difference of Arrival (TDoA) measurements.

---

## References

1. International Civil Aviation Organization (ICAO), "Aeronautical Telecommunications, Annex 10 to the Convention on International Civil Aviation, Volume IV: Surveillance and Collision Avoidance Systems," 5th ed., ICAO, Montreal, Canada, 2014.
2. Federal Aviation Administration (FAA), "Automatic Dependent Surveillance-Broadcast (ADS-B) Out Performance Requirements to Support Air Traffic Control (ATC) Service," *Federal Register*, Final Rule, 14 CFR Part 91, Docket No. FAA-2007-29305, May 2010.
3. RTCA, "Minimum Operational Performance Standards for $1090\,\text{MHz}$ Extended Squitter Automatic Dependent Surveillance - Broadcast (ADS-B) and Traffic Information Services - Broadcast (TIS-B)," RTCA DO-260B, Dec. 2009.
4. M. Strohmeier, M. Schäfer, V. Lenders, and I. Martinovic, "Real-World ADS-B Attacks," *IEEE Security & Privacy*, vol. 15, no. 5, pp. 16–22, Sep.–Oct. 2017.
5. A. Costin and A. Francillon, "Ghost in the Air(Traffic): On Insecurity of ADS-B Protocol and Practical Attacks on ADS-B Devices," in *Proc. Black Hat USA*, Las Vegas, NV, 2012.
6. M. Schäfer, M. Strohmeier, V. Lenders, and I. Martinovic, "Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research," in *Proc. 13th ACM/IEEE Int. Conf. Inf. Process. Sens. Netw. (IPSN)*, Berlin, Germany, 2014, pp. 83–94.
7. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, Oct. 2001.
8. E. Habler and A. Shabtai, "Using LSTM encoder-decoder algorithm for detecting anomalous ADS-B messages," *Computers & Security*, vol. 78, pp. 155–173, Sep. 2018.
9. K. Cho, B. van Merriënboer, Ç. Gülçehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Doha, Qatar, 2014, pp. 1724–1734.
10. K. Cai, Y. Li, Y.-P. Fang, and Y. Zhu, "A Deep Learning Approach for Flight Delay Prediction Through Time-Evolving Graphs," *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 8, pp. 11397–11407, Aug. 2022.
11. T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," in *Proc. 5th Int. Conf. Learn. Represent. (ICLR)*, Toulon, France, 2017.
12. S. Brody, U. Alon, and E. Yahav, "How Attentive are Graph Attention Networks?," in *Proc. 10th Int. Conf. Learn. Represent. (ICLR)*, Virtual, 2022.
13. Z. Ying, D. Bourgeois, J. You, M. Zitnik, and J. Leskovec, "GNNExplainer: Generating Explanations for Graph Neural Networks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, Vancouver, Canada, 2019, pp. 9240–9251.
