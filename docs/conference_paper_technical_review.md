# AeroGuard AI: Technical Implementation Review for Scholarly Conference Submission

This document outlines the design, implementation, and evaluation of **AeroGuard AI**, structured to support the drafting of a peer-reviewed conference paper. The core contribution of this work is a **multi-layer comparison framework for ADS-B spoofing detection**, benchmarking isolated tabular classification against relational Graph Neural Networks (GNNs) operating on dynamic airspace topologies.

---

## 1. Introduction & Threat Model

Automatic Dependent Surveillance–Broadcast (ADS-B) is the digital surveillance standard for global Air Traffic Control (ATC). Operating on an unauthenticated line-of-sight frequency of **1090 MHz**, ADS-B packets are transmitted without encryption or source verification. This lack of cryptographic authentication enables attackers using Software Defined Radios (SDRs) to broadcast fabricated messages.

AeroGuard AI models and detects the following threat vectors:
1. **Kinematic Attacks**:
   - *Position Jump*: Discontinuous shifts in geographic coordinates.
   - *Altitude / Velocity Override*: Step-function modifications or gradual drifts injected into reported altitude and groundspeed fields.
2. **Advanced Protocol Attacks**:
   - *Identity Inconsistency (Cloning)*: Replaying valid transponder packets to create duplicate flights using the same ICAO24 hex address.
   - *Doppler-Consistent Ghost Injections*: Injecting synthesized aircraft trajectories where physical RF propagation parameters (carrier shifts, power levels) are dynamically modeled to match flight kinematics, simulating a sophisticated attacker.

---

## 2. Software Defined Communication (SDC) & Signal Physics Modeling

To ground the project in the physical properties of Software Defined Communication (SDC), the dataset generation pipeline simulates a ground station receiver located at the Lisbon Airport Portela FIR center ($38.7756^\circ \text{ N}$, $-9.1354^\circ \text{ W}$, altitude $114.0\text{ m}$).

The signal physics engine computes the following properties for every observation:

### A. Geodetic to ECEF Coordinate Mapping
To calculate exact 3D distances over the WGS84 ellipsoid, reported geodetic coordinates $(\phi, \lambda, h)$ are projected into Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates $(X, Y, Z)$ using:
$$X = (N(\phi) + h) \cos\phi \cos\lambda$$
$$Y = (N(\phi) + h) \cos\phi \sin\lambda$$
$$Z = \left(N(\phi)(1 - e^2) + h\right) \sin\phi$$
where $N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$ is the prime vertical radius of curvature, $a$ is the semi-major axis ($6378137.0\text{ m}$), and $e^2$ is the eccentricity squared ($0.00669437999014$).

### B. Slant-Range Vector Calculation
The 3D range vector $\vec{R}$ is the difference between the aircraft ECEF position $\vec{P}_{ac}$ and the Lisbon receiver ECEF position $\vec{P}_{rx}$:
$$\vec{R} = \vec{P}_{ac} - \vec{P}_{rx}$$
$$d = \|\vec{R}\|_2$$

### C. 3D ECEF Velocity Rotation
Reported flight parameters (groundspeed $v_g$, heading $\theta$, vertical rate $v_z$) are first converted to local East-North-Up (ENU) coordinates:
$$v_E = v_g \sin\theta, \quad v_N = v_g \cos\theta, \quad v_U = v_z$$
The ENU velocity vector $\vec{v}_{enu}$ is then rotated to the ECEF frame $\vec{v}_{ecef}$ via:
$$\vec{v}_{ecef} = R^T \vec{v}_{enu}$$
where $R$ is the rotation matrix derived from geodetic latitude $\phi$ and longitude $\lambda$.

### D. Doppler Shift Estimation
The radial velocity $v_r$ of the aircraft relative to the receiver is computed as the projection of the velocity vector onto the unit range vector:
$$v_r = \vec{v}_{ecef} \cdot \frac{\vec{R}}{\|\vec{R}\|}$$
Using the carrier frequency $f_c = 1090\text{ MHz}$ and the speed of light $c \approx 3 \times 10^8\text{ m/s}$, the estimated Doppler shift $f_D$ is:
$$f_D = - \frac{v_r}{c} f_c$$

### E. Signal Power Propagation (FSPL, RSS, and SNR)
- **Free Space Path Loss (FSPL)**:
  $$\text{FSPL }(\text{dB}) = 20 \log_{10}(d) + 20 \log_{10}(f_c) - 147.56$$
- **Received Signal Strength (RSS)**: Assuming an omnidirectional transmitter power $P_{tx} = 250\text{ W}$ ($54\text{ dBm}$) and $0\text{ dBi}$ antenna gains:
  $$\text{RSS }(\text{dBm}) = P_{tx} - \text{FSPL}$$
- **Signal-to-Noise Ratio (SNR)**: Calculated using a typical SDR receiver noise floor ($N_0 = -107\text{ dBm}$):
  $$\text{SNR }(\text{dB}) = \text{RSS} - N_0$$

---

## 3. Kinematic Derivative & Temporal Feature Engineering

To capture short-term flight dynamics and identify kinematic anomalies, the pipeline computes numerical derivatives across successive time steps ($\Delta t$):
- **Acceleration**: Speed rate of change: $a = \frac{\Delta v_g}{\Delta t}$ ($m/s^2$).
- **Turn Rate**: Heading rate of change: $\omega = \frac{\Delta \theta}{\Delta t}$ ($deg/s$).
- **Vertical Acceleration**: Climbing rate change: $a_z = \frac{\Delta v_z}{\Delta t}$ ($m/s^2$).
- **Local Airspace Density**:
  - `neighbor_count`: Number of active flights within $100\text{ km}$.
  - `local_aircraft_density`: Number of active flights within $50\text{ km}$.

---

## 4. Dynamic Spatial Graph Construction

To transition from single-aircraft sequence modeling to cooperative spatial analysis, the airspace is modeled as a dynamic, time-evolving graph:
$$G(t) = (V(t), E(t))$$

### A. Graph Snapshots
Observation sequences are sliced into discrete **1-second temporal snapshots** at $1\text{ Hz}$ resolution. Flights not transmitting within the snapshot window are pruned, and new entries are added dynamically.

### B. Node Definitions
Each node in $V(t)$ represents a single active flight. The node feature vector $\mathbf{x}_i \in \mathbb{R}^{13}$ contains:
- Standard reported kinematics (altitude, groundspeed, heading, vertical rate).
- Derived kinematics (acceleration, vertical acceleration, turn rate, climb rate change).
- Communication signal characteristics (Doppler shift, RSS, SNR, range to receiver).
*Note: Absolute flight identifiers (ICAO24, callsign) and absolute coordinates are omitted from GNN node features to prevent overfitting to specific flight paths.*

### C. Proximity-Based Edge Formulation
An undirected edge $e_{ij} \in E(t)$ is established between aircraft $i$ and $j$ if their horizontal separation $d_h$ is within $100\text{ km}$:
$$E(t) = \{e_{ij} \mid d_h(i, j) \le 100\text{ km}\}$$

### D. Relational Edge Features
Each edge is assigned a 4-dimensional relative feature vector $\mathbf{e}_{ij}$ to capture spatial and kinematic inconsistencies between neighboring flights:
1. **Range**: 3D Euclidean distance between aircraft.
2. **Relative Altitude**: $\Delta h_{ij} = h_i - h_j$.
3. **Relative Velocity**: Difference in groundspeed magnitudes.
4. **Relative Heading**: Angular separation normalized to $[-180^\circ, 180^\circ]$.

---

## 5. AI Modeling & PyTorch Geometric GCN Classifier

We compared two machine learning approaches: a tabular baseline and a Graph Neural Network.

### A. Feature Scaling & Data Partitioning
Features are scaled using `StandardScaler` instances fitted strictly on the training partition and applied to validation and test sets.
- **Training Partition**: Jan 1, 2020 (first 80% chronological split, $t = 0\text{ s} - 2880\text{ s}$, 10:00:00–10:47:59): **$77,910$ records** across 27 unique aircraft.
- **Validation Partition**: Jan 1, 2020 (last 20% chronological split, $t = 2881\text{ s} - 3600\text{ s}$, 10:48:00–10:59:59): **$19,440$ records** across 27 unique aircraft. Used strictly for threshold calibration ($t^* = 0.20$) and fusion tuning without test set exposure.
- **Out-of-Distribution (OOD) Testing Partition**: Jan 2, 2020 (10:00:00–10:59:59): **$32,550$ records** ($27,544$ normal flights, $5,006$ anomalies) across 8 commercial aircraft and 1 injected phantom identity (`e00002`).
*The test partition is isolated chronologically and geographically ($\approx 1,500\text{ km}$ domain shift in a northern European FIR corridor) to evaluate generalization to unseen airspace sectors.*

### B. Baseline Tabular Classifier (Random Forest)
A classical Random Forest classifier is trained on isolated node-level features, establishing a baseline that ignores relational dependencies between aircraft.

### C. Graph Convolutional Network (GCN) Architecture
Built using **PyTorch Geometric**, the AeroGuard GNN consists of two Graph Convolution (GCN) layers followed by a linear classification layer:
1. **Layer 1 (GCN Conv)**: Takes the 13-dimensional node features and maps them to 32 hidden dimensions, performing neighborhood feature aggregation:
   $$\mathbf{h}_i^{(1)} = \text{ReLU}\left(\mathbf{W}^{(1)} \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} \mathbf{x}_j\right)$$
   where $\tilde{d}_i$ is the node degree with added self-loops.
2. **Layer 2 (GCN Conv)**: Maps the 32-dimensional representations to 16 hidden dimensions.
3. **Layer 3 (Linear Readout)**: A fully connected layer mapping the 16-dimensional node embeddings to 2 output classes (Normal vs. Anomalous).

### D. Weight-Balanced Loss Function
To address class imbalance (~16.5% anomaly ratio), the model is optimized using Weighted Cross-Entropy Loss (weighting the anomalous class by a factor of 10.0):
$$\mathcal{L} = - \sum_{i \in V} \left[ w_0 y_i \log(\hat{y}_{i,0}) + w_1 (1 - y_i) \log(\hat{y}_{i,1}) \right]$$

---

## 6. Exploratory Data Analysis (EDA) & Validation

To verify the physics simulator, statistical checks were run on the generated datasets:

### A. Doppler Shift vs. Velocity Consistency
We verified the linear relationship between Doppler shift and radial velocity. Radial velocities toward the receiver yield positive Doppler shifts, while receding targets show negative shifts.
- **Radial Velocity to Doppler Correlation**: **-1.0000** (matching physical theory).

### B. Path Loss Decay Verification
We checked Received Signal Strength (RSS) against target distance. RSS decays logarithmically as distance increases, validating the free-space propagation model.
- **Distance to RSS Correlation**: **-0.9611** (verifying logarithmic signal decay).

### C. Airspace Graph Density Statistics
Analysis of the 100 consecutive snapshots (12:40:41 to 12:42:20) shows:
- **Active Nodes per Snapshot**: Mean of **6.20** (Min 6, Max 7).
- **Edges per Snapshot**: Mean of **0.90** (Min 0, Max 1).
- **Anomalous Connectivity**: **90.00%** of anomalous nodes have active neighboring nodes, confirming that message passing is highly feasible for threat propagation in the GNN.

---

## 7. Benchmark Results & Domain Generalization Performance

Evaluating all architectures strictly on the identical out-of-distribution (OOD) test set (Jan 2, 2020, $32,550$ observations located $\approx 1,500\text{ km}$ away from training flights) yielded the following audited results (`research/metrics/benchmark_comparison.json`):

| Model Architecture | Input Features & Structure | Decision Threshold ($t$) | Precision | Recall | F1-Score | Accuracy | Specificity | FPR | ROC-AUC | PR-AUC |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Tabular Baseline (RF)** | Node-level 13D (with coordinates) | Default ($0.50$) | $0.5742$ | $0.0757$ | $0.1338$ | $0.8492$ | $0.9898$ | $0.0102$ | $0.5528$ | $0.2029$ |
| **Tabular Baseline (RF)** | Node-level 13D (with coordinates) | Calibrated ($0.15$) | $0.4263$ | $0.0815$ | $0.1368$ | $0.8419$ | $0.9801$ | $0.0199$ | $0.5528$ | $0.2029$ |
| **Original GCN Baseline** | Dynamic Graph (13D Node, unweighted edges) | Default ($0.50$) / Calibrated ($0.12$) | $0.0000$ | $0.0000$ | $0.0000$ | $0.8462$ | $1.0000$ | $0.0000$ | $0.5808$ | $0.2671$ |
| **Historical Edge-Aware GNN** | GATv2 (12D Node + 4D Edge, double-scaled) | Default ($0.50$) / Calibrated ($0.99$) | $0.2070$ | $0.1306$ | $0.1602$ | $0.7893$ | $0.9090$ | $0.0910$ | $0.4578$ | $0.1884$ |
| **Repaired Edge-Aware GNN (Spatial)** | Dynamic Graph (12D Coord-free + 4D Relational Edge) | Default ($0.50$) | $0.8772$ | $0.1484$ | $0.2539$ | $0.8658$ | $0.9962$ | $0.0038$ | **$0.7635$** | **$0.4347$** |
| **Tri-Stream ST-GNN (Official Ours)** | **Spatial GATv2 + Temporal GRU + Kinematic MLP** | **Calibrated ($0.20$)** | **$0.7306$** | **$0.1804$** | **$0.2893$** | **$0.8637$** | **$0.9879$** | **$0.0121$** | $0.6235$ | $0.3720$ |

### Key Analysis for Reviewers
1. **Majority Class Collapse of Unweighted GCN**: Under standard decision thresholds, the unweighted GCN baseline predicts 0 anomalies ($0.00$ F1). Ignoring continuous edge metrics causes neighborhood representations to over-smooth across the dominant normal traffic class.
2. **Relational Generalization via 4D Edge Attributes**: By eliminating absolute coordinates and conditioning attention on 4D relational edge attributes (range, relative altitude, velocity difference, heading separation), the Repaired Spatial GATv2 achieves **ROC-AUC $0.7635$** and **PR-AUC $0.4347$**, generalizing effectively across distant airspace sectors.
3. **Tri-Stream Operational Performance & False Alarm Control**: Operating the weighted Tri-Stream ST-GNN at the calibrated operational threshold $t^* = 0.20$ achieves **$73.06\%$ precision** and **$0.2893$ F1** with an operational false alarm rate of just **$1.21\%$** (specificity $98.79\%$).
4. **Per-Attack Operational Nuances & Empirical Boundaries**:
   - *Altitude Overrides*: $100.0\%$ detection ($660 / 660$) due to joint spatial-temporal and kinematic residual detection.
   - *Transponder Cloning (`identity_inconsistency`)*: $100.0\%$ detection ($150 / 150$), driven by the kinematic stream's causal duplicate-broadcast detector (`icao_collision`).
   - *Ghost Aircraft Cruise Mimicry*: Low operational recall ($2.56\%$, $92 / 3,600$) under $t=0.20$ because kinematically smooth synthetic flight profiles mimic nominal cruising aerodynamics without physical step discontinuities.
   - *Position Jumps*: Low operational recall ($0.17\%$, $1 / 596$) at $t=0.20$ despite high kinematic ranking ROC-AUC ($0.9697$), because only the initial 1-second teleportation step triggers kinematic residuals, while subsequent observations reflect nominal cruising flight.
