# AeroGuard AI — Scientific Research Figures

This directory contains the publication-grade scientific figures and evaluation visualizations for **AeroGuard AI**, generated from genuine project artifacts and benchmark predictions.

---

## 1. Directory Inventory

| File | Description | Resolution / Dimensions | Primary Data Source |
|---|---|---|---|
| [`figure1_seven_class_anomaly_coverage.png`](./figure1_seven_class_anomaly_coverage.png) | Seven-class synthetic anomaly coverage (Training/Validation Dataset vs. Out-of-Distribution Test Dataset) distinguishing evaluated benchmark classes from training-only classes. | 4200 × 2400 px (300 DPI) | `data/aero_guard_train.csv`<br>`data/aero_guard_test.csv` |
| [`figure2_confusion_matrix_calibrated.png`](./figure2_confusion_matrix_calibrated.png) | High-contrast confusion matrix at official operational calibrated threshold ($t=0.20$), with cell counts and row/airspace proportions. | 2850 × 2880 px (300 DPI) | `data/st_gnn_test_predictions.csv`<br>`models/st_gnn_calibrated_threshold.json` |
| [`figure3_roc_and_pr_curves.png`](./figure3_roc_and_pr_curves.png) | Receiver Operating Characteristic (ROC-AUC = 0.6235) and Precision–Recall (PR-AUC = 0.3720) continuous ranking curves with operating points. | 4200 × 1950 px (300 DPI) | `data/st_gnn_test_predictions.csv` |
| [`figure4_st_gnn_architecture.png`](./figure4_st_gnn_architecture.png) | Tri-Stream Spatio-Temporal GNN architecture diagram with spatial GATv2, temporal GRU, kinematic MLP, and calibrated fusion ($0.40 / 0.35 / 0.25$). | 4500 × 2550 px (300 DPI) | `models/st_gnn_model.py`<br>`models/st_gnn_calibrated_threshold.json` |
| [`figure5_flight_trajectory_visualization.png`](./figure5_flight_trajectory_visualization.png) | Multi-aircraft airspace map and temporal anomaly signatures (position jump, altitude teleport, ghost flight, transponder collision). | 4500 × 2550 px (300 DPI) | `data/aero_guard_test.csv`<br>`data/st_gnn_test_predictions.csv` |

---

## 2. Reproduction Instructions

All figures can be reproduced directly using the consolidated generator script in `scripts/generate_scientific_figures.py`.

### Prerequisites
Activate the Python virtual environment containing `matplotlib`, `seaborn`, `scikit-learn`, `pandas`, and `numpy`:
```bash
# Windows
.\.venv\Scripts\python.exe scripts/generate_scientific_figures.py
```

### Selective Figure Generation
To regenerate specific figures individually:
```bash
# Figure 1: Seven-Class Anomaly Coverage
python scripts/generate_scientific_figures.py --fig1

# Figure 2: Calibrated Confusion Matrix (t=0.20)
python scripts/generate_scientific_figures.py --fig2

# Figure 3: ROC & PR Curves
python scripts/generate_scientific_figures.py --fig3

# Figure 4: ST-GNN Architecture Diagram
python scripts/generate_scientific_figures.py --fig4

# Figure 5: Flight Trajectory Visualization
python scripts/generate_scientific_figures.py --fig5

# All Figures (Default)
python scripts/generate_scientific_figures.py --all
```

---

## 3. Scientific Invariants & Figure Descriptions

### Figure 1 — Seven-Class Synthetic Anomaly Coverage
- **Scientific Rationale**: Top-level project documentation promised 7 anomaly types. This figure confirms that all 7 types were implemented in [`scripts/generate_aeroguard_dataset.py`](../../scripts/generate_aeroguard_dataset.py) and trained into the model weights (`data/aero_guard_train.csv`: 7,314 anomalies across all 7 classes).
- **Test Sector Constraint**: It visually distinguishes the **4 classes evaluated in the held-out OOD test benchmark** (`ghost_aircraft`: 3,600, `altitude_anomaly`: 660, `position_jump`: 596, `identity_inconsistency`: 150) from the **3 training-only classes** (`velocity_anomaly`: 890, `vertical_rate_anomaly`: 725, `heading_anomaly`: 572), making clear that their absence from the test set reflects the 8-aircraft test sector constraint (`num_anomaly_flights = 2`) rather than an unimplemented feature.

### Figure 2 — Calibrated Operational Confusion Matrix ($t = 0.20$)
- **Official Ground Truth Invariant**: Evaluated on exactly 32,550 out-of-distribution test records (27,544 Normal, 5,006 Anomaly).
- **Calibrated Operational Breakdown**:
  - **True Negatives (TN)**: 27,211 (98.79% of normal traffic passed)
  - **False Positives (FP)**: 333 (1.21% false alarm rate)
  - **False Negatives (FN)**: 4,103 (81.96% undetected threats, predominantly low-detectability smooth ghost flights)
  - **True Positives (TP)**: 903 (18.04% true threat intercept rate)
  - **Operational Metrics**: Precision = 73.06%, Recall = 18.04%, F1-Score = 0.2893, Accuracy = 86.37%.

### Figure 3 — ROC and Precision–Recall (PR) Curves
- **Dynamic Curves**: Plotted directly from continuous predicted probabilities in `data/st_gnn_test_predictions.csv`.
- **Ranking Performance**:
  - **ROC-AUC**: **0.6235** (exceeds random chance of 0.5000 under severe geographic domain shift).
  - **PR-AUC**: **0.3720** (2.42× higher than the 15.38% class prevalence baseline of 0.1538).
- **Marked Operating Points**: Compares the operational calibrated threshold ($t = 0.20$: Precision 73.06%, Recall 18.04%) against the default binary threshold ($t = 0.50$: Precision 88.25%, Recall 14.70%).

### Figure 4 — ST-GNN Architecture Diagram
- **Exact Implementation Match**: Diagrams the exact layers of `AeroGuardSTGNN` in [`models/st_gnn_model.py`](../../models/st_gnn_model.py):
  1. **Spatial Stream**: Repaired Edge-Aware GATv2Conv ($\text{dim}=64 \to 32$) with 4D relative edge attributes (weight = 0.40).
  2. **Temporal Stream**: Causal recurrent GRU ($\text{dim}=32$) with $T=5$ history sequence masking (weight = 0.35).
  3. **Kinematic Stream**: Physical invariant MLP ($11 \to 32 \to 16$) evaluating causal motion residuals (weight = 0.25).
  4. **Fusion & Output**: Weighted probability integration $p_{\text{fused}} = 0.40 \cdot p_{\text{sp}} + 0.35 \cdot p_{\text{temp}} + 0.25 \cdot p_{\text{kin}}$, feeding into the operational threshold comparator ($t^* = 0.20$) and multi-tier threat levels (NOMINAL, ELEVATED, CRITICAL).

### Figure 5 — Genuine Flight Trajectory Visualization
- **Non-Fabricated Data**: Rendered directly from authentic aircraft coordinates in `data/aero_guard_test.csv`.
- **Corridor Mapping**: Displays the Northern European commercial airspace corridor ($50.5^\circ\text{--}54.0^\circ\,\text{N}$, $-11.5^\circ\text{--}-6.5^\circ\,\text{W}$):
  - Normal background flights (`500001`, `500003`, `500004`, `500007`, `500008`).
  - Position jump teleportation offset on flight `500002`.
  - Parallel corridor insertion of phantom flight `e00002` (`GHOST02`).
  - Transponder address collision duplicate transmissions on `500002`.
  - Temporal altitude step teleportation override on flight `500006`.
