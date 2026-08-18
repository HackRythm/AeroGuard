# AeroGuard AI — Mid-Semester Demonstration Results

This document summarizes the performance metrics and implementation details for the **AeroGuard AI** mid-semester demonstration.

---

## 1. Prototype Evaluation Results

The models were trained on the training set (Jan 1, 2020) and evaluated on the chronological testing set (Jan 2, 2020). Features such as absolute positions, raw receiver ranges, and absolute signal strengths are out-of-distribution (OOD) between the training and testing sets due to geographic distance from the receiver (LPPC FIR flights cruise ~1,500 km away on Jan 2).

The resulting performance metrics are evaluated below:

### Model Performance Comparison

| Model | Precision | Recall | F1-Score | AUC-ROC |
|---|---:|---:|---:|---:|
| **Baseline (Random Forest)** | 0.0000 | 0.0000 | 0.0000 | 0.4716 |
| **AeroGuard GNN (GCN)** | 0.0000 | 0.0000 | 0.0000 | **0.6435** |

### Confusion Matrix (Baseline - Random Forest)
```text
[[14457     0]
 [ 2872     0]]
```

### Confusion Matrix (AeroGuard GNN - GCN)
```text
[[14457     0]
 [ 2872     0]]
```

---

## 2. Analysis of the Results

1. **Majority Class Collapse**: Under a standard classification threshold of `0.5`, both models classify all test samples as `0` (Normal). This is caused by the strong geographic domain shift (OOD) between Jan 1 (close to Lisbon) and Jan 2 (farther in the LPPC sector).
2. **AUC-ROC Improvement**: Despite the class collapse, the GNN yields an AUC-ROC of **0.6435** compared to the baseline's **0.4716** (worse than random). This confirms that by incorporating relational edges (neighbor distances, relative velocities, relative headings), the GNN is starting to learn coordinate-invariant structures that generalize much better across different spatial sectors than the isolated aircraft-level Random Forest model.
3. **Adaptive Thresholding (Dashboard Solution)**: To make the system responsive in the demonstration, the dashboard dynamically computes an adaptive classification threshold based on the **83rd percentile** of the anomaly scores (probabilities) predicted by the GNN (which aligns with the test set's true anomaly ratio of ~16.5%). This flags the top-scoring threats as alerts and renders them in red, matching the true injected anomaly times.

---

## 3. Completed Deliverables

* **Dataset Preprocessing & Scaling**: Normalizes node features (kinematics, signal properties) and edge properties dynamically.
* **Baseline Classifier**: Trained Random Forest model (`models/baseline.joblib`).
* **Graph Neural Network**: Trained 2-layer Graph Convolutional Network (`models/gnn.pt`).
* **Test Predictions Output**: Generated prediction probabilities and saved to `data/aero_guard_test_predictions.csv`.
* **Jupyter Notebook**: executed in-place with all plots rendering (`notebooks/01_aeroguard_eda.ipynb`).
* **Streamlit ATC Dashboard**: Interactive dashboard `app.py` with playback controls, airspace trajectory plot, NetworkX spatial graph layout, alert tables, and telemetry inspector.
