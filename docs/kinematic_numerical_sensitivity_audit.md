# AeroGuard AI — Kinematic Feature Numerical Sensitivity Audit & Future Retraining Plan

**Author**: AeroGuard AI Maintenance Team
**Date**: October 2026
**Scope**: Scientific investigation of numerical stability in kinematic features `accel_residual` and `turn_residual` in `models/scaler_kinematic.joblib`.

---

## 1. Executive Summary & Root Cause Analysis

During fresh-clone replication and cross-platform validation audits, a numerical sensitivity was identified in the kinematic residual stream:
- The kinematic scaler (`models/scaler_kinematic.joblib`) contains near-zero scale parameters for two discrete residual features:
  - `accel_residual`: $\sigma = 6.8941 \times 10^{-15}$
  - `turn_residual`: $\sigma = 8.5545 \times 10^{-15}$
- All other 9 kinematic features have standard deviations well within typical engineering scales ($\sigma \in [0.1, 8000]$).

### Mathematical Mechanism:
1. In `scripts/generate_aeroguard_dataset.py`, instantaneous reported acceleration was computed as:
   $$\text{acceleration}[i] = \frac{\Delta v_{\text{groundspeed}}}{\Delta t}$$
2. In `scripts/train_st_gnn.py`, the expected acceleration consistency residual was computed as:
   $$\text{expected\_accel} = \frac{\Delta v_{\text{groundspeed}}}{\Delta t}$$
   $$\text{accel\_residual} = |\text{acceleration} - \text{expected\_accel}|$$
3. Because both formulas calculate discrete finite differences from the identical trajectory column, their mathematical difference on smooth synthetic trajectories is identically zero:
   $$\text{accel\_residual}_{\text{exact}} \equiv 0.0$$
4. In floating-point arithmetic ($64$-bit IEEE 754), slight operator ordering differences between vector operations produced values on the order of machine precision:
   $$\text{accel\_residual} \approx 0.0 \pm 10^{-16}$$
5. Standard scaling (`StandardScaler`) was fitted across the training partition without an explicit variance lower bound. Consequently, it calculated:
   $$\sigma_{\text{scaler}} = \text{std}(\text{accel\_residual}) \approx 6.8941 \times 10^{-15}$$
6. During downstream inference, dividing by $\sigma \approx 7 \times 10^{-15}$ yields a noise amplification factor of:
   $$\kappa = \frac{1}{\sigma} \approx 1.45 \times 10^{14}$$
   A minute roundoff difference of $1.4 \times 10^{-14}$ (common across different BLAS implementations, CPU microarchitectures, or 32-bit float casts) is magnified into a **$2.0$ to $3.5$ standard deviation** shift in the scaled feature vector.

---

## 2. Empirical Impact on Anomaly Detection

- The isolated Kinematic-Only MLP stream (`KinematicStream`) experiences variations in ROC-AUC between **$0.5193$** and **$0.5468$** on test evaluation depending on hardware floating-point roundoff.
- In the full fused Tri-Stream ST-GNN (`AeroGuardSTGNN`), this variation is damped by weighted probability fusion ($w_{\text{spatial}} = 0.50$, $w_{\text{temporal}} = 0.35$, $w_{\text{kinematic}} = 0.15$), maintaining the overall test ROC-AUC at $0.6235$.
- Model outputs remain strictly finite, with no overflow, `NaN`, or `Inf` generated under any tested conditions.

---

## 3. Why Clamping Without Retraining is Prohibited in v1.0.0

A candidate post-processing fix is clamping the scaler scale factor:
$$\sigma_{\text{clamped}} = \max(\sigma, 1.0)$$

However, testing confirms:
1. The official model weights in `models/st_gnn_official.pt` were trained directly against the unregularized, scaled feature representations.
2. Clamping the scale to $1.0$ without retraining collapses the Kinematic-Only ROC to **$0.4985$** (below random chance), altering the model's prediction distribution and invalidating the published, peer-reviewed benchmark metrics.
3. In accordance with strict reproducibility and academic integrity rules, **official model checkpoints, scalers, and recorded metrics are frozen and preserved for v1.0.0**.

---

## 4. Formal Resolution & Retraining Plan for v1.1.0

For the next planned release cycle (v1.1.0):

1. **Synthetic Noise Injection**:
   In `generate_aeroguard_dataset.py`, add realistic sensor quantization noise ($\pm 0.5\,\text{kt}$, $\pm 0.1^\circ$) to groundspeed and heading reports so that finite-difference residuals reflect realistic measurement variance ($\sigma \sim 0.5 - 2.0\,\text{m/s}^2$).
2. **Scaler Variance Floor**:
   In `scripts/train_st_gnn.py`, enforce an explicit variance floor during scaler fitting:
   ```python
   scaler_kinematic.scale_ = np.maximum(scaler_kinematic.scale_, 1.0)
   ```
3. **Retraining & Checkpoint Publishing**:
   Retrain `KinematicStream` and `AeroGuardSTGNN` with the regularized scaler, audit all benchmark metrics, and publish an updated checkpoint `models/st_gnn_v1.1.0.pt` alongside updated comparative tables.
