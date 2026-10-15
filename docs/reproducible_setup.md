# AeroGuard AI — Reproducible Setup & Workflow Guide

This guide describes how to set up the AeroGuard AI repository from a clean clone, install dependencies, prepare datasets, and run all validation, evaluation, and dashboard workflows.

---

## 1. Prerequisites

- **Python**: Version `3.10` or `3.11` (Tested on Python 3.11 with Windows and Linux).
- **Node.js**: Version `>= 18.0.0` with `npm` (Tested with Node.js v20+).
- **Git**: For version control.

---

## 2. Environment Setup

### A. Python Virtual Environment

```bash
# 1. Clone repository
git clone https://github.com/spsharvesh/AeroGuard.git
cd AeroGuard

# 2. Create isolated Python 3.11 virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 4. Install core dependencies
pip install -r requirements.txt
```

> [!NOTE]
> PyTorch CPU wheel is installed by default (`torch>=2.2.0`). For GPU CUDA acceleration, install the corresponding CUDA wheel from [pytorch.org](https://pytorch.org/) prior to running `requirements.txt`.

### B. Frontend Setup

The frontend is a React 19 application built with Vite and Tailwind CSS.

```bash
cd frontend

# Install exact pinned dependencies from lockfile
npm ci

cd ..
```

---

## 3. Required Models & Checkpoints

All production model checkpoints and preprocessing scalers are tracked directly in Git (`models/`, total size ~1.1 MB). No manual model downloads are required for standard evaluation or dashboard execution.

| Model / Checkpoint | File Size | Description |
| :--- | :---: | :--- |
| `models/st_gnn_official.pt` | 58.4 KB | Official trained Tri-Stream ST-GNN weights |
| `models/st_gnn_calibrated_threshold.json` | 719 B | Calibrated decision threshold configuration ($t^*=0.20$) |
| `models/gnn_edge_aware_repaired.pt` | 23.7 KB | Repaired spatial GNN baseline encoder |
| `models/gnn_edge_aware.pt` | 23.4 KB | Historical Week 1 edge-aware baseline model |
| `models/gnn_baseline.pt` | 7.2 KB | Baseline GCN model |
| `models/baseline_rf.joblib` | 959.8 KB | Baseline Random Forest classifier |
| `models/calibrated_threshold.json` | 506 B | Decision thresholds for baseline models |
| `models/scaler_node_repaired.joblib` | 1.3 KB | RobustScaler for coordinate-free node features |
| `models/scaler_edge_repaired.joblib` | 679 B | RobustScaler for relational edge features |
| `models/scaler_kinematic.joblib` | 1.3 KB | RobustScaler for discrete kinematic residuals |

---

## 4. Dataset Matrix & Workflows

Because large CSV datasets (>15 MB) are excluded from Git to prevent repository bloat, workflows are categorized into:
1. **Fresh Clone Workflows (Zero Datasets Required)**
2. **Evaluation & Replay Workflows (Generated via Script or Local Files)**

| Dataset Filename | Size | Required By | Role in Workflow | Fresh Clone Status | How to Obtain / Generate |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `data/st_gnn_test_predictions.csv` | 23.5 MB | `scripts/aeroguard_server.py` | Dashboard live telemetry replay | Excluded | Generated via Step 5B (`python scripts/audit_st_gnn_metrics.py`), or supplied locally. |
| `data/aero_guard_test.csv` | 18.9 MB | `scripts/audit_st_gnn_metrics.py`, `research/evaluate_benchmarks.py`, `research/ablation_protocol.py` | Out-of-distribution evaluation (32,550 test samples) | Excluded | Generated via Step 5A (`python scripts/generate_aeroguard_dataset.py`), or supplied locally. |
| `data/aero_guard_train.csv` | 57.7 MB | `scripts/train_st_gnn.py`, `scripts/train_models.py` | Retraining models from scratch | Excluded | Generated via Step 5A (`python scripts/generate_aeroguard_dataset.py`). |
| `data/aero_guard_train_backup.csv` | 18.9 MB | `scripts/generate_aeroguard_dataset.py` | Uncorrupted base training flight telemetry | Excluded | Auto-seeded by dataset generator if absent. |
| `data/aero_guard_test_backup.csv` | 5.8 MB | `scripts/generate_aeroguard_dataset.py` | Uncorrupted base test flight telemetry | Excluded | Auto-seeded by dataset generator if absent. |
| `data/aero_guard_test_predictions.csv` | 22.8 MB | `research/evaluate_benchmarks.py` | Baseline model predictions cache | Excluded | Generated automatically on first run of benchmark evaluation. |
| `data/reproduced_st_gnn_predictions.csv` | 25.0 MB | Independent verification | Output artifact of independent audit | Excluded | Output artifact created by `python scripts/audit_st_gnn_metrics.py`. |

---

## 5. Workflow Execution Commands

### Workflow 1: Fresh Clone Validation (Zero Datasets Required)
The entire test suite uses in-memory fixtures, mock message packets, and audited JSON dossiers. No CSV datasets are required:

```bash
# Run all 51 automated unit and integration tests
python -m unittest discover -s tests -p "test_*.py"
```

### Workflow 2: Generating Local Datasets
To generate the training and OOD test splits with physics-consistent Doppler/RSS signals and kinematic spoofing injections:

```bash
# Step 5A: Generate aero_guard_train.csv (57.7 MB) and aero_guard_test.csv (18.9 MB)
python scripts/generate_aeroguard_dataset.py
```

### Workflow 3: Reproducing Model Metrics & Generating Replay Data
Once `data/aero_guard_test.csv` is present, run model inference using the pre-trained official checkpoint:

```bash
# Step 5B: Verify exact test metrics and generate data/st_gnn_test_predictions.csv
python scripts/audit_st_gnn_metrics.py
```
*Expected Output:*
- Accuracy: `86.37%`
- Precision: `73.06%`
- Recall: `18.04%`
- F1-Score: `0.2893`
- ROC-AUC: `0.6235`

### Workflow 4: Running Multi-Model Benchmarks & Ablations
To evaluate all baselines (Random Forest, original GCN, Edge-Aware GNN, and Tri-Stream ST-GNN):

```bash
# Benchmark all 5 models on identical 32,550 test samples
python research/evaluate_benchmarks.py

# Run stream ablation analysis (Spatial vs Temporal vs Kinematic)
python research/ablation_protocol.py
```

### Workflow 5: Launching Command Center Dashboard
Once `data/st_gnn_test_predictions.csv` exists:

```bash
# Terminal 1: Launch Backend WebSocket Server (Port 8000)
python scripts/aeroguard_server.py

# Terminal 2: Launch Vite Frontend GUI (Port 5173)
cd frontend
npm run dev
```
Open `http://localhost:5173` in a web browser.
