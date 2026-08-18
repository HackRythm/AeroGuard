import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI(title="AeroGuard AI - ML Service", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load prediction dataset
PRED_PATH = "../data/aero_guard_test_predictions.csv"
if not os.path.exists(PRED_PATH):
    # Fallback to local root path
    PRED_PATH = "data/aero_guard_test_predictions.csv"

if not os.path.exists(PRED_PATH):
    raise FileNotFoundError(f"Predictions dataset not found at {PRED_PATH}. Run training script first.")

print(f"Loading predictions dataset from {PRED_PATH}...")
df_preds = pd.read_csv(PRED_PATH)
df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])
df_preds = df_preds.sort_values(by=['timestamp', 'icao24']).reset_index(drop=True)

# Cache model performance metrics from predictions
# RF Baseline metrics
baseline_auc = 0.4716
gnn_auc = 0.6435

# Set constant receiver coordinates
RECEIVER_LAT = 38.7756
RECEIVER_LON = -9.1354
PROXIMITY_THRESHOLD_M = 100000.0  # 100 km

# Calculate global threshold for classifications
GNN_THRESHOLD = df_preds['anomaly_score'].quantile(0.83)

@app.get("/api/health")
def health():
    return {"status": "online"}

@app.get("/api/timeline")
def get_timeline():
    # Convert timestamps back to ISO strings
    ts_list = df_preds['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z').unique().tolist()
    return {"timestamps": ts_list, "threshold": float(GNN_THRESHOLD)}

@app.get("/api/metrics")
def get_metrics():
    return {
        "baseline": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": baseline_auc
        },
        "gnn": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": gnn_auc
        }
    }

@app.get("/api/aircraft")
def get_aircraft(timestamp: str = Query(...)):
    try:
        ts = pd.to_datetime(timestamp)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
        
    active = df_preds[df_preds['timestamp'] == ts].copy()
    if active.empty:
        return {"aircraft": []}
        
    # Classify labels based on scores
    active['pred_label'] = 0
    active.loc[active['anomaly_score'] >= GNN_THRESHOLD, 'pred_label'] = 2  # Critical
    active.loc[(active['anomaly_score'] >= 0.3) & (active['anomaly_score'] < GNN_THRESHOLD), 'pred_label'] = 1  # Warning
    
    # Fill in SDC values
    records = []
    for _, row in active.iterrows():
        record = row.to_dict()
        # Convert timestamp to string
        record['timestamp'] = str(row['timestamp'])
        records.append(record)
        
    return {"aircraft": records}

@app.get("/api/graph/{timestamp}")
def get_graph(timestamp: str):
    try:
        ts = pd.to_datetime(timestamp)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
        
    active = df_preds[df_preds['timestamp'] == ts].copy().reset_index(drop=True)
    n = len(active)
    if n == 0:
        return {"nodes": [], "edges": []}
        
    # Construct nodes list
    active['pred_label'] = 0
    active.loc[active['anomaly_score'] >= GNN_THRESHOLD, 'pred_label'] = 2  # Critical
    active.loc[(active['anomaly_score'] >= 0.3) & (active['anomaly_score'] < GNN_THRESHOLD), 'pred_label'] = 1  # Warning
    
    nodes = []
    for i, row in active.iterrows():
        nodes.append({
            "id": i,
            "icao24": row['icao24'],
            "callsign": row['callsign'],
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "anomaly_score": float(row['anomaly_score']),
            "pred_label": int(row['pred_label']),
            "anomaly_type": str(row['anomaly_type'])
        })
        
    # Construct edges
    edges = []
    if n > 1:
        x_coords = active['x'].values
        y_coords = active['y'].values
        dx = x_coords[:, None] - x_coords[None, :]
        dy = y_coords[:, None] - y_coords[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        for i in range(n):
            for j in range(i+1, n):
                if dist[i, j] <= PROXIMITY_THRESHOLD_M:
                    edges.append({"source": i, "target": j})
                    
    return {"nodes": nodes, "edges": edges}
