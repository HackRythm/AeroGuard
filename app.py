import os
# Fix OpenMP duplicate linking issue in Windows python environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import time

# Set Page Config
st.set_page_config(
    page_title="AeroGuard AI - ATC Space Monitor",
    layout="wide",
    page_icon="📡",
    initial_sidebar_state="expanded"
)

# Custom Premium ATC Styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #0b0f19;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #38bdf8 !important;
        font-family: 'Courier New', monospace;
    }
    div[data-testid="stMetricValue"] {
        color: #0ea5e9;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    .alert-card {
        border-left: 4px solid #ef4444 !important;
    }
    .normal-card {
        border-left: 4px solid #10b981 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to render cards
def render_card(title, value, is_alert=False):
    border_color = "#ef4444" if is_alert else "#10b981"
    st.markdown(f"""
    <div style="background-color: #0f172a; border: 1px solid #1e293b; border-left: 5px solid {border_color}; border-radius: 8px; padding: 15px; text-align: center;">
        <div style="color: #94a3b8; font-size: 0.85rem; font-weight: bold; text-transform: uppercase;">{title}</div>
        <div style="color: #f8fafc; font-size: 1.8rem; font-weight: bold; font-family: 'Courier New', monospace; margin-top: 5px;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING & INITIALIZATION
# ==============================================================================
@st.cache_data
def load_predictions():
    pred_path = "data/aero_guard_test_predictions.csv"
    if not os.path.exists(pred_path):
        return None
    df = pd.read_csv(pred_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['timestamp', 'icao24']).reset_index(drop=True)
    return df

df_preds = load_predictions()

if df_preds is None:
    st.error("Predictions dataset `data/aero_guard_test_predictions.csv` not found. Please run the model training script `scripts/train_models.py` first.")
    st.stop()

# Get unique sorted timestamps
timestamps = sorted(df_preds['timestamp'].unique())
n_timestamps = len(timestamps)

# Calculate threshold for classification based on 83rd percentile of GNN scores
# This matches the expected true anomaly ratio in the test set
GNN_THRESHOLD = df_preds['anomaly_score'].quantile(0.83)

# ==============================================================================
# SIDEBAR / PLAYBACK CONTROLS
# ==============================================================================
st.sidebar.markdown("### 📡 SYSTEM CONTROLS")

# Playback state management
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

def toggle_play():
    st.session_state.playing = not st.session_state.playing

# Playback controls row
play_label = "⏸️ PAUSE" if st.session_state.playing else "▶️ PLAY MONITOR"
st.sidebar.button(play_label, on_click=toggle_play, use_container_width=True)

# Playback speed
playback_speed = st.sidebar.slider("Playback interval (seconds)", min_value=0.2, max_value=2.0, value=0.5, step=0.1)

# Time slider linked to session state
current_idx = st.sidebar.slider(
    "Active Time Step", 
    min_value=0, 
    max_value=n_timestamps - 1, 
    value=st.session_state.current_index, 
    key="slider_index"
)
st.session_state.current_index = current_idx

# Update index in session state if playing
if st.session_state.playing:
    if st.session_state.current_index < n_timestamps - 1:
        st.session_state.current_index += 1
        time.sleep(playback_speed)
        st.rerun()
    else:
        st.session_state.playing = False

current_ts = timestamps[st.session_state.current_index]

# Standard threshold selector
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ DETECTION CONFIG")
custom_threshold = st.sidebar.slider("GNN Anomaly Threshold", min_value=0.01, max_value=0.99, value=float(GNN_THRESHOLD), step=0.01)

# Filter active aircraft at current time
active_df = df_preds[df_preds['timestamp'] == current_ts].copy().reset_index(drop=True)
active_df['pred_label'] = (active_df['anomaly_score'] >= custom_threshold).astype(int)

# Active flights details
n_active = len(active_df)
n_anoms = (active_df['pred_label'] == 1).sum()
n_normal = n_active - n_anoms

# Active graph edges count
edge_count = 0
if n_active > 1:
    x_c, y_c = active_df['x'].values, active_df['y'].values
    dx = x_c[:, None] - x_c[None, :]
    dy = y_c[:, None] - y_c[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    edge_count = np.sum(dist <= 100000.0) // 2  # Undirected edges

# ==============================================================================
# MAIN PAGE LAYOUT
# ==============================================================================
# Header
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>📡 AeroGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #94a3b8 !important; margin-top: 0;'>ADS-B Airspace Relational Threat Detection System</h4>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center; font-family: monospace; color: #38bdf8;'>CURRENT MONITOR TIME: {current_ts}</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 1. Summary Cards (KPIs)
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_card("Aircraft Tracked", n_active)
with col2:
    render_card("Normal Aircraft", n_normal)
with col3:
    render_card("Active Alerts", n_anoms, is_alert=(n_anoms > 0))
with col4:
    render_card("Active Graph Edges", edge_count)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Main Visualizations (Map vs Graph Topology)
viz_col, graph_col = st.columns([3, 2])

with viz_col:
    st.markdown("### ✈️ AIRSPACE TRAJECTORY MAP")
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0b0f19')
    
    # Grid lines
    ax.grid(color='#1e293b', linestyle='--', linewidth=0.5)
    
    # Plot historical trajectories of active flights
    for _, row in active_df.iterrows():
        icao = row['icao24']
        callsign = row['callsign']
        
        # Get historical trail
        trail = df_preds[(df_preds['icao24'] == icao) & (df_preds['timestamp'] <= current_ts)].sort_values('timestamp').tail(60)
        
        # Determine trail color (red if currently anomalous)
        color = '#ef4444' if row['pred_label'] == 1 else '#38bdf8'
        ax.plot(trail['longitude'], trail['latitude'], color=color, alpha=0.4, linewidth=1.5)
        
        # Draw current position dot
        ax.scatter(row['longitude'], row['latitude'], color=color, s=80, edgecolors='#ffffff', linewidths=0.5, zorder=5)
        
        # Text label
        ax.text(row['longitude'] + 0.05, row['latitude'] + 0.03, f"{callsign}", color='#f1f5f9', fontsize=9, fontfamily='monospace', weight='bold')

    ax.set_xlim(-16.0, -6.5)
    ax.set_ylim(50.8, 56.0)
    ax.set_xlabel("Longitude (deg)", color='#94a3b8')
    ax.set_ylabel("Latitude (deg)", color='#94a3b8')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
        
    st.pyplot(fig)

with graph_col:
    st.markdown("### 🕸️ MULTI-AIRCRAFT GRAPH RELATIONS")
    
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0b0f19')
    
    # Construct networkx graph
    G = nx.Graph()
    colors = []
    labels = {}
    
    for i, row in active_df.iterrows():
        G.add_node(i, label=row['icao24'])
        labels[i] = row['callsign']
        colors.append('#ef4444' if row['pred_label'] == 1 else '#10b981')
        
    if n_active > 1:
        x_c, y_c = active_df['x'].values, active_df['y'].values
        dx = x_c[:, None] - x_c[None, :]
        dy = y_c[:, None] - y_c[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        for i in range(n_active):
            for j in range(i+1, n_active):
                if dist[i, j] <= 100000.0:
                    G.add_edge(i, j)
                    
    pos = nx.kamada_kawai_layout(G) if len(G.edges) > 0 else nx.circular_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=600, edgecolors='#ffffff', linewidths=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='#f8fafc', font_family='monospace', font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#475569', width=1.5, ax=ax)
    
    ax.axis('off')
    st.pyplot(fig)

# 3. Alert Panel & Selected Flight Inspector
st.markdown("---")
alert_col, inspect_col = st.columns([3, 2])

with alert_col:
    st.markdown("### ⚠️ THREAT DETECTOR & ALERTS")
    
    alert_display = []
    for _, row in active_df.iterrows():
        status = "🚨 ALERT" if row['pred_label'] == 1 else "✅ NORMAL"
        threat_type = row['anomaly_type'] if row['pred_label'] == 1 else "normal"
        alert_display.append({
            'Callsign': row['callsign'],
            'ICAO24': row['icao24'],
            'Status': status,
            'Anomaly Score': f"{row['anomaly_score']:.4f}",
            'Threat Category': threat_type.upper()
        })
        
    df_alerts = pd.DataFrame(alert_display)
    st.dataframe(
        df_alerts,
        use_container_width=True,
        hide_index=True
    )

with inspect_col:
    st.markdown("### 🔍 FLIGHT TELEMETRY INSPECTOR")
    
    # Dropdown to select active flight
    selected_callsign = st.selectbox("Select Flight Target", active_df['callsign'].unique())
    
    if selected_callsign:
        target_row = active_df[active_df['callsign'] == selected_callsign].iloc[0]
        
        status_text = "🚨 SPREADING SPOOF THREAT" if target_row['pred_label'] == 1 else "✅ SECURE KINEMATICS"
        status_color = "red" if target_row['pred_label'] == 1 else "green"
        
        st.markdown(f"**Target Status**: :{status_color}[{status_text}]")
        if target_row['pred_label'] == 1:
            st.markdown(f"**Identified Attack Profile**: :red[{target_row['anomaly_type'].upper()}]")
            st.markdown(f"**Model Anomaly Score**: :red[{target_row['anomaly_score']:.4f}]")
        else:
            st.markdown(f"**Model Anomaly Score**: :green[{target_row['anomaly_score']:.4f}]")
            
        # Detail grid
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown(f"**ICAO24**: `{target_row['icao24']}`")
            st.markdown(f"**Altitude**: `{target_row['altitude']:.1f} ft` (Geo: `{target_row['geoaltitude']:.1f} ft`)")
            st.markdown(f"**Speed**: `{target_row['groundspeed']:.1f} knots`")
            st.markdown(f"**Heading**: `{target_row['heading']:.1f}°`")
            st.markdown(f"**Vertical Rate**: `{target_row['vertical_rate']:.1f} fpm`")
        with dcol2:
            st.markdown(f"**Acceleration**: `{target_row['acceleration']:.3f} m/s²`")
            st.markdown(f"**Turn Rate**: `{target_row['turn_rate']:.3f}°/s`")
            st.markdown(f"**Doppler Shift (Est)**: `{target_row['estimated_doppler_hz']:.2f} Hz`")
            st.markdown(f"**Signal Power (Est)**: `{target_row['estimated_rss_dbm']:.2f} dBm`")
            st.markdown(f"**SNR (Est)**: `{target_row['estimated_snr_db']:.2f} dB`")
            
        # Multi-Aircraft Relational features
        st.markdown("**Relational Context (G(t))**:")
        st.markdown(f"- Neighbor Count (100km): `{target_row['neighbor_count']}` | Local Density (50km): `{target_row['local_aircraft_density']}`")
        if target_row['nearest_aircraft_distance'] < 500000.0:
            st.markdown(f"- Distance to Nearest Flight: `{target_row['nearest_aircraft_distance']:.1f} meters`")
            st.markdown(f"- Rel. Altitude: `{target_row['relative_altitude']:.1f} ft` | Rel. Velocity: `{target_row['relative_velocity']:.1f} knots` | Rel. Heading: `{target_row['relative_heading']:.1f}°`")
        else:
            st.markdown("- Nearest Flight: `None in detection range`")
