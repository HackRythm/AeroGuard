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
        background-color: #030712;
        color: #f8fafc;
    }
    .stApp {
        background-color: #030712;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #38bdf8 !important;
        font-family: 'Courier New', monospace;
    }
    div[data-testid="stMetricValue"] {
        color: #0ea5e9;
        font-family: 'Courier New', monospace;
    }
    
    /* Header Pulse Animation */
    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1.0; }
        100% { opacity: 0.5; }
    }
    .status-online {
        color: #10b981;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        animation: pulse 2.5s infinite;
        font-size: 1.0rem;
    }
    
    /* Sleek container boxes */
    .atc-panel {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to render cards
def render_card(title, value, border_color="#10b981"):
    st.markdown(f"""
    <div style="background-color: #0f172a; border: 1px solid #1e293b; border-left: 5px solid {border_color}; border-radius: 8px; padding: 15px; text-align: center;">
        <div style="color: #94a3b8; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; font-family: 'Courier New', monospace;">{title}</div>
        <div style="color: #f8fafc; font-size: 1.7rem; font-weight: bold; font-family: 'Courier New', monospace; margin-top: 5px;">{value}</div>
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
GNN_THRESHOLD = df_preds['anomaly_score'].quantile(0.83)

# ==============================================================================
# SESSION STATE MANAGEMENT
# ==============================================================================
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "selected_callsign" not in st.session_state:
    st.session_state.selected_callsign = ""
if "substep" not in st.session_state:
    st.session_state.substep = 0.0

# ==============================================================================
# PLAYBACK & DEMO LOOP (LINEAR INTERPOLATION)
# ==============================================================================
# Increment playback frame
if st.session_state.playing:
    # 2 sub-steps per timestamp to make movement glide instead of teleport
    if st.session_state.substep < 0.5:
        st.session_state.substep += 0.5
    else:
        st.session_state.substep = 0.0
        if st.session_state.current_index < n_timestamps - 1:
            st.session_state.current_index += 1
        else:
            st.session_state.playing = False
    time.sleep(0.2)  # High frame rate redraw
    st.rerun()

current_ts_idx = st.session_state.current_index
current_ts = timestamps[current_ts_idx]
alpha = st.session_state.substep

# ==============================================================================
# INTERPOLATE AIRCRAFT MOVEMENT
# ==============================================================================
# Get active rows at current and next step (if available)
active_t = df_preds[df_preds['timestamp'] == current_ts].copy()

if current_ts_idx < n_timestamps - 1 and alpha > 0.0:
    next_ts = timestamps[current_ts_idx + 1]
    active_t1 = df_preds[df_preds['timestamp'] == next_ts].copy()
    
    # Merge to find overlapping aircraft
    merged = pd.merge(active_t, active_t1, on='icao24', suffixes=('_t', '_t1'))
    
    interpolated_rows = []
    for _, row in merged.iterrows():
        # Linear interpolation
        lat = row['latitude_t'] * (1 - alpha) + row['latitude_t1'] * alpha
        lon = row['longitude_t'] * (1 - alpha) + row['longitude_t1'] * alpha
        alt = row['altitude_t'] * (1 - alpha) + row['altitude_t1'] * alpha
        geoalt = row['geoaltitude_t'] * (1 - alpha) + row['geoaltitude_t1'] * alpha
        gs = row['groundspeed_t'] * (1 - alpha) + row['groundspeed_t1'] * alpha
        vr = row['vertical_rate_t'] * (1 - alpha) + row['vertical_rate_t1'] * alpha
        
        # Heading interpolation (angular shortest path)
        h_t = row['heading_t']
        h_t1 = row['heading_t1']
        diff = (h_t1 - h_t + 180) % 360 - 180
        heading = (h_t + diff * alpha) % 360
        
        # Keep features from state t for simplicity
        interp_row = row.to_dict()
        # Clean suffix names
        for key in list(interp_row.keys()):
            if key.endswith('_t'):
                base_key = key[:-2]
                interp_row[base_key] = interp_row[key]
                
        interp_row.update({
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'geoaltitude': geoalt,
            'groundspeed': gs,
            'heading': heading,
            'vertical_rate': vr
        })
        interpolated_rows.append(interp_row)
        
    # Append flights that exist only in t
    only_t = active_t[~active_t['icao24'].isin(merged['icao24'])].to_dict('records')
    interpolated_rows.extend(only_t)
    
    active_df = pd.DataFrame(interpolated_rows)
else:
    active_df = active_t.copy()

# Ensure standard threshold values are applied
active_df['pred_label'] = 0
active_df.loc[active_df['anomaly_score'] >= GNN_THRESHOLD, 'pred_label'] = 2  # Critical
active_df.loc[(active_df['anomaly_score'] >= 0.3) & (active_df['anomaly_score'] < GNN_THRESHOLD), 'pred_label'] = 1  # Warning

# Count metrics
n_active = len(active_df)
n_crit = (active_df['pred_label'] == 2).sum()
n_warn = (active_df['pred_label'] == 1).sum()
n_normal = n_active - n_crit - n_warn

# Calculate Active Undirected Edges count (Proximity <= 100km)
edge_count = 0
if n_active > 1:
    x_c, y_c = active_df['x'].values, active_df['y'].values
    dx = x_c[:, None] - x_c[None, :]
    dy = y_c[:, None] - y_c[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    edge_count = np.sum(dist <= 100000.0) // 2

# ==============================================================================
# TOP HEADER & PRESENTATION GRID
# ==============================================================================
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.markdown("<h1 style='margin: 0; padding: 0;'>AEROGUARD AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-family: monospace; margin: 0; padding: 0;'>Dynamic Airspace Intelligence & ADS-B Security Monitoring</p>", unsafe_allow_html=True)
with h_col2:
    st.markdown("<div style='text-align: right; margin-top: 10px;'><span class='status-online'>● SYSTEM MONITOR ONLINE</span></div>", unsafe_allow_html=True)

st.markdown("<hr style='margin: 10px 0 20px 0; border-color: #1e293b;'>", unsafe_allow_html=True)

# Metric KPI cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    render_card("AIRCRAFT TRACKED", n_active, "#38bdf8")
with kpi2:
    render_card("SECURE TRAJECTORIES", n_normal, "#10b981")
with kpi3:
    render_card("ANOMALOUS ALERTS", n_crit + n_warn, "#ef4444" if (n_crit + n_warn > 0) else "#1e293b")
with kpi4:
    render_card("ACTIVE GRAPH EDGES", edge_count, "#64748b")

st.markdown("<br>", unsafe_allow_html=True)

# Selectable aircraft dropdown in sidebar
st.sidebar.markdown("### 🔍 FLIGHT INTERCEPT")
all_active_callsigns = sorted(active_df['callsign'].unique())
if st.session_state.selected_callsign not in all_active_callsigns:
    st.session_state.selected_callsign = all_active_callsigns[0] if all_active_callsigns else ""
    
selected_callsign = st.sidebar.selectbox(
    "Select Target Aircraft", 
    all_active_callsigns,
    index=all_active_callsigns.index(st.session_state.selected_callsign) if st.session_state.selected_callsign in all_active_callsigns else 0
)
st.session_state.selected_callsign = selected_callsign

# Retrieve selected aircraft row
selected_row = None
if selected_callsign in active_df['callsign'].values:
    selected_row = active_df[active_df['callsign'] == selected_callsign].iloc[0]

# ==============================================================================
# MAIN DISPLAY PANEL - AIRSPACE MAP
# ==============================================================================
st.markdown("### ✈️ LIVE SURVEILLANCE RADAR DISPLAY")

fig, ax = plt.subplots(figsize=(15, 6.5))
fig.patch.set_facecolor('#030712')
ax.set_facecolor('#0f172a')

# Faint radar grid and spines
ax.grid(color='#1e293b', linestyle=':', linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color('#1e293b')

# Concentric range rings around Lisbon Receiver coordinates
# Plot as faint range arcs centered at (38.7756, -9.1354)
for r_km in [1300, 1400, 1500, 1600]:
    bearings = np.linspace(np.radians(280), np.radians(360), 100)
    # Simple projection rings matching degree boundaries
    r_deg = r_km / 111.0
    lat_ring = RECEIVER_LAT + r_deg * np.cos(bearings)
    lon_ring = RECEIVER_LON + r_deg * np.sin(bearings)
    ax.plot(lon_ring, lat_ring, color='#334155', linestyle='--', linewidth=0.8, alpha=0.5)
    # Ring label
    ax.text(lon_ring[50], lat_ring[50] + 0.05, f"{r_km} km", color='#475569', fontsize=8, fontfamily='monospace')

# Plot active aircraft trails and markers
for _, row in active_df.iterrows():
    icao = row['icao24']
    callsign = row['callsign']
    pred = row['pred_label']
    
    # 1. Plot historical trajectory trail
    trail = df_preds[(df_preds['icao24'] == icao) & (df_preds['timestamp'] <= current_ts)].sort_values('timestamp').tail(40)
    is_selected = (selected_row is not None and icao == selected_row['icao24'])
    
    # Trace color based on status
    if pred == 2:
        trace_color = '#ef4444'  # Critical Red
    elif pred == 1:
        trace_color = '#f59e0b'  # Warning Amber
    else:
        trace_color = '#10b981'  # Normal Green
        
    trail_alpha = 0.8 if is_selected else 0.2
    trail_width = 2.5 if is_selected else 1.2
    
    ax.plot(trail['longitude'], trail['latitude'], color=trace_color, alpha=trail_alpha, linewidth=trail_width)
    
    # 2. Draw target marker
    marker_shape = 'o'
    marker_size = 70
    if pred == 2:
        marker_shape = 'D'  # Diamond
        marker_size = 90
    elif pred == 1:
        marker_shape = '^'  # Triangle
        marker_size = 80
        
    # Standard arrowhead heading vector
    h_rad = np.radians(90.0 - row['heading'])
    u = np.cos(h_rad) * 0.15
    v = np.sin(h_rad) * 0.15
    ax.quiver(row['longitude'], row['latitude'], u, v, color=trace_color, scale=3, scale_units='xy', width=0.003, zorder=6)
    
    ax.scatter(row['longitude'], row['latitude'], color=trace_color, marker=marker_shape, s=marker_size, edgecolors='#ffffff', linewidths=0.5, zorder=7)
    
    # Labels (only for selected or anomalous targets to prevent clutter)
    if is_selected or pred > 0:
        label_color = '#ef4444' if pred == 2 else ('#f59e0b' if pred == 1 else '#38bdf8')
        label_text = f"{callsign}\n[{row['anomaly_type'].upper()}]" if pred > 0 else f"{callsign}"
        ax.text(row['longitude'] + 0.08, row['latitude'] + 0.04, label_text, color=label_color, fontsize=8, fontfamily='monospace', weight='bold', bbox=dict(boxstyle='square,pad=0.2', facecolor='#090d16', alpha=0.8, edgecolor='#1e293b', lw=0.5))

# Draw neighbor links for selected aircraft
if selected_row is not None and n_active > 1:
    x_sel, y_sel = selected_row['x'], selected_row['y']
    for _, row in active_df.iterrows():
        if row['icao24'] == selected_row['icao24']:
            continue
        h_d = np.sqrt((x_sel - row['x'])**2 + (y_sel - row['y'])**2)
        if h_d <= 100000.0:  # Within proximity edge
            ax.plot([selected_row['longitude'], row['longitude']], [selected_row['latitude'], row['latitude']], color='#ef4444' if (selected_row['pred_label'] == 2 or row['pred_label'] == 2) else '#38bdf8', linestyle=':', linewidth=1.5, alpha=0.8)

ax.set_xlim(-16.0, -7.0)
ax.set_ylim(50.5, 56.2)
ax.tick_params(colors='#64748b', labelsize=8)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily('monospace')
    
st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# BOTTOM PANELS: GRAPH & ACTIVE THREATS
# ==============================================================================
graph_col, threat_col = st.columns([4, 3])

with graph_col:
    st.markdown("<div class='atc-panel'>", unsafe_allow_html=True)
    st.markdown("### 🕸️ DYNAMIC GRAPH TOPOLOGY")
    
    fig_g, ax_g = plt.subplots(figsize=(8, 5.5))
    fig_g.patch.set_facecolor('#0f172a')
    ax_g.set_facecolor('#0f172a')
    
    G = nx.Graph()
    colors = []
    labels = {}
    node_sizes = []
    line_widths = []
    
    # Stabilize graph node coordinates by mapping directly to flight longitude/latitude
    pos = {}
    
    for i, row in active_df.iterrows():
        G.add_node(i, label=row['icao24'])
        labels[i] = row['callsign']
        pos[i] = (row['longitude'], row['latitude'])
        
        is_selected = (selected_row is not None and row['icao24'] == selected_row['icao24'])
        
        if row['pred_label'] == 2:
            colors.append('#ef4444')
        elif row['pred_label'] == 1:
            colors.append('#f59e0b')
        else:
            colors.append('#10b981')
            
        node_sizes.append(600 if is_selected else 250)
        line_widths.append(1.5 if is_selected else 0.5)
        
    if n_active > 1:
        x_c, y_c = active_df['x'].values, active_df['y'].values
        dx = x_c[:, None] - x_c[None, :]
        dy = y_c[:, None] - y_c[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        for i in range(n_active):
            for j in range(i+1, n_active):
                if dist[i, j] <= 100000.0:
                    G.add_edge(i, j)
                    
    # Draw graph elements
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes, edgecolors='#ffffff', linewidths=line_widths, ax=ax_g)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color='#f8fafc', font_family='monospace', font_weight='bold', ax=ax_g)
    
    # Emphasize edges connected to selected flight
    edge_list = list(G.edges)
    if len(edge_list) > 0:
        edge_colors = []
        widths = []
        for u_node, v_node in edge_list:
            u_icao = active_df.loc[u_node, 'icao24']
            v_icao = active_df.loc[v_node, 'icao24']
            
            is_sel_edge = (selected_row is not None and (u_icao == selected_row['icao24'] or v_icao == selected_row['icao24']))
            edge_colors.append('#38bdf8' if is_sel_edge else '#475569')
            widths.append(2.0 if is_sel_edge else 1.0)
            
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_colors, width=widths, alpha=0.7, ax=ax_g)
        
    ax_g.axis('off')
    st.pyplot(fig_g)
    st.markdown("</div>", unsafe_allow_html=True)

with threat_col:
    st.markdown("<div class='atc-panel'>", unsafe_allow_html=True)
    st.markdown("### ⚠ ACTIVE THREATS")
    
    threat_list = []
    for _, row in active_df.iterrows():
        if row['pred_label'] > 0:
            status = "🚨 CRITICAL" if row['pred_label'] == 2 else "⚠️ WARNING"
            threat_list.append({
                'Callsign': row['callsign'],
                'Ident': row['icao24'],
                'Threat Profile': row['anomaly_type'].upper().replace('_', ' '),
                'Status': status,
                'Anomaly Score': f"{row['anomaly_score']:.3f}"
            })
            
    if threat_list:
        df_threats = pd.DataFrame(threat_list)
        # Interactive table selection
        selection = st.dataframe(
            df_threats,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single_row",
            key="threat_table"
        )
        
        # Sync selection back to selected aircraft
        selected_rows = selection.get("selection", {}).get("rows", [])
        if selected_rows:
            target_callsign = df_threats.iloc[selected_rows[0]]['Callsign']
            if target_callsign != st.session_state.selected_callsign:
                st.session_state.selected_callsign = target_callsign
                st.rerun()
    else:
        st.markdown("<div style='color: #475569; font-family: monospace; text-align: center; padding: 50px 0;'>NO SECURE VIOLATIONS DETECTED IN MONITOR BLOCK</div>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# INFORMATION PANELS: FLIGHT INSPECTOR & SDC TELEMETRY
# ==============================================================================
inspect_col, comm_col = st.columns(2)

with inspect_col:
    st.markdown("<div class='atc-panel'>", unsafe_allow_html=True)
    st.markdown("### 🔍 TARGET KINEMATICS")
    
    if selected_row is not None:
        p_label = selected_row['pred_label']
        status_text = "🚨 CRITICAL SPOOF THREAT" if p_label == 2 else ("⚠️ FLIGHT INCONSISTENCY" if p_label == 1 else "✅ SECURE KINEMATICS")
        status_color = "red" if p_label == 2 else ("orange" if p_label == 1 else "green")
        
        st.markdown(f"**Surveillance Status**: :{status_color}[{status_text}]")
        st.markdown(f"**Callsign**: `{selected_row['callsign']}` | **ICAO24**: `{selected_row['icao24']}`")
        
        # Telemetry detail
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown(f"**Latitude**: `{selected_row['latitude']:.4f}°`")
            st.markdown(f"**Longitude**: `{selected_row['longitude']:.4f}°`")
            st.markdown(f"**Altitude**: `{selected_row['altitude']:.0f} ft` (Geo: `{selected_row['geoaltitude']:.0f} ft`)")
            st.markdown(f"**Speed**: `{selected_row['groundspeed']:.1f} knots`")
        with dcol2:
            st.markdown(f"**Heading**: `{selected_row['heading']:.1f}°`")
            st.markdown(f"**Vertical Rate**: `{selected_row['vertical_rate']:.0f} fpm`")
            st.markdown(f"**Acceleration**: `{selected_row['acceleration']:.3f} m/s²`")
            st.markdown(f"**Turn Rate**: `{selected_row['turn_rate']:.3f}°/s`")
            
        st.markdown(f"**Threat Category**: `{selected_row['anomaly_type'].upper()}` | **GNN Anomaly Score**: `{selected_row['anomaly_score']:.4f}`")
    else:
        st.info("Select a flight target to display kinematics data.")
        
    st.markdown("</div>", unsafe_allow_html=True)

with comm_col:
    st.markdown("<div class='atc-panel'>", unsafe_allow_html=True)
    st.markdown("### 📡 SDC COMMUNICATION TELEMETRY (SIMULATED)")
    
    if selected_row is not None:
        st.markdown("<span style='color: #64748b; font-size: 0.8rem; font-family: monospace;'>SIGNAL ATTRIBUTES DERIVED FROM SYSTEM PARAMETERS</span>", unsafe_allow_html=True)
        st.markdown(f"**Nominal Carrier Frequency**: `1090 MHz` (ADS-B 1090ES)")
        
        # SDC telemetry detail
        scol1, scol2 = st.columns(2)
        with scol1:
            st.markdown(f"**Doppler Shift (Est)**: `{selected_row['estimated_doppler_hz']:.2f} Hz`")
            st.markdown(f"**Signal Power (Est)**: `{selected_row['estimated_rss_dbm']:.2f} dBm`")
            st.markdown(f"**SNR (Est)**: `{selected_row['estimated_snr_db']:.2f} dB`")
        with scol2:
            st.markdown(f"**Distance to Receiver**: `{selected_row['distance_to_receiver']/1000.0:.2f} km`")
            st.markdown(f"**Path Loss (Est)**: `{selected_row['path_loss_db']:.2f} dB`")
            st.markdown(f"**Radial Velocity**: `{selected_row['radial_velocity']:.1f} m/s`")
    else:
        st.info("Select a flight target to display communication signal telemetry.")
        
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# FOOTER / TIMELINE PLAYBACK CONTROLS
# ==============================================================================
st.markdown("<hr style='margin: 20px 0 10px 0; border-color: #1e293b;'>", unsafe_allow_html=True)

p_col1, p_col2, p_col3 = st.columns([1, 4, 1])

with p_col1:
    demo_mode = st.toggle("📺 DEMO AUTOPLAY", value=st.session_state.playing, key="demo_mode_toggle")
    if demo_mode != st.session_state.playing:
        st.session_state.playing = demo_mode
        st.rerun()

with p_col2:
    # Time step manual slider
    manual_idx = st.slider(
        "Playback Timeline",
        min_value=0,
        max_value=n_timestamps - 1,
        value=st.session_state.current_index,
        key="timeline_slider",
        label_visibility="collapsed"
    )
    if manual_idx != st.session_state.current_index:
        st.session_state.current_index = manual_idx
        st.session_state.substep = 0.0
        st.rerun()

with p_col3:
    if st.button("🔄 RESET DEMO", use_container_width=True):
        st.session_state.current_index = 0
        st.session_state.substep = 0.0
        st.session_state.playing = False
        st.rerun()
