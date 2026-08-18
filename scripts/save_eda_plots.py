import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Configure Receiver Constants
RECEIVER_LAT = 38.7756
RECEIVER_LON = -9.1354
RECEIVER_ALT = 114.0

def save_plots():
    os.makedirs("docs/images", exist_ok=True)
    
    # Load dataset
    df_train = pd.read_csv("data/aero_guard_train.csv")
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    
    sns.set_theme(style='darkgrid')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    # 1. Physics Validation Plot (Distance vs RSS & Radial Velocity vs Doppler)
    print("Generating physics validation plot...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sample_df = df_train.sample(n=min(5000, len(df_train)), random_state=42)
    
    # Distance vs RSS
    sns.scatterplot(data=sample_df, x='distance_to_receiver', y='estimated_rss_dbm', ax=axes[0], alpha=0.5, color='coral')
    axes[0].set_title('Receiver Distance vs. Received Signal Strength (RSS)')
    axes[0].set_xlabel('Distance to Receiver (meters)')
    axes[0].set_ylabel('RSS (dBm)')
    
    # Radial Velocity vs Doppler
    sns.scatterplot(data=sample_df, x='radial_velocity', y='estimated_doppler_hz', ax=axes[1], alpha=0.5, color='purple')
    axes[1].set_title('Radial Velocity vs. Doppler Shift')
    axes[1].set_xlabel('Radial Velocity (m/s)')
    axes[1].set_ylabel('Doppler Shift (Hz)')
    
    plt.tight_layout()
    plt.savefig("docs/images/physics_validation.png", dpi=150)
    plt.close()
    
    # 2. Anomaly Signatures Plot (Comparative boxplots)
    print("Generating anomaly comparative boxplot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    comp_features = [
        ('groundspeed', 'knots', 'groundspeed (knots)'),
        ('estimated_rss_dbm', 'dBm', 'RSS (dBm)'),
        ('estimated_doppler_hz', 'Hz', 'Doppler Shift (Hz)')
    ]
    for i, (col, unit, name) in enumerate(comp_features):
        sns.boxplot(data=df_train, x='label', y=col, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Normal vs. Anomalous: {name}')
        axes[i].set_xticklabels(['Normal (0)', 'Anomaly (1)'])
        axes[i].set_xlabel('')
        axes[i].set_ylabel(col)
        
    plt.tight_layout()
    plt.savefig("docs/images/anomaly_comparison.png", dpi=150)
    plt.close()
    
    # 3. Trajectory Plot (Normal vs Anomalous)
    print("Generating trajectory plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot A: Normal Traffic
    normal_df = df_train[df_train['label'] == 0]
    for icao in normal_df['icao24'].unique()[:8]:
        ac_traj = normal_df[normal_df['icao24'] == icao]
        ax1.plot(ac_traj['longitude'], ac_traj['latitude'], label=f'ICAO: {icao}', linewidth=2)
        ax1.scatter(ac_traj['longitude'].iloc[0], ac_traj['latitude'].iloc[0], marker='o', color='green', s=30)
        ax1.scatter(ac_traj['longitude'].iloc[-1], ac_traj['latitude'].iloc[-1], marker='x', color='red', s=30)
        
    ax1.scatter(RECEIVER_LON, RECEIVER_LAT, color='black', marker='^', s=150, zorder=5, label='Lisbon Receiver')
    ax1.set_title('Plot A: Concurrent Normal Air Traffic Trajectories')
    ax1.set_xlabel('Longitude (deg)')
    ax1.set_ylabel('Latitude (deg)')
    ax1.legend(loc='lower left', ncol=2)
    
    # Plot B: Injected Anomalies
    all_active_ac = df_train['icao24'].unique()
    for icao in all_active_ac:
        ac_traj = df_train[df_train['icao24'] == icao].sort_values('timestamp')
        if ac_traj['label'].any():
            norm_seg = ac_traj[ac_traj['label'] == 0]
            if not norm_seg.empty:
                ax2.scatter(norm_seg['longitude'], norm_seg['latitude'], color='blue', s=3, alpha=0.2, label='Normal Segment' if icao == all_active_ac[0] else '')
            
            anom_seg = ac_traj[ac_traj['label'] == 1]
            if not anom_seg.empty:
                for anom_type in anom_seg['anomaly_type'].unique():
                    anom_sub = anom_seg[anom_seg['anomaly_type'] == anom_type]
                    ax2.scatter(anom_sub['longitude'], anom_sub['latitude'], s=12, alpha=0.8, label=f'Anomaly: {anom_type}')
        else:
            ax2.scatter(ac_traj['longitude'], ac_traj['latitude'], color='blue', s=2, alpha=0.1)
            
    ax2.scatter(RECEIVER_LON, RECEIVER_LAT, color='black', marker='^', s=150, zorder=5, label='Lisbon Receiver')
    ax2.set_title('Plot B: Air Traffic with Injected Anomaly Trajectories')
    ax2.set_xlabel('Longitude (deg)')
    ax2.set_ylabel('Latitude (deg)')
    
    # Clean unique labels for legend
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='lower left')
    
    plt.tight_layout()
    plt.savefig("docs/images/aircraft_trajectories.png", dpi=150)
    plt.close()
    
    # 4. Dynamic Graph Snapshot Visualization
    print("Generating graph snapshot visualization...")
    # Find a good snapshot with edges and anomalies
    vis_file = None
    snap_dir = "data/graphs/sample_graphs"
    for filename in sorted(os.listdir(snap_dir)):
        if filename.endswith('.json'):
            with open(os.path.join(snap_dir, filename), 'r') as f:
                snap = json.load(f)
            n_anom = sum(1 for node in snap['nodes'] if node['label'] == 1)
            n_edges = len(snap['edges'])
            if n_anom > 0 and n_edges > 0:
                vis_file = filename
                break
                
    if vis_file:
        with open(os.path.join(snap_dir, vis_file), 'r') as f:
            snap = json.load(f)
            
        fig = plt.figure(figsize=(10, 8))
        G = nx.Graph()
        labels = {}
        colors = []
        
        for node in snap['nodes']:
            nid = node['node_id']
            icao = node['icao24']
            callsign = node['callsign']
            anom_type = node['anomaly_type']
            G.add_node(nid, label=icao, anomaly=node['label'])
            labels[nid] = f"{callsign}\n({icao})\n[{anom_type}]" if node['label'] == 1 else f"{callsign}\n({icao})"
            colors.append('tomato' if node['label'] == 1 else 'lightgreen')
            
        for edge in snap['edges']:
            G.add_edge(edge['source'], edge['target'])
            
        pos = {}
        for node in snap['nodes']:
            pos[node['node_id']] = (node['features'][1], node['features'][0])  # (lon, lat)
            
        nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, node_size=1800,
                font_size=8, font_weight='bold', edge_color='gray', width=1.5)
        
        plt.title(f"AeroGuard Airspace Dynamic Graph Snapshot: {snap['timestamp']}", fontsize=14)
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")
        plt.tight_layout()
        plt.savefig("docs/images/dynamic_graph_snapshot.png", dpi=150)
        plt.close()
        print("All plots saved successfully in docs/images/")
    else:
        print("No active graph snapshots with edges and anomalies found.")

if __name__ == "__main__":
    save_plots()
