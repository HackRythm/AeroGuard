import os
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
# Virtual Receiver Location (Lisbon Portela Airport - LPPC FIR center)
RECEIVER_LAT = 38.7756
RECEIVER_LON = -9.1354
RECEIVER_ALT = 114.0  # meters

# Communication Physics Parameters
CARRIER_FREQUENCY_HZ = 1090e6  # 1090 MHz nominal ADS-B frequency
C = 299792458.0  # Speed of light in m/s

# Transmitter & Receiver Assumptions
TX_POWER_DBM = 54.0  # 250 W peak transmit power
G_TX_DB = 0.0  # Omnidirectional aircraft antenna gain
G_RX_DB = 3.0  # Ground station antenna gain
L_SYS_DB = 2.0  # Cable/system losses
N_FLOOR_DBM = -107.0  # Thermal noise floor (2 MHz bandwidth, 4 dB noise figure)

# Spatial & Temporal Thresholds
PROXIMITY_THRESHOLD_M = 100000.0  # 100 km threshold for edge relationships
DENSITY_THRESHOLD_M = 50000.0  # 50 km threshold for local density
SEED = 42

# ==============================================================================
# MATHEMATICAL AND PHYSICS HELPER FUNCTIONS
# ==============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine formula to compute horizontal distance in meters.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371000.0 * c

def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """
    Convert WGS84 Geodetic coordinates (lat, lon, alt) to ECEF X, Y, Z coordinates.
    Vectorized for numpy arrays.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    a = 6378137.0  # semi-major axis
    e2 = 0.00669437999014  # eccentricity squared
    
    N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad)**2)
    x = (N + alt_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((1.0 - e2) * N + alt_m) * np.sin(lat_rad)
    return x, y, z

def get_ecef_velocity(lat_deg, lon_deg, gs_kt, heading_deg, vr_fpm):
    """
    Convert groundspeed (knots), heading (degrees), and vertical rate (fpm) 
    to ECEF velocity vector components vx, vy, vz.
    """
    # Convert inputs to SI units
    gs_m_s = gs_kt * 0.514444
    vr_m_s = vr_fpm * 0.00508
    
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    heading_rad = np.radians(heading_deg)
    
    # Local ENU velocity components
    v_east = gs_m_s * np.sin(heading_rad)
    v_north = gs_m_s * np.cos(heading_rad)
    v_up = vr_m_s
    
    # ENU to ECEF rotation matrix elements
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    vx = -sin_lon * v_east - sin_lat * cos_lon * v_north + cos_lat * cos_lon * v_up
    vy = cos_lon * v_east - sin_lat * sin_lon * v_north + cos_lat * sin_lon * v_up
    vz = cos_lat * v_north + sin_lat * v_up
    
    return vx, vy, vz

# ==============================================================================
# ANOMALY INJECTION PIPELINE
# ==============================================================================
def inject_anomalies(df, is_train=True):
    """
    Inject context-aware synthetic anomalies into the flights.
    First modifies 4 core anomaly types, then expands to 3 advanced anomaly types.
    """
    df = df.copy()
    
    # Initialize labels
    df['label'] = 0
    df['anomaly_type'] = 'normal'
    
    # Keep true physical state columns internal (will be dropped before saving)
    df['true_latitude'] = df['latitude']
    df['true_longitude'] = df['longitude']
    df['true_altitude'] = df['altitude']
    df['true_groundspeed'] = df['groundspeed']
    df['true_heading'] = df['heading_unwrapped']
    df['true_vertical_rate'] = df['vertical_rate']
    
    np.random.seed(SEED)
    aircraft = df['icao24'].unique()
    
    # Determine which flights will receive core anomalies
    # Train has 26 aircraft, Test has 7 aircraft
    if is_train:
        num_anomaly_flights = 5
    else:
        num_anomaly_flights = 2
        
    anomaly_aircraft = np.random.choice(aircraft, size=num_anomaly_flights, replace=False)
    
    # 4 Core anomaly types + 1 vertical rate anomaly
    core_types = ['position_jump', 'altitude_anomaly', 'velocity_anomaly', 'heading_anomaly', 'vertical_rate_anomaly']
    
    for i, icao in enumerate(anomaly_aircraft):
        ac_type = core_types[i % len(core_types)]
        mask = df['icao24'] == icao
        indices = df[mask].index.tolist()
        n_points = len(indices)
        
        if n_points > 100:
            # Select a random segment (15% to 25% of the flight duration)
            seg_len = int(n_points * np.random.uniform(0.15, 0.25))
            start_idx = np.random.randint(0, n_points - seg_len)
            anomaly_indices = indices[start_idx : start_idx + seg_len]
            
            if ac_type == 'position_jump':
                # Abrupt coordinates displacement (0.08 to 0.15 degrees shift, approx. 9-16 km)
                df.loc[anomaly_indices, 'latitude'] += np.random.uniform(0.08, 0.15)
                df.loc[anomaly_indices, 'longitude'] += np.random.uniform(0.08, 0.15)
                # Physical transmitter location remains at the original true coords
                
            elif ac_type == 'altitude_anomaly':
                # Abrupt altitude shift (e.g. 6000 to 12000 feet)
                shift = np.random.choice([-1, 1]) * np.random.uniform(6000, 12000)
                df.loc[anomaly_indices, 'altitude'] += shift
                df.loc[anomaly_indices, 'geoaltitude'] += shift
                # Physical transmitter location remains at the original true coords
                
            elif ac_type == 'velocity_anomaly':
                # Sudden scaling of reported groundspeed
                scale = np.random.choice([0.5, 1.5])
                df.loc[anomaly_indices, 'groundspeed'] *= scale
                
            elif ac_type == 'heading_anomaly':
                # Sudden shift in reported heading
                df.loc[anomaly_indices, 'heading_unwrapped'] += np.random.uniform(90, 180)
                
            elif ac_type == 'vertical_rate_anomaly':
                # Set reported vertical rate to climb/descent while altitude remains flat
                df.loc[anomaly_indices, 'vertical_rate'] = np.random.choice([-4000.0, 3000.0])
                
            df.loc[anomaly_indices, 'label'] = 1
            df.loc[anomaly_indices, 'anomaly_type'] = ac_type

    # Advanced Anomaly 1: Ghost Aircraft Scenario
    # Copy a legitimate trajectory, shift it, and assign a fake identity.
    # The transmitter is moving along this new trajectory, so true state = reported state.
    ghost_source_icao = np.random.choice(aircraft)
    ghost_df = df[df['icao24'] == ghost_source_icao].copy()
    
    # Assign fake identity
    ghost_df['icao24'] = 'e00001' if is_train else 'e00002'
    ghost_df['callsign'] = 'GHOST01' if is_train else 'GHOST02'
    
    # Shift position so it flies in a parallel airspace lane
    ghost_df['latitude'] += 0.12
    ghost_df['longitude'] -= 0.12
    
    # For a moving ghost aircraft, Doppler is simulated based on its reported (moving) coordinates
    # So its true transmitter trajectory is identical to its reported trajectory
    ghost_df['true_latitude'] = ghost_df['latitude']
    ghost_df['true_longitude'] = ghost_df['longitude']
    ghost_df['true_altitude'] = ghost_df['altitude']
    ghost_df['true_groundspeed'] = ghost_df['groundspeed']
    ghost_df['true_heading'] = ghost_df['heading_unwrapped']
    ghost_df['true_vertical_rate'] = ghost_df['vertical_rate']
    
    ghost_df['label'] = 1
    ghost_df['anomaly_type'] = 'ghost_aircraft'
    
    df = pd.concat([df, ghost_df], ignore_index=True)
    
    # Advanced Anomaly 2: Identity Inconsistency
    # Duplicate a segment of a flight, shift its reported position, but keep the original icao24.
    # This simulates spoofing the identity of a legitimate active aircraft.
    ident_source_icao = np.random.choice(aircraft)
    ident_source_df = df[df['icao24'] == ident_source_icao].copy()
    n_points = len(ident_source_df)
    
    if n_points > 200:
        seg_len = 150
        start_idx = n_points // 2 - seg_len // 2
        ident_seg = ident_source_df.iloc[start_idx : start_idx + seg_len].copy()
        
        # Shift reported coordinates
        ident_seg['latitude'] -= 0.1
        ident_seg['longitude'] += 0.1
        
        # True transmitter location is equal to reported coordinates (moving transmitter)
        ident_seg['true_latitude'] = ident_seg['latitude']
        ident_seg['true_longitude'] = ident_seg['longitude']
        ident_seg['true_altitude'] = ident_seg['altitude']
        ident_seg['true_groundspeed'] = ident_seg['groundspeed']
        ident_seg['true_heading'] = ident_seg['heading_unwrapped']
        ident_seg['true_vertical_rate'] = ident_seg['vertical_rate']
        
        ident_seg['label'] = 1
        ident_seg['anomaly_type'] = 'identity_inconsistency'
        
        df = pd.concat([df, ident_seg], ignore_index=True)
        
    return df

# ==============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ==============================================================================
def process_dataset(filepath, is_train=True):
    print(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Clean and interpolate missing heading fields per flight sequence
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['icao24', 'timestamp']).reset_index(drop=True)
    df['heading_unwrapped'] = df.groupby('icao24')['heading_unwrapped'].ffill().bfill().fillna(0.0)
    
    # 1. Inject Anomalies
    print("Injecting synthetic anomalies...")
    df = inject_anomalies(df, is_train=is_train)
    
    # Clean headings of newly appended rows and ensure normalized 0-360 heading is available
    df['heading_unwrapped'] = df.groupby('icao24')['heading_unwrapped'].ffill().bfill().fillna(0.0)
    df['heading'] = df['heading_unwrapped'] % 360.0
    df['true_heading'] = df['true_heading'].ffill().bfill().fillna(0.0) % 360.0
    
    # Re-sort to perform flight-level temporal calculations
    df = df.sort_values(by=['icao24', 'timestamp']).reset_index(drop=True)
    
    # 2. Temporal Features (Calculated from reported/public columns)
    print("Calculating temporal features...")
    # Speed & rate conversions to SI units for derivatives
    df['gs_m_s'] = df['groundspeed'] * 0.514444
    df['vr_m_s'] = df['vertical_rate'] * 0.00508
    
    df['time_delta'] = df.groupby('icao24')['timestamp'].diff().dt.total_seconds()
    df['delta_gs'] = df.groupby('icao24')['gs_m_s'].diff()
    df['delta_vr'] = df.groupby('icao24')['vr_m_s'].diff()
    df['delta_heading'] = df.groupby('icao24')['heading_unwrapped'].diff()
    df['delta_vr_fpm'] = df.groupby('icao24')['vertical_rate'].diff()
    
    mask = (df['time_delta'] > 0) & (~df['time_delta'].isna())
    
    df['acceleration'] = 0.0
    df.loc[mask, 'acceleration'] = df.loc[mask, 'delta_gs'] / df.loc[mask, 'time_delta']
    
    df['vertical_acceleration'] = 0.0
    df.loc[mask, 'vertical_acceleration'] = df.loc[mask, 'delta_vr'] / df.loc[mask, 'time_delta']
    
    df['turn_rate'] = 0.0
    df.loc[mask, 'turn_rate'] = df.loc[mask, 'delta_heading'] / df.loc[mask, 'time_delta']
    
    df['climb_rate_change'] = 0.0
    df.loc[mask, 'climb_rate_change'] = df.loc[mask, 'delta_vr_fpm'] / df.loc[mask, 'time_delta']
    
    # Fill temporal NaNs
    df['time_delta'] = df['time_delta'].fillna(0.0)
    
    # Drop intermediate temporal columns
    df = df.drop(columns=['gs_m_s', 'vr_m_s', 'delta_gs', 'delta_vr', 'delta_heading', 'delta_vr_fpm'])
    
    # 3. Spatial & Multi-Aircraft Features (Calculated from reported/public columns)
    print("Calculating spatial & multi-aircraft features...")
    
    # Pre-allocate feature columns
    df['nearest_aircraft_distance'] = 500000.0  # 500 km default
    df['horizontal_distance'] = 500000.0
    df['3d_distance'] = 500000.0
    df['relative_altitude'] = 50000.0  # feet
    df['nearest_aircraft_altitude_difference'] = 50000.0
    df['relative_velocity'] = 500.0  # knots
    df['nearest_aircraft_velocity_difference'] = 500.0
    df['relative_heading'] = 180.0  # degrees
    df['nearest_aircraft_heading_difference'] = 180.0
    df['neighbor_count'] = 0
    df['local_aircraft_density'] = 0
    
    # Group by timestamp to compute concurrent spatial features
    grouped = df.groupby('timestamp')
    
    # Store indices and updates to write in batch
    updates = []
    
    for ts, group in grouped:
        n = len(group)
        if n <= 1:
            continue
            
        indices = group.index.values
        lats = group['latitude'].values
        lons = group['longitude'].values
        alts = group['altitude'].values
        vels = group['groundspeed'].values
        heads = group['heading'].values
        
        # Calculate horizontal distance matrix using vectorized Haversine
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        dlat = lats_rad[:, None] - lats_rad[None, :]
        dlon = lons_rad[:, None] - lons_rad[None, :]
        a = np.sin(dlat / 2.0)**2 + np.cos(lats_rad[:, None]) * np.cos(lats_rad[None, :]) * np.sin(dlon / 2.0)**2
        dist_matrix = 2.0 * 6371000.0 * np.arcsin(np.sqrt(a))
        
        # Exclude self by setting diagonal to infinity
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find nearest neighbor indices
        nearest_indices = np.argmin(dist_matrix, axis=1)
        
        for i in range(n):
            idx = indices[i]
            nearest_idx = nearest_indices[i]
            
            h_dist = dist_matrix[i, nearest_idx]
            
            # Altitude difference in feet
            alt_diff = alts[i] - alts[nearest_idx]
            alt_diff_m = alt_diff * 0.3048
            
            # 3D distance
            dist_3d = np.sqrt(h_dist**2 + alt_diff_m**2)
            
            # Velocity difference in knots
            vel_diff = vels[i] - vels[nearest_idx]
            
            # Shortest angular heading difference
            head_diff = (heads[i] - heads[nearest_idx] + 180.0) % 360.0 - 180.0
            
            # Neighbor count within proximity threshold (100 km)
            n_neighbors = np.sum(dist_matrix[i] <= PROXIMITY_THRESHOLD_M)
            # Local density count within 50 km
            n_density = np.sum(dist_matrix[i] <= DENSITY_THRESHOLD_M)
            
            updates.append({
                'index': idx,
                'nearest_aircraft_distance': h_dist,
                'horizontal_distance': h_dist,
                '3d_distance': dist_3d,
                'relative_altitude': alt_diff,
                'nearest_aircraft_altitude_difference': abs(alt_diff),
                'relative_velocity': vel_diff,
                'nearest_aircraft_velocity_difference': abs(vel_diff),
                'relative_heading': head_diff,
                'nearest_aircraft_heading_difference': abs(head_diff),
                'neighbor_count': int(n_neighbors),
                'local_aircraft_density': int(n_density)
            })
            
    # Apply updates
    if updates:
        updates_df = pd.DataFrame(updates).set_index('index')
        df.update(updates_df)
        df['neighbor_count'] = df['neighbor_count'].astype(int)
        df['local_aircraft_density'] = df['local_aircraft_density'].astype(int)

    # 4. Simulated Communication/Physics Features (Calculated from true physical columns)
    print("Simulating physics-based communication features...")
    # Convert true altitudes to meters
    true_alt_m = df['true_altitude'] * 0.3048
    
    # ECEF coords of Lisbon Airport Ground Receiver
    rec_x, rec_y, rec_z = geodetic_to_ecef(RECEIVER_LAT, RECEIVER_LON, RECEIVER_ALT)
    
    # ECEF coords of aircraft physical locations
    ac_x, ac_y, ac_z = geodetic_to_ecef(df['true_latitude'], df['true_longitude'], true_alt_m)
    
    # Path coordinates relative to receiver
    dx = ac_x - rec_x
    dy = ac_y - rec_y
    dz = ac_z - rec_z
    dist_to_receiver = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Guard against division by zero (unlikely in real files)
    dist_safe = np.where(dist_to_receiver == 0.0, 1.0, dist_to_receiver)
    ux = dx / dist_safe
    uy = dy / dist_safe
    uz = dz / dist_safe
    
    # ECEF velocity of physical aircraft
    vx, vy, vz = get_ecef_velocity(
        df['true_latitude'], 
        df['true_longitude'], 
        df['true_groundspeed'], 
        df['true_heading'], 
        df['true_vertical_rate']
    )
    
    # Radial velocity relative to receiver
    radial_velocity = vx * ux + vy * uy + vz * uz
    
    # Theoretical Doppler shift in Hz
    df['radial_velocity'] = radial_velocity
    df['estimated_doppler_hz'] = - (radial_velocity / C) * CARRIER_FREQUENCY_HZ
    
    # Free Space Path Loss (FSPL) model
    df['distance_to_receiver'] = dist_to_receiver
    df['path_loss_db'] = 20 * np.log10(dist_safe) + 33.20
    
    # Received Signal Power (RSS)
    df['estimated_rss_dbm'] = TX_POWER_DBM + G_TX_DB + G_RX_DB - L_SYS_DB - df['path_loss_db']
    
    # Signal-to-Noise Ratio (SNR)
    df['estimated_snr_db'] = df['estimated_rss_dbm'] - N_FLOOR_DBM
    
    # 5. Clean Data & Prevent Model Cheating
    # Drop all internal true physical columns
    df = df.drop(columns=[
        'true_latitude', 'true_longitude', 'true_altitude', 
        'true_groundspeed', 'true_heading', 'true_vertical_rate'
    ])
    
    return df

# ==============================================================================
# QUALITY CHECKS & VALIDATION
# ==============================================================================
def run_quality_checks(df, filename):
    print(f"\n--- Quality Checks for {filename} ---")
    
    # 1. Missing-value check
    missing = df.isnull().sum()
    print("Missing values per column:")
    for col, count in missing.items():
        print(f"  {col}: {count}")
        
    # 2. Duplicate check (icao24 + timestamp)
    # Note: identity_inconsistency intentionally creates duplicates
    total_duplicates = df.duplicated(subset=['icao24', 'timestamp']).sum()
    expected_duplicates = (df['anomaly_type'] == 'identity_inconsistency').sum()
    print(f"Duplicate icao24 + timestamp records: {total_duplicates} (Expected from identity_inconsistency: {expected_duplicates})")
    
    # 3. Range validation
    print("Range validation checks:")
    print(f"  Latitude range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f} (Valid: -90 to 90)")
    print(f"  Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f} (Valid: -180 to 180)")
    print(f"  Altitude range: {df['altitude'].min():.1f} to {df['altitude'].max():.1f} ft")
    print(f"  Groundspeed range: {df['groundspeed'].min():.1f} to {df['groundspeed'].max():.1f} knots")
    print(f"  Heading range: {df['heading'].min():.1f} to {df['heading'].max():.1f} degrees")
    print(f"  Vertical rate range: {df['vertical_rate'].min():.1f} to {df['vertical_rate'].max():.1f} fpm")
    
    # 4. Physics sanity checks
    print("Physics sanity checks:")
    print(f"  Doppler Shift range: {df['estimated_doppler_hz'].min():.2f} Hz to {df['estimated_doppler_hz'].max():.2f} Hz")
    print(f"  Path Loss range: {df['path_loss_db'].min():.2f} dB to {df['path_loss_db'].max():.2f} dB")
    print(f"  RSS range: {df['estimated_rss_dbm'].min():.2f} dBm to {df['estimated_rss_dbm'].max():.2f} dBm")
    print(f"  SNR range: {df['estimated_snr_db'].min():.2f} dB to {df['estimated_snr_db'].max():.2f} dB")
    
    # Verify that RSS behaves consistently with distance (correlation check)
    corr = df['distance_to_receiver'].corr(df['estimated_rss_dbm'])
    print(f"  Distance vs RSS correlation: {corr:.4f} (Expected close to -1.0)")
    
    # 5. Check if any true state leaked
    leaked_cols = [c for c in df.columns if 'true_' in c]
    if leaked_cols:
        print(f"  [WARNING] Leaked internal state columns found: {leaked_cols}")
    else:
        print("  No internal true physical state columns leaked.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    raw_train_path = "data/aero_guard_train.csv"
    raw_test_path = "data/aero_guard_test.csv"
    
    # Make backups of original files if not already done, to use as the source dataset.
    # We will read from the backups if they exist, otherwise rename original files to backup
    # and then read from them. This ensures we can run this script repeatedly.
    backup_train_path = "data/aero_guard_train_backup.csv"
    backup_test_path = "data/aero_guard_test_backup.csv"
    
    if not os.path.exists(backup_train_path):
        print("Creating backup of original training data...")
        os.rename(raw_train_path, backup_train_path)
    if not os.path.exists(backup_test_path):
        print("Creating backup of original testing data...")
        os.rename(raw_test_path, backup_test_path)
        
    # Process Train Set
    print("\n==================== PROCESSING TRAINING SET ====================")
    df_train_final = process_dataset(backup_train_path, is_train=True)
    run_quality_checks(df_train_final, "aero_guard_train.csv")
    
    # Process Test Set
    print("\n==================== PROCESSING TESTING SET ====================")
    df_test_final = process_dataset(backup_test_path, is_train=False)
    run_quality_checks(df_test_final, "aero_guard_test.csv")
    
    # Verification of chronological split (Time leakage check)
    print("\n==================== DATA LEAKAGE CHECK ====================")
    train_max_time = df_train_final['timestamp'].max()
    test_min_time = df_test_final['timestamp'].min()
    print(f"Training Max Timestamp: {train_max_time}")
    print(f"Testing Min Timestamp:  {test_min_time}")
    if train_max_time < test_min_time:
        print("Chronological split is perfectly respected! No time overlap/leakage.")
    else:
        print("[WARNING] Overlap in timestamps detected between Train and Test sets!")
        
    # Save the files
    print("\nSaving processed train and test sets...")
    df_train_final.to_csv(raw_train_path, index=False)
    df_test_final.to_csv(raw_test_path, index=False)
    print("Files successfully generated:")
    print(f"  - {raw_train_path}")
    print(f"  - {raw_test_path}")
    
    # Output final summary statistics
    print("\n==================== FINAL SUMMARY STATISTICS ====================")
    print(f"Original rows (Train/Test backup): {len(pd.read_csv(backup_train_path))} / {len(pd.read_csv(backup_test_path))}")
    print(f"Original aircraft count (Train/Test backup): {pd.read_csv(backup_train_path)['icao24'].nunique()} / {pd.read_csv(backup_test_path)['icao24'].nunique()}")
    print(f"Training rows: {len(df_train_final)}")
    print(f"Testing rows: {len(df_test_final)}")
    print(f"Training aircraft count: {df_train_final['icao24'].nunique()}")
    print(f"Testing aircraft count: {df_test_final['icao24'].nunique()}")
    
    print("\nClass Balance:")
    print("Training normal/anomalous:")
    print(df_train_final['label'].value_counts())
    print("Testing normal/anomalous:")
    print(df_test_final['label'].value_counts())
    
    print("\nAnomaly Types in Training:")
    print(df_train_final['anomaly_type'].value_counts())
    print("Anomaly Types in Testing:")
    print(df_test_final['anomaly_type'].value_counts())
    
    print("\nFinal Column List:")
    print(list(df_train_final.columns))
