import { create } from 'zustand';

let wsClient = null;
let wsReconnectTimeout = null;

const useAeroStore = create((set, get) => ({
  // Backend connection & configuration
  backendConnected: false,
  systemStatus: 'CONNECTING',
  threshold: 0.20,
  activeModel: 'Tri-Stream ST-GNN',
  receiverMode: 'software_defined_simulation',
  receiverStatus: {
    receiver_mode: 'synthetic_iq',
    sample_rate_msps: 2.0,
    center_frequency_mhz: 1090.0,
    hardware_connected: false,
    hardware_status: 'No physical SDR device required (Software-Defined Mode)',
    samples_processed: 0,
    messages_decoded: 0,
    crc_passed: 0,
    crc_failed: 0,
    last_message_timestamp: 'No frames decoded yet',
  },
  burstDecoding: false,
  recentDecodedFrames: [],
  simulationAttackActive: null,

  // Aircraft & telemetry state
  aircraft: [],
  graphEdges: [],
  currentTimestamp: 'Waiting for stream...',
  timestepIndex: 0,

  // Alert state
  alerts: [],

  // Metrics (derived + backend)
  metrics: {
    totalTracked: 0,
    modelConfidence: 0.0,
    activeAnomalies: 0,
    signalIntegrity: 100.0,
  },

  // Playback state
  isPlaying: false,
  tickCount: 0,

  // UI state
  selectedAircraft: null,
  sidebarCollapsed: true,

  // XAI Explanation state
  activeExplanation: null,
  explanationLoading: false,
  explanationError: null,
  explanationCache: {},

  // ── Actions ────────────────────────────────────

  togglePlayback: () => {
    const nextState = !get().isPlaying;
    set({ isPlaying: nextState });
    if (nextState) {
      get().tick();
    }
  },

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  selectAircraft: (id) => {
    const prev = get().selectedAircraft;
    const newId = prev === id ? null : id;
    set({ selectedAircraft: newId });
    if (newId) {
      get().fetchExplanation(newId);
    } else {
      set({ activeExplanation: null, explanationError: null });
    }
  },

  dismissAlert: (alertId) =>
    set((s) => ({
      alerts: s.alerts.map((a) =>
        a.id === alertId ? { ...a, dismissed: true } : a
      ),
    })),

  // Initialize WebSocket for real-time telemetry streaming
  initWebSocket: () => {
    if (wsClient && (wsClient.readyState === WebSocket.OPEN || wsClient.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;

    try {
      wsClient = new WebSocket(wsUrl);

      wsClient.onopen = () => {
        set({ backendConnected: true, systemStatus: 'ONLINE' });
        if (wsReconnectTimeout) {
          clearTimeout(wsReconnectTimeout);
          wsReconnectTimeout = null;
        }
      };

      wsClient.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'STATUS') {
            set({
              backendConnected: true,
              systemStatus: msg.data.status,
              threshold: msg.data.calibrated_threshold,
              activeModel: msg.data.active_model,
              receiverMode: msg.data.receiver_mode,
            });
          } else if (msg.type === 'SNAPSHOT') {
            const snapData = msg.data;
            const activeCount = snapData.stats?.active_anomalies ?? snapData.aircraft.filter(a => a.isAnomaly).length;
            const meanProb = snapData.stats?.mean_anomaly_prob ?? 0.05;

            set((s) => ({
              aircraft: snapData.aircraft || [],
              graphEdges: snapData.graph_edges || [],
              currentTimestamp: snapData.timestamp,
              timestepIndex: snapData.timestep_index,
              tickCount: s.tickCount + 1,
              metrics: {
                totalTracked: snapData.stats?.total_tracked ?? snapData.aircraft.length,
                modelConfidence: Math.round((1 - meanProb) * 1000) / 10,
                activeAnomalies: activeCount,
                signalIntegrity: 95.8,
              },
            }));

            // Sync alerts from backend
            get().fetchAlerts();

            // Refresh explanation if aircraft selected
            const sel = get().selectedAircraft;
            if (sel) {
              get().fetchExplanation(sel);
            }
          } else if (msg.type === 'SIMULATION_ATTACK_BURST') {
            set({ simulationAttackActive: msg });
            if (msg.predictions && msg.predictions.length > 0) {
              set((s) => ({
                aircraft: msg.predictions,
                metrics: {
                  ...s.metrics,
                  totalTracked: msg.predictions.length,
                  activeAnomalies: msg.predictions.filter(p => p.isAnomaly || p.is_anomaly).length,
                }
              }));
            }
            get().fetchAlerts();
            if (msg.target_icao24) {
              get().fetchExplanation(msg.target_icao24);
            }
          }
        } catch {}
      };

      wsClient.onclose = () => {
        set({ backendConnected: false, systemStatus: 'BACKEND_OFFLINE' });
        wsClient = null;
        // Attempt reconnection after 3 seconds
        if (!wsReconnectTimeout) {
          wsReconnectTimeout = setTimeout(() => {
            wsReconnectTimeout = null;
            get().initWebSocket();
          }, 3000);
        }
      };

      wsClient.onerror = () => {
        set({ backendConnected: false, systemStatus: 'BACKEND_OFFLINE' });
      };
    } catch {
      set({ backendConnected: false, systemStatus: 'BACKEND_OFFLINE' });
    }
  },

  // Fetch system status, snapshot, alerts, and receiver info
  fetchInitialData: async () => {
    // Start WebSocket
    get().initWebSocket();

    try {
      const statusRes = await fetch('/api/status');
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        set({
          backendConnected: true,
          systemStatus: statusData.status,
          threshold: statusData.calibrated_threshold,
          activeModel: statusData.active_model,
          receiverMode: statusData.receiver_mode,
        });

        // Telemetry snapshot
        const snapRes = await fetch('/api/telemetry/snapshot');
        if (snapRes.ok) {
          const snapData = await snapRes.json();
          const activeCount = snapData.stats?.active_anomalies ?? snapData.aircraft.filter(a => a.isAnomaly).length;
          const meanProb = snapData.stats?.mean_anomaly_prob ?? 0.05;

          set({
            aircraft: snapData.aircraft || [],
            graphEdges: snapData.graph_edges || [],
            currentTimestamp: snapData.timestamp,
            timestepIndex: snapData.timestep_index,
            metrics: {
              totalTracked: snapData.stats?.total_tracked ?? snapData.aircraft.length,
              modelConfidence: Math.round((1 - meanProb) * 1000) / 10,
              activeAnomalies: activeCount,
              signalIntegrity: 95.8,
            },
          });
        }

        // Alerts & receiver status
        get().fetchAlerts();
        get().fetchReceiverStatus();
      } else {
        set({ backendConnected: false, systemStatus: 'BACKEND_OFFLINE', aircraft: [] });
      }
    } catch {
      set({ backendConnected: false, systemStatus: 'BACKEND_OFFLINE', aircraft: [] });
    }
  },

  fetchAlerts: async () => {
    try {
      const alertRes = await fetch('/api/alerts');
      if (alertRes.ok) {
        const alerts = await alertRes.json();
        set({ alerts });
      }
    } catch {}
  },

  fetchReceiverStatus: async () => {
    try {
      const res = await fetch('/api/receiver/status');
      if (res.ok) {
        const data = await res.json();
        set({ receiverStatus: data });
      }
    } catch {}
  },

  triggerSyntheticBurst: async () => {
    set({ burstDecoding: true });
    try {
      const res = await fetch('/api/receiver/decode_synthetic', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        set({
          recentDecodedFrames: data.messages || [],
          burstDecoding: false,
        });
        if (data.predictions && data.predictions.length > 0) {
          set({ aircraft: data.predictions });
        }
        get().fetchReceiverStatus();
        get().fetchAlerts();
      } else {
        set({ burstDecoding: false });
      }
    } catch {
      set({ burstDecoding: false });
    }
  },

  triggerAttackSimulation: async (attackType) => {
    try {
      const res = await fetch('/api/simulation/attack_burst', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attack_type: attackType }),
      });
      if (res.ok) {
        const data = await res.json();
        set({ simulationAttackActive: data });
        if (data.predictions && data.predictions.length > 0) {
          set((s) => ({
            aircraft: data.predictions,
            metrics: {
              ...s.metrics,
              totalTracked: data.predictions.length,
              activeAnomalies: data.predictions.filter(p => p.isAnomaly || p.is_anomaly).length,
            }
          }));
        }
        get().fetchAlerts();
        if (data.target_icao24) {
          get().fetchExplanation(data.target_icao24);
        }
      }
    } catch {}
  },

  // On-demand XAI Explanation fetcher with cache
  fetchExplanation: async (icao24) => {
    const { explanationCache, currentTimestamp } = get();
    const cacheKey = `${icao24}_${currentTimestamp}`;

    if (explanationCache[cacheKey]) {
      set({
        activeExplanation: explanationCache[cacheKey],
        explanationLoading: false,
        explanationError: null,
      });
      return;
    }

    if ((!currentTimestamp || currentTimestamp === 'Waiting for stream...') && explanationCache[icao24]) {
      set({
        activeExplanation: explanationCache[icao24],
        explanationLoading: false,
        explanationError: null,
      });
      return;
    }

    set({ explanationLoading: true, explanationError: null });

    try {
      const res = await fetch(`/api/xai/explain/${icao24}?timestamp=${encodeURIComponent(currentTimestamp)}`);
      if (res.ok) {
        const dossier = await res.json();
        set((s) => ({
          activeExplanation: dossier,
          explanationLoading: false,
          explanationError: null,
          explanationCache: {
            ...s.explanationCache,
            [cacheKey]: dossier,
            [icao24]: dossier,
          },
        }));
      } else {
        const errJson = await res.json().catch(() => ({}));
        set({
          explanationLoading: false,
          explanationError: errJson.error || 'Explanation unavailable for this observation context.',
        });
      }
    } catch (err) {
      set({
        explanationLoading: false,
        explanationError: 'Unable to reach backend XAI service. Ensure AeroGuard server is running.',
      });
    }
  },

  // Real-time or Replay Tick
  tick: async () => {
    const { backendConnected } = get();

    // If WebSocket is active, send tick request over socket
    if (wsClient && wsClient.readyState === WebSocket.OPEN) {
      wsClient.send(JSON.stringify({ action: 'tick', step: 1 }));
      return;
    }

    // Fallback to HTTP POST /api/telemetry/tick
    if (backendConnected) {
      try {
        const tickRes = await fetch('/api/telemetry/tick', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ step: 1 }),
        });
        if (tickRes.ok) {
          const snapData = await tickRes.json();
          const activeCount = snapData.stats?.active_anomalies ?? snapData.aircraft.filter(a => a.isAnomaly).length;
          const meanProb = snapData.stats?.mean_anomaly_prob ?? 0.05;

          set((s) => ({
            aircraft: snapData.aircraft || [],
            graphEdges: snapData.graph_edges || [],
            currentTimestamp: snapData.timestamp,
            timestepIndex: snapData.timestep_index,
            tickCount: s.tickCount + 1,
            metrics: {
              totalTracked: snapData.stats?.total_tracked ?? snapData.aircraft.length,
              modelConfidence: Math.round((1 - meanProb) * 1000) / 10,
              activeAnomalies: activeCount,
              signalIntegrity: 95.8,
            },
          }));

          // Fetch alerts for updated timestamp
          get().fetchAlerts();

          // If an aircraft is currently selected, refresh its explanation
          const sel = get().selectedAircraft;
          if (sel) {
            get().fetchExplanation(sel);
          }
        } else {
          set({ backendConnected: false, systemStatus: 'BACKEND_OFFLINE' });
        }
      } catch {
        set({ backendConnected: false, systemStatus: 'BACKEND_OFFLINE' });
      }
    }
  },
}));

export default useAeroStore;
