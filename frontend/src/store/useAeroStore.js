import { create } from 'zustand';
import mockFlights from '../data/mockFlights';

const initialAlerts = [
  {
    id: 'ALT-001',
    timestamp: '2026-01-01T14:23:17Z',
    aircraftId: 'AC005',
    callsign: 'UAP-099',
    anomalyType: 'position_jump',
    severity: 'critical',
    message: 'Erratic pathing detected — 12km position jump in 1s window',
    dismissed: false,
  },
  {
    id: 'ALT-002',
    timestamp: '2026-01-01T14:24:02Z',
    aircraftId: 'AC007',
    callsign: 'GHOST-7',
    anomalyType: 'ghost_aircraft',
    severity: 'critical',
    message: 'Ghost injection — no correlated Mode-S interrogation history',
    dismissed: false,
  },
  {
    id: 'ALT-003',
    timestamp: '2026-01-01T14:24:45Z',
    aircraftId: 'AC010',
    callsign: 'VEL-ERR',
    anomalyType: 'velocity_anomaly',
    severity: 'warning',
    message: 'Velocity inconsistency — 150 kts GS with +3200 fpm climb rate',
    dismissed: false,
  },
];

const useAeroStore = create((set, get) => ({
  // Aircraft state
  aircraft: mockFlights.map((f) => ({ ...f })),

  // Alert state
  alerts: [...initialAlerts],

  // Metrics (derived + static)
  metrics: {
    totalTracked: mockFlights.length,
    modelConfidence: 94.2,
    activeAnomalies: mockFlights.filter((f) => f.isAnomaly).length,
    signalIntegrity: 87.6,
  },

  // Playback state
  isPlaying: false,
  tickCount: 0,

  // UI state
  selectedAircraft: null,
  sidebarCollapsed: true,

  // ── Actions ────────────────────────────────────

  togglePlayback: () => set((s) => ({ isPlaying: !s.isPlaying })),

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  selectAircraft: (id) =>
    set((s) => ({
      selectedAircraft: s.selectedAircraft === id ? null : id,
    })),

  dismissAlert: (alertId) =>
    set((s) => ({
      alerts: s.alerts.map((a) =>
        a.id === alertId ? { ...a, dismissed: true } : a
      ),
    })),

  // Simulated real-time tick
  tick: () =>
    set((s) => {
      const newAircraft = s.aircraft.map((ac) => {
        const headingRad = (ac.heading * Math.PI) / 180;
        const speed = ac.groundspeed * 0.00015; // scaled movement per tick
        const jitter = ac.isAnomaly ? (Math.random() - 0.5) * 0.02 : 0;

        return {
          ...ac,
          lat: ac.lat + Math.cos(headingRad) * speed + jitter,
          lon: ac.lon + Math.sin(headingRad) * speed + jitter,
          doppler: ac.doppler + (Math.random() - 0.5) * 8,
          snr: Math.max(
            3,
            Math.min(30, ac.snr + (Math.random() - 0.5) * 1.2)
          ),
          rss: ac.rss + (Math.random() - 0.5) * 0.5,
        };
      });

      // Small chance to spawn a new alert during playback
      let newAlerts = [...s.alerts];
      if (Math.random() < 0.03) {
        const anomalyAC = newAircraft.filter((a) => a.isAnomaly);
        if (anomalyAC.length > 0) {
          const target =
            anomalyAC[Math.floor(Math.random() * anomalyAC.length)];
          newAlerts = [
            {
              id: `ALT-${Date.now()}`,
              timestamp: new Date().toISOString(),
              aircraftId: target.id,
              callsign: target.callsign,
              anomalyType: target.anomalyType,
              severity: Math.random() > 0.5 ? 'critical' : 'warning',
              message: `Recurring ${target.anomalyType.replace('_', ' ')} signature — tick ${s.tickCount + 1}`,
              dismissed: false,
            },
            ...newAlerts,
          ].slice(0, 20); // cap at 20
        }
      }

      return {
        aircraft: newAircraft,
        alerts: newAlerts,
        tickCount: s.tickCount + 1,
        metrics: {
          totalTracked: newAircraft.length,
          modelConfidence:
            94.2 + Math.sin(s.tickCount * 0.1) * 2.5,
          activeAnomalies: newAircraft.filter((a) => a.isAnomaly).length,
          signalIntegrity:
            87.6 + Math.cos(s.tickCount * 0.08) * 3.1,
        },
      };
    }),
}));

export default useAeroStore;
