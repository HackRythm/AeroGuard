import { useMemo } from 'react';
import { Radio, AlertTriangle, ShieldCheck, RefreshCw } from 'lucide-react';
import useAeroStore from '../store/useAeroStore';

export default function AirspaceMap() {
  const aircraft = useAeroStore((s) => s.aircraft);
  const selectedAircraft = useAeroStore((s) => s.selectedAircraft);
  const selectAircraft = useAeroStore((s) => s.selectAircraft);
  const backendConnected = useAeroStore((s) => s.backendConnected);
  const systemStatus = useAeroStore((s) => s.systemStatus);
  const receiverStatus = useAeroStore((s) => s.receiverStatus);
  const triggerSyntheticBurst = useAeroStore((s) => s.triggerSyntheticBurst);
  const simulationAttackActive = useAeroStore((s) => s.simulationAttackActive);

  // Dynamic geographic bounds centered around real observations
  const bounds = useMemo(() => {
    const lats = aircraft
      .map((a) => a.lat ?? a.latitude)
      .filter((v) => v != null && !isNaN(v));
    const lons = aircraft
      .map((a) => a.lon ?? a.longitude)
      .filter((v) => v != null && !isNaN(v));

    if (lats.length === 0 || lons.length === 0) {
      return {
        latMin: 52.1,
        latMax: 52.6,
        lonMin: -9.2,
        lonMax: -8.6,
        centerLat: 52.33,
        centerLon: -8.92,
      };
    }

    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);

    const spanLat = maxLat - minLat;
    const spanLon = maxLon - minLon;
    const padLat = Math.max(spanLat * 0.28, 0.06);
    const padLon = Math.max(spanLon * 0.28, 0.08);

    return {
      latMin: minLat - padLat,
      latMax: maxLat + padLat,
      lonMin: minLon - padLon,
      lonMax: maxLon + padLon,
      centerLat: (minLat + maxLat) / 2,
      centerLon: (minLon + maxLon) / 2,
    };
  }, [aircraft]);

  // Project lat/lon to SVG viewBox coordinates [0, 1000] x [0, 800]
  const project = useMemo(() => {
    const { latMin, latMax, lonMin, lonMax } = bounds;
    const latSpan = latMax - latMin || 0.1;
    const lonSpan = lonMax - lonMin || 0.1;

    return (lat, lon) => {
      const x = ((lon - lonMin) / lonSpan) * 1000;
      const y = 800 - ((lat - latMin) / latSpan) * 800;
      return {
        x: Math.round(Math.max(20, Math.min(980, x))),
        y: Math.round(Math.max(20, Math.min(780, y))),
      };
    };
  }, [bounds]);

  // Projected aircraft coordinates
  const projectedAircraft = useMemo(() => {
    return aircraft.map((ac) => {
      const lat = ac.lat ?? ac.latitude ?? bounds.centerLat;
      const lon = ac.lon ?? ac.longitude ?? bounds.centerLon;
      const isAnomaly = Boolean(ac.isAnomaly ?? ac.is_anomaly);
      const pos = project(lat, lon);

      // Trajectory heading tail
      const headingRad = (((ac.heading ?? 0) + 180) * Math.PI) / 180;
      const t1 = project(lat + Math.cos(headingRad) * 0.008, lon + Math.sin(headingRad) * 0.008);
      const t2 = project(lat + Math.cos(headingRad) * 0.016, lon + Math.sin(headingRad) * 0.016);

      return {
        ...ac,
        id: String(ac.id || ac.icao24),
        icao24: String(ac.icao24 || ac.id),
        callsign: ac.callsign || 'FLIGHT',
        altitude: ac.altitude ?? 0,
        groundspeed: ac.groundspeed ?? 0,
        heading: ac.heading ?? 0,
        isAnomaly,
        anomalyType: isAnomaly
          ? ((!ac.anomalyType && !ac.anomaly_type) || String(ac.anomalyType || ac.anomaly_type).toLowerCase() === 'normal'
              ? 'TRAJECTORY ANOMALY'
              : String(ac.anomalyType || ac.anomaly_type))
          : (String(ac.anomalyType || ac.anomaly_type || 'NORMAL')),
        signalSource: ac.signal_source || (ac.rf_observed ? 'SYNTHETIC_SDR' : 'REPLAY_TELEMETRY'),
        sx: pos.x,
        sy: pos.y,
        tail: [t2, t1],
      };
    });
  }, [aircraft, project, bounds]);

  // Determine current active source badge
  const sourceLabel = useMemo(() => {
    if (simulationAttackActive) {
      return `SIMULATION BURST: ${simulationAttackActive.attack_type?.toUpperCase().replace('_', ' ')}`;
    }
    const hasSDR = aircraft.some((a) => a.signal_source === 'SYNTHETIC_SDR' || a.rf_observed);
    if (hasSDR) return 'SOURCE: SYNTHETIC 1090 MHz SDR';
    return 'SOURCE: CHRONOLOGICAL REPLAY';
  }, [aircraft, simulationAttackActive]);

  return (
    <section className="flex-1 relative rounded-lg border border-outline-variant overflow-hidden bg-surface-container-lowest grid-pattern flex flex-col">
      {/* Status overlay - top left */}
      <div className="absolute top-3 left-3 flex flex-wrap gap-2 z-10">
        <div className="glass-panel px-3 py-1.5 rounded flex items-center gap-2 border border-outline-variant/60">
          <div
            className={`w-2 h-2 rounded-full ${
              backendConnected ? 'bg-emerald-400 glow-primary' : 'bg-amber-400 animate-pulse'
            }`}
          />
          <span className="text-mono-data text-on-surface text-[11px] font-semibold">
            SYS: {backendConnected ? 'ONLINE' : 'BACKEND OFFLINE'}
          </span>
        </div>

        <div className="glass-panel px-3 py-1.5 rounded border border-outline-variant/60">
          <span className="text-mono-data text-on-surface text-[11px]">
            CTR: {bounds.centerLat.toFixed(2)}°N {Math.abs(bounds.centerLon).toFixed(2)}°W
          </span>
        </div>

        <div className="glass-panel px-3 py-1.5 rounded border border-outline-variant/60">
          <span className="text-mono-data text-primary text-[11px] tracking-wider uppercase font-medium">
            {sourceLabel}
          </span>
        </div>
      </div>

      {/* SVG Radar Canvas */}
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 1000 800"
        preserveAspectRatio="xMidYMid slice"
      >
        <defs>
          <linearGradient id="radarSweepGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(56, 189, 248, 0.18)" />
            <stop offset="100%" stopColor="rgba(56, 189, 248, 0)" />
          </linearGradient>
          <filter id="aircraftGlow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Range Circles centered on radar display */}
        <g fill="none" stroke="rgba(66,71,84,0.18)" strokeWidth="0.8" strokeDasharray="4,4">
          <circle cx="500" cy="400" r="140" />
          <circle cx="500" cy="400" r="280" />
          <circle cx="500" cy="400" r="390" />
        </g>

        {/* Tactical Crosshairs */}
        <g stroke="rgba(66,71,84,0.22)" strokeWidth="0.6">
          <line x1="500" y1="20" x2="500" y2="780" />
          <line x1="20" y1="400" x2="980" y2="400" />
          <line x1="500" y1="400" x2="700" y2="200" strokeDasharray="2,4" />
          <line x1="500" y1="400" x2="300" y2="200" strokeDasharray="2,4" />
        </g>

        {/* Range Labels */}
        <text x="508" y="270" fill="rgba(173,198,255,0.4)" fontSize="9" fontFamily="monospace">
          50 NM
        </text>
        <text x="508" y="130" fill="rgba(173,198,255,0.4)" fontSize="9" fontFamily="monospace">
          100 NM
        </text>

        {/* Active Aircraft Layer */}
        {projectedAircraft.map((ac) => {
          const isSelected = selectedAircraft === ac.id || selectedAircraft === ac.icao24;
          const color = ac.isAnomaly ? '#ffb4ab' : '#adc6ff';
          const trailColor = ac.isAnomaly ? 'rgba(255,180,171,0.7)' : 'rgba(173,198,255,0.45)';

          const tailPoints = [...ac.tail, { x: ac.sx, y: ac.sy }];
          const pathD = tailPoints.map((p, i) => (i === 0 ? `M${p.x},${p.y}` : `L${p.x},${p.y}`)).join(' ');

          return (
            <g
              key={ac.id}
              onClick={() => selectAircraft(ac.id)}
              className="cursor-pointer group"
            >
              {/* Heading trajectory tail */}
              <path
                d={pathD}
                fill="none"
                stroke={trailColor}
                strokeWidth={ac.isAnomaly ? 2 : 1.4}
                strokeDasharray={ac.isAnomaly ? '5,3' : 'none'}
              />

              {/* Anomaly warning halo */}
              {ac.isAnomaly && (
                <circle
                  cx={ac.sx}
                  cy={ac.sy}
                  r="20"
                  fill="rgba(255,180,171,0.06)"
                  stroke="rgba(255,180,171,0.4)"
                  strokeWidth="1"
                >
                  <animate attributeName="r" values="14;24;14" dur="2.4s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.7;0.1;0.7" dur="2.4s" repeatCount="indefinite" />
                </circle>
              )}

              {/* Aircraft Icon (rotated triangle with compass heading) */}
              <g
                transform={`translate(${ac.sx}, ${ac.sy}) rotate(${ac.heading})`}
                filter={isSelected ? 'url(#aircraftGlow)' : undefined}
              >
                <polygon
                  points="0,-11 -6,7 0,3 6,7"
                  fill={color}
                  opacity={isSelected ? 1 : 0.9}
                />
              </g>

              {/* Callsign & Altitude Label */}
              <text
                x={ac.sx + 14}
                y={ac.sy - 5}
                fill={color}
                fontFamily="'JetBrains Mono', monospace"
                fontSize="10"
                fontWeight={ac.isAnomaly ? 'bold' : '500'}
              >
                {ac.callsign}
              </text>

              <text
                x={ac.sx + 14}
                y={ac.sy + 7}
                fill={color}
                fontFamily="'JetBrains Mono', monospace"
                fontSize="8"
                opacity="0.8"
              >
                FL{Math.round(ac.altitude / 100)} · {Math.round(ac.groundspeed)} kts
              </text>

              {/* Threat type tag */}
              {ac.isAnomaly && (
                <g>
                  <rect
                    x={ac.sx + 14}
                    y={ac.sy + 12}
                    width={String(ac.anomalyType).length * 5.4 + 10}
                    height="13"
                    rx="2"
                    fill="rgba(147,0,10,0.75)"
                    stroke="#ffb4ab"
                    strokeWidth="0.5"
                  />
                  <text
                    x={ac.sx + 19}
                    y={ac.sy + 22}
                    fill="#ffb4ab"
                    fontFamily="'JetBrains Mono', monospace"
                    fontSize="7.5"
                    fontWeight="bold"
                  >
                    {String(ac.anomalyType).toUpperCase().replace('_', ' ')}
                  </text>
                </g>
              )}

              {/* Selection Reticle */}
              {isSelected && (
                <circle
                  cx={ac.sx}
                  cy={ac.sy}
                  r="18"
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="1.5"
                  strokeDasharray="4,2"
                >
                  <animateTransform
                    attributeName="transform"
                    type="rotate"
                    from={`0 ${ac.sx} ${ac.sy}`}
                    to={`360 ${ac.sx} ${ac.sy}`}
                    dur="4s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
            </g>
          );
        })}

        {/* Continuous Radar Sweep line */}
        <g className="radar-line" transform="translate(500, 400)">
          <line x1="0" y1="0" x2="0" y2="-400" stroke="rgba(56,189,248,0.3)" strokeWidth="1.5" />
          <path d="M0,0 L0,-400 A400,400 0 0,1 120,-380 Z" fill="url(#radarSweepGrad)" />
        </g>
      </svg>

      {/* Empty State Overlay when 0 aircraft active */}
      {projectedAircraft.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface-container-lowest/70 backdrop-blur-sm z-20 p-6 text-center animate-fade-in">
          <div className="w-12 h-12 rounded-full border border-primary/40 bg-primary/10 flex items-center justify-center mb-3">
            <Radio className="w-6 h-6 text-primary animate-pulse" />
          </div>
          <h3 className="text-title-sm text-on-surface font-semibold mb-1">
            {backendConnected ? 'Awaiting Airspace Telemetry' : 'AeroGuard Server Offline'}
          </h3>
          <p className="text-body-sm text-on-surface-variant max-w-sm mb-4 font-mono text-[11px]">
            {backendConnected
              ? 'No active tracks in buffer. Click "Test SDR Burst" in the top bar to synthesize 1090 MHz Mode-S pulses or start playback.'
              : 'The backend server is not responding on http://127.0.0.1:8000. Launch `scripts/aeroguard_server.py` to connect.'}
          </p>

          {backendConnected && (
            <button
              onClick={triggerSyntheticBurst}
              className="px-4 py-2 rounded bg-primary/20 border border-primary/50 text-primary text-mono-label hover:bg-primary/30 transition-all flex items-center gap-2"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Trigger 1090 MHz SDR Burst
            </button>
          )}
        </div>
      )}

      {/* Tactical Legend - bottom right */}
      <div className="absolute bottom-3 right-3 glass-panel px-3.5 py-1.5 rounded flex items-center gap-4 border border-outline-variant/60 text-[11px] font-mono z-10">
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 bg-primary/30 border border-primary rounded-sm" />
          <span className="text-on-surface-variant">NOMINAL</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 bg-error/30 border border-error rounded-sm anomaly-pulse" />
          <span className="text-error font-medium">ST-GNN ANOMALY (t ≥ 0.20)</span>
        </div>
      </div>
    </section>
  );
}
