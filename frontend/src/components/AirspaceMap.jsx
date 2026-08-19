import { useMemo } from 'react';
import useAeroStore from '../store/useAeroStore';

// Project lat/lon to SVG viewBox coordinates
function project(lat, lon, viewBox = { x: 0, y: 0, w: 1000, h: 800 }) {
  const latMin = 38.2, latMax = 39.6;
  const lonMin = -9.9, lonMax = -8.5;
  const x = ((lon - lonMin) / (lonMax - lonMin)) * viewBox.w;
  const y = viewBox.h - ((lat - latMin) / (latMax - latMin)) * viewBox.h;
  return { x: Math.round(x), y: Math.round(y) };
}

// Generate a trajectory tail (3 past positions simulated)
function getTail(ac) {
  const headRad = ((ac.heading + 180) * Math.PI) / 180;
  const step = 0.04;
  const points = [];
  for (let i = 1; i <= 3; i++) {
    const jitter = ac.isAnomaly ? (Math.sin(i * 2.3) * 0.015) : 0;
    points.push({
      lat: ac.lat + Math.cos(headRad) * step * i + jitter,
      lon: ac.lon + Math.sin(headRad) * step * i + jitter * 1.5,
    });
  }
  return points;
}

export default function AirspaceMap() {
  const aircraft = useAeroStore((s) => s.aircraft);
  const selectedAircraft = useAeroStore((s) => s.selectedAircraft);
  const selectAircraft = useAeroStore((s) => s.selectAircraft);

  const projected = useMemo(() => {
    return aircraft.map((ac) => {
      const pos = project(ac.lat, ac.lon);
      const tail = getTail(ac).map((t) => project(t.lat, t.lon));
      return { ...ac, sx: pos.x, sy: pos.y, tail };
    });
  }, [aircraft]);

  return (
    <section className="flex-1 relative rounded-lg border border-outline-variant overflow-hidden bg-surface-container-lowest grid-pattern">
      {/* Status overlay - top left */}
      <div className="absolute top-4 left-4 flex gap-2 z-10">
        <div className="glass-panel px-3 py-1.5 rounded flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-primary glow-primary" />
          <span className="text-mono-data text-on-surface">SYS: ONLINE</span>
        </div>
        <div className="glass-panel px-3 py-1.5 rounded">
          <span className="text-mono-data text-on-surface">
            LPPC FIR — 38.78°N 9.14°W
          </span>
        </div>
      </div>

      {/* SVG Canvas */}
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 1000 800"
        preserveAspectRatio="xMidYMid slice"
      >
        <defs>
          <linearGradient id="radarGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(173,198,255,0.12)" />
            <stop offset="100%" stopColor="rgba(173,198,255,0)" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Grid lines */}
        <g stroke="rgba(66,71,84,0.2)" strokeWidth="0.5">
          {[160, 320, 480, 640].map((y) => (
            <line key={`h${y}`} x1="0" x2="1000" y1={y} y2={y} />
          ))}
          {[200, 400, 600, 800].map((x) => (
            <line key={`v${x}`} x1={x} x2={x} y1="0" y2="800" />
          ))}
        </g>

        {/* Range circles */}
        <g fill="none" stroke="rgba(66,71,84,0.15)" strokeWidth="0.5" strokeDasharray="4,4">
          <circle cx="500" cy="400" r="150" />
          <circle cx="500" cy="400" r="300" />
        </g>

        {/* Aircraft trajectories + icons */}
        {projected.map((ac) => {
          const isSelected = selectedAircraft === ac.id;
          const color = ac.isAnomaly ? '#ffb4ab' : '#adc6ff';
          const trailColor = ac.isAnomaly
            ? 'rgba(255,180,171,0.7)'
            : 'rgba(173,198,255,0.4)';

          // Build path from tail points to current position
          const pathPoints = [...ac.tail.reverse(), { x: ac.sx, y: ac.sy }];
          const pathD = pathPoints
            .map((p, i) => (i === 0 ? `M${p.x},${p.y}` : `L${p.x},${p.y}`))
            .join(' ');

          return (
            <g
              key={ac.id}
              onClick={() => selectAircraft(ac.id)}
              className="cursor-pointer"
            >
              {/* Trajectory trail */}
              <path
                d={pathD}
                fill="none"
                stroke={trailColor}
                strokeWidth={ac.isAnomaly ? 2 : 1.5}
                strokeDasharray={ac.isAnomaly ? '6,3' : 'none'}
              />

              {/* Anomaly pulse ring */}
              {ac.isAnomaly && (
                <>
                  <circle
                    cx={ac.sx}
                    cy={ac.sy}
                    r="20"
                    fill="rgba(255,180,171,0.08)"
                    stroke="rgba(255,180,171,0.3)"
                    strokeWidth="1"
                  >
                    <animate
                      attributeName="r"
                      values="14;22;14"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="0.6;0.1;0.6"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                  </circle>
                </>
              )}

              {/* Aircraft icon (rotated triangle) */}
              <g
                transform={`translate(${ac.sx}, ${ac.sy}) rotate(${ac.heading})`}
                filter={isSelected ? 'url(#glow)' : undefined}
              >
                <polygon
                  points="0,-10 -5,6 0,3 5,6"
                  fill={color}
                  opacity={isSelected ? 1 : 0.85}
                />
              </g>

              {/* Label */}
              <text
                x={ac.sx + 12}
                y={ac.sy - 6}
                fill={color}
                fontFamily="'JetBrains Mono', monospace"
                fontSize="10"
                fontWeight={ac.isAnomaly ? 'bold' : 'normal'}
              >
                {ac.callsign}
              </text>
              <text
                x={ac.sx + 12}
                y={ac.sy + 6}
                fill={color}
                fontFamily="'JetBrains Mono', monospace"
                fontSize="8"
                opacity="0.7"
              >
                FL{Math.round(ac.altitude / 100)}
              </text>

              {/* Anomaly type badge */}
              {ac.isAnomaly && (
                <g>
                  <rect
                    x={ac.sx + 10}
                    y={ac.sy + 10}
                    width={ac.anomalyType.length * 5.5 + 10}
                    height="14"
                    rx="2"
                    fill="rgba(147,0,10,0.6)"
                    stroke="#ffb4ab"
                    strokeWidth="0.5"
                  />
                  <text
                    x={ac.sx + 15}
                    y={ac.sy + 20}
                    fill="#ffb4ab"
                    fontFamily="'JetBrains Mono', monospace"
                    fontSize="8"
                    fontWeight="500"
                  >
                    {ac.anomalyType.toUpperCase().replace('_', ' ')}
                  </text>
                </g>
              )}

              {/* Selection ring */}
              {isSelected && (
                <circle
                  cx={ac.sx}
                  cy={ac.sy}
                  r="16"
                  fill="none"
                  stroke={color}
                  strokeWidth="1.5"
                  strokeDasharray="4,2"
                >
                  <animateTransform
                    attributeName="transform"
                    type="rotate"
                    from={`0 ${ac.sx} ${ac.sy}`}
                    to={`360 ${ac.sx} ${ac.sy}`}
                    dur="3s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
            </g>
          );
        })}

        {/* Radar sweep */}
        <g className="radar-line" transform="translate(500, 400)">
          <line
            x1="0"
            y1="0"
            x2="0"
            y2="-400"
            stroke="rgba(173,198,255,0.2)"
            strokeWidth="1.5"
          />
          <path
            d="M0,0 L0,-400 A400,400 0 0,1 104,-386 Z"
            fill="url(#radarGrad)"
          />
        </g>
      </svg>

      {/* Legend - bottom right */}
      <div className="absolute bottom-4 right-4 glass-panel px-4 py-2 rounded flex gap-5">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 border border-primary bg-primary/20" />
          <span className="text-mono-data text-on-surface-variant">
            COMMERCIAL
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 border border-error bg-error/20 anomaly-pulse" />
          <span className="text-mono-data text-error">ANOMALY</span>
        </div>
      </div>
    </section>
  );
}
