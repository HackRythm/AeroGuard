import { useMemo } from 'react';
import useAeroStore from '../store/useAeroStore';

// Graph layout positions (manually laid out for visual clarity)
const NODE_POSITIONS = [
  { x: 50, y: 30 },
  { x: 25, y: 65 },
  { x: 75, y: 55 },
  { x: 40, y: 100 },
  { x: 100, y: 30 },
  { x: 130, y: 70 },
  { x: 155, y: 35 },
  { x: 85, y: 95 },
  { x: 140, y: 110 },
  { x: 110, y: 130 },
  { x: 60, y: 135 },
  { x: 165, y: 95 },
];

// Precomputed edges (indices into NODE_POSITIONS; connects nearby aircraft)
const EDGES = [
  [0, 1], [0, 2], [0, 4],
  [1, 2], [1, 3],
  [2, 3], [2, 4], [2, 7],
  [3, 10],
  [4, 5], [4, 6],
  [5, 6], [5, 7], [5, 11],
  [6, 11],
  [7, 8], [7, 9],
  [8, 9], [8, 11],
  [9, 10],
];

export default function GNNPanel() {
  const aircraft = useAeroStore((s) => s.aircraft);
  const tickCount = useAeroStore((s) => s.tickCount);

  const nodes = useMemo(() => {
    return aircraft.map((ac, i) => ({
      ...ac,
      pos: NODE_POSITIONS[i] || { x: 90, y: 75 },
    }));
  }, [aircraft]);

  return (
    <div className="glass-panel rounded-lg p-4 flex flex-col h-[260px] animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 border-b border-outline-variant pb-2">
        <h2 className="text-mono-label text-on-surface tracking-widest uppercase">
          Relational Graph
        </h2>
        <span className="text-mono-data text-on-surface-variant">
          GNN · L2 GCN
        </span>
      </div>

      {/* Graph visualization */}
      <div className="flex-1 relative w-full bg-surface-container-lowest rounded border border-outline-variant/50 overflow-hidden">
        <svg width="100%" height="100%" viewBox="0 0 190 155" className="p-1">
          {/* Edges */}
          {EDGES.map(([from, to], i) => {
            const a = nodes[from];
            const b = nodes[to];
            if (!a || !b) return null;

            const isAnomalyEdge = a.isAnomaly || b.isAnomaly;
            const bothAnomaly = a.isAnomaly && b.isAnomaly;

            return (
              <line
                key={`e${i}`}
                x1={a.pos.x}
                y1={a.pos.y}
                x2={b.pos.x}
                y2={b.pos.y}
                stroke={isAnomalyEdge ? '#ffb4ab' : '#424754'}
                strokeWidth={isAnomalyEdge ? 1.2 : 0.8}
                strokeDasharray={isAnomalyEdge ? '3,2' : 'none'}
                opacity={bothAnomaly ? 0.9 : 0.6}
              />
            );
          })}

          {/* Nodes */}
          {nodes.map((node, i) => {
            const nodeColor = node.isAnomaly ? '#ffb4ab' : '#adc6ff';
            const fillColor = node.isAnomaly ? '#93000a' : '#004395';
            const r = node.isAnomaly ? 6 : 4;

            return (
              <g key={node.id}>
                {/* Anomaly pulse */}
                {node.isAnomaly && (
                  <circle
                    cx={node.pos.x}
                    cy={node.pos.y}
                    r="10"
                    fill="none"
                    stroke="#ffb4ab"
                    strokeWidth="0.5"
                    opacity="0.4"
                  >
                    <animate
                      attributeName="r"
                      values="8;14;8"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="0.5;0;0.5"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                  </circle>
                )}

                {/* Node circle */}
                <circle
                  cx={node.pos.x}
                  cy={node.pos.y}
                  r={r}
                  fill={fillColor}
                  stroke={nodeColor}
                  strokeWidth={node.isAnomaly ? 1.5 : 1}
                />

                {/* Label (only for anomaly or selected) */}
                {node.isAnomaly && (
                  <text
                    x={node.pos.x + 8}
                    y={node.pos.y + 3}
                    fill={nodeColor}
                    fontFamily="'JetBrains Mono', monospace"
                    fontSize="6"
                    fontWeight="500"
                  >
                    {node.callsign}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* Graph stats overlay */}
        <div className="absolute bottom-1 left-2 right-2 flex justify-between">
          <span className="text-mono-data text-on-surface-variant opacity-70">
            N={nodes.length} E={EDGES.length}
          </span>
          <span className="text-mono-data text-error opacity-70">
            Δ={nodes.filter((n) => n.isAnomaly).length}
          </span>
        </div>
      </div>
    </div>
  );
}
