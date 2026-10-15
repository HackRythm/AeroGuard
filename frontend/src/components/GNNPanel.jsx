import { useState, useMemo } from 'react';
import { Network, Layers, GitBranch, Cpu, ShieldCheck } from 'lucide-react';
import useAeroStore from '../store/useAeroStore';

// Circular fallback layout positions for up to 16 nodes
const DEFAULT_POSITIONS = [
  { x: 95, y: 30 },
  { x: 135, y: 40 },
  { x: 160, y: 75 },
  { x: 145, y: 115 },
  { x: 105, y: 135 },
  { x: 65, y: 130 },
  { x: 35, y: 100 },
  { x: 30, y: 60 },
  { x: 60, y: 35 },
  { x: 95, y: 80 },
  { x: 125, y: 80 },
  { x: 70, y: 80 },
];

export default function GNNPanel() {
  const [viewMode, setViewMode] = useState('graph'); // 'graph' | 'architecture'

  const aircraft = useAeroStore((s) => s.aircraft);
  const graphEdges = useAeroStore((s) => s.graphEdges);
  const selectedAircraft = useAeroStore((s) => s.selectedAircraft);
  const selectAircraft = useAeroStore((s) => s.selectAircraft);
  const activeExplanation = useAeroStore((s) => s.activeExplanation);
  const threshold = useAeroStore((s) => s.threshold);
  const activeModel = useAeroStore((s) => s.activeModel);

  // Position nodes based on relative lat/lon or circular fallback
  const nodes = useMemo(() => {
    if (aircraft.length === 0) return [];

    const lats = aircraft
      .map((a) => a.lat ?? a.latitude)
      .filter((v) => v != null && !isNaN(v));
    const lons = aircraft
      .map((a) => a.lon ?? a.longitude)
      .filter((v) => v != null && !isNaN(v));

    const minLat = lats.length > 0 ? Math.min(...lats) : 0;
    const maxLat = lats.length > 0 ? Math.max(...lats) : 1;
    const minLon = lons.length > 0 ? Math.min(...lons) : 0;
    const maxLon = lons.length > 0 ? Math.max(...lons) : 1;

    const latSpan = maxLat - minLat || 0.1;
    const lonSpan = maxLon - minLon || 0.1;

    return aircraft.map((ac, i) => {
      const lat = ac.lat ?? ac.latitude ?? 0;
      const lon = ac.lon ?? ac.longitude ?? 0;
      const isAnomaly = Boolean(ac.isAnomaly ?? ac.is_anomaly);

      const x = Math.round(25 + ((lon - minLon) / lonSpan) * 140);
      const y = Math.round(135 - ((lat - minLat) / latSpan) * 110);

      const defaultPos = DEFAULT_POSITIONS[i % DEFAULT_POSITIONS.length];
      const finalX = isNaN(x) || x < 15 || x > 175 ? defaultPos.x : x;
      const finalY = isNaN(y) || y < 15 || y > 140 ? defaultPos.y : y;

      return {
        ...ac,
        id: String(ac.id || ac.icao24),
        icao24: String(ac.icao24 || ac.id),
        callsign: ac.callsign || 'FLIGHT',
        isAnomaly,
        pos: { x: finalX, y: finalY },
      };
    });
  }, [aircraft]);

  const nodeMap = useMemo(() => {
    const map = new Map();
    nodes.forEach((n) => {
      map.set(String(n.id), n);
      map.set(String(n.icao24), n);
    });
    return map;
  }, [nodes]);

  const criticalNeighborIds = useMemo(() => {
    if (!activeExplanation?.spatial_explanation?.critical_edges) return new Set();
    return new Set(
      activeExplanation.spatial_explanation.critical_edges.map((e) => String(e.neighbor_icao24))
    );
  }, [activeExplanation]);

  const renderedEdges = useMemo(() => {
    if (graphEdges.length > 0) {
      return graphEdges
        .map((e, idx) => {
          const a = nodeMap.get(String(e.from));
          const b = nodeMap.get(String(e.to));
          if (!a || !b) return null;

          const isSelectedIncident =
            a.id === selectedAircraft ||
            a.icao24 === selectedAircraft ||
            b.id === selectedAircraft ||
            b.icao24 === selectedAircraft;

          const isCriticalXAI =
            isSelectedIncident &&
            (criticalNeighborIds.has(String(a.icao24)) || criticalNeighborIds.has(String(b.icao24)));

          return {
            key: `ge_${idx}`,
            x1: a.pos.x,
            y1: a.pos.y,
            x2: b.pos.x,
            y2: b.pos.y,
            isSelectedIncident,
            isCriticalXAI,
            isAnomalyEdge: e.is_anomaly_edge || a.isAnomaly || b.isAnomaly,
            dist: e.distance_km,
          };
        })
        .filter(Boolean);
    }

    // Default proximity fallback mesh
    const edges = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < Math.min(i + 3, nodes.length); j++) {
        const a = nodes[i],
          b = nodes[j];
        edges.push({
          key: `fe_${i}_${j}`,
          x1: a.pos.x,
          y1: a.pos.y,
          x2: b.pos.x,
          y2: b.pos.y,
          isSelectedIncident: a.id === selectedAircraft || b.id === selectedAircraft,
          isCriticalXAI: false,
          isAnomalyEdge: a.isAnomaly || b.isAnomaly,
          dist: null,
        });
      }
    }
    return edges;
  }, [graphEdges, nodeMap, selectedAircraft, criticalNeighborIds, nodes]);

  return (
    <div className="glass-panel rounded-lg p-3.5 flex flex-col h-[280px] animate-fade-in border border-outline-variant/60">
      {/* Header with View Toggle */}
      <div className="flex items-center justify-between mb-2.5 border-b border-outline-variant/60 pb-2">
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setViewMode('graph')}
            className={`px-2 py-0.5 rounded text-[11px] font-mono flex items-center gap-1.5 transition-all ${
              viewMode === 'graph'
                ? 'bg-primary/20 text-primary border border-primary/40 font-semibold'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <Network className="w-3 h-3" />
            Graph Snapshot
          </button>
          <button
            onClick={() => setViewMode('architecture')}
            className={`px-2 py-0.5 rounded text-[11px] font-mono flex items-center gap-1.5 transition-all ${
              viewMode === 'architecture'
                ? 'bg-primary/20 text-primary border border-primary/40 font-semibold'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <Cpu className="w-3 h-3" />
            ST-GNN Architecture
          </button>
        </div>
        <span className="text-mono-data text-on-surface-variant text-[10px]">
          t={threshold.toFixed(2)}
        </span>
      </div>

      {/* VIEW 1: DYNAMIC RELATIONAL GRAPH */}
      {viewMode === 'graph' && (
        <div className="flex-1 relative w-full bg-surface-container-lowest rounded border border-outline-variant/50 overflow-hidden flex flex-col">
          <svg width="100%" height="100%" viewBox="0 0 190 155" className="p-1">
            {/* Edges */}
            {renderedEdges.map((e) => {
              let stroke = '#424754';
              let strokeWidth = 0.8;
              let strokeDash = 'none';
              let opacity = 0.5;

              if (e.isCriticalXAI) {
                stroke = '#38bdf8';
                strokeWidth = 2.0;
                opacity = 1.0;
              } else if (e.isSelectedIncident) {
                stroke = '#adc6ff';
                strokeWidth = 1.4;
                opacity = 0.85;
              } else if (e.isAnomalyEdge) {
                stroke = '#ffb4ab';
                strokeWidth = 1.2;
                strokeDash = '3,2';
                opacity = 0.7;
              }

              return (
                <line
                  key={e.key}
                  x1={e.x1}
                  y1={e.y1}
                  x2={e.x2}
                  y2={e.y2}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                  strokeDasharray={strokeDash}
                  opacity={opacity}
                />
              );
            })}

            {/* Nodes */}
            {nodes.map((node) => {
              const isSelected = selectedAircraft === node.id || selectedAircraft === node.icao24;
              const isCriticalNeighbor = criticalNeighborIds.has(String(node.icao24));

              const nodeColor = isSelected
                ? '#38bdf8'
                : node.isAnomaly
                ? '#ffb4ab'
                : isCriticalNeighbor
                ? '#38bdf8'
                : '#adc6ff';

              const fillColor = isSelected
                ? '#0369a1'
                : node.isAnomaly
                ? '#93000a'
                : isCriticalNeighbor
                ? '#075985'
                : '#004395';

              const r = isSelected ? 6.5 : node.isAnomaly ? 5.5 : 4;

              return (
                <g
                  key={node.id}
                  onClick={() => selectAircraft(node.id)}
                  className="cursor-pointer group"
                >
                  {(node.isAnomaly || isSelected) && (
                    <circle
                      cx={node.pos.x}
                      cy={node.pos.y}
                      r={isSelected ? 10 : 8}
                      fill="none"
                      stroke={isSelected ? '#38bdf8' : '#ffb4ab'}
                      strokeWidth="0.7"
                      opacity="0.5"
                    >
                      <animate
                        attributeName="r"
                        values={isSelected ? '8;14;8' : '7;12;7'}
                        dur="2s"
                        repeatCount="indefinite"
                      />
                      <animate
                        attributeName="opacity"
                        values="0.6;0;0.6"
                        dur="2s"
                        repeatCount="indefinite"
                      />
                    </circle>
                  )}

                  <circle
                    cx={node.pos.x}
                    cy={node.pos.y}
                    r={r}
                    fill={fillColor}
                    stroke={nodeColor}
                    strokeWidth={isSelected ? 2 : node.isAnomaly ? 1.5 : 1}
                  />

                  {(isSelected || node.isAnomaly || isCriticalNeighbor) && (
                    <text
                      x={node.pos.x + 8}
                      y={node.pos.y + 3}
                      fill={nodeColor}
                      fontFamily="'JetBrains Mono', monospace"
                      fontSize={isSelected ? '7' : '6'}
                      fontWeight={isSelected ? 'bold' : '500'}
                    >
                      {node.callsign}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          {/* Graph footer stats */}
          <div className="absolute bottom-1 left-2 right-2 flex justify-between text-[9px] font-mono bg-surface-container-lowest/80 px-1 py-0.5 rounded">
            <span className="text-on-surface-variant">
              N={nodes.length} E={renderedEdges.length} (d ≤ 100km)
            </span>
            <span className="text-error font-semibold">
              Δ={nodes.filter((n) => n.isAnomaly).length} anomalies
            </span>
          </div>
        </div>
      )}

      {/* VIEW 2: VERIFIED 3-STREAM ST-GNN ARCHITECTURE */}
      {viewMode === 'architecture' && (
        <div className="flex-1 bg-surface-container-lowest rounded border border-outline-variant/50 p-2.5 flex flex-col justify-between overflow-y-auto font-mono text-[10px]">
          {/* Stream 1 */}
          <div className="p-1.5 rounded bg-surface-container border border-outline-variant/60 flex items-center justify-between">
            <div>
              <span className="text-primary font-bold">STREAM 1: SPATIAL GNN</span>
              <div className="text-[9px] text-on-surface-variant">
                12D Node Feats + 4D Edge Attr (100 km GATv2Conv)
              </div>
            </div>
            <span className="bg-primary/10 text-primary px-1.5 py-0.5 rounded text-[9px] font-bold">
              h_s: 16D
            </span>
          </div>

          {/* Stream 2 */}
          <div className="p-1.5 rounded bg-surface-container border border-outline-variant/60 flex items-center justify-between">
            <div>
              <span className="text-cyan-400 font-bold">STREAM 2: TEMPORAL GRU</span>
              <div className="text-[9px] text-on-surface-variant">
                T=5 History Window · 25D [h_s(16) || kin(9)]
              </div>
            </div>
            <span className="bg-cyan-500/10 text-cyan-400 px-1.5 py-0.5 rounded text-[9px] font-bold">
              h_t: 32D
            </span>
          </div>

          {/* Stream 3 */}
          <div className="p-1.5 rounded bg-surface-container border border-outline-variant/60 flex items-center justify-between">
            <div>
              <span className="text-amber-400 font-bold">STREAM 3: KINEMATIC MLP</span>
              <div className="text-[9px] text-on-surface-variant">
                11 Causal Residuals (disp_ratio, alt_res, accel)
              </div>
            </div>
            <span className="bg-amber-500/10 text-amber-400 px-1.5 py-0.5 rounded text-[9px] font-bold">
              h_k: 16D
            </span>
          </div>

          {/* Fusion Footer */}
          <div className="pt-1.5 border-t border-outline-variant/60 flex items-center justify-between text-[9px]">
            <span className="text-on-surface font-semibold">
              FUSION: Weighted Prob (w=[0.33, 0.34, 0.33])
            </span>
            <span className="text-emerald-400 bg-emerald-500/15 px-1.5 py-0.5 rounded font-bold">
              Threshold t=0.20
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
