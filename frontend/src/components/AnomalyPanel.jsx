import { useState } from 'react';
import { Activity, Clock, Cpu, AlertTriangle, ShieldCheck, HelpCircle, ChevronRight, Layers } from 'lucide-react';
import useAeroStore from '../store/useAeroStore';

export default function AnomalyPanel() {
  const [activeTab, setActiveTab] = useState('spatial'); // 'spatial' | 'temporal' | 'kinematic' | 'limitations'

  const selectedAircraft = useAeroStore((s) => s.selectedAircraft);
  const aircraft = useAeroStore((s) => s.aircraft);
  const activeExplanation = useAeroStore((s) => s.activeExplanation);
  const explanationLoading = useAeroStore((s) => s.explanationLoading);
  const explanationError = useAeroStore((s) => s.explanationError);
  const threshold = useAeroStore((s) => s.threshold);

  const selectedFlight = aircraft.find((a) => a.id === selectedAircraft || a.icao24 === selectedAircraft);

  // 1. Loading State
  if (explanationLoading) {
    return (
      <div className="glass-panel rounded-lg p-4 animate-fade-in border border-primary/30">
        <div className="flex items-center justify-between mb-3 border-b border-outline-variant pb-2">
          <h2 className="text-mono-label text-primary tracking-widest uppercase flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 animate-spin" />
            Computing XAI Dossier...
          </h2>
          <span className="text-mono-data text-on-surface-variant text-[11px]">
            {selectedFlight?.callsign || selectedAircraft}
          </span>
        </div>
        <div className="py-6 flex flex-col items-center justify-center gap-2 text-center">
          <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          <span className="text-mono-data text-primary text-[11px] uppercase tracking-wider">
            Optimizing GNNExplainer & GRU Saliency
          </span>
          <span className="text-[10px] text-on-surface-variant font-mono max-w-[280px]">
            Decomposing spatial graph edges, 5-step temporal gradients, and motion residuals...
          </span>
        </div>
      </div>
    );
  }

  // 2. Error State
  if (explanationError) {
    return (
      <div className="glass-panel rounded-lg p-4 animate-fade-in border border-error/40">
        <div className="flex items-center justify-between mb-2 border-b border-outline-variant pb-2">
          <h2 className="text-mono-label text-error tracking-widest uppercase flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5" />
            Explanation Notice
          </h2>
          <span className="text-mono-data text-on-surface-variant text-[11px]">
            {selectedFlight?.callsign || selectedAircraft}
          </span>
        </div>
        <p className="text-mono-data text-error text-[11px] mb-2">{explanationError}</p>
        <div className="p-2 bg-surface-container-lowest rounded border border-outline-variant text-[10px] text-on-surface-variant font-mono">
          Threshold: t={threshold.toFixed(2)} · Select an active track with historical observations.
        </div>
      </div>
    );
  }

  // 3. No Aircraft Selected State
  if (!activeExplanation && !selectedFlight) {
    return (
      <div className="glass-panel rounded-lg p-4 animate-fade-in">
        <div className="flex items-center justify-between mb-3 border-b border-outline-variant pb-2">
          <h2 className="text-mono-label text-on-surface tracking-widest uppercase flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-primary" />
            XAI Incident Inspector
          </h2>
          <span className="text-mono-data text-on-surface-variant text-[11px]">t={threshold.toFixed(2)}</span>
        </div>
        <div className="flex flex-col items-center justify-center py-6 text-center text-on-surface-variant">
          <HelpCircle className="w-6 h-6 mb-2 opacity-40 text-primary" />
          <p className="text-mono-data text-[11px] max-w-[280px]">
            Select any flight marker on the radar or click an active alert below to inspect its multi-stream attribution dossier.
          </p>
        </div>
      </div>
    );
  }

  const exp = activeExplanation;
  const prob = exp?.anomaly_probability ?? (selectedFlight?.st_gnn_prob ?? 0.0);
  const isThreat = prob >= threshold;
  const threatLevel = exp?.threat_level || (prob >= 0.70 ? 'CRITICAL' : isThreat ? 'ELEVATED' : 'NOMINAL');
  const confidence = exp?.attribution_confidence || 'HIGH';
  const primaryDriver = exp?.primary_root_cause || exp?.primary_driver?.model_derived_driver || 'Nominal Flight Envelope';

  const threatColor = threatLevel === 'CRITICAL' ? 'text-error bg-error/15 border-error/40' :
                      threatLevel === 'ELEVATED' ? 'text-warning bg-warning/15 border-warning/40' :
                      'text-emerald-400 bg-emerald-500/15 border-emerald-500/40';

  const confColor = confidence.includes('INCONCLUSIVE') ? 'text-warning border-warning/40' :
                    confidence === 'HIGH' ? 'text-primary border-primary/40' :
                    'text-on-surface-variant border-outline-variant';

  return (
    <div className="glass-panel rounded-lg p-4 animate-fade-in flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-outline-variant pb-2">
        <div className="flex items-center gap-2">
          {isThreat ? (
            <AlertTriangle className="w-4 h-4 text-error" />
          ) : (
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          )}
          <h2 className="text-mono-label tracking-widest uppercase text-on-surface">
            {exp?.callsign || selectedFlight?.callsign || 'FLIGHT'}
          </h2>
          <span className="text-[10px] text-on-surface-variant font-mono">
            ({exp?.icao24 || selectedFlight?.icao24})
          </span>
        </div>
        <span className="text-[10px] font-mono text-on-surface-variant">
          {exp?.timestamp?.slice(11, 19) || 'LIVE'}
        </span>
      </div>

      {/* Probability Gauge & Threat Badges */}
      <div className="grid grid-cols-3 gap-2 bg-surface-container-lowest p-2.5 rounded border border-outline-variant/60">
        <div className="flex flex-col">
          <span className="text-[9px] font-mono text-on-surface-variant uppercase">ST-GNN Score</span>
          <div className="flex items-baseline gap-1 mt-0.5">
            <span className={`text-mono-data font-bold text-sm ${isThreat ? 'text-error' : 'text-emerald-400'}`}>
              {(prob * 100).toFixed(1)}%
            </span>
            <span className="text-[9px] font-mono text-on-surface-variant">
              (t={(threshold * 100).toFixed(0)}%)
            </span>
          </div>
        </div>

        <div className="flex flex-col">
          <span className="text-[9px] font-mono text-on-surface-variant uppercase">Threat Tier</span>
          <span className={`text-[10px] font-mono font-medium mt-0.5 px-1.5 py-0.5 rounded border inline-block text-center ${threatColor}`}>
            {threatLevel}
          </span>
        </div>

        <div className="flex flex-col">
          <span className="text-[9px] font-mono text-on-surface-variant uppercase">Confidence</span>
          <span className={`text-[10px] font-mono mt-0.5 px-1 py-0.5 rounded border truncate text-center ${confColor}`}>
            {confidence.split(' ')[0]}
          </span>
        </div>
      </div>

      {/* Inferred Primary Driver */}
      <div className="p-2 bg-surface-container-lowest rounded border border-outline-variant/50 flex flex-col gap-0.5">
        <span className="text-[9px] font-mono text-on-surface-variant uppercase tracking-wider">Primary Attributed Driver</span>
        <span className="text-mono-data text-[11px] font-medium text-on-surface truncate">
          {primaryDriver}
        </span>
      </div>

      {/* Multi-Stream Navigation Tabs */}
      <div className="flex gap-1 border-b border-outline-variant pb-1">
        <button
          onClick={() => setActiveTab('spatial')}
          className={`flex-1 text-[10px] font-mono py-1 rounded transition-colors ${
            activeTab === 'spatial' ? 'bg-primary/20 text-primary font-bold' : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          Spatial
        </button>
        <button
          onClick={() => setActiveTab('temporal')}
          className={`flex-1 text-[10px] font-mono py-1 rounded transition-colors ${
            activeTab === 'temporal' ? 'bg-primary/20 text-primary font-bold' : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          Temporal
        </button>
        <button
          onClick={() => setActiveTab('kinematic')}
          className={`flex-1 text-[10px] font-mono py-1 rounded transition-colors ${
            activeTab === 'kinematic' ? 'bg-primary/20 text-primary font-bold' : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          Kinematics
        </button>
        <button
          onClick={() => setActiveTab('limitations')}
          className={`flex-1 text-[10px] font-mono py-1 rounded transition-colors ${
            activeTab === 'limitations' ? 'bg-primary/20 text-primary font-bold' : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          Limits
        </button>
      </div>

      {/* Tab 1: Spatial Stream */}
      {activeTab === 'spatial' && (
        <div className="flex flex-col gap-2 animate-fade-in">
          <div className="flex justify-between items-center text-[10px] font-mono text-on-surface-variant">
            <span>Top Node Saliency (GNNExplainer)</span>
            <span>Relational Weight</span>
          </div>
          <div className="flex flex-col gap-1.5">
            {exp?.spatial_explanation?.top_node_features?.slice(0, 4).map((f, i) => (
              <div key={i} className="flex flex-col gap-0.5">
                <div className="flex justify-between text-[10px] font-mono">
                  <span className="text-on-surface">{f.feature}</span>
                  <span className="text-primary font-bold">{(f.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-1 bg-surface-container-highest rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, Math.max(5, f.importance * 100))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Critical Neighbors */}
          <div className="mt-1 pt-2 border-t border-outline-variant/40">
            <span className="text-[9px] font-mono text-on-surface-variant uppercase tracking-wider block mb-1">
              Critical Relational Neighbors ({exp?.spatial_explanation?.critical_edges?.length || 0})
            </span>
            {exp?.spatial_explanation?.critical_edges?.length > 0 ? (
              <div className="flex flex-col gap-1 max-h-[80px] overflow-y-auto pr-1">
                {exp.spatial_explanation.critical_edges.map((edge, i) => (
                  <div key={i} className="flex items-center justify-between text-[10px] font-mono bg-surface-container-lowest p-1 rounded border border-outline-variant/40">
                    <span className="text-on-surface font-medium">{edge.neighbor_callsign} ({edge.neighbor_icao24})</span>
                    <span className="text-primary font-bold">Edge Imp: {(edge.edge_importance * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-[10px] font-mono text-on-surface-variant/70 italic p-1 bg-surface-container-lowest rounded">
                Isolated flight: 0 neighbors within 100 km radius.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 2: Temporal Stream */}
      {activeTab === 'temporal' && (
        <div className="flex flex-col gap-2 animate-fade-in">
          <div className="flex justify-between items-center text-[10px] font-mono text-on-surface-variant">
            <span>5-Timestep Window (GRU Saliency)</span>
            <span className="text-primary font-medium">Peak: {exp?.temporal_explanation?.peak_timestep || 't'}</span>
          </div>

          <div className="grid grid-cols-5 gap-1.5 mt-1">
            {exp?.temporal_explanation?.temporal_step_importance?.map((step, i) => {
              const isPeak = step.timestep === exp?.temporal_explanation?.peak_timestep;
              const isPadded = !step.is_valid_observation;
              const barHeight = Math.min(100, Math.max(10, step.importance * 100));

              return (
                <div key={i} className="flex flex-col items-center gap-1">
                  <div className="w-full h-14 bg-surface-container-lowest rounded border border-outline-variant flex items-end justify-center p-0.5">
                    <div
                      className={`w-full rounded-sm transition-all duration-500 ${
                        isPadded ? 'bg-surface-container-highest opacity-20' :
                        isPeak ? 'bg-error shadow-sm shadow-error/50' : 'bg-primary'
                      }`}
                      style={{ height: `${barHeight}%` }}
                    />
                  </div>
                  <span className={`text-[9px] font-mono ${isPeak ? 'text-error font-bold' : 'text-on-surface-variant'}`}>
                    {step.timestep}
                  </span>
                  <span className="text-[8px] font-mono text-on-surface-variant">
                    {isPadded ? 'pad' : `${(step.importance * 100).toFixed(0)}%`}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="mt-1 pt-1.5 border-t border-outline-variant/40 text-[10px] font-mono text-on-surface-variant">
            Top Feature: <span className="text-on-surface font-medium">{exp?.temporal_explanation?.top_temporal_features?.[0]?.feature}</span>
          </div>
        </div>
      )}

      {/* Tab 3: Kinematic Stream */}
      {activeTab === 'kinematic' && (
        <div className="flex flex-col gap-2 animate-fade-in">
          <div className="flex justify-between items-center text-[10px] font-mono text-on-surface-variant">
            <span>Physical Invariant Residuals</span>
            <span>Gradient Saliency</span>
          </div>

          <div className="flex flex-col gap-1">
            {exp?.kinematic_explanation?.top_kinematic_residuals?.slice(0, 3).map((r, i) => (
              <div key={i} className="flex justify-between items-center text-[10px] font-mono bg-surface-container-lowest p-1.5 rounded border border-outline-variant/40">
                <span className="text-on-surface">{r.residual_feature}</span>
                <div className="flex items-center gap-2">
                  <span className="text-on-surface-variant text-[9px]">Val: {r.raw_value.toFixed(1)}</span>
                  <span className="text-primary font-bold">{(r.importance * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>

          {/* Independent Rule Warnings (Heuristics) */}
          {exp?.kinematic_explanation?.independent_rule_warnings?.length > 0 && (
            <div className="mt-1 pt-1.5 border-t border-outline-variant/40">
              <span className="text-[9px] font-mono text-warning uppercase tracking-wider block mb-1">
                Rule-Based Diagnostic Warnings
              </span>
              <div className="flex flex-col gap-1">
                {exp.kinematic_explanation.independent_rule_warnings.map((rule, i) => (
                  <div key={i} className="p-1.5 rounded bg-warning/10 border border-warning/30 text-[9.5px] font-mono text-warning">
                    {rule.message}
                  </div>
                ))}
              </div>
            </div>
          )}

          <p className="text-[8.5px] font-mono text-on-surface-variant/70 italic mt-0.5">
            Rule diagnostics represent independent sanity checks and do not constitute proof of spoofing alone.
          </p>
        </div>
      )}

      {/* Tab 4: Limitations & Domain Uncertainty */}
      {activeTab === 'limitations' && (
        <div className="flex flex-col gap-2 animate-fade-in text-[10px] font-mono">
          <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant text-on-surface-variant leading-relaxed">
            <span className="text-primary font-bold block mb-1 uppercase tracking-wider text-[9px]">
              Operational Limitation Statement:
            </span>
            {exp?.limitations_and_uncertainty || 'Single-sensor geometry; evaluate localized barometric adjustments and multipath fading before dispatching advisory.'}
          </div>
          <div className="flex justify-between items-center text-on-surface-variant text-[9px] px-1">
            <span>Model Mode: Causal Tri-Stream ST-GNN</span>
            <span>Threshold t={threshold.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
