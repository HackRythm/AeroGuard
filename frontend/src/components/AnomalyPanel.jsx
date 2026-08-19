import useAeroStore from '../store/useAeroStore';

export default function AnomalyPanel() {
  const aircraft = useAeroStore((s) => s.aircraft);
  const selectedAircraft = useAeroStore((s) => s.selectedAircraft);

  // Show details for the selected anomaly aircraft, or the first anomaly
  const anomaly = selectedAircraft
    ? aircraft.find((a) => a.id === selectedAircraft && a.isAnomaly)
    : aircraft.find((a) => a.isAnomaly);

  if (!anomaly) {
    return (
      <div className="glass-panel rounded-lg p-4 animate-fade-in">
        <div className="flex items-center justify-between mb-3 border-b border-outline-variant pb-2">
          <h2 className="text-mono-label text-on-surface tracking-widest uppercase flex items-center gap-2">
            <span className="text-primary">◆</span>
            Anomaly Analysis
          </h2>
        </div>
        <div className="flex items-center justify-center h-20 text-mono-data text-on-surface-variant opacity-50">
          No anomalies selected
        </div>
      </div>
    );
  }

  const confidenceMap = {
    position_jump: 94,
    ghost_aircraft: 88,
    velocity_anomaly: 76,
  };

  const typeLabels = {
    position_jump: 'Erratic Pathing Signature',
    ghost_aircraft: 'Ghost Injection — Synthetic Track',
    velocity_anomaly: 'Velocity Vector Inconsistency',
  };

  const confidence = confidenceMap[anomaly.anomalyType] || 80;
  const typeLabel = typeLabels[anomaly.anomalyType] || anomaly.anomalyType;
  const isCritical = confidence > 85;

  return (
    <div className="glass-panel rounded-lg p-4 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 border-b border-outline-variant pb-2">
        <h2 className="text-mono-label text-error tracking-widest uppercase flex items-center gap-2">
          <span>⚠</span>
          Anomaly Analysis
        </h2>
        <span className="text-mono-data text-on-surface-variant">
          {anomaly.callsign}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {/* Confidence */}
        <div className="bg-surface-container-lowest p-3 rounded border border-outline-variant flex justify-between items-center">
          <span className="text-mono-data text-on-surface-variant">
            Confidence
          </span>
          <div className="flex items-center gap-2">
            <div className="w-24 h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
              <div
                className="h-full bg-error rounded-full transition-all duration-700"
                style={{ width: `${confidence}%` }}
              />
            </div>
            <span className="text-mono-data text-on-surface font-medium">
              {confidence}%
            </span>
          </div>
        </div>

        {/* Classification */}
        <div className="bg-surface-container-lowest p-3 rounded border border-outline-variant flex flex-col gap-1">
          <span className="text-mono-data text-on-surface-variant">
            Classification Type
          </span>
          <span className="text-body-sm text-on-surface font-medium">
            {typeLabel}
          </span>
        </div>

        {/* Telemetry snapshot */}
        <div className="bg-surface-container-lowest p-3 rounded border border-outline-variant grid grid-cols-3 gap-2">
          <div className="flex flex-col items-center">
            <span className="text-mono-data text-on-surface-variant text-[10px]">ALT</span>
            <span className="text-mono-data text-on-surface">
              FL{Math.round(anomaly.altitude / 100)}
            </span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-mono-data text-on-surface-variant text-[10px]">GS</span>
            <span className="text-mono-data text-error">
              {anomaly.groundspeed} kts
            </span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-mono-data text-on-surface-variant text-[10px]">VS</span>
            <span className="text-mono-data text-error">
              {anomaly.verticalRate > 0 ? '+' : ''}
              {anomaly.verticalRate} fpm
            </span>
          </div>
        </div>

        {/* Alert Level */}
        <div
          className={`p-3 rounded border flex items-center justify-between mt-1 ${
            isCritical
              ? 'bg-error-container/20 border-error/50'
              : 'bg-warning/10 border-warning/50'
          }`}
        >
          <span
            className={`text-mono-label ${
              isCritical ? 'text-error' : 'text-warning'
            }`}
          >
            ALERT LEVEL
          </span>
          <span
            className={`text-mono-label px-2 py-0.5 rounded animate-pulse ${
              isCritical
                ? 'bg-error text-on-error'
                : 'bg-warning text-on-warning'
            }`}
          >
            {isCritical ? 'CRITICAL' : 'WARNING'}
          </span>
        </div>
      </div>
    </div>
  );
}
