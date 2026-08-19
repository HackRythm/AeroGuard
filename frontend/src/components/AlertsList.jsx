import { X } from 'lucide-react';
import useAeroStore from '../store/useAeroStore';

function formatTimestamp(iso) {
  try {
    const d = new Date(iso);
    return d.toISOString().replace('T', ' ').slice(11, 19);
  } catch {
    return '—';
  }
}

export default function AlertsList() {
  const alerts = useAeroStore((s) => s.alerts);
  const dismissAlert = useAeroStore((s) => s.dismissAlert);

  const visible = alerts.filter((a) => !a.dismissed).slice(0, 8);

  if (visible.length === 0) {
    return (
      <div className="glass-panel rounded-lg p-4 animate-fade-in">
        <div className="flex items-center justify-between mb-3 border-b border-outline-variant pb-2">
          <h2 className="text-mono-label text-on-surface tracking-widest uppercase">
            Flight Alerts
          </h2>
          <span className="text-mono-data text-on-surface-variant">0</span>
        </div>
        <div className="flex items-center justify-center h-12 text-mono-data text-on-surface-variant opacity-50">
          All clear — no active alerts
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel rounded-lg p-4 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 border-b border-outline-variant pb-2">
        <h2 className="text-mono-label text-on-surface tracking-widest uppercase">
          Flight Alerts
        </h2>
        <span className="text-mono-data text-error">{visible.length}</span>
      </div>

      {/* Alert list */}
      <div className="flex flex-col gap-1.5 max-h-[200px] overflow-y-auto pr-1">
        {visible.map((alert) => {
          const isCritical = alert.severity === 'critical';

          return (
            <div
              key={alert.id}
              className={`p-2.5 rounded border flex items-start gap-2 transition-all duration-150 group ${
                isCritical
                  ? 'bg-error-container/10 border-error/30 hover:border-error/60'
                  : 'bg-warning/5 border-warning/20 hover:border-warning/50'
              }`}
            >
              {/* Severity dot */}
              <div
                className={`w-2 h-2 rounded-full mt-1 shrink-0 ${
                  isCritical ? 'bg-error anomaly-pulse' : 'bg-warning'
                }`}
              />

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-mono-data text-on-surface-variant">
                    {formatTimestamp(alert.timestamp)}
                  </span>
                  <span
                    className={`text-mono-data font-medium ${
                      isCritical ? 'text-error' : 'text-warning'
                    }`}
                  >
                    {alert.callsign}
                  </span>
                  <span
                    className={`text-[9px] px-1.5 py-0.5 rounded font-mono uppercase ${
                      isCritical
                        ? 'bg-error/20 text-error'
                        : 'bg-warning/20 text-warning'
                    }`}
                  >
                    {alert.anomalyType.replace('_', ' ')}
                  </span>
                </div>
                <p className="text-mono-data text-on-surface-variant truncate text-[11px]">
                  {alert.message}
                </p>
              </div>

              {/* Dismiss */}
              <button
                onClick={() => dismissAlert(alert.id)}
                className="p-1 text-on-surface-variant hover:text-error opacity-0 group-hover:opacity-100 transition-opacity"
                aria-label="Dismiss alert"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
