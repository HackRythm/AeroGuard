import { Plane, BrainCircuit, AlertTriangle, Radio } from 'lucide-react';
import useAeroStore from '../store/useAeroStore';

const cards = [
  {
    key: 'totalTracked',
    label: 'TOTAL TRACKED',
    icon: Plane,
    format: (v) => String(Math.round(v)),
    accent: 'primary',
  },
  {
    key: 'modelConfidence',
    label: 'MODEL CONFIDENCE',
    icon: BrainCircuit,
    format: (v) => `${v.toFixed(1)}%`,
    accent: 'primary',
    hasBar: true,
  },
  {
    key: 'activeAnomalies',
    label: 'ACTIVE ANOMALIES',
    icon: AlertTriangle,
    format: (v) => String(Math.round(v)),
    accent: 'error',
    isAlert: true,
  },
  {
    key: 'signalIntegrity',
    label: 'SIGNAL INTEGRITY',
    icon: Radio,
    format: (v) => `${v.toFixed(1)}%`,
    accent: 'primary',
    hasBar: true,
  },
];

export default function MetricCards() {
  const metrics = useAeroStore((s) => s.metrics);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {cards.map((card) => {
        const value = metrics[card.key];
        const Icon = card.icon;
        const isError = card.accent === 'error';

        return (
          <div
            key={card.key}
            className={`glass-panel rounded-lg p-4 flex flex-col gap-3 transition-all duration-150 hover:border-${card.accent} ${
              card.isAlert && value > 0 ? 'border-error/50' : ''
            }`}
          >
            {/* Header */}
            <div className="flex items-center justify-between">
              <span className="text-mono-data text-on-surface-variant tracking-wider uppercase">
                {card.label}
              </span>
              <Icon
                className={`w-4 h-4 ${
                  isError ? 'text-error' : 'text-on-surface-variant'
                }`}
                strokeWidth={1.5}
              />
            </div>

            {/* Value */}
            <div className="flex items-end gap-2">
              <span
                className={`text-headline-md font-semibold ${
                  isError ? 'text-error' : 'text-on-surface'
                } ${card.isAlert && value > 0 ? 'anomaly-pulse' : ''}`}
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                {card.format(value)}
              </span>
            </div>

            {/* Progress bar */}
            {card.hasBar && (
              <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isError ? 'bg-error' : 'bg-primary'
                  }`}
                  style={{ width: `${Math.min(100, value)}%` }}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
