import { useMemo } from 'react';
import useAeroStore from '../store/useAeroStore';

export default function SignalTelemetry() {
  const aircraft = useAeroStore((s) => s.aircraft);
  const tickCount = useAeroStore((s) => s.tickCount);

  // Aggregate signal metrics from all aircraft
  const stats = useMemo(() => {
    const anomalies = aircraft.filter((a) => a.isAnomaly);
    const normal = aircraft.filter((a) => !a.isAnomaly);

    const avgSnr =
      aircraft.reduce((s, a) => s + a.snr, 0) / aircraft.length;
    const avgRss =
      aircraft.reduce((s, a) => s + a.rss, 0) / aircraft.length;
    const maxDoppler = anomalies.length > 0
      ? Math.max(...anomalies.map((a) => Math.abs(a.doppler)))
      : 0;

    return {
      freq: '1090 MHz',
      doppler: maxDoppler > 0 ? `+${Math.round(maxDoppler)} Hz` : '0 Hz',
      snr: `${avgSnr.toFixed(1)} dB`,
      rss: `${avgRss.toFixed(1)} dBm`,
      isDopplerAnomaly: maxDoppler > 300,
    };
  }, [aircraft]);

  // Generate waveform path based on tick count for visual animation
  const waveformPath = useMemo(() => {
    const points = [];
    const segments = 50;
    for (let i = 0; i <= segments; i++) {
      const x = (i / segments) * 100;
      const y =
        10 +
        Math.sin((i * 0.6) + tickCount * 0.2) * 4 +
        Math.sin((i * 1.3) + tickCount * 0.15) * 2.5 +
        Math.cos((i * 0.4) + tickCount * 0.1) * 1.5;
      points.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return points.join(' ');
  }, [tickCount]);

  return (
    <div className="glass-panel rounded-lg p-4 flex-1 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 border-b border-outline-variant pb-2">
        <h2 className="text-mono-label text-on-surface tracking-widest uppercase">
          Signal Telemetry
        </h2>
        <span className="text-mono-data text-on-surface-variant">ADS-B</span>
      </div>

      {/* Stat grid */}
      <div className="grid grid-cols-2 gap-2">
        {/* FREQ */}
        <div className="bg-surface-container-lowest p-2.5 rounded border border-outline-variant flex flex-col justify-center items-center h-16">
          <span className="text-mono-data text-on-surface-variant mb-1">
            FREQ
          </span>
          <span className="text-mono-data text-primary font-medium">
            {stats.freq}
          </span>
        </div>

        {/* DOPPLER */}
        <div className="bg-surface-container-lowest p-2.5 rounded border border-outline-variant flex flex-col justify-center items-center h-16">
          <span className="text-mono-data text-on-surface-variant mb-1">
            DOPPLER
          </span>
          <span
            className={`text-mono-data font-medium ${
              stats.isDopplerAnomaly ? 'text-error' : 'text-primary'
            }`}
          >
            {stats.doppler}
          </span>
        </div>

        {/* RSS */}
        <div className="bg-surface-container-lowest p-2.5 rounded border border-outline-variant flex flex-col justify-center items-center h-16">
          <span className="text-mono-data text-on-surface-variant mb-1">
            RSS
          </span>
          <span className="text-mono-data text-on-surface">
            {stats.rss}
          </span>
        </div>

        {/* SNR */}
        <div className="bg-surface-container-lowest p-2.5 rounded border border-outline-variant flex flex-col justify-center items-center h-16">
          <span className="text-mono-data text-on-surface-variant mb-1">
            SNR
          </span>
          <span className="text-mono-data text-primary font-medium">
            {stats.snr}
          </span>
        </div>

        {/* Waveform (spans full width) */}
        <div className="col-span-2 bg-surface-container-lowest p-2.5 rounded border border-outline-variant mt-1">
          <div className="flex justify-between mb-1">
            <span className="text-mono-data text-on-surface-variant">
              Signal Waveform
            </span>
            <span className="text-mono-data text-primary">
              {stats.snr}
            </span>
          </div>
          <div className="w-full h-8 relative overflow-hidden rounded">
            <svg
              className="absolute inset-0 w-full h-full"
              viewBox="0 0 100 20"
              preserveAspectRatio="none"
            >
              {/* Main waveform */}
              <path
                d={waveformPath}
                fill="none"
                stroke="#adc6ff"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
              {/* Shadow waveform */}
              <path
                d={waveformPath}
                fill="none"
                stroke="#adc6ff"
                strokeWidth="0.5"
                strokeLinecap="round"
                opacity="0.3"
                transform="translate(0,2)"
              />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
