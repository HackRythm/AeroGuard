import { useState } from 'react';
import { Search, Bell, HelpCircle, Play, Pause, Radio, Zap, ShieldAlert, CheckCircle2 } from 'lucide-react';
import useAeroStore from '../store/useAeroStore';

export default function TopBar() {
  const [showAttackMenu, setShowAttackMenu] = useState(false);

  const collapsed = useAeroStore((s) => s.sidebarCollapsed);
  const isPlaying = useAeroStore((s) => s.isPlaying);
  const togglePlayback = useAeroStore((s) => s.togglePlayback);
  const alerts = useAeroStore((s) => s.alerts);
  const receiverStatus = useAeroStore((s) => s.receiverStatus);
  const triggerSyntheticBurst = useAeroStore((s) => s.triggerSyntheticBurst);
  const triggerAttackSimulation = useAeroStore((s) => s.triggerAttackSimulation);
  const burstDecoding = useAeroStore((s) => s.burstDecoding);
  const simulationAttackActive = useAeroStore((s) => s.simulationAttackActive);
  const backendConnected = useAeroStore((s) => s.backendConnected);

  const undismissedCount = alerts.filter((a) => !a.dismissed).length;
  const leftOffset = collapsed ? 'left-20' : 'left-60';

  const handleTriggerAttack = (type) => {
    setShowAttackMenu(false);
    triggerAttackSimulation(type);
  };

  return (
    <header
      className={`h-16 flex items-center px-6 bg-surface-container-lowest/85 backdrop-blur-xl border-b border-outline-variant fixed top-0 right-0 ${leftOffset} z-40 transition-all duration-300 ease-out`}
    >
      {/* Left: Controls & Badges */}
      <div className="flex items-center gap-3.5 flex-1">
        {/* Playback Replay / Live Feed Toggle */}
        <button
          onClick={togglePlayback}
          className={`flex items-center gap-2 px-3 py-1.5 rounded border transition-all duration-150 ${
            isPlaying
              ? 'bg-primary/10 border-primary text-primary font-semibold'
              : 'bg-surface-container border-outline-variant text-on-surface-variant hover:text-primary hover:border-primary'
          }`}
          title={isPlaying ? 'Pause replay stream' : 'Advance live telemetry replay'}
        >
          {isPlaying ? (
            <Pause className="w-3.5 h-3.5" strokeWidth={2} />
          ) : (
            <Play className="w-3.5 h-3.5" strokeWidth={2} />
          )}
          <span className="text-mono-label">{isPlaying ? 'STREAMING' : 'REPLAY'}</span>
          {isPlaying && <span className="w-2 h-2 rounded-full bg-error anomaly-pulse" />}
        </button>

        {/* Software-Defined Radio (SDC) Receiver Badge */}
        <div className="hidden md:flex items-center gap-2 bg-surface-container px-3 py-1.5 rounded border border-outline-variant text-[11px] font-mono">
          <span
            className={`w-2 h-2 rounded-full ${
              backendConnected ? 'bg-emerald-400 glow-primary' : 'bg-amber-400 animate-pulse'
            }`}
          />
          <span className="text-on-surface font-medium">SDC 1090 MHz</span>
          <span className="text-on-surface-variant">|</span>
          <span className="text-primary uppercase tracking-wider">
            {receiverStatus?.receiver_mode?.replace('_', ' ') || 'SYNTHETIC IQ'}
          </span>
          <span className="text-[9px] text-on-surface-variant/80 bg-surface-container-high px-1.5 py-0.5 rounded">
            HW: UNAVAILABLE (SOFTWARE MODE)
          </span>
        </div>

        {/* Test Synthetic ADS-B Waveform Burst Button */}
        <button
          onClick={triggerSyntheticBurst}
          disabled={burstDecoding}
          className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded border border-primary/40 bg-primary/10 hover:bg-primary/20 text-primary text-[10px] font-mono font-medium transition-all duration-150 disabled:opacity-50"
          title="Modulate and decode 1090 MHz PPM baseband burst into track buffer"
        >
          <Radio className={`w-3.5 h-3.5 ${burstDecoding ? 'animate-spin' : ''}`} />
          {burstDecoding ? 'DECODING IQ...' : 'TEST SDR BURST'}
          {receiverStatus?.crc_passed > 0 && (
            <span className="bg-primary/20 px-1 rounded text-[9px] text-primary">
              ✓{receiverStatus.crc_passed}
            </span>
          )}
        </button>

        {/* Attack Simulation Menu (Clearly Labeled Simulated Test Event) */}
        <div className="relative">
          <button
            onClick={() => setShowAttackMenu(!showAttackMenu)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded border border-warning/40 bg-warning/10 hover:bg-warning/20 text-warning text-[10px] font-mono font-medium transition-all"
            title="Inject simulated ADS-B attack burst into active track buffer"
          >
            <Zap className="w-3.5 h-3.5" />
            SIMULATE ATTACK BURST
          </button>

          {showAttackMenu && (
            <div className="absolute left-0 mt-2 w-64 bg-surface-container-high border border-outline-variant rounded-lg shadow-xl py-2 z-50 animate-fade-in text-[11px] font-mono">
              <div className="px-3 py-1 text-[9px] text-on-surface-variant uppercase tracking-wider border-b border-outline-variant/60 font-bold">
                ACADEMIC DEMO TEST INJECTIONS
              </div>
              <button
                onClick={() => handleTriggerAttack('altitude_teleport')}
                className="w-full text-left px-3 py-2 hover:bg-primary/10 hover:text-primary transition-colors flex items-center justify-between"
              >
                <span>Altitude Jump (+7,500 ft)</span>
                <span className="text-[9px] text-error font-semibold">t ≥ 0.20</span>
              </button>
              <button
                onClick={() => handleTriggerAttack('gps_velocity_drift')}
                className="w-full text-left px-3 py-2 hover:bg-primary/10 hover:text-primary transition-colors flex items-center justify-between"
              >
                <span>GPS Trajectory Drift</span>
                <span className="text-[9px] text-error font-semibold">disp_ratio</span>
              </button>
              <button
                onClick={() => handleTriggerAttack('ghost_aircraft')}
                className="w-full text-left px-3 py-2 hover:bg-primary/10 hover:text-primary transition-colors flex items-center justify-between"
              >
                <span>Ghost Aircraft (GHOST99)</span>
                <span className="text-[9px] text-error font-semibold">Unverified</span>
              </button>
              <button
                onClick={() => handleTriggerAttack('transponder_collision')}
                className="w-full text-left px-3 py-2 hover:bg-primary/10 hover:text-primary transition-colors flex items-center justify-between"
              >
                <span>Transponder Address Collision</span>
                <span className="text-[9px] text-error font-semibold">Duplicate</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Right: Actions & Indicators */}
      <div className="flex items-center gap-3">
        {/* Simulation Banner Badge */}
        {simulationAttackActive && (
          <div className="hidden xl:flex items-center gap-1.5 px-2.5 py-1 rounded bg-error/15 border border-error/40 text-error text-[10px] font-mono animate-pulse">
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>TEST BURST: {simulationAttackActive.attack_type?.toUpperCase().replace('_', ' ')}</span>
          </div>
        )}

        {/* Notifications */}
        <button className="relative p-2 text-on-surface-variant hover:text-primary transition-colors">
          <Bell className="w-5 h-5" strokeWidth={1.5} />
          {undismissedCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] flex items-center justify-center rounded-full bg-error text-on-error text-[10px] font-bold px-1">
              {undismissedCount}
            </span>
          )}
        </button>

        {/* Viva / SDC Demo Badge */}
        <div className="h-8 px-2.5 rounded bg-surface-container-high border border-outline-variant flex items-center justify-center text-[10px] font-mono text-primary font-semibold">
          SEM 3 · SDC
        </div>
      </div>
    </header>
  );
}
