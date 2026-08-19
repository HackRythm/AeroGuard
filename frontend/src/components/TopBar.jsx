import { Search, Bell, HelpCircle, Play, Pause } from 'lucide-react';
import useAeroStore from '../store/useAeroStore';

export default function TopBar() {
  const collapsed = useAeroStore((s) => s.sidebarCollapsed);
  const isPlaying = useAeroStore((s) => s.isPlaying);
  const togglePlayback = useAeroStore((s) => s.togglePlayback);
  const alerts = useAeroStore((s) => s.alerts);

  const undismissedCount = alerts.filter((a) => !a.dismissed).length;
  const leftOffset = collapsed ? 'left-20' : 'left-60';

  return (
    <header
      className={`h-16 flex items-center px-6 bg-surface-container-lowest/80 backdrop-blur-xl border-b border-outline-variant fixed top-0 right-0 ${leftOffset} z-40 transition-all duration-300 ease-out`}
    >
      {/* Left: Search */}
      <div className="flex items-center gap-4 flex-1">
        <div className="relative group">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-on-surface-variant" />
          <input
            type="text"
            placeholder="Search airspace data..."
            className="bg-surface-container w-64 h-9 pl-10 pr-4 rounded border border-outline-variant focus:border-primary focus:outline-none text-body-sm text-on-surface placeholder:text-on-surface-variant transition-colors"
          />
        </div>

        {/* Playback Control */}
        <button
          onClick={togglePlayback}
          className={`flex items-center gap-2 px-3 py-1.5 rounded border transition-all duration-150 ${
            isPlaying
              ? 'bg-primary/10 border-primary text-primary'
              : 'bg-surface-container border-outline-variant text-on-surface-variant hover:text-primary hover:border-primary'
          }`}
        >
          {isPlaying ? (
            <Pause className="w-3.5 h-3.5" strokeWidth={2} />
          ) : (
            <Play className="w-3.5 h-3.5" strokeWidth={2} />
          )}
          <span className="text-mono-label">
            {isPlaying ? 'LIVE' : 'PLAY'}
          </span>
          {isPlaying && (
            <span className="w-2 h-2 rounded-full bg-error anomaly-pulse" />
          )}
        </button>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-3">
        {/* Timestamp */}
        <span className="text-mono-data text-on-surface-variant hidden lg:block">
          {new Date().toISOString().replace('T', ' ').slice(0, 19)} UTC
        </span>

        {/* Notifications */}
        <button className="relative p-2 text-on-surface-variant hover:text-primary transition-colors">
          <Bell className="w-5 h-5" strokeWidth={1.5} />
          {undismissedCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] flex items-center justify-center rounded-full bg-error text-on-error text-[10px] font-bold px-1">
              {undismissedCount}
            </span>
          )}
        </button>

        {/* Help */}
        <button className="p-2 text-on-surface-variant hover:text-primary transition-colors">
          <HelpCircle className="w-5 h-5" strokeWidth={1.5} />
        </button>

        {/* Avatar */}
        <div className="h-8 w-8 rounded-full bg-surface-container-high border border-outline-variant flex items-center justify-center cursor-pointer hover:border-primary transition-colors overflow-hidden">
          <span className="text-mono-label text-primary text-xs">HM</span>
        </div>
      </div>
    </header>
  );
}
