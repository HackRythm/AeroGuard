import { useMemo } from 'react';
import {
  LayoutDashboard,
  Globe,
  Network,
  History,
  Settings,
  Plus,
  ChevronLeft,
  ChevronRight,
  Shield,
} from 'lucide-react';
import useAeroStore from '../store/useAeroStore';

const navItems = [
  { icon: LayoutDashboard, label: 'Dashboard', id: 'dashboard' },
  { icon: Globe, label: 'Airspace', id: 'airspace' },
  { icon: Network, label: 'Analysis', id: 'analysis' },
  { icon: History, label: 'History', id: 'history' },
];

export default function Sidebar() {
  const collapsed = useAeroStore((s) => s.sidebarCollapsed);
  const toggle = useAeroStore((s) => s.toggleSidebar);
  const activeAnomalies = useAeroStore((s) => s.metrics.activeAnomalies);

  const width = collapsed ? 'w-20' : 'w-60';

  return (
    <nav
      className={`${width} bg-surface-container-low/80 backdrop-blur-xl h-screen flex flex-col items-center py-6 border-r border-outline-variant fixed left-0 top-0 z-50 transition-all duration-300 ease-out`}
    >
      {/* Brand */}
      <div className="flex items-center gap-2 mb-8 px-4">
        <Shield className="w-7 h-7 text-primary shrink-0" strokeWidth={1.5} />
        {!collapsed && (
          <span className="text-headline-sm text-primary font-semibold tracking-tight whitespace-nowrap animate-fade-in">
            AeroGuard
          </span>
        )}
      </div>

      {/* Collapse toggle */}
      <button
        onClick={toggle}
        className="absolute -right-3 top-20 w-6 h-6 rounded-full bg-surface-container-high border border-outline-variant flex items-center justify-center text-on-surface-variant hover:text-primary hover:border-primary transition-colors z-50"
        aria-label="Toggle sidebar"
      >
        {collapsed ? (
          <ChevronRight className="w-3.5 h-3.5" />
        ) : (
          <ChevronLeft className="w-3.5 h-3.5" />
        )}
      </button>

      {/* Nav items */}
      <div className="flex-1 flex flex-col items-stretch gap-1 mt-4 w-full px-2">
        {navItems.map((item, i) => {
          const active = i === 0; // Dashboard active by default
          const Icon = item.icon;
          return (
            <a
              key={item.id}
              href="#"
              className={`relative flex items-center gap-3 py-2.5 rounded transition-all duration-150 ${
                collapsed ? 'justify-center px-0' : 'px-3'
              } ${
                active
                  ? 'text-primary bg-primary/8 border-r-2 border-primary'
                  : 'text-on-surface-variant hover:text-primary hover:bg-primary/5'
              }`}
              aria-label={item.label}
            >
              <Icon className="w-5 h-5 shrink-0" strokeWidth={1.5} />
              {!collapsed && (
                <span className="text-body-sm whitespace-nowrap animate-fade-in">
                  {item.label}
                </span>
              )}
              {/* Anomaly badge on Dashboard */}
              {item.id === 'dashboard' && activeAnomalies > 0 && (
                <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-error anomaly-pulse" />
              )}
            </a>
          );
        })}
      </div>

      {/* Bottom section */}
      <div className="flex flex-col items-stretch gap-1 w-full px-2 mt-auto">
        <a
          href="#"
          className={`flex items-center gap-3 py-2.5 rounded text-on-surface-variant hover:text-primary hover:bg-primary/5 transition-all duration-150 ${
            collapsed ? 'justify-center px-0' : 'px-3'
          }`}
          aria-label="Settings"
        >
          <Settings className="w-5 h-5 shrink-0" strokeWidth={1.5} />
          {!collapsed && (
            <span className="text-body-sm whitespace-nowrap animate-fade-in">
              Settings
            </span>
          )}
        </a>

        {/* New Mission CTA */}
        <button
          className={`mt-2 flex items-center justify-center gap-2 rounded bg-primary-container text-on-primary-container hover:bg-primary transition-colors font-medium ${
            collapsed ? 'p-2.5' : 'py-2.5 px-4'
          }`}
          aria-label="New Mission"
        >
          <Plus className="w-4 h-4 shrink-0" strokeWidth={2} />
          {!collapsed && (
            <span className="text-body-sm whitespace-nowrap animate-fade-in">
              New Mission
            </span>
          )}
        </button>
      </div>
    </nav>
  );
}
