import { useEffect } from 'react';
import useAeroStore from './store/useAeroStore';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import AirspaceMap from './components/AirspaceMap';
import GNNPanel from './components/GNNPanel';
import MetricCards from './components/MetricCards';
import AnomalyPanel from './components/AnomalyPanel';
import SignalTelemetry from './components/SignalTelemetry';
import AlertsList from './components/AlertsList';

export default function App() {
  const isPlaying = useAeroStore((s) => s.isPlaying);
  const tick = useAeroStore((s) => s.tick);
  const sidebarCollapsed = useAeroStore((s) => s.sidebarCollapsed);

  // Real-time simulation loop
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(tick, 1200);
    return () => clearInterval(interval);
  }, [isPlaying, tick]);

  const mainMargin = sidebarCollapsed ? 'ml-20' : 'ml-60';

  return (
    <div className="min-h-screen bg-background text-on-background overflow-hidden">
      <Sidebar />
      <TopBar />

      {/* Main content area */}
      <main
        className={`${mainMargin} mt-16 h-[calc(100vh-64px)] p-4 flex flex-col xl:flex-row gap-4 transition-all duration-300 ease-out overflow-hidden`}
      >
        {/* Left column: Map + Metrics */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          {/* Airspace Map — takes most of the vertical space */}
          <div className="flex-1 min-h-[300px] flex flex-col">
            <AirspaceMap />
          </div>

          {/* Metric cards row — below the map */}
          <MetricCards />
        </div>

        {/* Right column: Data panels */}
        <aside className="w-full xl:w-[380px] flex flex-col gap-3 overflow-y-auto pr-1 pb-1 shrink-0">
          <GNNPanel />
          <AnomalyPanel />
          <SignalTelemetry />
          <AlertsList />
        </aside>
      </main>
    </div>
  );
}
