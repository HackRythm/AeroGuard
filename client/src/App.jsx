import React, { useState, useEffect, useRef } from 'react';
import L from 'leaflet';

// Constants for Lisbon Receiver and map boundaries
const RECEIVER_LAT = 38.7756;
const RECEIVER_LON = -9.1354;
const PROXIMITY_THRESHOLD_M = 100000.0; // 100 km edge proximity
const MAP_BOUNDS = {
  minLon: -16.0,
  maxLon: -7.0,
  minLat: 50.5,
  maxLat: 56.2
};

export default function App() {
  // Playback & Timeline State
  const [timestamps, setTimestamps] = useState([]);
  const [currentTimeIdx, setCurrentTimeIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedMultiplier, setSpeedMultiplier] = useState(1.0);
  const [substep, setSubstep] = useState(0.0);
  
  // Data State
  const [aircraftT, setAircraftT] = useState([]);
  const [aircraftT1, setAircraftT1] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [metrics, setMetrics] = useState({ baseline: {}, gnn: {} });
  const [gnnThreshold, setGnnThreshold] = useState(0.5);
  const [historyMap, setHistoryMap] = useState({});
  
  // Interaction State
  const [selectedCallsign, setSelectedCallsign] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Refs
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const layerGroup = useRef(null);

  // 1. Fetch Timeline and Metrics on Mount
  useEffect(() => {
    setLoading(true);
    fetch('http://localhost:5000/api/timeline')
      .then(res => {
        if (!res.ok) throw new Error("Backend timeline API unavailable");
        return res.json();
      })
      .then(data => {
        setTimestamps(data.timestamps || []);
        setGnnThreshold(data.threshold || 0.5);
      })
      .catch(err => {
        console.error(err);
        setErrorMsg("Failed to connect to backend server. Make sure API is running.");
      });

    fetch('http://localhost:5000/api/metrics')
      .then(res => res.json())
      .then(data => setMetrics(data))
      .catch(err => console.error("Error fetching metrics:", err));
      
    setLoading(false);
  }, []);

  // 2. Fetch Active Aircraft and Graph on Timestamp change
  useEffect(() => {
    if (timestamps.length === 0) return;
    const currentTS = timestamps[currentTimeIdx];
    
    // Fetch active aircraft at t
    fetch(`http://localhost:5000/api/aircraft?timestamp=${encodeURIComponent(currentTS)}`)
      .then(res => res.json())
      .then(data => setAircraftT(data.aircraft || []))
      .catch(err => console.error("Error fetching aircraft t:", err));
      
    // Fetch graph topology at t
    fetch(`http://localhost:5000/api/graph/${encodeURIComponent(currentTS)}`)
      .then(res => res.json())
      .then(data => setGraphData(data))
      .catch(err => console.error("Error fetching graph:", err));

    // Fetch aircraft at t+1 (for linear interpolation)
    if (currentTimeIdx < timestamps.length - 1) {
      const nextTS = timestamps[currentTimeIdx + 1];
      fetch(`http://localhost:5000/api/aircraft?timestamp=${encodeURIComponent(nextTS)}`)
        .then(res => res.json())
        .then(data => setAircraftT1(data.aircraft || []))
        .catch(err => console.error("Error fetching aircraft t+1:", err));
    } else {
      setAircraftT1([]);
    }
  }, [currentTimeIdx, timestamps]);

  // 3. Autoplay / Substep Timer Loop
  useEffect(() => {
    if (!playing || timestamps.length === 0) return;
    
    const fps = 8; // Refresh rate
    const stepDuration = 1000 / speedMultiplier;
    const intervalTime = stepDuration / fps;
    const increment = 1.0 / fps;

    const timer = setInterval(() => {
      setSubstep(prev => {
        if (prev + increment >= 1.0) {
          setCurrentTimeIdx(idx => {
            if (idx < timestamps.length - 1) {
              return idx + 1;
            } else {
              setPlaying(false);
              return idx;
            }
          });
          return 0.0;
        }
        return prev + increment;
      });
    }, intervalTime);

    return () => clearInterval(timer);
  }, [playing, timestamps, speedMultiplier]);

  // 4. Calculate Interpolated Aircraft Positions
  const getInterpolatedAircraft = () => {
    if (aircraftT.length === 0) return [];
    if (!playing || substep === 0.0 || aircraftT1.length === 0) {
      return aircraftT;
    }

    const t1Map = new Map(aircraftT1.map(ac => [ac.icao24, ac]));
    
    return aircraftT.map(ac_t => {
      const ac_t1 = t1Map.get(ac_t.icao24);
      if (!ac_t1) return ac_t; // Keep stationary if disappeared next second

      // Interpolate kinematics
      const lat = ac_t.latitude * (1 - substep) + ac_t1.latitude * substep;
      const lon = ac_t.longitude * (1 - substep) + ac_t1.longitude * substep;
      const alt = ac_t.altitude * (1 - substep) + ac_t1.altitude * substep;
      const gs = ac_t.groundspeed * (1 - substep) + ac_t1.groundspeed * substep;
      const vr = ac_t.vertical_rate * (1 - substep) + ac_t1.vertical_rate * substep;
      
      // Shortest path heading interpolation
      const h_t = ac_t.heading;
      const h_t1 = ac_t1.heading;
      const diff = ((h_t1 - h_t + 180) % 360) - 180;
      const heading = (h_t + diff * substep + 360) % 360;

      return {
        ...ac_t,
        latitude: lat,
        longitude: lon,
        altitude: alt,
        groundspeed: gs,
        heading: heading,
        vertical_rate: vr
      };
    });
  };

  const interpolatedAircraft = getInterpolatedAircraft();

  // Find active selected target
  const selectedAircraft = interpolatedAircraft.find(ac => ac.callsign === selectedCallsign) || null;

  // Active counts for dashboard status cards
  const n_active = interpolatedAircraft.length;
  const n_crit = interpolatedAircraft.filter(ac => ac.pred_label === 2).length;
  const n_warn = interpolatedAircraft.filter(ac => ac.pred_label === 1).length;
  const n_normal = n_active - n_crit - n_warn;
  const edge_count = graphData.edges ? graphData.edges.length : 0;

  // Auto-select first aircraft on start
  useEffect(() => {
    if (interpolatedAircraft.length > 0 && !selectedCallsign) {
      setSelectedCallsign(interpolatedAircraft[0].callsign);
    }
  }, [interpolatedAircraft]);

  // Keep a buffer of past observations for inline sparkline charting
  useEffect(() => {
    if (interpolatedAircraft.length === 0) return;
    setHistoryMap(prev => {
      const nextMap = { ...prev };
      interpolatedAircraft.forEach(ac => {
        const arr = nextMap[ac.icao24] || [];
        const tsStr = timestamps[currentTimeIdx];
        if (arr.length === 0 || arr[arr.length - 1].timestamp !== tsStr) {
          nextMap[ac.icao24] = [...arr, {
            timestamp: tsStr,
            altitude: ac.altitude,
            groundspeed: ac.groundspeed
          }].slice(-30);
        }
      });
      return nextMap;
    });
  }, [interpolatedAircraft, currentTimeIdx, timestamps]);

  // 5. Leaflet Map Initialization and Rendering
  useEffect(() => {
    if (!mapRef.current) return;

    // Create map instance centered on the airspace sector
    mapInstance.current = L.map(mapRef.current, {
      center: [53.4, -11.5],
      zoom: 7,
      zoomControl: false,
      attributionControl: false
    });

    // Dark styled basemap
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19
    }).addTo(mapInstance.current);

    layerGroup.current = L.layerGroup().addTo(mapInstance.current);

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, []);

  // Redraw map overlays (markers, lines, circles) on data updates
  useEffect(() => {
    if (!layerGroup.current) return;
    layerGroup.current.clearLayers();

    // 1. Draw Faint Concentric Range Circles around Lisbon receiver
    const rangeRings = [1200, 1300, 1400, 1500, 1600];
    rangeRings.forEach(r => {
      L.circle([RECEIVER_LAT, RECEIVER_LON], {
        radius: r * 1000,
        color: '#334155',
        weight: 0.7,
        dashArray: '4, 8',
        fill: false,
        className: 'radar-ring'
      }).addTo(layerGroup.current);
    });

    // 2. Draw Aircraft markers
    interpolatedAircraft.forEach(ac => {
      const isSelected = selectedAircraft && ac.icao24 === selectedAircraft.icao24;
      const isAnomalous = ac.pred_label > 0;
      
      let color = '#10b981'; // Green (Secure)
      let symbol = '●';
      if (ac.pred_label === 2) {
        color = '#ef4444'; // Red (Critical)
        symbol = '◆';
      } else if (ac.pred_label === 1) {
        color = '#f59e0b'; // Amber (Warning)
        symbol = '▲';
      }

      // Rotate arrow indicator by heading angle
      const rotation = ac.heading;

      // ATC styled div marker containing vector arrow and threat symbol
      const htmlContent = `
        <div style="position: relative; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;">
          <!-- Heading Vector Arrow -->
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style="transform: rotate(${rotation}deg); position: absolute; transition: transform 0.2s linear;">
            <path d="M12 2L16 10H13V18H11V10H8L12 2Z" fill="${color}" fill-opacity="${isSelected ? 0.9 : 0.3}" />
          </svg>
          <!-- Core Node Symbol -->
          <span style="color: ${color}; font-size: 14px; font-weight: bold; z-index: 10; text-shadow: 0 0 3px #000;">
            ${symbol}
          </span>
          ${isSelected ? `<div style="position: absolute; border: 1.5px solid #38bdf8; border-radius: 50%; width: 26px; height: 26px; animation: status-pulse 2s infinite;"></div>` : ''}
        </div>
      `;

      const marker = L.marker([ac.latitude, ac.longitude], {
        icon: L.divIcon({
          html: htmlContent,
          className: 'atc-marker',
          iconSize: [30, 30],
          iconAnchor: [15, 15]
        })
      });

      marker.on('click', () => setSelectedCallsign(ac.callsign));
      marker.addTo(layerGroup.current);

      // Label only for anomalous or selected aircraft to prevent clutter
      if (isSelected || isAnomalous) {
        const labelText = isAnomalous ? `${ac.callsign}<br><span style="font-size: 7px; color: ${color};">${ac.anomaly_type.toUpperCase()}</span>` : ac.callsign;
        const labelColor = isAnomalous ? color : '#38bdf8';
        
        L.popup({
          closeButton: false,
          closeOnClick: false,
          autoClose: false,
          offset: [0, -12],
          className: 'atc-label-popup'
        })
        .setLatLng([ac.latitude, ac.longitude])
        .setContent(`<div style="background-color: #090d16; border: 1px solid #1e293b; color: ${labelColor}; padding: 3px 6px; font-family: monospace; font-size: 9px; font-weight: bold; border-radius: 2px; text-align: center;">${labelText}</div>`)
        .addTo(layerGroup.current);
      }
    });

    // 3. Draw Neighbor Links for Selected target
    if (selectedAircraft && interpolatedAircraft.length > 1) {
      interpolatedAircraft.forEach(ac => {
        if (ac.icao24 === selectedAircraft.icao24) return;
        
        // Dynamic edge threshold: <= 100km
        const dx = selectedAircraft.x - ac.x;
        const dy = selectedAircraft.y - ac.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        
        if (dist <= PROXIMITY_THRESHOLD_M) {
          const lineColor = (selectedAircraft.pred_label === 2 || ac.pred_label === 2) ? '#ef4444' : '#38bdf8';
          L.polyline([
            [selectedAircraft.latitude, selectedAircraft.longitude],
            [ac.latitude, ac.longitude]
          ], {
            color: lineColor,
            weight: 1.5,
            dashArray: '3, 6',
            opacity: 0.8
          }).addTo(layerGroup.current);
        }
      });
    }

  }, [interpolatedAircraft, selectedCallsign]);

  // ==============================================================================
  // UI HANDLERS
  // ==============================================================================
  const togglePlay = () => setPlaying(!playing);
  const handleReset = () => {
    setPlaying(false);
    setCurrentTimeIdx(0);
    setSubstep(0.0);
  };
  const handleNext = () => {
    if (currentTimeIdx < timestamps.length - 1) {
      setCurrentTimeIdx(prev => prev + 1);
      setSubstep(0.0);
    }
  };
  const handlePrev = () => {
    if (currentTimeIdx > 0) {
      setCurrentTimeIdx(prev => prev - 1);
      setSubstep(0.0);
    }
  };

  const handleAlertClick = (callsign) => {
    setSelectedCallsign(callsign);
  };

  // SVG dynamic graph panel: Geographically locked node position mapping
  const renderSVGGraph = () => {
    const width = 450;
    const height = 260;
    const padding = 20;

    const mapRange = (val, inMin, inMax, outMin, outMax) => {
      return outMin + ((val - inMin) / (inMax - inMin)) * (outMax - outMin);
    };

    const nodes = graphData.nodes || [];
    const edges = graphData.edges || [];

    // Precalculate pixel coordinates for nodes
    const nodeCoords = nodes.map(n => {
      const px = mapRange(n.longitude, MAP_BOUNDS.minLon, MAP_BOUNDS.maxLon, padding, width - padding);
      const py = mapRange(n.latitude, MAP_BOUNDS.minLat, MAP_BOUNDS.maxLat, height - padding, padding); // Flip Y
      return { ...n, px, py };
    });

    return (
      <svg width="100%" height={height} className="border border-slate-800 rounded bg-slate-950 p-2">
        {/* Draw Edges */}
        {edges.map((e, index) => {
          const src = nodeCoords[e.source];
          const tgt = nodeCoords[e.target];
          if (!src || !tgt) return null;

          const isSelectedEdge = selectedAircraft && 
            (src.callsign === selectedAircraft.callsign || tgt.callsign === selectedAircraft.callsign);
          
          return (
            <line
              key={`edge-${index}`}
              x1={src.px}
              y1={src.py}
              x2={tgt.px}
              y2={tgt.py}
              stroke={isSelectedEdge ? '#38bdf8' : '#334155'}
              strokeWidth={isSelectedEdge ? 2 : 1}
              strokeDasharray={isSelectedEdge ? 'none' : '3,3'}
              opacity={0.8}
            />
          );
        })}

        {/* Draw Nodes */}
        {nodeCoords.map(node => {
          const isSelected = selectedAircraft && node.callsign === selectedAircraft.callsign;
          
          let fill = '#10b981';
          if (node.pred_label === 2) fill = '#ef4444';
          else if (node.pred_label === 1) fill = '#f59e0b';

          return (
            <g 
              key={`node-${node.id}`} 
              className="cursor-pointer" 
              onClick={() => setSelectedCallsign(node.callsign)}
            >
              <circle
                cx={node.px}
                cy={node.py}
                r={isSelected ? 10 : 5}
                fill={fill}
                stroke={isSelected ? '#38bdf8' : '#ffffff'}
                strokeWidth={isSelected ? 2 : 0.5}
              />
              <text
                x={node.px + 10}
                y={node.py + 4}
                fill={isSelected ? '#38bdf8' : '#f8fafc'}
                fontSize="8px"
                fontFamily="monospace"
                fontWeight={isSelected ? 'bold' : 'normal'}
              >
                {node.callsign}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  // Render SVG Inline Telemetry Sparkline charts
  const renderSparkline = (key, title, color) => {
    if (!selectedAircraft) return null;
    
    const history = historyMap[selectedAircraft.icao24] || [];
      
    if (history.length < 2) {
      return <div className="text-slate-500 font-mono text-xs py-4 text-center">COLLECTING RADAR HISTORY... ({history.length}/2)</div>;
    }

    const values = history.map(h => h[key]);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const valRange = maxVal - minVal || 1.0;

    const width = 200;
    const height = 40;
    const padding = 4;

    const points = history.map((h, i) => {
      const x = padding + (i / (history.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((h[key] - minVal) / valRange) * (height - 2 * padding);
      return `${x},${y}`;
    }).join(' ');

    return (
      <div className="flex flex-col gap-1 border border-slate-800 rounded bg-slate-950 p-2 text-center">
        <span className="text-[10px] text-slate-400 font-mono font-bold uppercase">{title} HISTORY</span>
        <svg width="100%" height={height}>
          <polyline
            fill="none"
            stroke={color}
            strokeWidth="1.5"
            points={points}
          />
        </svg>
        <span className="text-[9px] text-slate-500 font-mono">
          Min: {minVal.toFixed(0)} | Max: {maxVal.toFixed(0)}
        </span>
      </div>
    );
  };

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight">AEROGUARD</h1>
          <p className="text-xs font-mono text-slate-400">Flight Trajectory Anomaly & ADS-B Spoofing Detection using GNN</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <span className="status-online">● SYSTEM MONITOR ONLINE</span>
            <div className="text-[10px] font-mono text-slate-500">UTC CLOCK: {timestamps[currentTimeIdx]}</div>
          </div>
        </div>
      </header>

      {errorMsg && (
        <div className="mb-4 p-3 bg-red-950/50 border border-red-800 text-red-200 rounded text-sm font-mono text-center">
          ⚠ {errorMsg}
        </div>
      )}

      {/* KPI Cards */}
      <section className="grid grid-cols-4 gap-4 mb-6">
        <div className="atc-panel py-3 text-center border-l-4 border-l-sky-500">
          <div className="text-[10px] font-mono font-bold text-slate-400">AIRCRAFT TRACKED</div>
          <div className="text-2xl font-mono font-bold text-sky-400 mt-1">{n_active}</div>
        </div>
        <div className="atc-panel py-3 text-center border-l-4 border-l-emerald-500">
          <div className="text-[10px] font-mono font-bold text-slate-400">SECURE TRAJECTORIES</div>
          <div className="text-2xl font-mono font-bold text-emerald-400 mt-1">{n_normal}</div>
        </div>
        <div className="atc-panel py-3 text-center border-l-4 border-l-red-500">
          <div className="text-[10px] font-mono font-bold text-slate-400">ANOMALOUS ALERTS</div>
          <div className="text-2xl font-mono font-bold text-red-500 mt-1">{n_crit + n_warn}</div>
        </div>
        <div className="atc-panel py-3 text-center border-l-4 border-l-slate-500">
          <div className="text-[10px] font-mono font-bold text-slate-400">ACTIVE GRAPH EDGES</div>
          <div className="text-2xl font-mono font-bold text-slate-400 mt-1">{edge_count}</div>
        </div>
      </section>

      {/* Main Map Panel */}
      <section className="atc-panel mb-6" style={{ height: '400px' }}>
        <div ref={mapRef} className="rounded" style={{ width: '100%', height: '100%' }}></div>
      </section>

      {/* Middle Grid: Graph & Threats */}
      <section className="grid grid-cols-2 gap-6 mb-6">
        {/* Dynamic Graph */}
        <div className="atc-panel flex flex-col gap-2">
          <h2 className="atc-panel-title">🕸️ Dynamic Aircraft Graph</h2>
          {renderSVGGraph()}
        </div>

        {/* Active Threats Log */}
        <div className="atc-panel flex flex-col gap-2">
          <h2 className="atc-panel-title">⚠ Active Threat alerts</h2>
          <div className="custom-scrollbar h-[260px] border border-slate-800 rounded bg-slate-950 p-2">
            <table className="w-full text-left font-mono text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-slate-800">
                  <th className="py-2 pl-2">CALLSIGN</th>
                  <th className="py-2">IDENT</th>
                  <th className="py-2">PROFILE</th>
                  <th className="py-2">STATUS</th>
                  <th className="py-2 pr-2">SCORE</th>
                </tr>
              </thead>
              <tbody>
                {interpolatedAircraft
                  .filter(ac => ac.pred_label > 0)
                  .map(ac => {
                    const statusText = ac.pred_label === 2 ? 'CRITICAL' : 'WARNING';
                    const textColor = ac.pred_label === 2 ? 'text-red-500' : 'text-amber-500';
                    const isSelected = selectedAircraft && ac.icao24 === selectedAircraft.icao24;
                    
                    return (
                      <tr 
                        key={ac.icao24} 
                        className={`hover:bg-slate-900 cursor-pointer ${isSelected ? 'bg-sky-950/40 border-l-2 border-l-sky-500' : ''}`}
                        onClick={() => handleAlertClick(ac.callsign)}
                      >
                        <td className="py-2 pl-2 font-bold">{ac.callsign}</td>
                        <td className="py-2">{ac.icao24}</td>
                        <td className="py-2 text-slate-300">{ac.anomaly_type.toUpperCase().replace('_', ' ')}</td>
                        <td className={`py-2 font-bold ${textColor}`}>{statusText}</td>
                        <td className="py-2 pr-2 font-bold">{ac.anomaly_score.toFixed(3)}</td>
                      </tr>
                    );
                  })}
                {interpolatedAircraft.filter(ac => ac.pred_label > 0).length === 0 && (
                  <tr>
                    <td colSpan="5" className="text-center py-20 text-slate-600">
                      NO AIRSPACE THREATS RECORDED IN THIS BLOCK
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Bottom Grid: Telemetry Inspector & Comm Signals */}
      <section className="grid grid-cols-3 gap-6 mb-6">
        {/* Kinematics Details */}
        <div className="atc-panel col-span-2 flex flex-col gap-2">
          <h2 className="atc-panel-title">🔍 Target Kinematics</h2>
          {selectedAircraft ? (
            <div className="grid grid-cols-3 gap-4 font-mono text-xs">
              <div className="flex flex-col gap-2 bg-slate-950 p-3 border border-slate-800 rounded">
                <div><span className="text-slate-500">TARGET:</span> <span className="text-sky-400 font-bold">{selectedAircraft.callsign}</span></div>
                <div><span className="text-slate-500">ICAO24:</span> <span>{selectedAircraft.icao24}</span></div>
                <div><span className="text-slate-500">STATUS:</span> <span className={selectedAircraft.pred_label === 2 ? 'text-red-500 font-bold' : (selectedAircraft.pred_label === 1 ? 'text-amber-500 font-bold' : 'text-emerald-500 font-bold')}>{selectedAircraft.pred_label === 2 ? '🚨 ALERT' : (selectedAircraft.pred_label === 1 ? '⚠️ WARNING' : '✅ SECURE')}</span></div>
                {selectedAircraft.pred_label > 0 && (
                  <div><span className="text-slate-500">PROFILE:</span> <span className="text-red-400">{selectedAircraft.anomaly_type.toUpperCase()}</span></div>
                )}
              </div>
              <div className="flex flex-col gap-1.5 bg-slate-950 p-3 border border-slate-800 rounded">
                <div><span className="text-slate-500">LATITUDE:</span> <span>{selectedAircraft.latitude.toFixed(4)}°</span></div>
                <div><span className="text-slate-500">LONGITUDE:</span> <span>{selectedAircraft.longitude.toFixed(4)}°</span></div>
                <div><span className="text-slate-500">ALTITUDE:</span> <span>{selectedAircraft.altitude.toFixed(0)} ft</span></div>
                <div><span className="text-slate-500">GEO ALT:</span> <span>{selectedAircraft.geoaltitude.toFixed(0)} ft</span></div>
              </div>
              <div className="flex flex-col gap-1.5 bg-slate-950 p-3 border border-slate-800 rounded">
                <div><span className="text-slate-500">SPEED:</span> <span>{selectedAircraft.groundspeed.toFixed(1)} kt</span></div>
                <div><span className="text-slate-500">HEADING:</span> <span>{selectedAircraft.heading.toFixed(1)}°</span></div>
                <div><span className="text-slate-500">VERT RATE:</span> <span>{selectedAircraft.vertical_rate.toFixed(0)} fpm</span></div>
                <div><span className="text-slate-500 font-bold">SCORE:</span> <span className="text-red-400 font-bold">{selectedAircraft.anomaly_score.toFixed(4)}</span></div>
              </div>
            </div>
          ) : (
            <div className="text-center text-slate-500 font-mono py-12">SELECT TARGET AIRCRAFT FOR TELEMETRY INSPECTION</div>
          )}
        </div>

        {/* Communication Telemetry */}
        <div className="atc-panel flex flex-col gap-2">
          <h2 className="atc-panel-title">📡 Communication (Simulated)</h2>
          {selectedAircraft ? (
            <div className="flex flex-col gap-1 bg-slate-950 p-3 border border-slate-800 rounded font-mono text-xs">
              <div className="text-[9px] text-slate-500 font-bold uppercase mb-2 border-b border-slate-800 pb-1">Derived signal parameters</div>
              <div className="flex justify-between"><span>CARRIER FREQ:</span> <span className="text-slate-300">1090 MHz</span></div>
              <div className="flex justify-between"><span>EST DOPPLER:</span> <span className="text-sky-400">{selectedAircraft.estimated_doppler_hz.toFixed(2)} Hz</span></div>
              <div className="flex justify-between"><span>EST POWER:</span> <span className="text-emerald-400">{selectedAircraft.estimated_rss_dbm.toFixed(2)} dBm</span></div>
              <div className="flex justify-between"><span>EST SNR:</span> <span className="text-emerald-400">{selectedAircraft.estimated_snr_db.toFixed(2)} dB</span></div>
              <div className="flex justify-between"><span>RANGE TO RX:</span> <span className="text-slate-300">{(selectedAircraft.distance_to_receiver / 1000.0).toFixed(2)} km</span></div>
            </div>
          ) : (
            <div className="text-center text-slate-500 font-mono py-12">NO COMMUNICATIONS TELEMETRY LOGGED</div>
          )}
        </div>
      </section>

      {/* Sparkline History Charts */}
      {selectedAircraft && (
        <section className="grid grid-cols-2 gap-6 mb-6">
          {renderSparkline('altitude', 'Altitude', '#10b981')}
          {renderSparkline('groundspeed', 'Speed', '#38bdf8')}
        </section>
      )}

      {/* Playback Timeline Controller */}
      <footer className="atc-panel flex flex-col gap-4">
        <div className="flex justify-between items-center gap-4">
          {/* Controls buttons */}
          <div className="flex items-center gap-2">
            <button className="atc-button" onClick={handlePrev} disabled={currentTimeIdx === 0}>◀ PREV</button>
            <button 
              className={`atc-button ${playing ? 'active' : ''}`} 
              onClick={togglePlay}
            >
              {playing ? '⏸ PAUSE' : '▶ AUTOPLAY'}
            </button>
            <button className="atc-button" onClick={handleNext} disabled={currentTimeIdx === timestamps.length - 1}>NEXT ▶</button>
            <button className="atc-button" onClick={handleReset}>🔄 RESET</button>
          </div>

          {/* Speed Selector */}
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-slate-500 font-bold">SPEED:</span>
            {[0.5, 1.0, 2.0].map(s => (
              <button
                key={`speed-${s}`}
                className={`px-2 py-1 rounded border border-slate-800 ${speedMultiplier === s ? 'bg-sky-500 text-slate-950 font-bold' : 'bg-slate-900 text-slate-300'}`}
                onClick={() => setSpeedMultiplier(s)}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Time slider */}
        <div className="w-full flex items-center gap-4">
          <input
            type="range"
            min="0"
            max={timestamps.length - 1}
            value={currentTimeIdx}
            onChange={(e) => {
              setCurrentTimeIdx(Number(e.target.value));
              setSubstep(0.0);
            }}
            className="w-full accent-sky-500 bg-slate-950 cursor-pointer h-1.5 rounded-lg border border-slate-800"
          />
        </div>
      </footer>
    </div>
  );
}


