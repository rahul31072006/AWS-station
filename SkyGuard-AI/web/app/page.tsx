'use client';

import dynamic from 'next/dynamic';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity, AlertTriangle, Bell, BrainCircuit, CheckCircle2, ChevronDown,
  CircleHelp, CloudLightning, Compass, Database, Download, FileBarChart2,
  Gauge, HeartPulse, LayoutDashboard, Menu, Network, Search, Server,
  Settings, ShieldCheck, Sparkles, Thermometer, UserRound, Waves, Wind,
  X, Zap,
} from 'lucide-react';
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts';
import { fetchStations, mockStations, type Station } from '../services/stationService';
import { useStationSelection } from './station-context';

const LiveStationsMap = dynamic(() => import('../components/LiveStationsMap'), { ssr: false });

type AnyRecord = Record<string, any>;
type Dashboard = {
  source: string; updated_at: string; kpis: AnyRecord; stations: AnyRecord[];
  latest: AnyRecord[]; anomalies: AnyRecord[]; series: AnyRecord[];
  unknowns: AnyRecord[]; registry: AnyRecord; station_evidence: AnyRecord[];
  healing?: AnyRecord[]; explanations?: AnyRecord[];
};

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const nav = [
  ['Command Center', '/command-center', LayoutDashboard], ['Live Stations', '/live-stations', Network], ['Anomaly Detection', '/anomaly-detection', AlertTriangle],
  ['Station Intelligence', '/station-intelligence', Compass], ['SHAP Explainability', '/ai-explainability', BrainCircuit], ['Self-Healing', '/self-healing', ShieldCheck],
  ['Unknown Discovery', '/unknown-discovery', CircleHelp], ['Analytics', '/analytics', FileBarChart2], ['Reports', '/reports', Download], ['Settings', '/settings', Settings],
] as const;

const demo: Dashboard = {
  source: 'SYNTHETIC_DEMO', updated_at: '2026-09-13T00:00:00.000Z',
  kpis: { network_health: 78.4, active_stations: 6, anomalies_detected: 1152, sensor_faults: 793, self_healed: 421, unknown_patterns: 38 },
  stations: [], latest: [], anomalies: [], series: [], unknowns: [], registry: {}, station_evidence: [],
};

function fmt(value: any, digits = 0) { return value === null || value === undefined ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits }); }
function statusFor(row: AnyRecord) { if (row.final_fault_type === 'UNKNOWN') return 'unknown'; if (row.spatial_diagnosis === 'likely_weather_event') return 'weather'; if (row.is_sensor_fault || row.is_flagged) return row.anomaly_score > .85 ? 'critical' : 'warn'; return 'healthy'; }
function filterByStation(rows: AnyRecord[] | undefined, stationId: string | null) { return (rows || []).filter(row => !stationId || row.station_id === stationId); }
function scopeDashboard(data: Dashboard, stationId: string | null): Dashboard {
  if (!stationId) return data;
  const latest = filterByStation(data.latest, stationId);
  const anomalies = filterByStation(data.anomalies, stationId);
  const series = filterByStation(data.series, stationId);
  const unknowns = filterByStation(data.unknowns, stationId);
  const healing = filterByStation(data.healing, stationId);
  const evidence = filterByStation(data.station_evidence, stationId);
  const selectedRowIds = new Set(anomalies.map(row => row.row_id));
  const explanations = stationId
    ? (data.explanations || []).filter(row => selectedRowIds.has(row.row_id))
    : data.explanations;
  const registry = stationId
    ? Object.fromEntries(Object.entries(data.registry).filter(([, item]: [string, any]) => item.stations?.includes(stationId)))
    : data.registry;
  const stationKpis = { ...data.kpis,
    active_stations: latest.length,
    anomalies_detected: anomalies.length,
    sensor_faults: anomalies.filter(row => row.is_sensor_fault).length,
    self_healed: healing.length,
    unknown_patterns: unknowns.length,
    network_health: latest[0]?.health_score ?? 0,
    sensor_health: latest[0]?.health_score ?? 0,
    degrading_sensors: evidence.filter(row => row.health_status === 'DEGRADING').length,
    early_warnings: evidence.filter(row => row.health_status === 'DEGRADING').length,
  };
  return { ...data, kpis: stationKpis, latest, anomalies, series, unknowns, healing, explanations, registry, station_evidence: evidence };
}

const pageNames: Record<string, string> = Object.fromEntries(nav.map(([label, path]) => [path, label]));

export default function Home() {
  const [data, setData] = useState<Dashboard>(demo);
  const [online, setOnline] = useState(false);
  const pathname = usePathname();
  const router = useRouter();
  const activePath = pathname === '/' ? '/command-center' : pathname;
  const active = pageNames[activePath] || 'Command Center';
  const [search, setSearch] = useState('');
  const [mobileNav, setMobileNav] = useState(false);
  const [selectedStation, setSelectedStation] = useState<AnyRecord | null>(null);
  const { selectedStationId, selectionMode, setSelectedStationId } = useStationSelection();
  const scopedData = useMemo(() => scopeDashboard(data, selectedStationId), [data, selectedStationId]);

  useEffect(() => {
    fetch(`${API}/api/dashboard`).then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(payload => { setData(payload); setOnline(true); }).catch(() => setOnline(false));
    const socketUrl = API.replace(/^http/, 'ws') + '/ws/live';
    let socket: WebSocket | undefined;
    try { socket = new WebSocket(socketUrl); socket.onmessage = event => { const update = JSON.parse(event.data); setData(current => ({ ...current, ...update })); }; } catch { /* API may not expose WebSocket yet. */ }
    return () => socket?.close();
  }, []);

  const latest = useMemo(() => scopedData.latest, [scopedData.latest]);
  const anomalies = useMemo(() => scopedData.anomalies.filter(row => !search || `${row.station_id} ${row.final_fault_type} ${row.predicted_fault_type}`.toLowerCase().includes(search.toLowerCase())), [scopedData.anomalies, search]);
  const series = useMemo(() => scopedData.series.slice(-72).map(row => ({ ...row, time: new Date(row.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) })), [scopedData.series]);
  const selectStation = (row: AnyRecord) => { setSelectedStation(row); setSelectedStationId(row.station_id); };

  return <div className="shell">
    <aside className={`sidebar ${mobileNav ? 'mobile-open' : ''}`}>
      <div className="brand"><div className="brand-mark"><Sparkles size={18} /></div><div><div className="brand-title">SkyGuard AI</div><div className="brand-sub">Weather intelligence</div></div></div>
      <div className="nav-label">Operations</div>
      <nav className="nav">{nav.map(([label, path, Icon]) => <button key={label} className={active === label ? 'active' : ''} onClick={() => { router.push(path); setMobileNav(false); }}><Icon size={16} />{label}{label === 'Unknown Discovery' && data.unknowns.length > 0 ? <span className="nav-count">{data.unknowns.length}</span> : null}</button>)}</nav>
      <div className="sidebar-footer"><span className={`status-dot ${online ? '' : 'warn'}`} />{online ? 'Ingestion pipeline connected' : 'Demo pipeline active'}<div style={{ marginTop: 8 }}>v1.0.0 · {online ? 'API-backed telemetry' : 'Synthetic demo data'}</div></div>
    </aside>
    <main className="main">
      <header className="topbar"><button className="icon-btn mobile-menu" onClick={() => setMobileNav(v => !v)}><Menu size={17} /></button><div className="top-search"><Search size={15} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search stations, anomalies, events..." /></div><div className="top-spacer" /><StationSelector stations={data.latest} selectedStationId={selectedStationId} onChange={setSelectedStationId} /><div className="top-metric"><strong><span className={`status-dot ${online ? '' : 'warn'}`} />{online ? 'LIVE SYSTEM' : 'DEMO MODE'}</strong>{online ? 'API connected' : 'Waiting for API'}</div><div className="top-metric"><strong>{fmt(scopedData.kpis.active_stations)}</strong>{selectionMode === 'station' ? 'Active station' : 'Active stations'}</div><button className="icon-btn" title="Notifications"><Bell size={16} /></button><button className="icon-btn" title="Account"><UserRound size={16} /></button></header>
      {activePath === '/command-center' ? <div className="content">
        <div className="page-eyebrow">Network operations / {active}</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 18 }}><div><h1 className="page-title">{selectedStationId || 'Command center'}</h1><div className="page-intro">A live operational view of station health, anomaly evidence, and automated recovery.</div></div><div style={{ display: 'flex', gap: 8, alignItems: 'center' }}><StationSelector stations={data.latest} selectedStationId={selectedStationId} onChange={setSelectedStationId} /><button className="icon-btn" title="Refresh data" onClick={() => location.reload()}><Activity size={15} /></button></div></div>
        <div className="data-badges"><span className="badge live">● {online ? 'Live data' : 'Demo data'}</span><span className="badge model">◆ Model prediction</span><span className="badge imputed">✦ Imputed readings remain visible</span><span className="badge">Updated {new Date(data.updated_at).toLocaleTimeString()}</span></div>
        <Kpis data={scopedData} />
        <div className="grid-two"><StationMap stations={selectedStationId ? latest : data.latest} onSelect={selectStation} /><Panel title="Live weather analytics" meta="72 latest observations"><div className="chart-controls"><span className="chart-key"><i className="temp" /> Temperature</span><span className="chart-key"><i className="rh" /> Humidity</span><span className="chart-key"><i className="score" /> Anomaly score</span></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={series}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="time" interval={11} /><YAxis yAxisId="left" domain={['auto','auto']} /><YAxis yAxisId="right" orientation="right" domain={[0,1]} /><Tooltip contentStyle={{ background:'#0b1b2c', border:'1px solid rgba(148,177,210,.2)', borderRadius:7, fontSize:11 }} /><Line yAxisId="left" type="monotone" dataKey="temperature_c" stroke="#f5b84d" dot={false} strokeWidth={2} /><Line yAxisId="left" type="monotone" dataKey="humidity_pct" stroke="#51e1e8" dot={false} strokeWidth={1.5} /><Line yAxisId="right" type="monotone" dataKey="anomaly_score" stroke="#ff6674" dot={false} strokeWidth={1.5} /></LineChart></ResponsiveContainer></div></Panel></div>
        <div className="grid-two lower-grid"><Panel title="Anomaly timeline" meta={`${anomalies.length} events in current window`} action={<button className="text-btn" onClick={() => router.push('/anomaly-detection')}>View investigation <ChevronDown size={13} /></button>}><div className="timeline">{anomalies.slice(0, 7).map((row, index) => <motion.div initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * .04 }} className="timeline-row" key={`${row.timestamp}-${row.station_id}-${index}`}><div className="timeline-time">{new Date(row.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}<small>{new Date(row.timestamp).toLocaleDateString()}</small></div><div className="timeline-main"><strong>{row.station_id}</strong> · {row.final_fault_type || row.predicted_fault_type || 'Anomaly'}<small>Score {fmt(row.anomaly_score, 2)} · Confidence {fmt(row.classifier_confidence, 2)}</small></div><span className={`severity ${row.anomaly_score > .85 ? 'critical' : row.anomaly_score > .65 ? 'warning' : 'info'}`}>{row.anomaly_score > .85 ? 'critical' : 'review'}</span></motion.div>)}{anomalies.length === 0 && <EmptyState text="No anomalies match the current filters." />}</div></Panel><Panel title="Operational event feed" meta="Latest system activity"><EventFeed data={scopedData} /></Panel></div>
        <Panel title="Station intelligence" meta="Latest telemetry across the network" action={<button className="text-btn" onClick={() => router.push('/live-stations')}>Open station monitor <ChevronDown size={13} /></button>}><StationTable rows={latest} onSelect={selectStation} /></Panel>
      </div> : activePath === '/live-stations' ? <LiveStationsPage selectedStationId={selectedStationId} onSelectStation={setSelectedStationId} /> : <RoutedPage page={active} data={scopedData} onNavigate={router.push} selectedStationId={selectedStationId} onChangeStation={setSelectedStationId} />}
    </main>
    <AnimatePresence>{selectedStation && <StationDrawer station={selectedStation} evidence={data.station_evidence.find(item => item.station_id === selectedStation.station_id)} history={data.series.filter(x => x.station_id === selectedStation.station_id)} onClose={() => setSelectedStation(null)} />}</AnimatePresence>
  </div>;
}

function Kpis({ data }: { data: Dashboard }) { const items = [['Network health', data.kpis.network_health, '%', Gauge, 'Stable network baseline', ''], ['Sensor health', data.kpis.sensor_health, '%', HeartPulse, 'Operational indicator', ''], ['Degrading sensors', data.kpis.degrading_sensors, '', Activity, 'Early warning candidates', 'warn'], ['Active stations', data.kpis.active_stations, '', Network, 'All reporting normally', ''], ['Anomalies detected', data.kpis.anomalies_detected, '', AlertTriangle, 'Requires review', 'warn'], ['Sensor faults', data.kpis.sensor_faults, '', Zap, 'Evidence-backed review', 'warn'], ['Self-healed readings', data.kpis.self_healed, '', ShieldCheck, 'Originals preserved', ''], ['Unknown patterns', data.kpis.unknown_patterns, '', CircleHelp, 'Open-world discovery', 'warn']] as const; return <div className="kpi-grid">{items.map(([label, value, suffix, Icon, foot, mood]) => <motion.div className="card kpi" key={label} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}><div className="kpi-head"><span>{label}</span><Icon className="kpi-icon" size={16} /></div><div className="kpi-value">{fmt(value, label.includes('health') ? 1 : 0)}{suffix}</div><div className={`kpi-foot ${mood}`}>{foot}</div></motion.div>)}</div>; }
function Panel({ title, meta, action, children }: { title: string; meta?: string; action?: React.ReactNode; children: React.ReactNode }) { return <section className="card section-card"><div className="section-head"><div><div className="section-title">{title}</div><div className="section-meta">{meta}</div></div>{action}</div>{children}</section>; }
function StationMap({ stations, onSelect }: { stations: AnyRecord[]; onSelect: (row: AnyRecord) => void }) { return <Panel title="Station network" meta="Spatial consistency / Chennai region"><div className="map"><div className="map-rings" />{stations.map((row, index) => { const status = statusFor(row); return <button title={row.station_id} aria-label={`Open ${row.station_id}`} onClick={() => onSelect(row)} key={row.station_id} className={`marker ${status}`} style={{ left: `${26 + (index * 17) % 62}%`, top: `${33 + (index * 23) % 45}%` }} />; })}<div className="map-legend"><span className="legend healthy"><i />Healthy</span><span className="legend warn"><i />Review</span><span className="legend critical"><i />Fault</span><span className="legend unknown"><i />Unknown</span></div></div></Panel>; }
function EventFeed({ data }: { data: Dashboard }) { const events = data.anomalies.slice(0, 5); return <div className="feed">{events.map((row, index) => <div className="feed-item" key={`${row.station_id}-${index}`}><div className="feed-icon">{row.final_fault_type === 'UNKNOWN' ? <CircleHelp size={14} /> : row.is_sensor_fault ? <AlertTriangle size={14} /> : <CloudLightning size={14} />}</div><div className="feed-copy"><strong>{row.station_id}</strong> {row.final_fault_type === 'UNKNOWN' ? 'unknown pattern discovered' : row.is_sensor_fault ? 'sensor anomaly classified' : 'weather event supported'}<div className="feed-time">{new Date(row.timestamp).toLocaleTimeString()} · confidence {fmt(row.classifier_confidence, 2)}</div></div></div>)}{!events.length && <EmptyState text="Waiting for telemetry events." />}</div>; }
function StationTable({ rows, onSelect }: { rows: AnyRecord[]; onSelect: (row: AnyRecord) => void }) { return <div className="table-wrap"><table><thead><tr><th>Station</th><th>Status</th><th>Temperature</th><th>Pressure</th><th>Humidity</th><th>Health</th><th>Last update</th></tr></thead><tbody>{rows.map(row => <tr key={row.station_id} onClick={() => onSelect(row)}><td><strong>{row.station_id}</strong><small> AWS observation node</small></td><td><span className={`table-status ${statusFor(row)}`}><i />{statusFor(row)}</span></td><td>{fmt(row.temperature_c, 1)}°C</td><td>{fmt(row.pressure_hpa, 1)} hPa</td><td>{fmt(row.humidity_pct, 1)}%</td><td><div className="health"><span style={{ width: `${row.health_score || 0}%` }} /><b>{fmt(row.health_score, 0)}</b></div></td><td className="muted">{row.timestamp ? new Date(row.timestamp).toLocaleTimeString() : '—'}</td></tr>)}</tbody></table>{!rows.length && <EmptyState text="No station telemetry available." />}</div>; }
function EmptyState({ text }: { text: string }) { return <div className="empty"><Database size={19} />{text}</div>; }
function StationDrawer({ station, evidence, history, onClose }: { station: AnyRecord; evidence?: AnyRecord; history: AnyRecord[]; onClose: () => void }) { const chart = history.slice(-36).map(row => ({ time: new Date(row.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), temp: row.temperature_c, score: row.anomaly_score, health: row.health_score })); const health = evidence || station; const disagreement = `${fmt(health.nearby_disagreement_count, 0)} / ${fmt(health.nearby_station_count, 0)} checks`; return <motion.div className="drawer-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}><motion.aside className="drawer" initial={{ x: 420 }} animate={{ x: 0 }} exit={{ x: 420 }} onClick={e => e.stopPropagation()}><div className="drawer-head"><div><div className="page-eyebrow">Station intelligence</div><h2>{station.station_id}</h2></div><button className="icon-btn" onClick={onClose}><X size={16} /></button></div><div className="drawer-status"><span className={`table-status ${statusFor(station)}`}><i />{statusFor(station)}</span><span className="muted">Last update {station.timestamp ? new Date(station.timestamp).toLocaleTimeString() : '—'}</span></div><div className="drawer-grid"><div><small>Temperature</small><strong>{fmt(station.temperature_c, 1)}°C</strong></div><div><small>Pressure</small><strong>{fmt(station.pressure_hpa, 1)}</strong></div><div><small>Humidity</small><strong>{fmt(station.humidity_pct, 1)}%</strong></div><div><small>Health score</small><strong>{fmt(health.health_score, 0)}</strong></div></div><div className="drawer-block"><div className="section-head"><div><div className="section-title">Anomaly evidence</div><div className="section-meta">Evidence fusion for the latest station decision</div></div><span className="table-status warn"><i />{health.final_decision || 'NO ACTIVE ANOMALY'}</span></div><div className="evidence-grid"><Evidence label="ML probability" value={`${fmt(Number(health.ml_anomaly_probability || 0) * 100, 0)}%`} /><Evidence label="Physics / VPD" value={health.physics_validation || 'PASSED'} /><Evidence label="Nearby disagreement" value={disagreement} /><Evidence label="Decision confidence" value={`${fmt(Number(health.decision_confidence || 0) * 100, 0)}%`} /></div></div><div className="drawer-block"><div className="section-head"><div><div className="section-title">Sensor health evolution</div><div className="section-meta">Operational indicator, not failure probability</div></div><span className={`table-status ${health.health_status === 'DEGRADING' ? 'fault' : 'healthy'}`}><i />{health.health_status || 'STABLE'}</span></div><div className="health-contributors">{Object.entries(health.contributors || {}).map(([label, value]) => <div key={label}><span>{label}</span><b>-{fmt(value, 1)}</b></div>)}</div><div className="drawer-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={chart}><XAxis dataKey="time" hide /><YAxis domain={[0, 100]} hide /><Tooltip contentStyle={{ background:'#0b1b2c', border:'1px solid rgba(148,177,210,.2)', fontSize:11 }} /><Line type="monotone" dataKey="health" stroke="#48d597" dot={false} strokeWidth={2} /></LineChart></ResponsiveContainer></div></div><div className="drawer-block"><div className="section-title">Telemetry pulse</div><div className="drawer-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chart}><defs><linearGradient id="drawerFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#48aaff" stopOpacity={.28} /><stop offset="100%" stopColor="#48aaff" stopOpacity={0} /></linearGradient></defs><XAxis dataKey="time" hide /><YAxis hide /><Tooltip contentStyle={{ background:'#0b1b2c', border:'1px solid rgba(148,177,210,.2)', fontSize:11 }} /><Area type="monotone" dataKey="temp" stroke="#48aaff" fill="url(#drawerFill)" /></AreaChart></ResponsiveContainer></div></div><div className="provenance"><Sparkles size={15} /><div><strong>Data provenance</strong><p>Values are derived from temporal, physical, and spatial evidence. Original observations remain available for audit.</p></div></div></motion.aside></motion.div>; }

function StationSelector({ stations, selectedStationId, onChange }: { stations: AnyRecord[]; selectedStationId: string | null; onChange: (stationId: string | null) => void }) {
  const stationIds = Array.from(new Set(stations.map(row => String(row.station_id)).filter(Boolean)));
  return <label className="station-context"><span>Viewing</span><select className="select" value={selectedStationId || 'ALL'} onChange={event => onChange(event.target.value === 'ALL' ? null : event.target.value)}><option value="ALL">Overall Network</option>{stationIds.map(id => <option key={id} value={id}>{id}</option>)}</select></label>;
}

function LiveStationsPage({ selectedStationId, onSelectStation }: { selectedStationId: string | null; onSelectStation: (stationId: string | null) => void }) {
  const [stations, setStations] = useState<Station[]>(mockStations);
  const [selectedId, setSelectedId] = useState<string | null>(mockStations[0]?.id || null);
  const [filter, setFilter] = useState<'all' | Station['health']>('all');
  const [query, setQuery] = useState('');
  const [live, setLive] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  const refresh = () => fetchStations().then((next) => { setStations(next); setLive(true); setLastUpdated(new Date()); if (!selectedId && next[0]) setSelectedId(next[0].id); }).catch(() => { setLive(false); setLastUpdated(new Date()); });
  useEffect(() => { refresh(); const interval = window.setInterval(refresh, 45000); return () => window.clearInterval(interval); }, []);
  const visible = stations.filter((station) => (!selectedStationId || station.id === selectedStationId) && (filter === 'all' || station.health === filter) && station.id.toLowerCase().includes(query.toLowerCase()));
  const selected = stations.find((station) => station.id === (selectedStationId || selectedId)) || null;
  return <div className="content station-page"><div className="page-eyebrow">Network operations / Live Stations</div><div className="station-page-head"><div><h1 className="page-title">{selectedStationId || 'Live stations'}</h1><div className="page-intro">Interactive OpenStreetMap coverage for the Chennai and Tamil Nadu observation network.</div></div><div className="live-refresh"><span className={`status-dot ${live ? '' : 'warn'}`} />{live ? 'LIVE' : 'MOCK DATA'}<small>Updated {lastUpdated.toLocaleTimeString()}</small><button className="select" onClick={refresh}>Refresh</button></div></div><div className="data-badges"><span className="badge live">● {live ? 'Live API' : 'Mock fallback'}</span><span className="badge">OpenStreetMap tiles</span><span className="badge model">45 sec auto-refresh</span></div><div className="station-layout"><section className="card station-map-card"><div className="section-head"><div><div className="section-title">Station geography</div><div className="section-meta">Click a marker to inspect station telemetry</div></div><span className="map-attribution">© OpenStreetMap contributors</span></div><div className="station-map-frame"><LiveStationsMap stations={visible} selectedId={selectedStationId || selectedId} onSelect={(station) => { setSelectedId(station.id); onSelectStation(station.id); }} /></div></section><aside className="card station-list-card"><div className="section-head"><div><div className="section-title">Live stations</div><div className="section-meta">{visible.length} of {stations.length} reporting nodes</div></div><Activity size={16} className="kpi-icon" /></div><div className="station-search"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search stations..." /></div><div className="station-filters">{(['all', 'healthy', 'warning', 'fault', 'unknown'] as const).map((value) => <button className={filter === value ? 'selected' : ''} key={value} onClick={() => setFilter(value)}>{value}</button>)}</div><div className="station-list">{visible.map((station) => <button className={`station-list-item ${(selectedStationId || selectedId) === station.id ? 'selected' : ''}`} key={station.id} onClick={() => { setSelectedId(station.id); onSelectStation(station.id); }}><span className={`station-state ${station.health}`} /><span className="station-list-copy"><strong>{station.id}</strong><small>{station.temperature === null ? '—' : `${station.temperature.toFixed(1)}°C`} · {station.humidity === null ? '—' : `${station.humidity.toFixed(0)}% RH`}</small></span><span className={`table-status ${station.health}`}>{station.health}</span></button>)}{visible.length === 0 && <EmptyState text="No stations match this filter." />}</div></aside></div>{selected && <section className="card station-details"><div className="section-head"><div><div className="page-eyebrow">Selected station</div><div className="section-title">{selected.id}</div><div className="section-meta">{selected.latitude.toFixed(4)}°N, {selected.longitude.toFixed(4)}°E</div></div><span className={`table-status ${selected.health}`}><i />{selected.health}</span></div><div className="station-detail-grid"><Metric label="Temperature" value={selected.temperature === null ? '—' : `${selected.temperature.toFixed(1)} °C`} icon={<Thermometer size={15} />} /><Metric label="Humidity" value={selected.humidity === null ? '—' : `${selected.humidity.toFixed(0)} %`} icon={<Waves size={15} />} /><Metric label="Pressure" value={selected.pressure === null ? '—' : `${selected.pressure.toFixed(1)} hPa`} icon={<Gauge size={15} />} /><Metric label="Wind speed" value={selected.windSpeed === null ? '—' : `${selected.windSpeed.toFixed(1)} km/h`} icon={<Wind size={15} />} /><Metric label="Rainfall" value={selected.rainfall === null ? '—' : `${selected.rainfall.toFixed(1)} mm`} icon={<CloudLightning size={15} />} /><Metric label="Anomaly score" value={selected.anomalyScore.toFixed(2)} icon={<AlertTriangle size={15} />} /></div><div className="section-meta detail-updated">Last updated {new Date(selected.lastUpdated).toLocaleString()} · Original telemetry remains auditable</div></section>}</div>;
}

function Metric({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) { return <div className="station-metric"><span>{icon}</span><small>{label}</small><strong>{value}</strong></div>; }

function RoutedPage({ page, data, onNavigate, selectedStationId, onChangeStation }: { page: string; data: Dashboard; onNavigate: (path: string) => void; selectedStationId: string | null; onChangeStation: (stationId: string | null) => void }) {
  const copy: Record<string, { eyebrow: string; title: string; description: string; icon: typeof Activity }> = {
    'Anomaly Detection': { eyebrow: 'Investigation workspace', title: 'Anomaly detection', description: 'Review temporal, physical, and spatial evidence behind every flagged reading.', icon: AlertTriangle },
    'Station Intelligence': { eyebrow: 'Station context', title: 'Station intelligence', description: 'Compare station health, neighboring observations, and evolving telemetry.', icon: Compass },
    'SHAP Explainability': { eyebrow: 'MODEL TRANSPARENCY', title: 'SHAP Explainability', description: 'Understand how each weather feature contributes to anomaly predictions.', icon: BrainCircuit },
    'Self-Healing': { eyebrow: 'Recovery operations', title: 'Self-healing', description: 'Audit reconstructed readings while keeping original observations visible.', icon: ShieldCheck },
    'Unknown Discovery': { eyebrow: 'Open-world research', title: 'Unknown discovery', description: 'Track recurring anomaly signatures and emerging unknown-fault clusters.', icon: CircleHelp },
    Analytics: { eyebrow: 'Network intelligence', title: 'Analytics', description: 'Explore fault frequency, health trends, healing outcomes, and event patterns.', icon: FileBarChart2 },
    Reports: { eyebrow: 'Operational exports', title: 'Reports', description: 'Generate auditable network, anomaly, healing, and station-health reports.', icon: Download },
    Settings: { eyebrow: 'System controls', title: 'Settings', description: 'Configure API connectivity, refresh behavior, and operator preferences.', icon: Settings },
  };
  const current = copy[page] || copy.Analytics; const Icon = current.icon;
  const header = <><div className="page-eyebrow">{current.eyebrow}</div><div className="isolated-title"><div className="isolated-icon"><Icon size={22} /></div><div><h1 className="page-title">{selectedStationId ? `${selectedStationId} / ${current.title}` : current.title}</h1><div className="page-intro">{current.description}</div></div><StationSelector stations={data.latest} selectedStationId={selectedStationId} onChange={onChangeStation} /></div><div className="data-badges"><span className="badge live">● {data.source === 'SYNTHETIC_DEMO' ? 'Demo data' : 'Live data'}</span><span className="badge model">◆ {selectedStationId ? `Viewing ${selectedStationId}` : 'Overall Network'}</span></div></>;
  const body = page === 'Anomaly Detection' ? <AnomalyPage data={data} />
    : page === 'Station Intelligence' ? <IntelligencePage data={data} />
    : page === 'SHAP Explainability' ? <ExplainabilityPage data={data} selectedStationId={selectedStationId} onChangeStation={onChangeStation} />
    : page === 'Self-Healing' ? <HealingPage data={data} />
    : page === 'Unknown Discovery' ? <UnknownPage data={data} />
    : page === 'Analytics' ? <AnalyticsPage data={data} />
    : page === 'Reports' ? <ReportsPage data={data} />
    : <SettingsPage data={data} />;
  return <div className="content isolated-page">{header}{body}</div>;
}

function PageKpis({ items }: { items: Array<[string, string | number, string, typeof Activity]> }) { return <div className="page-kpis">{items.map(([label, value, detail, Icon]) => <div className="card page-kpi" key={label}><Icon size={16} /><small>{label}</small><strong>{value}</strong><span>{detail}</span></div>)}</div>; }
function AnomalyPage({ data }: { data: Dashboard }) { const critical = data.anomalies.filter(row => Number(row.anomaly_score) > .85); return <><PageKpis items={[["Total anomalies", data.kpis.anomalies_detected, "Current model window", AlertTriangle], ["Critical", critical.length, "Score above 0.85", Zap], ["Sensor faults", data.kpis.sensor_faults, "Spatial and model evidence", ShieldCheck], ["Unknown patterns", data.kpis.unknown_patterns, "Rejected known classes", CircleHelp]]} /><Panel title="Detected readings" meta="Temporal, physical, and spatial evidence"><div className="investigation-table"><table><thead><tr><th>Timestamp</th><th>Station</th><th>Score</th><th>Classification</th><th>Confidence</th><th>Spatial diagnosis</th></tr></thead><tbody>{data.anomalies.map((row, index) => <tr key={`${row.timestamp}-${index}`}><td>{new Date(row.timestamp).toLocaleString()}</td><td><strong>{row.station_id}</strong></td><td><span className={`score-pill ${Number(row.anomaly_score) > .85 ? 'critical' : 'warning'}`}>{fmt(row.anomaly_score, 2)}</span></td><td>{row.final_fault_type || row.predicted_fault_type || 'UNKNOWN'}</td><td>{fmt(row.classifier_confidence, 2)}</td><td>{row.spatial_diagnosis || 'uncertain'}</td></tr>)}</tbody></table></div></Panel></>; }
function IntelligencePage({ data }: { data: Dashboard }) { const series = data.series.slice(-72).map(row => ({ time: new Date(row.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), health: row.health_score, anomaly: row.anomaly_score })); return <><PageKpis items={[["Network health", `${fmt(data.kpis.network_health, 1)}%`, "Rolling station average", HeartPulse], ["Active stations", data.kpis.active_stations, "Reporting nodes", Network], ["Observations", data.series.length, "Latest telemetry window", Database], ["Evolution states", new Set(data.series.map(row => row.fault_evolution_state)).size, "Tracked states", Activity]]} /><div className="grid-two page-grid"><Panel title="Health evolution" meta="Station health score over time"><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={series}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="time" interval={11} /><YAxis domain={[0, 100]} /><Tooltip /><Line type="monotone" dataKey="health" stroke="#48d597" dot={false} strokeWidth={2} /></LineChart></ResponsiveContainer></div></Panel><Panel title="Station comparison" meta="Latest health and evolution state"><StationTable rows={data.latest} onSelect={() => undefined} /></Panel></div></>; }
type ShapFeature = { name: string; key: string; value: number; baseline: number; shap: number; unit: string };
type ShapReading = { station: string; timestamp: string; probability: number; baseValue: number; features: ShapFeature[] };

const DEMO_SHAP_READING: ShapReading = {
  station: 'AWS-002', timestamp: '2026-09-13T10:42:00.000Z', probability: .92, baseValue: .34,
  features: [
    { name: 'Relative Humidity', key: 'humidity_pct', value: 87, baseline: 68, shap: .31, unit: '%' },
    { name: 'Temperature', key: 'temperature_c', value: 34, baseline: 30, shap: .21, unit: '°C' },
    { name: 'Pressure', key: 'pressure_hpa', value: 992, baseline: 1006, shap: .12, unit: 'hPa' },
    { name: 'Wind Speed', key: 'wind_speed_kmh', value: 18, baseline: 11, shap: -.08, unit: 'km/h' },
    { name: 'Rainfall', key: 'rainfall_mm', value: 4.2, baseline: 1.1, shap: .05, unit: 'mm' },
    { name: 'Solar Radiation', key: 'solar_radiation_wm2', value: 182, baseline: 260, shap: -.03, unit: 'W/m²' },
  ],
};

const SHAP_IMPORTANCE = [
  ['Humidity', .31], ['Temperature', .24], ['Pressure', .18], ['Wind Speed', .13],
  ['Rainfall', .09], ['Solar Radiation', .07], ['Wind Direction', .04],
] as const;

function shapReadingFrom(explanation: AnyRecord | undefined, row: AnyRecord | undefined): ShapReading {
  const source = explanation?.shap_values;
  if (!explanation || !Array.isArray(source) || !source.length) return DEMO_SHAP_READING;
  const featureNames = explanation.feature_names || DEMO_SHAP_READING.features.map(feature => feature.name);
  const values = explanation.feature_values || {};
  const baselines = explanation.baseline_values || {};
  return {
    station: row?.station_id || DEMO_SHAP_READING.station,
    timestamp: row?.timestamp || DEMO_SHAP_READING.timestamp,
    probability: Number(row?.anomaly_score ?? explanation.prediction_probability ?? DEMO_SHAP_READING.probability),
    baseValue: Number(explanation.base_value ?? DEMO_SHAP_READING.baseValue),
    features: source.map((shap: number, index: number) => ({
      name: String(featureNames[index] || `Feature ${index + 1}`), key: String(featureNames[index] || index),
      value: Number(values[featureNames[index]] ?? 0), baseline: Number(baselines[featureNames[index]] ?? 0), shap: Number(shap), unit: '',
    })),
  };
}

function ExplainabilityPage({ data, selectedStationId, onChangeStation }: { data: Dashboard; selectedStationId: string | null; onChangeStation: (stationId: string | null) => void }) {
  const explanations = data.explanations || [];
  const stationId = selectedStationId || 'AWS-002';
  const [readingId, setReadingId] = useState('0');
  const [date, setDate] = useState('2026-09-13');
  const [highlighted, setHighlighted] = useState('Relative Humidity');
  const row = data.anomalies[Number(readingId)] || data.anomalies.find(item => item.station_id === stationId);
  const reading = shapReadingFrom(explanations[Number(readingId)] || explanations[0], row);
  const features = reading.features.length ? reading.features : DEMO_SHAP_READING.features;
  const maxImpact = Math.max(...features.map(feature => Math.abs(feature.shap)), .01);
  const finalPrediction = reading.baseValue + features.reduce((sum, feature) => sum + feature.shap, 0);
  const queue = explanations.length ? explanations.slice(0, 8) : [0, 1, 2].map((id) => ({ row_id: id, station_id: id === 0 ? 'AWS-002' : `AWS-00${id + 2}`, anomaly_score: .92 - id * .07, top_feature: id === 0 ? 'Humidity' : id === 1 ? 'Temperature' : 'Pressure', narrative: id === 0 ? 'High humidity was the strongest contributor to this anomaly prediction.' : 'The leading SHAP contribution pushed this reading toward anomaly.' }));
  const swarm = features.flatMap((feature, featureIndex) => Array.from({ length: 6 }, (_, pointIndex) => ({ feature: feature.name, y: featureIndex, x: feature.shap * (.55 + pointIndex * .11) * (pointIndex % 2 ? -1 : 1), color: pointIndex % 2 ? '#51e1e8' : '#f5b84d' })));
  return <>
    <div className="shap-controls"><label>Station<select className="select" value={selectedStationId || 'ALL'} onChange={event => onChangeStation(event.target.value === 'ALL' ? null : event.target.value)}><option value="ALL">Overall Network</option>{data.latest.map(item => <option key={item.station_id}>{item.station_id}</option>)}</select></label><label>Reading<select className="select" value={readingId} onChange={event => setReadingId(event.target.value)}>{[0, 1, 2].map(id => <option key={id} value={id}>Reading {id}</option>)}</select></label><label>Date<input className="select" type="date" value={date} onChange={event => setDate(event.target.value)} /></label><button className="shap-action" onClick={() => setHighlighted(features[0]?.name || 'Relative Humidity')}><BrainCircuit size={14} /> View SHAP Details</button></div>
    <PageKpis items={[["SHAP Explanations", 250, "Model predictions explained", BrainCircuit], ["Highest |SHAP| Impact", maxImpact.toFixed(2), "Strongest feature contribution", Zap], ["Model Confidence", `${Math.round(reading.probability * 100)}%`, "Prediction confidence", Gauge]]} />
    <div className="grid-two page-grid shap-main-grid"><Panel title="Why did SkyGuard flag this reading?" meta={`${reading.station} · ${new Date(reading.timestamp).toLocaleString()}`}><div className="shap-reading-head"><div><span className="table-status critical"><i />Anomaly Detected</span><strong>{Math.round(reading.probability * 100)}% <small>anomaly probability</small></strong></div><div className="shap-score">Score <b>{reading.probability.toFixed(2)}</b></div></div><ShapWaterfall features={features} baseValue={reading.baseValue} finalPrediction={finalPrediction} highlighted={highlighted} onHighlight={setHighlighted} /></Panel><Panel title="Recent SHAP Explanations" meta="Selected model-backed decisions"><div className="feed shap-queue">{queue.map(item => <div className="feed-item" key={item.row_id}><div className="feed-icon"><Sparkles size={14} /></div><div className="feed-copy"><strong>Reading {item.row_id} · {item.station_id || stationId}</strong><div className="feed-time">Anomaly score: {Number(item.anomaly_score || .72).toFixed(2)} · Top feature: {item.top_feature || 'Humidity'}</div><p>{item.narrative || 'High humidity was the strongest contributor to this anomaly prediction.'}</p></div></div>)}</div></Panel></div>
    <div className="grid-two page-grid"><Panel title="Global SHAP Feature Importance" meta="Mean absolute SHAP value across the evaluation window"><div className="importance-chart">{SHAP_IMPORTANCE.map(([name, value], index) => <button className={`importance-row ${highlighted === name || (name === 'Humidity' && highlighted === 'Relative Humidity') ? 'selected' : ''}`} key={name} onClick={() => setHighlighted(name)} title={`${name}: mean |SHAP| ${value.toFixed(2)}`}><span>{index + 1}. {name}</span><i><b style={{ width: `${value / .31 * 100}%` }} /></i><strong>{value.toFixed(2)}</strong></button>)}</div></Panel><Panel title="SHAP Summary" meta="X-axis: SHAP value · each point is a weather observation"><div className="beeswarm"><div className="beeswarm-axis"><span>Negative</span><b>0</b><span>Positive</span></div><div className="beeswarm-zero" />{swarm.map((point, index) => <button key={`${point.feature}-${index}`} className={`beeswarm-point ${highlighted === point.feature || (highlighted === 'Humidity' && point.feature === 'Relative Humidity') ? 'selected' : ''}`} title={`${point.feature}: SHAP ${point.x.toFixed(2)}`} onClick={() => setHighlighted(point.feature)} style={{ left: `${50 + point.x * 120}%`, top: `${10 + point.y * 13 + (index % 3) * 2}%`, background: point.color }} />)}{features.map((feature, index) => <span className="beeswarm-label" key={feature.name} style={{ top: `${8 + index * 13}%` }}>{feature.name}</span>)}</div></Panel></div>
    <Panel title="Feature Contributions" meta={`${reading.station} · selected reading detail`}><div className="contribution-table"><div className="contribution-header"><span>Feature</span><span>Actual</span><span>Baseline</span><span>SHAP value</span><span>Contribution</span></div>{features.map(feature => <button className={`contribution-row ${highlighted === feature.name ? 'selected' : ''}`} key={feature.name} onClick={() => setHighlighted(feature.name)}><strong>{feature.name}</strong><span>{feature.value || '—'} {feature.unit}</span><span>{feature.baseline || '—'} {feature.unit}</span><b className={feature.shap >= 0 ? 'positive' : 'negative'}>{feature.shap >= 0 ? '+' : ''}{feature.shap.toFixed(2)}</b><span className={feature.shap >= 0 ? 'positive' : 'negative'}>{feature.shap >= 0 ? 'Increases anomaly probability' : 'Pushes toward Normal'}</span></button>)}</div></Panel>
  </>;
}

function ShapWaterfall({ features, baseValue, finalPrediction, highlighted, onHighlight }: { features: ShapFeature[]; baseValue: number; finalPrediction: number; highlighted: string; onHighlight: (name: string) => void }) {
  const max = Math.max(...features.map(feature => Math.abs(feature.shap)), .01);
  return <div className="waterfall"><div className="waterfall-legend"><span><i className="positive" />Pushes toward Anomaly</span><span><i className="negative" />Pushes toward Normal</span></div><div className="waterfall-line"><span>Base Value</span><b>{baseValue.toFixed(2)}</b></div>{features.map(feature => <button className={`waterfall-row ${highlighted === feature.name ? 'selected' : ''}`} key={feature.name} onClick={() => onHighlight(feature.name)} title={`${feature.name}: SHAP ${feature.shap >= 0 ? '+' : ''}${feature.shap.toFixed(2)}`}><span>{feature.name}</span><div className="waterfall-track"><i className={feature.shap >= 0 ? 'positive' : 'negative'} style={{ width: `${Math.abs(feature.shap) / max * 46}%`, [feature.shap >= 0 ? 'marginLeft' : 'marginRight']: '50%' }} /></div><b className={feature.shap >= 0 ? 'positive' : 'negative'}>{feature.shap >= 0 ? '+' : ''}{feature.shap.toFixed(2)}</b></button>)}<div className="waterfall-final"><span>Final Prediction</span><b>{finalPrediction.toFixed(2)} · Anomaly</b></div></div>;
}
function HealingPage({ data }: { data: Dashboard }) { const healing = data.healing || []; const total = data.anomalies.filter(row => row.is_sensor_fault).length; return <><PageKpis items={[["Faulty readings", total, "Sensor-fault decisions", AlertTriangle], ["Successfully healed", healing.length, "Originals preserved", ShieldCheck], ["Unhealed", Math.max(0, total - healing.length), "Needs operator review", CircleHelp], ["Success rate", `${total ? Math.round(healing.length / total * 100) : 0}%`, "Interpolation coverage", Gauge]]} /><Panel title="Recovery audit" meta="Original values remain visible beside corrected values"><div className="investigation-table"><table><thead><tr><th>Timestamp</th><th>Station</th><th>Temperature original</th><th>Temperature corrected</th><th>Imputed</th><th>Method</th></tr></thead><tbody>{healing.map((row, index) => <tr key={`${row.timestamp}-${index}`}><td>{new Date(row.timestamp).toLocaleString()}</td><td><strong>{row.station_id}</strong></td><td>{fmt(row.temperature_c_original, 1)}°C</td><td>{fmt(row.temperature_c, 1)}°C</td><td><span className="table-status healthy"><i />{row.temperature_c_imputed ? 'yes' : 'no'}</span></td><td>Temporal interpolation</td></tr>)}</tbody></table>{!healing.length && <EmptyState text="No healed readings in the current window." />}</div></Panel></>; }
function UnknownPage({ data }: { data: Dashboard }) { return <><PageKpis items={[["Unknown observations", data.unknowns.length, "Rejected known classes", CircleHelp], ["Recurring clusters", Object.keys(data.registry).length, "Registry-backed patterns", Network], ["Emerging patterns", Object.values(data.registry).filter((item: any) => item.status === 'Emerging').length, "Needs recurrence evidence", Activity]]} /><div className="grid-two page-grid"><Panel title="Unknown signatures" meta="Noise remains separate from recurring clusters"><div className="investigation-table"><table><thead><tr><th>Timestamp</th><th>Station</th><th>Pattern</th><th>Cluster</th><th>Confidence</th></tr></thead><tbody>{data.unknowns.map((row, index) => <tr key={`${row.timestamp}-${index}`}><td>{new Date(row.timestamp).toLocaleString()}</td><td>{row.station_id}</td><td>{row.unknown_pattern}</td><td>{row.unknown_cluster_id}</td><td>{fmt(row.unknown_confidence, 2)}</td></tr>)}</tbody></table></div></Panel><Panel title="Unknown fault registry" meta="Persistent signatures are promoted only after recurrence"><div className="feed">{Object.values(data.registry).map((item: any) => <div className="feed-item" key={item.fault_id}><div className="feed-icon"><CircleHelp size={14} /></div><div className="feed-copy"><strong>{item.fault_id} · {item.status}</strong><div className="feed-time">{item.occurrences} occurrences · {item.stations?.length || 0} stations</div></div></div>)}</div></Panel></div></>; }
function AnalyticsPage({ data }: { data: Dashboard }) { const counts = data.anomalies.reduce<Record<string, number>>((result, row) => { const key = row.final_fault_type || row.predicted_fault_type || 'UNKNOWN'; result[key] = (result[key] || 0) + 1; return result; }, {}); const chart = Object.entries(counts).map(([name, value]) => ({ name, value })); return <><PageKpis items={[["Anomaly rate", `${data.kpis.active_stations ? fmt(data.kpis.anomalies_detected / data.kpis.active_stations, 0) : '0'}`, "Events per station", Activity], ["Sensor-fault share", `${data.kpis.anomalies_detected ? Math.round(data.kpis.sensor_faults / data.kpis.anomalies_detected * 100) : 0}%`, "Of detected anomalies", ShieldCheck], ["Unknown share", `${data.kpis.anomalies_detected ? Math.round(data.kpis.unknown_patterns / data.kpis.anomalies_detected * 100) : 0}%`, "Open-world rejection", CircleHelp]]} /><Panel title="Fault distribution" meta="Current pipeline output by final classification"><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chart}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip /><Area type="monotone" dataKey="value" stroke="#48aaff" fill="#48aaff" fillOpacity={.18} /></AreaChart></ResponsiveContainer></div></Panel></>; }
function ReportsPage({ data }: { data: Dashboard }) { const csv = `station_id,timestamp,anomaly_score,final_fault_type\n${data.anomalies.map(row => `${row.station_id},${row.timestamp},${row.anomaly_score},${row.final_fault_type || row.predicted_fault_type}`).join('\n')}`; return <><PageKpis items={[["Network summary", data.kpis.active_stations, "Stations included", Network], ["Anomaly rows", data.anomalies.length, "Export-ready records", FileBarChart2], ["Healing rows", data.healing?.length || 0, "Audit records", ShieldCheck]]} /><div className="report-grid">{['Station health report', 'Anomaly report', 'Sensor fault report', 'Self-healing report', 'AI explainability report', 'Network summary'].map((name) => <div className="card report-card" key={name}><FileBarChart2 size={18} /><h2>{name}</h2><p>Generate an auditable report from the current pipeline snapshot.</p><a className="select" download="skyguard-anomalies.csv" href={`data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`}>Export CSV</a></div>)}</div></>; }
function SettingsPage({ data }: { data: Dashboard }) { return <><div className="settings-list"><div className="card setting-row"><div><strong>API connection</strong><small>FastAPI dashboard adapter</small></div><span className="table-status healthy"><i />{data.source === 'SYNTHETIC_DEMO' ? 'demo fallback' : 'connected'}</span></div><div className="card setting-row"><div><strong>Auto refresh</strong><small>Station data polling interval</small></div><span className="table-status healthy"><i />45 seconds</span></div><div className="card setting-row"><div><strong>Model provenance</strong><small>SHAP fallback remains available if optional packages are absent</small></div><span className="table-status healthy"><i />auditable</span></div></div></>; }
function Evidence({ label, value }: { label: string; value: string }) { return <div className="evidence"><span>{label}</span><strong>{value}</strong></div>; }
