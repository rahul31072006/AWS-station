export type StationHealth = 'healthy' | 'warning' | 'fault' | 'unknown';

export type Station = {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  temperature: number | null;
  humidity: number | null;
  pressure: number | null;
  windSpeed: number | null;
  rainfall: number | null;
  health: StationHealth;
  anomalyScore: number;
  lastUpdated: string;
  history: Array<{ timestamp: string; temperature: number | null; anomalyScore: number }>;
};

const DEMO_TIMESTAMP = '2026-09-13T00:00:00.000Z';

export const mockStations: Station[] = [
  { id: 'AWS-001', name: 'AWS-001', latitude: 13.0827, longitude: 80.2707, temperature: 31.2, humidity: 72, pressure: 1004, windSpeed: 12, rainfall: 0, health: 'healthy', anomalyScore: 0.12, lastUpdated: DEMO_TIMESTAMP, history: [] },
  { id: 'AWS-002', name: 'AWS-002', latitude: 13.2257, longitude: 79.9888, temperature: 30.8, humidity: 75, pressure: 1004, windSpeed: 12, rainfall: 0, health: 'warning', anomalyScore: 0.72, lastUpdated: DEMO_TIMESTAMP, history: [] },
  { id: 'AWS-003', name: 'AWS-003', latitude: 11.6643, longitude: 78.1460, temperature: 32.1, humidity: 68, pressure: 1006, windSpeed: 9, rainfall: 0, health: 'healthy', anomalyScore: 0.18, lastUpdated: DEMO_TIMESTAMP, history: [] },
  { id: 'AWS-004', name: 'AWS-004', latitude: 10.7905, longitude: 78.7047, temperature: 29.7, humidity: 81, pressure: 1002, windSpeed: 16, rainfall: 1.2, health: 'fault', anomalyScore: 0.91, lastUpdated: DEMO_TIMESTAMP, history: [] },
];

type DashboardPayload = {
  stations: Array<Record<string, unknown>>;
  latest: Array<Record<string, unknown>>;
  series: Array<Record<string, unknown>>;
};

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

function numberValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function healthFrom(row: Record<string, unknown>): StationHealth {
  if (row.final_fault_type === 'UNKNOWN') return 'unknown';
  if (row.is_sensor_fault || row.spatial_diagnosis === 'likely_sensor_fault') {
    return Number(row.anomaly_score || 0) > 0.85 ? 'fault' : 'warning';
  }
  return 'healthy';
}

export function mapStations(payload: DashboardPayload): Station[] {
  return payload.latest.map((latest) => {
    const station = payload.stations.find((item) => item.station_id === latest.station_id) || {};
    const history = payload.series
      .filter((item) => item.station_id === latest.station_id)
      .slice(-36)
      .map((item) => ({
        timestamp: String(item.timestamp),
        temperature: numberValue(item.temperature_c),
        anomalyScore: numberValue(item.anomaly_score) || 0,
      }));
    return {
      id: String(latest.station_id),
      name: String(latest.station_id),
      latitude: numberValue(station.latitude) || numberValue(latest.latitude) || 13.0827,
      longitude: numberValue(station.longitude) || numberValue(latest.longitude) || 80.2707,
      temperature: numberValue(latest.temperature_c),
      humidity: numberValue(latest.humidity_pct),
      pressure: numberValue(latest.pressure_hpa),
      windSpeed: numberValue(latest.wind_speed_kmh),
      rainfall: numberValue(latest.rainfall_mm),
      health: healthFrom(latest),
      anomalyScore: numberValue(latest.anomaly_score) || 0,
      lastUpdated: String(latest.timestamp || new Date().toISOString()),
      history,
    };
  });
}

export async function fetchStations(signal?: AbortSignal): Promise<Station[]> {
  const response = await fetch(`${API}/api/dashboard`, { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(`Station API returned ${response.status}`);
  return mapStations(await response.json());
}
