'use client';

import { useEffect } from 'react';
import { CircleMarker, MapContainer, Popup, TileLayer, useMap } from 'react-leaflet';
import type { LatLngExpression } from 'leaflet';
import type { Station } from '../services/stationService';
import 'leaflet/dist/leaflet.css';

type Props = { stations: Station[]; selectedId: string | null; onSelect: (station: Station) => void };

const colors: Record<Station['health'], string> = {
  healthy: '#48d597', warning: '#f5b84d', fault: '#ff6674', unknown: '#a589ff',
};

function FocusStation({ station }: { station: Station | null }) {
  const map = useMap();
  useEffect(() => {
    if (station) map.flyTo([station.latitude, station.longitude], 10, { duration: 0.7 });
  }, [map, station]);
  return null;
}

function StationPopup({ station }: { station: Station }) {
  return <div className="osm-popup">
    <strong>{station.id}</strong><span className={`popup-health ${station.health}`}>{station.health}</span>
    <div className="popup-grid"><span>Temperature<b>{station.temperature === null ? '—' : `${station.temperature.toFixed(1)} °C`}</b></span><span>Humidity<b>{station.humidity === null ? '—' : `${station.humidity.toFixed(1)} %`}</b></span><span>Pressure<b>{station.pressure === null ? '—' : `${station.pressure.toFixed(1)} hPa`}</b></span><span>Anomaly score<b>{station.anomalyScore.toFixed(2)}</b></span></div>
    <small>Last updated {new Date(station.lastUpdated).toLocaleString()}</small>
  </div>;
}

export default function LiveStationsMap({ stations, selectedId, onSelect }: Props) {
  const selected = stations.find((station) => station.id === selectedId) || null;
  const center: LatLngExpression = selected ? [selected.latitude, selected.longitude] : [13.0827, 80.2707];
  return <MapContainer center={center} zoom={7} scrollWheelZoom className="osm-map">
    <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
    <FocusStation station={selected} />
    {stations.map((station) => <CircleMarker key={station.id} center={[station.latitude, station.longitude]} radius={selectedId === station.id ? 11 : 8} pathOptions={{ color: colors[station.health], fillColor: colors[station.health], fillOpacity: 0.85, weight: selectedId === station.id ? 4 : 2 }} eventHandlers={{ click: () => onSelect(station) }}><Popup><StationPopup station={station} /></Popup></CircleMarker>)}
  </MapContainer>;
}
