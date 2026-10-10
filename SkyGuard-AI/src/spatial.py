"""
Module 3 — Lightweight Spatial Check.

Compares a flagged station's readings against its 3-5 nearest neighboring
stations at the same timestamp, using a precomputed Haversine distance
matrix (no external GIS service required, no network dependency).

If neighbors show a similar shift -> evidence for a genuine regional weather
event. If the flagged station diverges from its neighbors -> evidence for an
isolated sensor/data fault. This matches the official SIH example use case.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def build_distance_matrix(stations: pd.DataFrame) -> pd.DataFrame:
    """
    stations: dataframe with columns station_id, latitude, longitude (one row per station).
    Returns a station_id x station_id distance matrix (km).
    """
    ids = stations["station_id"].to_list()
    n = len(ids)
    mat = np.zeros((n, n))
    lats = stations["latitude"].to_numpy()
    lons = stations["longitude"].to_numpy()
    for i in range(n):
        mat[i, :] = haversine_km(lats[i], lons[i], lats, lons)
    return pd.DataFrame(mat, index=ids, columns=ids)


def nearest_neighbors(distance_matrix: pd.DataFrame, station_id: str, k: int = 3, max_km: float = 300) -> list:
    """Return up to k nearest neighboring station ids within max_km, excluding self."""
    if station_id not in distance_matrix.index:
        return []
    row = distance_matrix.loc[station_id].drop(labels=[station_id])
    row = row[row <= max_km].sort_values()
    return row.index[:k].to_list()


def spatial_consistency_evidence(
    df: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    station_id: str,
    timestamp: pd.Timestamp,
    param: str,
    k: int = 3,
    max_km: float = 300,
    tolerance_std: float = 2.0,
) -> dict:
    """
    Compares the target station's value for `param` at `timestamp` against its
    nearest neighbors' values at the same timestamp (nearest available reading).

    Returns a dict with:
      - neighbor_ids
      - neighbor_mean, neighbor_std
      - target_value
      - agrees_with_neighbors (bool): whether the target is within tolerance
      - evidence: "supports_weather_event" | "supports_sensor_fault" | "insufficient_data"
    """
    neighbors = nearest_neighbors(distance_matrix, station_id, k=k, max_km=max_km)
    if not neighbors:
        return {"neighbor_ids": [], "evidence": "insufficient_data"}

    window = pd.Timedelta(minutes=30)
    neighbor_rows = df[
        (df["station_id"].isin(neighbors))
        & (df["timestamp"] >= timestamp - window)
        & (df["timestamp"] <= timestamp + window)
    ]
    if neighbor_rows.empty:
        return {"neighbor_ids": neighbors, "evidence": "insufficient_data"}

    neighbor_vals = neighbor_rows.groupby("station_id")[param].mean()
    neighbor_mean = neighbor_vals.mean()
    neighbor_std = neighbor_vals.std() if len(neighbor_vals) > 1 else neighbor_rows[param].std()
    neighbor_std = neighbor_std if neighbor_std and not np.isnan(neighbor_std) else 1.0

    target_rows = df[
        (df["station_id"] == station_id)
        & (df["timestamp"] >= timestamp - window)
        & (df["timestamp"] <= timestamp + window)
    ]
    if target_rows.empty:
        return {"neighbor_ids": neighbors, "evidence": "insufficient_data"}
    target_value = target_rows[param].mean()

    deviation = abs(target_value - neighbor_mean) / (neighbor_std + 1e-6)
    agrees = deviation <= tolerance_std

    return {
        "neighbor_ids": neighbors,
        "neighbor_mean": round(float(neighbor_mean), 2),
        "neighbor_std": round(float(neighbor_std), 2),
        "target_value": round(float(target_value), 2),
        "deviation_in_std": round(float(deviation), 2),
        "agrees_with_neighbors": bool(agrees),
        "evidence": "supports_weather_event" if agrees else "supports_sensor_fault",
    }
