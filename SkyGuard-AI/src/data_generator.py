"""
Synthetic Automatic Weather Station (AWS) data generator.

Generates plausible, clearly-labeled SYNTHETIC Temperature / Pressure / Relative
Humidity time series for multiple stations, with realistic diurnal cycles, and
optionally injects known sensor faults for evaluation.

This data is for demonstration and evaluation only and must never be presented
as real observations.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class StationDef:
    station_id: str
    latitude: float
    longitude: float
    elevation_m: float
    base_temp_c: float      # mean annual temperature, drives diurnal cycle
    base_pressure_hpa: float
    base_humidity_pct: float


def default_stations(num_stations: int = 6, seed: int = 42) -> List[StationDef]:
    """Create a small synthetic network of stations roughly spread across a region."""
    rng = np.random.default_rng(seed)
    stations = []
    # Roughly centered around Tamil Nadu, India, for plausible synthetic geography
    center_lat, center_lon = 9.9, 78.1
    for i in range(num_stations):
        lat = center_lat + rng.uniform(-1.5, 1.5)
        lon = center_lon + rng.uniform(-1.5, 1.5)
        elevation = rng.uniform(20, 450)
        base_temp = rng.uniform(26, 33) - 0.0065 * elevation  # lapse rate
        base_pressure = 1013.25 * (1 - 2.25577e-5 * elevation) ** 5.25588
        base_humidity = rng.uniform(55, 80)
        stations.append(StationDef(
            station_id=f"AWS-{i+1:03d}",
            latitude=round(lat, 4),
            longitude=round(lon, 4),
            elevation_m=round(elevation, 1),
            base_temp_c=round(base_temp, 2),
            base_pressure_hpa=round(base_pressure, 2),
            base_humidity_pct=round(base_humidity, 2),
        ))
    return stations


def _diurnal_cycle(hours: np.ndarray, amplitude: float, phase_shift: float = -6.0) -> np.ndarray:
    """Simple sinusoidal diurnal cycle peaking in the afternoon."""
    return amplitude * np.sin(2 * np.pi * (hours - phase_shift) / 24.0)


def generate_station_series(
    station: StationDef,
    start: str = "2026-01-01",
    days: int = 14,
    interval_minutes: int = 10,
    seed: int = 42,
    regional_weather_event: bool = False,
) -> pd.DataFrame:
    """Generate a clean synthetic T/P/RH series for one station."""
    rng = np.random.default_rng(abs(hash(station.station_id)) % (2**32) + seed)
    periods = int(days * 24 * 60 / interval_minutes)
    timestamps = pd.date_range(start=start, periods=periods, freq=f"{interval_minutes}min")
    hours = np.asarray(timestamps.hour + timestamps.minute / 60.0, dtype=float)
    day_index = np.arange(periods) / (24 * 60 / interval_minutes)

    # Seasonal drift across the simulated window (very mild) + diurnal cycle + noise
    seasonal = 1.5 * np.sin(2 * np.pi * day_index / 30.0)
    temp = (
        station.base_temp_c
        + _diurnal_cycle(hours, amplitude=6.0)
        + seasonal
        + rng.normal(0, 0.3, periods)
    )
    pressure = (
        station.base_pressure_hpa
        + _diurnal_cycle(hours, amplitude=1.2, phase_shift=2.0) * -1
        + rng.normal(0, 0.15, periods)
    )
    humidity = (
        station.base_humidity_pct
        - _diurnal_cycle(hours, amplitude=15.0)
        + rng.normal(0, 1.5, periods)
    )
    humidity = np.clip(humidity, 5, 100)

    if regional_weather_event:
        # Simulate a genuine, physically-consistent regional weather event
        # (e.g. an approaching low-pressure system): temp drops, humidity rises,
        # pressure falls together, consistently, for several hours across the network.
        event_start = periods // 3
        event_len = int(6 * 60 / interval_minutes)  # 6 hours
        idx = slice(event_start, event_start + event_len)
        temp[idx] -= np.linspace(0, 4.0, event_len)
        pressure[idx] -= np.linspace(0, 6.0, event_len)
        humidity[idx] = np.clip(humidity[idx] + np.linspace(0, 20.0, event_len), 5, 100)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station.station_id,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "temperature_c": np.round(temp, 2),
        "pressure_hpa": np.round(pressure, 2),
        "humidity_pct": np.round(humidity, 2),
    })
    df["source"] = "SYNTHETIC_NORMAL"
    return df


def generate_network(
    num_stations: int = 6,
    start: str = "2026-01-01",
    days: int = 14,
    interval_minutes: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate clean synthetic data for a whole network of stations, including
    one shared genuine regional weather event so spatial-consistency logic has
    something real to validate against."""
    stations = default_stations(num_stations, seed)
    frames = []
    for i, st in enumerate(stations):
        frames.append(generate_station_series(
            st, start=start, days=days, interval_minutes=interval_minutes,
            seed=seed + i, regional_weather_event=True,
        ))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Known fault injection
# ---------------------------------------------------------------------------

KNOWN_FAULT_TYPES = [
    "spike", "sudden_drop", "flatline", "bias", "gradual_drift",
    "missing_data", "intermittent_failure", "comm_error",
]


def inject_known_faults(
    df: pd.DataFrame,
    station_id: str,
    fault_type: str,
    seed: int = 42,
    n_events: int = 3,
) -> pd.DataFrame:
    """
    Inject a known fault type into ONE station's data, returning a NEW dataframe
    (the original clean copy is never mutated). A ground-truth label column is
    added so this can be used for evaluation; it must never be fed to the model.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["true_state"] = "HEALTHY"
    out["fault_type"] = "none"
    mask_station = out["station_id"] == station_id
    station_idx = out.index[mask_station].to_numpy()

    if len(station_idx) < 50:
        return out

    for _ in range(n_events):
        start_i = rng.integers(20, len(station_idx) - 30)
        if fault_type == "spike":
            i = station_idx[start_i]
            out.loc[i, "temperature_c"] += rng.choice([-1, 1]) * rng.uniform(8, 15)
            out.loc[i, "true_state"] = "FAULT"
            out.loc[i, "fault_type"] = "spike"

        elif fault_type == "sudden_drop":
            i = station_idx[start_i]
            out.loc[i, "pressure_hpa"] -= rng.uniform(15, 25)
            out.loc[i, "true_state"] = "FAULT"
            out.loc[i, "fault_type"] = "sudden_drop"

        elif fault_type == "flatline":
            span = station_idx[start_i:start_i + 24]
            frozen_val = out.loc[span[0], "humidity_pct"]
            out.loc[span, "humidity_pct"] = frozen_val
            out.loc[span, "true_state"] = "FAULT"
            out.loc[span, "fault_type"] = "flatline"

        elif fault_type == "bias":
            span = station_idx[start_i:start_i + 48]
            out.loc[span, "temperature_c"] += rng.uniform(3, 6)
            out.loc[span, "true_state"] = "FAULT"
            out.loc[span, "fault_type"] = "bias"

        elif fault_type == "gradual_drift":
            span = station_idx[start_i:start_i + 72]
            drift = np.linspace(0, rng.uniform(5, 9), len(span))
            out.loc[span, "pressure_hpa"] += drift
            out.loc[span, "true_state"] = "FAULT"
            out.loc[span, "fault_type"] = "gradual_drift"

        elif fault_type == "missing_data":
            span = station_idx[start_i:start_i + 12]
            out.loc[span, ["temperature_c", "pressure_hpa", "humidity_pct"]] = np.nan
            out.loc[span, "true_state"] = "FAULT"
            out.loc[span, "fault_type"] = "missing_data"

        elif fault_type == "intermittent_failure":
            span = station_idx[start_i:start_i + 40:4]
            out.loc[span, "humidity_pct"] = np.nan
            out.loc[span, "true_state"] = "FAULT"
            out.loc[span, "fault_type"] = "intermittent_failure"

        elif fault_type == "comm_error":
            span = station_idx[start_i:start_i + 8]
            out.loc[span, "temperature_c"] = np.nan
            out.loc[span, "pressure_hpa"] = np.nan
            out.loc[span, "true_state"] = "FAULT"
            out.loc[span, "fault_type"] = "comm_error"

    out.loc[mask_station, "source"] = "SYNTHETIC_FAULT_INJECTED"
    return out
