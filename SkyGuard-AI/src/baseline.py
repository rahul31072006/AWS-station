"""
Module 1 — Temporal & Seasonal Baseline.

Learns the normal diurnal pattern per station, per parameter, using rolling
statistics (mean/std by hour-of-day), and derives temporal features used
downstream by the anomaly detector and classifier.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

PARAMS = ["temperature_c", "pressure_hpa", "humidity_pct"]


def add_temporal_features(df: pd.DataFrame, rolling_window: int = 6) -> pd.DataFrame:
    """
    Adds, per station and per parameter:
      - first difference (rate of change per reading)
      - rolling mean / std / min / max
      - deviation from local rolling baseline (z-score)
    """
    out = df.sort_values(["station_id", "timestamp"]).copy()
    out["hour_of_day"] = out["timestamp"].dt.hour

    grouped = out.groupby("station_id", group_keys=False)

    for p in PARAMS:
        out[f"{p}_diff"] = grouped[p].diff()
        out[f"{p}_roll_mean"] = grouped[p].transform(
            lambda s: s.rolling(rolling_window, min_periods=2).mean()
        )
        out[f"{p}_roll_std"] = grouped[p].transform(
            lambda s: s.rolling(rolling_window, min_periods=2).std()
        )
        out[f"{p}_roll_min"] = grouped[p].transform(
            lambda s: s.rolling(rolling_window, min_periods=2).min()
        )
        out[f"{p}_roll_max"] = grouped[p].transform(
            lambda s: s.rolling(rolling_window, min_periods=2).max()
        )
        # Z-score uses a PAST-ONLY baseline (shifted window) so a genuine
        # anomaly doesn't get averaged into and dampen its own reference
        # mean/std -- otherwise a single-reading spike would pull the rolling
        # window's own stats toward itself and mask the deviation.
        past_mean = grouped[p].transform(
            lambda s: s.shift(1).rolling(rolling_window, min_periods=2).mean()
        )
        past_std = grouped[p].transform(
            lambda s: s.shift(1).rolling(rolling_window, min_periods=2).std()
        )
        # Floor the denominator relative to the parameter's overall variability
        # so that a coincidentally near-flat short window (a handful of very
        # similar healthy readings) doesn't blow up into a huge spurious
        # z-score once divided by a near-zero std.
        global_std = out[p].std()
        std_floor = max(global_std * 0.05, 1e-3)
        out[f"{p}_zscore"] = (out[p] - past_mean) / (past_std.clip(lower=std_floor))

    return out


def add_diurnal_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Learns an hour-of-day baseline (mean/std) per station per parameter from
    the data itself, and computes each reading's deviation from that expected
    diurnal baseline. This is the "normal pattern" the system considers healthy.
    """
    out = df.copy()
    for p in PARAMS:
        baseline = (
            out.groupby(["station_id", "hour_of_day"])[p]
            .transform("mean")
        )
        baseline_std = (
            out.groupby(["station_id", "hour_of_day"])[p]
            .transform("std")
        )
        out[f"{p}_diurnal_baseline"] = baseline
        out[f"{p}_diurnal_deviation"] = (out[p] - baseline) / (baseline_std + 1e-6)
    return out


def add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sensor relationship features (independent of the physics module) —
    e.g. does humidity move in the expected inverse relation to temperature."""
    out = df.copy()
    out["temp_humidity_corr_local"] = (
        out.groupby("station_id")
        .apply(lambda g: g["temperature_c_diff"].rolling(12, min_periods=4).corr(g["humidity_pct_diff"]))
        .reset_index(level=0, drop=True)
    )
    out["temp_humidity_corr_local"] = out["temp_humidity_corr_local"].fillna(0)
    return out


def build_features(df: pd.DataFrame, rolling_window: int = 6) -> pd.DataFrame:
    """Full Module 1 feature pipeline."""
    out = add_temporal_features(df, rolling_window=rolling_window)
    out = add_diurnal_baseline(out)
    out = add_cross_sensor_features(out)
    return out
