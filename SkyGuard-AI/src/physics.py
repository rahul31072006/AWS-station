"""
Module 2 — Multivariate Physical Consistency (VPD / Psychrometric check).

Core innovation: rather than treating Temperature, Pressure, and Humidity as
independent signals, this module checks whether a given (T, P, RH) triple is
physically co-possible, using standard meteorological psychrometric formulas
(Tetens/Magnus saturation vapour pressure, and Vapour Pressure Deficit).

A physically consistent triple (even an extreme one, e.g. a genuine heatwave)
will still respect these relationships. A sensor fault frequently produces
combinations that are individually plausible per-channel but jointly
inconsistent (e.g. very high temperature with an implausibly high humidity
reading, or a humidity value that doesn't respond at all as temperature swings).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def saturation_vapor_pressure_kpa(temp_c: np.ndarray) -> np.ndarray:
    """Tetens' formula for saturation vapor pressure (kPa) given temperature (°C)."""
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def actual_vapor_pressure_kpa(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Actual vapor pressure from RH and saturation vapor pressure."""
    es = saturation_vapor_pressure_kpa(temp_c)
    return es * (np.clip(rh_pct, 0, 100) / 100.0)


def vapor_pressure_deficit_kpa(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """VPD = es - ea. Always >= 0 for physically valid inputs."""
    es = saturation_vapor_pressure_kpa(temp_c)
    ea = actual_vapor_pressure_kpa(temp_c, rh_pct)
    return es - ea


def dew_point_c(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Magnus-formula dew point, °C. Physically, dew point must never exceed air temperature."""
    rh_clipped = np.clip(rh_pct, 0.1, 100)
    a, b = 17.27, 237.3
    alpha = (a * temp_c) / (b + temp_c) + np.log(rh_clipped / 100.0)
    return (b * alpha) / (a - alpha)


def pressure_altitude_expected_hpa(elevation_m: np.ndarray, sea_level_hpa: float = 1013.25) -> np.ndarray:
    """Expected station pressure from elevation via the barometric formula."""
    return sea_level_hpa * (1 - 2.25577e-5 * elevation_m) ** 5.25588


def compute_physics_features(df: pd.DataFrame, elevation_col: str = None) -> pd.DataFrame:
    """
    Adds physics-consistency features to a dataframe with temperature_c,
    pressure_hpa, humidity_pct columns.

    Returns a copy with new columns:
      - vpd_kpa
      - dew_point_c
      - dew_point_violation  (1 if dew point > air temp, physically impossible)
      - pressure_expected_hpa (only if elevation provided)
      - pressure_anomaly_hpa
    """
    out = df.copy()
    t = out["temperature_c"].to_numpy(dtype=float)
    rh = out["humidity_pct"].to_numpy(dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        out["vpd_kpa"] = vapor_pressure_deficit_kpa(t, rh)
        out["dew_point_c"] = dew_point_c(t, rh)

    out["dew_point_violation"] = (out["dew_point_c"] > out["temperature_c"] + 0.3).astype(int)

    if elevation_col and elevation_col in out.columns:
        elev = out[elevation_col].to_numpy(dtype=float)
        out["pressure_expected_hpa"] = pressure_altitude_expected_hpa(elev)
        out["pressure_anomaly_hpa"] = out["pressure_hpa"] - out["pressure_expected_hpa"]

    return out


def physical_consistency_score(df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """
    Produces a normalized inconsistency score in [0, 1] per row summarizing how
    "physically co-possible" the T/P/RH triple is. Higher = less physically
    consistent. This does NOT by itself decide fault vs. weather event — it is
    evidence, fed into Module 4/5 alongside temporal and spatial evidence.
    """
    out = compute_physics_features(df)

    # Dew point violation is a hard physical impossibility -> strong signal
    violation_score = out["dew_point_violation"].astype(float)

    # VPD relative to a rolling per-station median acts as a soft consistency signal:
    # sudden implausible jumps in VPD unaccompanied by proportional T change are suspicious.
    vpd_z = (
        out.groupby("station_id")["vpd_kpa"]
        .transform(lambda s: (s - s.rolling(48, min_periods=6).median()) / (s.rolling(48, min_periods=6).std() + 1e-6))
        .abs()
        .fillna(0)
    )
    vpd_soft_score = np.clip(vpd_z / 6.0, 0, 1)

    combined = np.clip(0.7 * violation_score + 0.3 * vpd_soft_score, 0, 1)
    out["physics_inconsistency_score"] = combined
    out["physics_flag"] = (combined > threshold).astype(int)
    return out
