"""
Fuses Module 1 (temporal), Module 2 (physics), and rule-based heuristics into a
single anomaly score per reading, and applies simple rule-based sub-typing
(spike / frozen / drift) that seeds the ground-truth-free `is_flagged` column
consumed by the classifier and spatial modules.

This score-fusion step is intentionally simple and transparent (not another
black box) so operators can trust *why* something was flagged before the
root-cause classifier and SHAP add further detail.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

PARAMS = ["temperature_c", "pressure_hpa", "humidity_pct"]


def compute_anomaly_score(
    df: pd.DataFrame,
    score_threshold: float = 0.6,
    frozen_std_threshold: float = 0.01,
    spike_zscore_threshold: float = 4.0,
) -> pd.DataFrame:
    out = df.copy()

    # Multivariate deviation: combine per-parameter z-scores (temporal) with
    # the physics inconsistency score. The diurnal-baseline deviation (which
    # already accounts for the expected hour-of-day cycle per station) is used
    # here rather than a raw rolling z-score, since a raw rolling window lags
    # behind fast, entirely normal diurnal transitions (e.g. morning warm-up)
    # and would otherwise generate frequent false positives.
    z_cols = [f"{p}_zscore" for p in PARAMS]
    diurnal_cols_for_score = [f"{p}_diurnal_deviation" for p in PARAMS]
    if all(c in out.columns for c in diurnal_cols_for_score):
        temporal_matrix = out[diurnal_cols_for_score].abs().fillna(0).to_numpy()
        temporal_denom = spike_zscore_threshold * 0.75  # diurnal z is typically tighter than raw z
    else:
        temporal_matrix = out[z_cols].abs().fillna(0).to_numpy()
        temporal_denom = spike_zscore_threshold
    temporal_score = np.clip(temporal_matrix.max(axis=1) / temporal_denom, 0, 1)

    physics_score = out.get("physics_inconsistency_score", pd.Series(0, index=out.index)).fillna(0)

    # A "frozen sensor" (stuck reading) shows near-zero short-term variance
    # while it *should* be varying according to the learned diurnal baseline --
    # this is a distinct evidence signal from a raw z-score spike, since a
    # flatlined value can sit close to its own recent rolling mean.
    diurnal_cols = [f"{p}_diurnal_deviation" for p in PARAMS]
    has_diurnal = all(c in out.columns for c in diurnal_cols)
    if has_diurnal:
        diurnal_dev = out[diurnal_cols].abs().fillna(0).to_numpy()
    else:
        diurnal_dev = np.zeros((len(out), len(PARAMS)))
    roll_std_matrix = out[[f"{p}_roll_std" for p in PARAMS]].fillna(1).to_numpy()
    frozen_hint_matrix = (roll_std_matrix < frozen_std_threshold) & (diurnal_dev > 1.0)
    frozen_score = frozen_hint_matrix.any(axis=1).astype(float)

    # Missing-data readings are unambiguous evidence of a data/comm fault.
    missing_score = out[PARAMS].isna().any(axis=1).astype(float)

    # Evidence fusion: each signal is independently sufficient to raise
    # suspicion (a max-style ensemble), rather than requiring several weak
    # signals to add up -- a single strong piece of evidence (e.g. an
    # impossible dew point, a frozen sensor, or missing data) should be
    # enough to flag a reading even if the others are quiet.
    out["anomaly_score"] = np.clip(
        np.maximum.reduce([
            0.9 * temporal_score,
            physics_score,
            0.75 * frozen_score,
            0.95 * missing_score,
        ]),
        0, 1,
    )
    out["is_flagged"] = out["anomaly_score"] > score_threshold

    # Rule-based quick sub-type hints (used as extra features / sanity checks,
    # the final label still comes from Module 4's learned classifier)
    out["hint_frozen"] = frozen_hint_matrix.any(axis=1)
    out["hint_spike"] = (out[z_cols].abs().fillna(0) > spike_zscore_threshold).any(axis=1)
    out["hint_missing"] = out[PARAMS].isna().any(axis=1)

    return out


def health_score(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Derives an explainable operational sensor-health indicator from the
    available evidence. The component columns are retained so operators can
    see why a score changed; this is not a calibrated failure probability.
    """
    out = df.sort_values(["station_id", "timestamp"]).copy()
    persistence = (
        out.groupby("station_id")["anomaly_score"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    anomaly_penalty = np.clip(out["anomaly_score"].fillna(0) * 55, 0, 55)
    physics_penalty = np.clip(
        out.get("physics_inconsistency_score", pd.Series(0.0, index=out.index)).fillna(0) * 20,
        0,
        20,
    )
    spatial_penalty = np.clip(
        out.get("spatial_disagreement_score", pd.Series(0.0, index=out.index)).fillna(0) * 15,
        0,
        15,
    )
    persistence_penalty = np.clip(persistence * 10, 0, 10)
    out["health_anomaly_penalty"] = anomaly_penalty.round(1)
    out["health_physics_penalty"] = physics_penalty.round(1)
    out["health_spatial_penalty"] = spatial_penalty.round(1)
    out["health_persistence_penalty"] = persistence_penalty.round(1)
    out["health_score"] = np.clip(
        100 - anomaly_penalty - physics_penalty - spatial_penalty - persistence_penalty,
        0,
        100,
    ).round(1)
    out["health_trend"] = out.groupby("station_id")["health_score"].transform(
        lambda s: s - s.shift(min(window, max(len(s) - 1, 1))).fillna(s.iloc[0])
    ).round(1)
    out["health_status"] = np.select(
        [out["health_trend"] <= -10, out["health_score"] < 50, out["health_score"] < 75],
        ["DEGRADING", "AT_RISK", "WATCH"],
        default="STABLE",
    )
    return out


FAULT_EVOLUTION_STATES = [
    "HEALTHY", "EARLY_DEVIATION", "EMERGING_ANOMALY", "PERSISTENT_FAULT", "RECOVERY",
]


def fault_evolution_state(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Assigns a simple evolution state per row per station based on recent
    persistence of flags, so operators see whether an issue is emerging,
    persistent, or recovering rather than a single binary flag.
    """
    out = df.sort_values(["station_id", "timestamp"]).copy()
    flagged_rate = (
        out.groupby("station_id")["is_flagged"]
        .transform(lambda s: s.astype(float).rolling(window, min_periods=1).mean())
    )
    prev_rate = flagged_rate.groupby(out["station_id"]).shift(1).fillna(0)

    def classify(rate, prev, current_flag):
        if rate == 0:
            return "HEALTHY"
        if current_flag and prev < 0.3 and rate < 0.5:
            return "EARLY_DEVIATION"
        if rate >= 0.5 and prev < rate:
            return "EMERGING_ANOMALY"
        if rate >= 0.5 and prev >= rate:
            return "PERSISTENT_FAULT"
        if rate < prev:
            return "RECOVERY"
        return "EARLY_DEVIATION"

    out["fault_evolution_state"] = [
        classify(r, p, f) for r, p, f in zip(flagged_rate, prev_rate, out["is_flagged"])
    ]
    return out
