"""Convert anomalous observations into comparable unknown-fault signatures."""

from __future__ import annotations

import pandas as pd


SIGNATURE_COLUMNS = [
    "anomaly_score", "temperature_c_zscore", "pressure_hpa_zscore",
    "humidity_pct_zscore", "temperature_c_diurnal_deviation",
    "pressure_hpa_diurnal_deviation", "humidity_pct_diurnal_deviation",
    "physics_inconsistency_score", "spatial_fault_votes",
]


def build_signatures(unknowns: pd.DataFrame) -> pd.DataFrame:
    """Add a stable numeric signature vector and human-readable pattern text."""
    out = unknowns.copy()
    for column in SIGNATURE_COLUMNS:
        if column not in out:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    def describe(row: pd.Series) -> str:
        deviations = {
            "temperature": abs(row["temperature_c_zscore"]),
            "pressure": abs(row["pressure_hpa_zscore"]),
            "humidity": abs(row["humidity_pct_zscore"]),
        }
        dominant = max(deviations, key=deviations.get)
        physics = row["physics_inconsistency_score"] > 0.5
        return f"{dominant} deviation" + (" with multivariate inconsistency" if physics else "")

    out["unknown_pattern"] = out.apply(describe, axis=1)
    out["signature_vector"] = out[SIGNATURE_COLUMNS].round(4).values.tolist()
    return out