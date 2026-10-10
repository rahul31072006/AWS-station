"""
SkyGuard AI — Module 6: Optional Self-Healing.

When a reading is classified as a probable sensor fault, this module
reconstructs a plausible corrected value using temporal interpolation
from the same station's healthy history.

CRITICAL RULE:
Imputed values are NEVER silently substituted.

Every imputed value:
    - has <param>_original preserved
    - is marked through <param>_imputed
    - causes the row-level `imputed` flag to become True
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PARAMS = [
    "temperature_c",
    "pressure_hpa",
    "humidity_pct",
]


def self_heal(
    df: pd.DataFrame,
    fault_mask_col: str = "is_sensor_fault",
    max_gap: int = 6,
) -> pd.DataFrame:
    """
    Heal probable sensor faults using temporal interpolation.

    Requirements:
        station_id
        timestamp
        temperature_c
        pressure_hpa
        humidity_pct
        fault_mask_col

    Adds:
        <param>_original
        <param>_imputed
        imputed

    Original values are ALWAYS preserved.
    """

    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------
    required = [
        "station_id",
        "timestamp",
        fault_mask_col,
        *PARAMS,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"self_heal() is missing required columns: {missing}"
        )

    # ------------------------------------------------------------
    # Preserve original row identity
    # ------------------------------------------------------------
    out = df.copy()

    out["_original_index"] = out.index

    # Make sure timestamps are datetime
    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        errors="coerce",
    )

    # ------------------------------------------------------------
    # Preserve original values BEFORE any modification
    # ------------------------------------------------------------
    for param in PARAMS:
        out[f"{param}_original"] = out[param]

        # Explicit per-parameter imputation flag
        out[f"{param}_imputed"] = False

    # Row-level flag
    out["imputed"] = False

    # ------------------------------------------------------------
    # Sort for temporal interpolation
    # ------------------------------------------------------------
    out = out.sort_values(
        ["station_id", "timestamp"]
    ).copy()

        # ------------------------------------------------------------
        # Heal one station at a time
        # ------------------------------------------------------------
    healed_groups = []

    for station_id, group in out.groupby(
        "station_id",
        sort=False,
        dropna=False,
    ):
        g = group.copy()

        fault_rows = (
            g[fault_mask_col]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        for param in PARAMS:

            # Convert to numeric safely
            series = pd.to_numeric(
                g[param],
                errors="coerce",
            )

            # Faulty observations must NOT influence interpolation
            healthy_series = series.where(
                ~fault_rows,
                other=np.nan,
            )

            # Temporal interpolation
            interpolated = healthy_series.interpolate(
                method="linear",
                limit=max_gap,
                limit_direction="both",
            )

            # Determine which faulty values can actually be healed
            can_heal = (
                fault_rows
                & interpolated.notna().to_numpy()
            )

            # Start with original values
            healed_values = series.to_numpy(
                copy=True
            )

            # Replace ONLY values that can safely be interpolated
            healed_values[can_heal] = (
                interpolated.to_numpy()[can_heal]
            )

            g[param] = healed_values

            # Per-parameter flag
            g.loc[
                can_heal,
                f"{param}_imputed"
            ] = True

            # Row-level flag
            g.loc[
                can_heal,
                "imputed"
            ] = True

        healed_groups.append(g)

    # ------------------------------------------------------------
    # Recombine groups
    # ------------------------------------------------------------
    if healed_groups:
        out = pd.concat(
            healed_groups,
            axis=0,
        )
    else:
        out = out.copy()

    # ------------------------------------------------------------
    # Restore original row ordering
    # ------------------------------------------------------------
    out = (
        out.sort_values("_original_index")
        .drop(columns=["_original_index"])
    )

    # ------------------------------------------------------------
    # Final safety checks
    # ------------------------------------------------------------
    if "station_id" not in out.columns:
        raise RuntimeError(
            "Self-healing accidentally removed station_id."
        )

    if "timestamp" not in out.columns:
        raise RuntimeError(
            "Self-healing accidentally removed timestamp."
        )

    return out