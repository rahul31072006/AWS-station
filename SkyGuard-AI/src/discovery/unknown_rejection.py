"""Reject anomalous readings that do not confidently match known faults."""

from __future__ import annotations

import pandas as pd


def reject_unknowns(
    flagged: pd.DataFrame,
    confidence_threshold: float = 0.70,
    known_faults: set[str] | None = None,
) -> pd.DataFrame:
    """Label flagged rows as known faults or UNKNOWN.

    The classifier's predicted label is accepted only when its confidence is
    above the configured threshold and the label is in the known fault set.
    Missing classifier output is deliberately treated as unknown.
    """
    out = flagged.copy()
    known = known_faults or {
        "spike", "flatline", "sudden_jump", "sudden_drop", "gradual_drift",
        "bias", "missing_data", "intermittent_failure", "comm_error",
        "calibration_drift", "frozen_sensor", "genuine_event", "none",
    }
    label = out.get("predicted_fault_type", pd.Series("unknown", index=out.index))
    confidence = pd.to_numeric(
        out.get("classifier_confidence", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    is_known = label.isin(known) & confidence.ge(confidence_threshold)
    out["known_fault_status"] = is_known.map({True: "KNOWN", False: "UNKNOWN"})
    out["final_fault_type"] = label.where(is_known, "UNKNOWN")
    out["unknown_confidence"] = (1.0 - confidence).clip(0, 1)
    return out