"""
SkyGuard AI — end-to-end pipeline orchestrator.

    AWS DATA STREAM (T, P, RH)
            |
    Module 1: Temporal & Seasonal Baseline
            |
    Module 2: Multivariate Consistency (VPD/Physics)
            |
    Anomaly scoring (fusion) --> is_flagged
            |
    Module 3: Lightweight Spatial Check (for flagged rows)
            |
    Module 4: Root-Cause Classifier
            |
    Module 5: Confidence + SHAP Explainability
            |
    Module 6: Optional Self-Healing
            |
    Dashboard / outputs
"""

from __future__ import annotations
import logging
import pandas as pd

from .baseline import build_features
from .physics import physical_consistency_score
from .anomaly import compute_anomaly_score, health_score, fault_evolution_state
from .spatial import build_distance_matrix, spatial_consistency_evidence
from .classifier import predict_root_cause, FEATURE_COLUMNS
from .explainability import build_explainer, explain_row
from .self_healing import self_heal
from .config import load_config
from .discovery import (
    UnknownFaultRegistry,
    build_signatures,
    cluster_signatures,
    reject_unknowns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("skyguard.pipeline")


def run_pipeline(
    raw_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    model=None,
    config: dict = None,
    explain_top_flagged: int = 25,
) -> dict:
    """
    Runs the full SkyGuard AI pipeline on a batch of raw AWS observations.

    Returns a dict with:
      - features: full feature dataframe (Modules 1-2 + anomaly score + evolution)
      - flagged: subset of rows that were flagged, with root-cause + spatial evidence
      - healed: self-healed dataframe (Module 6)
      - explanations: dict of row-index -> explanation (Module 5), for a sample
        of flagged rows (bounded by explain_top_flagged for performance)
    """
    cfg = config or load_config()
    logger.info("Step 1/6: building temporal & seasonal baseline features")
    feats = build_features(raw_df, rolling_window=cfg.get("rolling_window", 6))

    logger.info("Step 2/6: computing physical (VPD/psychrometric) consistency")
    feats = physical_consistency_score(feats, threshold=cfg["vpd"]["inconsistency_threshold"])

    logger.info("Step 3/6: fusing anomaly evidence")
    feats = compute_anomaly_score(
        feats,
        score_threshold=cfg["anomaly"]["score_threshold"],
        frozen_std_threshold=cfg["anomaly"]["frozen_std_threshold"],
        spike_zscore_threshold=cfg["anomaly"]["spike_zscore_threshold"],
    )
    logger.info("Step 4/6: spatial cross-check for flagged readings")
    dist_matrix = build_distance_matrix(stations_df)
    flagged = feats[feats["is_flagged"]].copy()

    spatial_results = []
    for _, row in flagged.iterrows():
        for param in ["temperature_c", "pressure_hpa", "humidity_pct"]:
            ev = spatial_consistency_evidence(
                feats, dist_matrix, row["station_id"], row["timestamp"], param,
                k=cfg["spatial"]["num_neighbors"], max_km=cfg["spatial"]["max_distance_km"],
            )
            ev.update({"index": row.name, "param": param})
            spatial_results.append(ev)
    spatial_df = pd.DataFrame(spatial_results)

    if not spatial_df.empty:
        weather_votes = (spatial_df["evidence"] == "supports_weather_event").groupby(spatial_df["index"]).sum()
        fault_votes = (spatial_df["evidence"] == "supports_sensor_fault").groupby(spatial_df["index"]).sum()
        flagged["spatial_weather_votes"] = flagged.index.map(weather_votes).fillna(0)
        flagged["spatial_fault_votes"] = flagged.index.map(fault_votes).fillna(0)
        flagged["spatial_diagnosis"] = flagged.apply(
            lambda r: (
                "likely_weather_event" if r["spatial_weather_votes"] > r["spatial_fault_votes"]
                else "likely_sensor_fault" if r["spatial_fault_votes"] > r["spatial_weather_votes"]
                else "uncertain"
            ), axis=1,
        )
    else:
        flagged["spatial_weather_votes"] = 0
        flagged["spatial_fault_votes"] = 0
        flagged["spatial_diagnosis"] = "uncertain"

    # Preserve a normalized spatial signal on the full feature frame so the
    # health indicator reflects the same evidence used for diagnosis.
    feats["spatial_fault_votes"] = 0
    feats["spatial_weather_votes"] = 0
    feats["spatial_disagreement_score"] = 0.0
    if not spatial_df.empty:
        fault_by_index = flagged["spatial_fault_votes"]
        weather_by_index = flagged["spatial_weather_votes"]
        feats.loc[flagged.index, "spatial_fault_votes"] = fault_by_index
        feats.loc[flagged.index, "spatial_weather_votes"] = weather_by_index
        total_votes = (fault_by_index + weather_by_index).replace(0, 1)
        feats.loc[flagged.index, "spatial_disagreement_score"] = (
            fault_by_index / total_votes
        )
    feats = health_score(feats)
    feats = fault_evolution_state(feats)

    logger.info("Step 5/6: root-cause classification + explainability")
    explanations = {}
    if model is not None and len(flagged) > 0:
        flagged = predict_root_cause(model, flagged)
        explainer = build_explainer(model)
        for idx, row in flagged.head(explain_top_flagged).iterrows():
            explanations[idx] = explain_row(model, explainer, row)
    else:
        flagged["predicted_fault_type"] = "unclassified (no trained model loaded)"
        flagged["classifier_confidence"] = 0.0

    discovery_cfg = cfg.get("unknown_discovery", {})
    if discovery_cfg.get("enabled", True) and not flagged.empty:
        flagged = reject_unknowns(
            flagged,
            confidence_threshold=discovery_cfg.get("confidence_threshold", 0.70),
        )
        unknowns = flagged[flagged["final_fault_type"] == "UNKNOWN"].copy()
        unknowns = build_signatures(unknowns)
        unknowns = cluster_signatures(
            unknowns,
            eps=discovery_cfg.get("dbscan_eps", 1.25),
            min_samples=discovery_cfg.get("dbscan_min_samples", 2),
        )
        registry = UnknownFaultRegistry(discovery_cfg.get("registry_path", "data/unknown_faults/registry.json"))
        registry_records = registry.update(unknowns)
        flagged = flagged.join(
            unknowns[["unknown_pattern", "signature_vector", "unknown_cluster_id"]],
            how="left",
        )
    else:
        unknowns = flagged.iloc[0:0].copy()
        registry_records = {}
        flagged["known_fault_status"] = "KNOWN"
        flagged["final_fault_type"] = flagged.get("predicted_fault_type", "UNKNOWN")

    flagged["is_sensor_fault"] = (
        (flagged["spatial_diagnosis"] != "likely_weather_event")
        & (flagged["final_fault_type"] != "genuine_event")
    )
    flagged["ml_anomaly_probability"] = flagged["anomaly_score"].round(3)
    flagged["physics_validation"] = flagged["physics_inconsistency_score"].map(
        lambda score: "FAILED" if score >= cfg["vpd"]["inconsistency_threshold"] else "PASSED"
    )
    if not spatial_df.empty:
        spatial_groups = spatial_df.groupby("index")
        nearby_ids = spatial_groups["neighbor_ids"].apply(
            lambda values: sorted({station for ids in values for station in ids})
        )
        disagreement_count = spatial_groups["evidence"].apply(
            lambda values: int((values == "supports_sensor_fault").sum())
        )
        flagged["nearby_station_ids"] = [
            value if isinstance(value, list) else []
            for value in flagged.index.map(nearby_ids)
        ]
        flagged["nearby_disagreement_count"] = flagged.index.map(disagreement_count).fillna(0).astype(int)
    else:
        flagged["nearby_station_ids"] = [[] for _ in range(len(flagged))]
        flagged["nearby_disagreement_count"] = 0
    flagged["nearby_station_count"] = flagged["nearby_station_ids"].map(len)
    flagged["final_decision"] = flagged.apply(
        lambda row: (
            "UNKNOWN ANOMALY PATTERN" if row["final_fault_type"] == "UNKNOWN"
            else "LIKELY GENUINE WEATHER EVENT" if row["spatial_diagnosis"] == "likely_weather_event"
            and row["physics_validation"] == "PASSED"
            else "LIKELY SENSOR FAULT" if row["is_sensor_fault"]
            else "REVIEW REQUIRED"
        ),
        axis=1,
    )
    flagged["decision_confidence"] = flagged.apply(
        lambda row: round(
            max(float(row.get("classifier_confidence", 0.0)), float(row["anomaly_score"])), 3
        ),
        axis=1,
    )

    logger.info("Step 6/6: self-healing imputation")
    feats["is_sensor_fault"] = False
    feats.loc[flagged.index, "is_sensor_fault"] = flagged["is_sensor_fault"]
    if cfg["self_healing"]["enabled"]:
        healed = self_heal(
            feats,
            fault_mask_col="is_sensor_fault",
            max_gap=cfg["self_healing"]["max_gap_readings"],
        )
    else:
        healed = feats.copy()

    required_healed_columns = ["station_id", "timestamp"]
    missing_healed = [
        column for column in required_healed_columns
        if column not in healed.columns
    ]
    if missing_healed:
        raise RuntimeError(
            f"Self-healing output is missing required columns: {missing_healed}"
        )

    logger.info(f"Pipeline complete: {len(feats)} readings processed, {len(flagged)} flagged.")
    return {
        "features": feats,
        "flagged": flagged,
        "healed": healed,
        "explanations": explanations,
        "distance_matrix": dist_matrix,
        "unknowns": unknowns,
        "unknown_registry": registry_records,
        "spatial_evidence": spatial_df,
    }
