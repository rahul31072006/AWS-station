"""Focused tests for open-world unknown-fault discovery."""

import pandas as pd

from src.discovery import UnknownFaultRegistry, build_signatures, cluster_signatures, reject_unknowns


def test_low_confidence_prediction_is_unknown():
    rows = pd.DataFrame({
        "predicted_fault_type": ["spike", "new_pattern"],
        "classifier_confidence": [0.95, 0.30],
    })
    result = reject_unknowns(rows, confidence_threshold=0.70)
    assert result["final_fault_type"].tolist() == ["spike", "UNKNOWN"]


def test_signature_clustering_preserves_noise_label():
    rows = pd.DataFrame({
        "anomaly_score": [0.8, 0.81],
        "temperature_c_zscore": [1.0, 1.01],
        "pressure_hpa_zscore": [0.0, 0.01],
        "humidity_pct_zscore": [2.0, 2.01],
        "temperature_c_diurnal_deviation": [1.0, 1.01],
        "pressure_hpa_diurnal_deviation": [0.0, 0.01],
        "humidity_pct_diurnal_deviation": [2.0, 2.01],
        "physics_inconsistency_score": [0.7, 0.71],
        "spatial_fault_votes": [1, 1],
    })
    result = cluster_signatures(build_signatures(rows), min_samples=2)
    assert result["unknown_cluster_id"].iloc[0] == result["unknown_cluster_id"].iloc[1]


def test_registry_persists_recurring_cluster(tmp_path):
    path = tmp_path / "registry.json"
    registry = UnknownFaultRegistry(path)
    rows = pd.DataFrame({
        "unknown_cluster_id": [0, 0],
        "station_id": ["AWS-001", "AWS-002"],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
    })
    records = registry.update(rows)
    assert records["0"]["fault_id"] == "UF-001"
    assert records["0"]["occurrences"] == 2
    assert path.exists()