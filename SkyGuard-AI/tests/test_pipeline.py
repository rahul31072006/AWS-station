"""Basic smoke tests for the SkyGuard AI pipeline. Run with: pytest tests/"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_generator import generate_network, default_stations, inject_known_faults
from src.baseline import build_features
from src.physics import physical_consistency_score, vapor_pressure_deficit_kpa
from src.anomaly import compute_anomaly_score, health_score
from src.spatial import build_distance_matrix, haversine_km


def test_data_generation_shape():
    df = generate_network(num_stations=3, days=2, interval_minutes=30, seed=1)
    assert not df.empty
    assert set(["timestamp", "station_id", "temperature_c", "pressure_hpa", "humidity_pct"]).issubset(df.columns)


def test_haversine_zero_distance_same_point():
    d = haversine_km(10.0, 78.0, 10.0, 78.0)
    assert abs(d) < 1e-6


def test_vpd_non_negative():
    import numpy as np
    t = np.array([20.0, 30.0, 15.0])
    rh = np.array([50.0, 80.0, 40.0])
    vpd = vapor_pressure_deficit_kpa(t, rh)
    assert (vpd >= 0).all()


def test_known_fault_injection_flags_rows():
    df = generate_network(num_stations=2, days=3, interval_minutes=30, seed=2)
    station_id = df["station_id"].unique()[0]
    injected = inject_known_faults(df, station_id, "spike", seed=1, n_events=2)
    assert (injected["fault_type"] == "spike").sum() > 0


def test_feature_pipeline_runs():
    df = generate_network(num_stations=2, days=2, interval_minutes=30, seed=3)
    feats = build_features(df, rolling_window=4)
    feats = physical_consistency_score(feats)
    feats = compute_anomaly_score(feats)
    feats = health_score(feats)
    assert "anomaly_score" in feats.columns
    assert "health_score" in feats.columns
    assert feats["health_score"].between(0, 100).all()


def test_distance_matrix_symmetric():
    stations = default_stations(4, seed=1)
    stations_df = pd.DataFrame([s.__dict__ for s in stations])
    mat = build_distance_matrix(stations_df)
    diff = (mat.to_numpy() - mat.to_numpy().T)
    assert (abs(diff) < 1e-6).all()
