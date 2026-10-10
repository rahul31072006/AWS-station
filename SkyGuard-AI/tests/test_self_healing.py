"""Regression tests for metadata-safe self-healing."""

import pandas as pd

from src.self_healing import self_heal


def test_self_heal_preserves_station_identity_and_original_values():
    data = pd.DataFrame({
        "station_id": ["AWS-002", "AWS-001", "AWS-001", "AWS-002"],
        "timestamp": pd.to_datetime([
            "2026-01-01 00:10", "2026-01-01 00:00",
            "2026-01-01 00:10", "2026-01-01 00:00",
        ]),
        "temperature_c": [20.0, 10.0, 30.0, 40.0],
        "pressure_hpa": [1000.0, 1001.0, 1003.0, 1004.0],
        "humidity_pct": [50.0, 60.0, 70.0, 80.0],
        "is_sensor_fault": [False, False, True, False],
    })

    healed = self_heal(data, max_gap=1)

    assert healed["station_id"].tolist() == data["station_id"].tolist()
    assert healed["temperature_c_original"].tolist() == data["temperature_c"].tolist()
    assert healed.loc[2, "temperature_c"] == 10.0
    assert healed.loc[2, "temperature_c_imputed"]
    assert healed.loc[2, "imputed"]
    assert set(["station_id", "timestamp"]).issubset(healed.columns)