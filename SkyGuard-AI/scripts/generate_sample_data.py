"""
Generates a synthetic demo AWS network (clean data) and saves it to
data/sample/. Clearly labeled SYNTHETIC — for demonstration only.

Usage: python scripts/generate_sample_data.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_generator import generate_network, default_stations
from src.config import load_config


def main():
    cfg = load_config()
    stations = default_stations(cfg.get("num_stations", 6), seed=cfg.get("random_seed", 42))
    stations_df = pd.DataFrame([s.__dict__ for s in stations])

    df = generate_network(
        num_stations=cfg.get("num_stations", 6),
        days=14,
        interval_minutes=cfg.get("sampling_interval_minutes", 10),
        seed=cfg.get("random_seed", 42),
    )

    os.makedirs("data/sample", exist_ok=True)
    df.to_csv("data/sample/sample_aws_data.csv", index=False)
    stations_df.to_csv("data/sample/stations.csv", index=False)
    print(f"Generated {len(df)} synthetic readings across {len(stations)} stations.")
    print("Saved: data/sample/sample_aws_data.csv, data/sample/stations.csv")


if __name__ == "__main__":
    main()
