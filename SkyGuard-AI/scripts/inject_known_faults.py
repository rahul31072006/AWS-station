"""
Loads the clean synthetic dataset and injects known fault types across
different stations, producing a labeled evaluation dataset. Ground-truth
labels (true_state, fault_type) are saved for evaluation only and must never
be used as model input features at inference time.

Usage: python scripts/inject_known_faults.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_generator import inject_known_faults, KNOWN_FAULT_TYPES
from src.config import load_config


def main():
    cfg = load_config()
    src_path = "data/sample/sample_aws_data.csv"
    if not os.path.exists(src_path):
        raise FileNotFoundError("Run scripts/generate_sample_data.py first.")

    df = pd.read_csv(src_path, parse_dates=["timestamp"])
    station_ids = df["station_id"].unique().tolist()

    result = df.copy()
    result["true_state"] = "HEALTHY"
    result["fault_type"] = "none"

    for i, fault_type in enumerate(KNOWN_FAULT_TYPES):
        station_id = station_ids[i % len(station_ids)]
        injected = inject_known_faults(result, station_id, fault_type, seed=cfg.get("random_seed", 42) + i)
        # Only overwrite the specific rows this fault actually touched, so that
        # stations reused across multiple fault types (more fault types than
        # stations) don't have earlier injected faults wiped out.
        touched = injected["fault_type"] == fault_type
        cols = ["temperature_c", "pressure_hpa", "humidity_pct", "true_state", "fault_type", "source"]
        result.loc[touched, cols] = injected.loc[touched, cols]

    os.makedirs("data/sample", exist_ok=True)
    result.to_csv("data/sample/labeled_evaluation_data.csv", index=False)
    print(f"Injected known faults: {KNOWN_FAULT_TYPES}")
    print(f"Fault distribution:\n{result['fault_type'].value_counts()}")
    print("Saved: data/sample/labeled_evaluation_data.csv")


if __name__ == "__main__":
    main()
