"""
Runs the full pipeline against the labeled synthetic evaluation dataset and
reports detection accuracy metrics (precision/recall/F1 against known faults,
plus basic health-score / evolution-state sanity summaries).

Note: this evaluation always uses synthetic fault injection; no results here
should be presented as validated against real-world sensor faults.

Usage: python evaluate.py
"""
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.pipeline import run_pipeline
from src.classifier import load_classifier
from src.config import load_config


def main():
    cfg = load_config()
    data_path = "data/sample/labeled_evaluation_data.csv"
    stations_path = "data/sample/stations.csv"
    model_path = "models/root_cause_classifier.joblib"

    if not os.path.exists(data_path) or not os.path.exists(stations_path):
        raise FileNotFoundError("Run scripts/generate_sample_data.py and scripts/inject_known_faults.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Run train.py first.")

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    stations_df = pd.read_csv(stations_path)
    model = load_classifier(model_path)

    result = run_pipeline(df, stations_df, model=model, config=cfg, explain_top_flagged=5)
    feats = result["features"]

    ground_truth_flag = (feats["true_state"] == "FAULT")
    predicted_flag = feats["is_flagged"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth_flag, predicted_flag, average="binary", zero_division=0
    )

    print("=== SkyGuard AI — Synthetic Fault Injection Evaluation ===\n")
    print(f"Total readings evaluated: {len(feats)}")
    print(f"Ground-truth faulty readings: {int(ground_truth_flag.sum())}")
    print(f"Flagged by system: {int(predicted_flag.sum())}\n")
    print(f"Detection Precision: {precision:.3f}")
    print(f"Detection Recall:    {recall:.3f}")
    print(f"Detection F1:        {f1:.3f}\n")

    print("Confusion matrix (Healthy vs Fault, row=truth, col=predicted):")
    print(confusion_matrix(ground_truth_flag, predicted_flag))

    print(f"\nStations processed: {feats['station_id'].nunique()}")
    print(f"Mean sensor health score: {feats['health_score'].mean():.1f} / 100")
    print("\nFault evolution state distribution:")
    print(feats["fault_evolution_state"].value_counts())

    print("\nNote: Synthetic fault injection evaluation.")


if __name__ == "__main__":
    main()
