import os
import pandas as pd

from src.baseline import build_features
from src.physics import physical_consistency_score
from src.anomaly import compute_anomaly_score
from src.classifier import train_classifier
from src.config import load_config


def main():
    cfg = load_config()
    path = "data/sample/labeled_evaluation_data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Run scripts/generate_sample_data.py and scripts/inject_known_faults.py first.")

    df = pd.read_csv(path, parse_dates=["timestamp"])

    feats = build_features(df, rolling_window=cfg.get("rolling_window", 6))
    feats = physical_consistency_score(feats, threshold=cfg["vpd"]["inconsistency_threshold"])
    feats = compute_anomaly_score(
        feats,
        score_threshold=cfg["anomaly"]["score_threshold"],
        frozen_std_threshold=cfg["anomaly"]["frozen_std_threshold"],
        spike_zscore_threshold=cfg["anomaly"]["spike_zscore_threshold"],
    )

    model, metrics = train_classifier(feats, model_path="models/root_cause_classifier.joblib")

    print("Training complete. Saved model to models/root_cause_classifier.joblib")
    print("\nPer-class metrics:")
    for label, m in metrics["classification_report"].items():
        if isinstance(m, dict):
            print(f"  {label:>20s}  precision={m['precision']:.2f}  recall={m['recall']:.2f}  f1={m['f1-score']:.2f}")
    print(f"\nOverall accuracy: {metrics['classification_report']['accuracy']:.3f}")
    print("(Note: synthetic fault injection evaluation.)")


if __name__ == "__main__":
    main()
