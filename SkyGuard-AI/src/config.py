"""
Central configuration for SkyGuard AI.

All tunable parameters live here (or in config.yaml) so that no module
hard-codes arbitrary thresholds inline.
"""

import yaml
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML configuration. Falls back to sane defaults if the file is missing."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {
        "rolling_window": 6,
        "baseline_window_hours": 24,
        "spatial": {"num_neighbors": 4, "max_distance_km": 150},
        "vpd": {"inconsistency_threshold": 0.35},
        "anomaly": {"score_threshold": 0.6},
        "self_healing": {"enabled": True},
        "random_seed": 42,
    }
