"""
Module 4 — Root-Cause Classifier.

Combines the temporal features (Module 1), physics-consistency evidence
(Module 2), and spatial evidence (Module 3) into a single feature vector, and
classifies each flagged reading into one of:

  spike | frozen_sensor | comm_error | calibration_drift | genuine_event | healthy

Uses a gradient-boosted tree model (LightGBM if available, else scikit-learn's
GradientBoostingClassifier as a lightweight, dependency-light fallback) so the
project runs on an ordinary laptop without requiring LightGBM to be installed.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import os

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_LGBM = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLUMNS = [
    "temperature_c_zscore", "pressure_hpa_zscore", "humidity_pct_zscore",
    "temperature_c_roll_std", "pressure_hpa_roll_std", "humidity_pct_roll_std",
    "temperature_c_diurnal_deviation", "pressure_hpa_diurnal_deviation", "humidity_pct_diurnal_deviation",
    "temp_humidity_corr_local",
    "physics_inconsistency_score", "dew_point_violation",
]

LABEL_COLUMN = "fault_type"


def make_model():
    if _HAS_LGBM:
        return LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, verbose=-1,
        )
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)


def prepare_training_data(feature_df: pd.DataFrame):
    df = feature_df.dropna(subset=FEATURE_COLUMNS).copy()
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df[LABEL_COLUMN].fillna("none")
    return X, y, df


def train_classifier(feature_df: pd.DataFrame, model_path: str = None):
    """Trains the root-cause classifier. Requires a `fault_type` ground-truth
    column (present only for synthetic evaluation data — never used at real
    inference time as an input feature)."""
    X, y, _ = prepare_training_data(feature_df)
    if y.nunique() < 2:
        raise ValueError("Need at least 2 distinct fault_type classes to train.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = make_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

    return model, {"classification_report": report, "confusion_matrix": cm.tolist(), "labels": sorted(y.unique().tolist())}


def load_classifier(model_path: str):
    return joblib.load(model_path)


def predict_root_cause(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    out = feature_df.copy()
    X = out[FEATURE_COLUMNS].fillna(0)
    out["predicted_fault_type"] = model.predict(X)
    proba = model.predict_proba(X)
    out["classifier_confidence"] = proba.max(axis=1)
    return out
