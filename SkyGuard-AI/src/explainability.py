"""
Module 5 — Confidence Score + Explainable AI (SHAP).

Wraps the trained root-cause classifier with SHAP (TreeExplainer where
possible) to produce a human-readable explanation of why a reading
was flagged, in addition to the raw confidence score from Module 4.

Falls back gracefully to feature-importance-based explanations if SHAP
is unavailable or produces an unexpected output shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from .classifier import FEATURE_COLUMNS


# ---------------------------------------------------------------------
# Human-readable feature names
# ---------------------------------------------------------------------

READABLE_NAMES = {
    "temperature_c_zscore":
        "temperature deviation from its recent baseline",

    "pressure_hpa_zscore":
        "pressure deviation from its recent baseline",

    "humidity_pct_zscore":
        "humidity deviation from its recent baseline",

    "temperature_c_roll_std":
        "short-term temperature variance",

    "pressure_hpa_roll_std":
        "short-term pressure variance",

    "humidity_pct_roll_std":
        "short-term humidity variance",

    "temperature_c_diurnal_deviation":
        "deviation from the expected daily temperature cycle",

    "pressure_hpa_diurnal_deviation":
        "deviation from the expected daily pressure cycle",

    "humidity_pct_diurnal_deviation":
        "deviation from the expected daily humidity cycle",

    "temp_humidity_corr_local":
        "loss of the usual temperature-humidity relationship",

    "physics_inconsistency_score":
        "physical VPD/psychrometric inconsistency between T, P and RH",

    "dew_point_violation":
        "an impossible dew-point-above-air-temperature reading",
}


# ---------------------------------------------------------------------
# SHAP Explainer
# ---------------------------------------------------------------------

def build_explainer(model):
    """
    Build a SHAP TreeExplainer when possible.

    Returns:
        SHAP explainer object, or None if SHAP is unavailable/
        incompatible with the supplied model.
    """

    if not _HAS_SHAP:
        return None

    try:
        return shap.TreeExplainer(model)

    except Exception:
        return None


# ---------------------------------------------------------------------
# Main explanation function
# ---------------------------------------------------------------------

def explain_row(
    model,
    explainer,
    row: pd.Series,
    top_k: int = 3
) -> dict:
    """
    Explain why a particular row was flagged.

    Returns:
        {
            "top_features": [
                {
                    "feature": "...",
                    "contribution": 0.1234,
                    "readable_name": "..."
                }
            ],
            "narrative": "Flagged mainly due to ..."
        }
    """

    # -------------------------------------------------------------
    # Prepare model input
    # -------------------------------------------------------------

    try:
        x = (
            row[FEATURE_COLUMNS]
            .fillna(0)
            .to_numpy()
            .reshape(1, -1)
        )

    except Exception:
        # If something unexpected happens while preparing the row,
        # return a safe fallback explanation.
        return {
            "top_features": [],
            "narrative": "Unable to generate an explanation for this reading.",
        }

    contributions = None

    # -------------------------------------------------------------
    # Try SHAP
    # -------------------------------------------------------------

    if explainer is not None:

        try:

            shap_values = explainer.shap_values(x)

            # -----------------------------------------------------
            # Different SHAP versions/models can return:
            #
            # 1. list of arrays       -> multiclass
            # 2. 2D numpy array       -> normal output
            # 3. 3D numpy array       -> newer SHAP multiclass
            # -----------------------------------------------------

            if isinstance(shap_values, list):

                # Multiclass SHAP output
                probabilities = model.predict_proba(x)[0]

                pred_class_idx = int(
                    np.argmax(probabilities)
                )

                selected_values = shap_values[pred_class_idx]

                selected_values = np.asarray(
                    selected_values
                )

                # Usually shape = (1, n_features)
                if selected_values.ndim > 1:
                    selected_values = selected_values[0]

                sv = selected_values

            else:

                shap_array = np.asarray(shap_values)

                # -------------------------------------------------
                # Handle different SHAP output dimensions
                # -------------------------------------------------

                if shap_array.ndim == 1:

                    # Shape:
                    # (n_features,)
                    sv = shap_array

                elif shap_array.ndim == 2:

                    # Shape:
                    # (1, n_features)
                    sv = shap_array[0]

                elif shap_array.ndim == 3:

                    # Possible shape:
                    # (1, n_features, n_classes)
                    #
                    # Select the predicted class.
                    probabilities = model.predict_proba(x)[0]

                    pred_class_idx = int(
                        np.argmax(probabilities)
                    )

                    sv = shap_array[
                        0,
                        :,
                        pred_class_idx
                    ]

                else:

                    raise ValueError(
                        f"Unexpected SHAP output shape: "
                        f"{shap_array.shape}"
                    )

            # -----------------------------------------------------
            # Convert every contribution to a scalar
            # -----------------------------------------------------

            sv = np.asarray(sv).reshape(-1)

            contributions = []

            for feature, value in zip(
                FEATURE_COLUMNS,
                sv
            ):

                # Force NumPy scalar/array into Python float
                scalar_value = float(
                    np.asarray(value).reshape(-1)[0]
                )

                contributions.append(
                    (
                        feature,
                        scalar_value
                    )
                )

        except Exception:

            # SHAP failed.
            # Use feature importance fallback.
            contributions = _fallback_importance(
                model,
                x
            )

    else:

        # SHAP isn't installed/available.
        contributions = _fallback_importance(
            model,
            x
        )

    # -------------------------------------------------------------
    # Final safety check
    # -------------------------------------------------------------

    if not contributions:

        contributions = [
            (
                feature,
                0.0
            )
            for feature in FEATURE_COLUMNS
        ]

    # -------------------------------------------------------------
    # Sort by absolute contribution
    #
    # IMPORTANT:
    # Every contribution is now guaranteed to be a scalar.
    # -------------------------------------------------------------

    contributions.sort(
        key=lambda t: abs(float(t[1])),
        reverse=True
    )

    # Select top K
    top = contributions[:top_k]

    # -------------------------------------------------------------
    # Generate human-readable narrative
    # -------------------------------------------------------------

    narrative_parts = []

    for feature, contribution in top:

        readable_name = READABLE_NAMES.get(
            feature,
            feature
        )

        narrative_parts.append(
            readable_name
        )

    if narrative_parts:

        narrative = (
            "Flagged mainly due to "
            + ", ".join(narrative_parts)
            + "."
        )

    else:

        narrative = (
            "Unable to determine the main contributing features."
        )

    # -------------------------------------------------------------
    # Return final explanation
    # -------------------------------------------------------------

    return {
        "top_features": [
            {
                "feature": feature,

                "contribution": round(
                    float(contribution),
                    4
                ),

                "readable_name": READABLE_NAMES.get(
                    feature,
                    feature
                ),
            }

            for feature, contribution in top
        ],

        "narrative": narrative,
    }


# ---------------------------------------------------------------------
# Fallback explanation
# ---------------------------------------------------------------------

def _fallback_importance(
    model,
    x
):
    """
    Feature-importance-weighted fallback explanation.

    Used when:
      - SHAP is not installed
      - SHAP cannot explain the model
      - SHAP returns an unexpected output

    Returns:
        list of (feature, contribution)
    """

    # Try to get tree-model feature importances
    importances = getattr(
        model,
        "feature_importances_",
        None
    )

    # -------------------------------------------------------------
    # If model has no feature_importances_, use equal weighting
    # -------------------------------------------------------------

    if importances is None:

        importances = (
            np.ones(
                len(FEATURE_COLUMNS)
            )
            /
            len(FEATURE_COLUMNS)
        )

    else:

        importances = np.asarray(
            importances
        ).reshape(-1)

    # -------------------------------------------------------------
    # Make sure number of importances matches features
    # -------------------------------------------------------------

    if len(importances) != len(FEATURE_COLUMNS):

        importances = (
            np.ones(
                len(FEATURE_COLUMNS)
            )
            /
            len(FEATURE_COLUMNS)
        )

    # -------------------------------------------------------------
    # Convert x into a flat numeric array
    # -------------------------------------------------------------

    values = np.asarray(
        x,
        dtype=float
    ).reshape(-1)

    # -------------------------------------------------------------
    # Calculate contribution
    # -------------------------------------------------------------

    contributions = []

    for feature, importance, value in zip(
        FEATURE_COLUMNS,
        importances,
        values
    ):

        contribution = (
            float(importance)
            *
            float(value)
        )

        contributions.append(
            (
                feature,
                contribution
            )
        )

    return contributions