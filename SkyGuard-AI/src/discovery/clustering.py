"""Similarity-based clustering for recurring unknown signatures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .signature import SIGNATURE_COLUMNS


def cluster_signatures(
    signatures: pd.DataFrame,
    eps: float = 1.25,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Assign cluster IDs; ``-1`` means isolated noise, not a new fault."""
    out = signatures.copy()
    if out.empty:
        out["unknown_cluster_id"] = pd.Series(dtype=int)
        return out
    matrix = StandardScaler().fit_transform(out[SIGNATURE_COLUMNS].fillna(0.0))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
    out["unknown_cluster_id"] = labels.astype(int)
    return out