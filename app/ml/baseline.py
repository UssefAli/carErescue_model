"""
Per-vehicle baselining + robust normalization.

This is what makes the detector generic across cars: instead of learning
absolute sensor values (which differ per make/model), we normalize every
reading against a baseline. A baseline is the robust centre (median) and
spread (MAD) of each feature for a given car's healthy driving.

  robust_z = (value - median) / (1.4826 * MAD)

A z-score means the same thing on every vehicle, so the trained model
transfers without per-make retraining. If a car has no baseline yet, the
population baseline (built from the bundled Etios data) is used as a fallback.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.ml.features import FEATURE_ORDER

MAD_SCALE = 1.4826  # makes MAD a consistent estimator of std for normal data
EPS = 1e-9


def compute_baseline(features_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Robust per-feature stats from a frame of (already featurized) rows."""
    stats: Dict[str, Dict[str, float]] = {}
    for col in FEATURE_ORDER:
        if col not in features_df.columns:
            continue
        s = features_df[col].dropna()
        if s.empty:
            continue
        median = float(s.median())
        mad = float((s - median).abs().median())
        std = float(s.std()) if len(s) > 1 else 0.0
        # fall back to std when MAD collapses (e.g. discrete-ish features)
        scale = MAD_SCALE * mad if mad > EPS else (std if std > EPS else 0.0)
        stats[col] = {
            "median": round(median, 6),
            "mad": round(mad, 6),
            "scale": round(scale, 6),
            "p01": round(float(s.quantile(0.01)), 6),
            "p99": round(float(s.quantile(0.99)), 6),
            "n": int(len(s)),
        }
    return stats


def normalize_vector(
    feature_values: Dict[str, Optional[float]],
    baseline: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Feature dict -> ordered robust-z vector. Missing/zero-scale -> 0 (neutral)."""
    vec = np.zeros(len(FEATURE_ORDER), dtype=float)
    for i, col in enumerate(FEATURE_ORDER):
        stat = baseline.get(col)
        val = feature_values.get(col)
        if stat is None or val is None or np.isnan(val):
            vec[i] = 0.0
            continue
        scale = stat.get("scale", 0.0)
        if scale <= EPS:
            vec[i] = 0.0
            continue
        z = (float(val) - stat["median"]) / scale
        # clip extreme values so one wild sensor can't dominate everything
        vec[i] = float(np.clip(z, -30.0, 30.0))
    return vec


def normalize_dataframe(
    features_df: pd.DataFrame,
    baseline: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Featurized frame -> matrix of robust-z rows (for training)."""
    rows = np.zeros((len(features_df), len(FEATURE_ORDER)), dtype=float)
    medians = np.array([baseline.get(c, {}).get("median", 0.0) for c in FEATURE_ORDER])
    scales = np.array([baseline.get(c, {}).get("scale", 0.0) for c in FEATURE_ORDER])
    arr = features_df[FEATURE_ORDER].to_numpy(dtype=float)
    for j in range(len(FEATURE_ORDER)):
        scale = scales[j]
        col = arr[:, j]
        if scale <= EPS:
            rows[:, j] = 0.0
            continue
        z = (col - medians[j]) / scale
        z = np.where(np.isnan(z), 0.0, z)
        rows[:, j] = np.clip(z, -30.0, 30.0)
    return rows
