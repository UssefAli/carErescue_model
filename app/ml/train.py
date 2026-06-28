"""
Train + calibrate the anomaly detector from the bundled Toyota Etios data.

Steps:
  1. Load all CSVs, keep engine-running rows.
  2. Featurize (raw + relational) and build the robust population baseline.
  3. Train an IsolationForest (secondary multivariate signal).
  4. Calibrate on WINDOWED normal data (rolling median, like production):
       - z_full: z-score that maps to anomaly_score = 1.0 (fixed at 10).
       - threshold: chosen so the false-positive rate on normal windows is low.
  5. Validate by injecting SUSTAINED faults into health features of windowed
     normal data; report detection rate vs false-positive rate.
  6. Save model.pkl, baseline.json, meta.json.

IMPORTANT (honesty): this bundled dataset is a single car, unlabeled, and the
raw PIDs are mis-scaled (a known Carloop logging artifact). So these numbers are
ILLUSTRATIVE of the pipeline, not a production benchmark. On properly-decoded
real OBD data with a clean warmed-up per-vehicle baseline, separation is better.

Run:  python -m app.ml.train
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.data import loader
from app.ml import baseline as baseline_mod
from app.ml.detector import ARTIFACT_DIR, BASELINE_PATH, META_PATH, MODEL_PATH
from app.ml.features import FEATURE_ORDER, HEALTH_FEATURES, featurize_dataframe

RANDOM_STATE = 42
WINDOW = 15          # ~15 s rolling-median window (transient removal)
WARMUP = 60          # skip first ~60 s of each session (warmup drift)
Z_FULL = 10.0        # robust z that maps to anomaly_score = 1.0
TARGET_FP_PERCENTILE = 98.0
HEALTH_IDX = [FEATURE_ORDER.index(f) for f in HEALTH_FEATURES]


def _healthy_frames():
    frames = []
    for f in loader.list_files():
        try:
            d = loader.load_dataframe(f)
            d = d[d["engine_run_time"].fillna(0) > 0]
            frames.append(featurize_dataframe(d).reset_index(drop=True))
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No dataset files available to train on.")
    return frames


def _max_health_z(matrix: np.ndarray) -> np.ndarray:
    return np.max(np.abs(matrix[:, HEALTH_IDX]), axis=1)


def _per_session(frames):
    """
    Production-mode calibration: each session builds its OWN baseline from its
    warmed-up first half, and is tested on windowed readings from its second
    half. Returns (normal_units, inject_fn) where inject_fn(mag) -> detection.
    """
    normal_units = []
    fault_units = {6: [], 8: [], 10: []}
    rng = np.random.default_rng(RANDOM_STATE)

    for ft in frames:
        warm = ft.iloc[WARMUP:].reset_index(drop=True)
        if len(warm) < 4 * WINDOW:
            continue
        half = len(warm) // 2
        base_s = baseline_mod.compute_baseline(warm.iloc[:half])
        test = warm.iloc[half:].rolling(WINDOW, min_periods=WINDOW).median().dropna()
        if len(test) < 20:
            continue

        Xn = baseline_mod.normalize_dataframe(test, base_s)
        normal_units.append(np.clip(_max_health_z(Xn) / Z_FULL, 0.0, 1.0))

        idx = rng.choice(len(test), size=min(1000, len(test)), replace=False)
        for mag in fault_units:
            sample = test.iloc[idx].reset_index(drop=True).copy()
            for i in range(len(sample)):
                feat = HEALTH_FEATURES[i % len(HEALTH_FEATURES)]
                stat = base_s.get(feat)
                if stat and stat.get("scale", 0) > 0:
                    sample.loc[i, feat] = stat["median"] + mag * stat["scale"]
            Xf = baseline_mod.normalize_dataframe(sample, base_s)
            fault_units[mag].append(np.clip(_max_health_z(Xf) / Z_FULL, 0.0, 1.0))

    normal = np.concatenate(normal_units) if normal_units else np.array([])
    faults = {m: (np.concatenate(v) if v else np.array([])) for m, v in fault_units.items()}
    return normal, faults


def train() -> Dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    frames = _healthy_frames()
    all_feats = pd.concat(frames, ignore_index=True)

    # population baseline — coarse cold-start fallback when a car has no baseline
    baseline = baseline_mod.compute_baseline(all_feats)

    # IsolationForest (secondary multivariate signal)
    X = baseline_mod.normalize_dataframe(all_feats, baseline)
    model = IsolationForest(
        n_estimators=200, contamination=0.02, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X)
    raw_scores = model.score_samples(X)
    score_min = float(np.percentile(raw_scores, 1))
    score_max = float(np.percentile(raw_scores, 99))

    # --- calibrate + validate in PRIMARY (per-vehicle baseline) mode ---
    normal_unit, fault_units = _per_session(frames)
    threshold = float(np.clip(np.percentile(normal_unit, TARGET_FP_PERCENTILE), 0.45, 0.9)) if normal_unit.size else 0.6
    false_positive_rate = float((normal_unit >= threshold).mean()) if normal_unit.size else None
    detection = {
        f"mag_{m}": (round(float((u >= threshold).mean()), 4) if u.size else None)
        for m, u in fault_units.items()
    }

    # population-fallback FP, for transparency (expected to be higher)
    fb_parts = [
        baseline_mod.normalize_dataframe(
            ft.iloc[WARMUP:].rolling(WINDOW, min_periods=WINDOW).median().dropna(), baseline
        )
        for ft in frames
    ]
    fb = np.vstack([p for p in fb_parts if len(p)]) if fb_parts else np.empty((0, len(FEATURE_ORDER)))
    fallback_fp = float((np.clip(_max_health_z(fb) / Z_FULL, 0.0, 1.0) >= threshold).mean()) if len(fb) else None

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "vehicle": "Toyota Etios 2014",
        "n_rows": int(len(all_feats)),
        "window": WINDOW,
        "z_full": Z_FULL,
        "threshold": round(threshold, 4),
        "score_min": round(score_min, 6),
        "score_max": round(score_max, 6),
        "feature_order": FEATURE_ORDER,
        "health_features": HEALTH_FEATURES,
        "validation": {
            "note": (
                "Illustrative only: single unlabeled, mis-scaled car; faults are "
                "synthetic sustained shifts. Primary mode = per-vehicle baseline."
            ),
            "mode": "per_vehicle_baseline",
            "false_positive_rate": round(false_positive_rate, 4) if false_positive_rate is not None else None,
            "detection_rate": detection,
            "population_fallback_false_positive_rate": round(fallback_fp, 4) if fallback_fp is not None else None,
        },
    }

    joblib.dump(model, MODEL_PATH)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
