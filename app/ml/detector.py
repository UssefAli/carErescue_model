"""
Anomaly detector.

Scoring is a robust, interpretable statistic — the largest robust z-score
across the HEALTH features (coolant, fuel trims, voltage, catalyst, AFR) after
normalizing against the vehicle's baseline. Driver/operating-point signals are
context only, never direct anomaly evidence. An IsolationForest is loaded as a
secondary multivariate vote.

Single instantaneous readings are noisy (transients), so the recommended path
is `analyze_window`, which median-aggregates a short window first.

Artifacts (from train.py, in app/ml/artifacts/):
  model.pkl · baseline.json (population) · meta.json (threshold, z_full, info)
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

import joblib
import numpy as np

from app.ml import baseline as baseline_mod
from app.ml.features import (
    FEATURE_ORDER,
    FEATURE_TO_SYSTEM,
    HEALTH_FEATURES,
    featurize_sensors,
)
from app.schemas import OBDHealth, OBDSensors, OBDSnapshot

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
BASELINE_PATH = ARTIFACT_DIR / "baseline.json"
META_PATH = ARTIFACT_DIR / "meta.json"

_HEALTH_IDX = [FEATURE_ORDER.index(f) for f in HEALTH_FEATURES]


class Detector:
    def __init__(self) -> None:
        self.model = None
        self.population_baseline: Dict[str, Dict[str, float]] = {}
        self.meta: Dict = {}
        self._vehicle_baselines: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ---- lifecycle ----
    @property
    def ready(self) -> bool:
        return bool(self.population_baseline)

    def load(self) -> bool:
        if not (BASELINE_PATH.exists() and META_PATH.exists()):
            return False
        self.population_baseline = json.loads(BASELINE_PATH.read_text())
        self.meta = json.loads(META_PATH.read_text())
        self.model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
        return True

    def ensure_loaded(self) -> None:
        if self.ready:
            return
        if not self.load():
            from app.ml.train import train  # lazy import to avoid cycle
            train()
            self.load()

    # ---- per-vehicle baselines ----
    def register_vehicle_baseline(self, vehicle_id: str, baseline: Dict) -> None:
        self._vehicle_baselines[vehicle_id] = baseline

    def _baseline_for(self, vehicle_id: Optional[str]):
        if vehicle_id and vehicle_id in self._vehicle_baselines:
            return self._vehicle_baselines[vehicle_id], vehicle_id
        return self.population_baseline, "population"

    # ---- scoring core ----
    def _score_vector(self, vec: np.ndarray):
        """Return (anomaly_score 0..1, max health z, per-feature z list)."""
        z_full = float(self.meta.get("z_full", 10.0))
        health_abs = [abs(vec[i]) for i in _HEALTH_IDX]
        max_health_z = max(health_abs) if health_abs else 0.0
        z_unit = float(np.clip(max_health_z / z_full, 0.0, 1.0))

        if_unit = 0.0
        if self.model is not None:
            raw = float(self.model.score_samples(vec.reshape(1, -1))[0])
            lo = self.meta.get("score_min", -0.6)
            hi = self.meta.get("score_max", -0.4)
            if hi - lo > 1e-9:
                if_unit = float(np.clip((hi - raw) / (hi - lo), 0.0, 1.0))

        # robust statistic leads; IF nudges the score up a little when it agrees
        anomaly = max(z_unit, 0.6 * if_unit)
        return anomaly, max_health_z, if_unit

    def _diagnose(self, sensors: Dict, health: OBDHealth, vehicle_id: str, source: str) -> Dict:
        baseline, baseline_used = self._baseline_for(vehicle_id)
        feats = featurize_sensors(sensors)
        vec = baseline_mod.normalize_vector(feats, baseline)

        anomaly_score, max_health_z, if_unit = self._score_vector(vec)
        threshold = float(self.meta.get("threshold", 0.6))

        mil_on = bool(getattr(health, "mil_on", False))
        dtc_codes = list(getattr(health, "dtc_codes", []) or [])
        model_flag = anomaly_score >= threshold
        is_anomaly = bool(model_flag or mil_on or dtc_codes)

        # deviations among HEALTH features only (what actually drives the score)
        deviations = []
        for i in _HEALTH_IDX:
            col = FEATURE_ORDER[i]
            z = vec[i]
            if abs(z) < 1e-6:
                continue
            deviations.append({
                "feature": col,
                "z_score": round(float(z), 3),
                "value": feats.get(col),
                "system": FEATURE_TO_SYSTEM.get(col, "Computer & Sensors"),
            })
        deviations.sort(key=lambda d: abs(d["z_score"]), reverse=True)
        top = deviations[:5]

        likely_system = None
        if model_flag and top:
            votes: Dict[str, float] = {}
            for d in top:
                votes[d["system"]] = votes.get(d["system"], 0.0) + abs(d["z_score"])
            likely_system = max(votes, key=votes.get)
        elif dtc_codes or mil_on:
            likely_system = "Computer & Sensors"

        confidence = "low"
        if is_anomaly:
            if mil_on or anomaly_score >= threshold + 0.25:
                confidence = "high"
            elif anomaly_score >= threshold + 0.1:
                confidence = "medium"

        if mil_on or dtc_codes:
            message = "Vehicle self-diagnostics report a fault (MIL/DTC)."
        elif model_flag:
            message = f"Sustained abnormal readings, most consistent with the {likely_system}."
        else:
            message = "No anomaly detected; readings are within this vehicle's normal range."

        return {
            "vehicle_id": vehicle_id,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "threshold": round(threshold, 4),
            "likely_system": likely_system,
            "confidence": confidence,
            "baseline_used": baseline_used,
            "health_gate": {
                "mil_on": mil_on,
                "dtc_codes": dtc_codes,
                "passed": not (mil_on or bool(dtc_codes)),
            },
            "top_deviations": top,
            "message": message,
        }

    # ---- public API ----
    def analyze(self, snapshot: OBDSnapshot) -> Dict:
        self.ensure_loaded()
        sensors = snapshot.sensors.model_dump()
        return self._diagnose(sensors, snapshot.health, snapshot.vehicle_id, snapshot.source or "live")

    def analyze_window(self, vehicle_id: str, snapshots: List[OBDSnapshot]) -> Dict:
        """Median-aggregate a window of readings, then diagnose (transient-robust)."""
        self.ensure_loaded()
        if not snapshots:
            raise ValueError("window is empty")

        # robust median per sensor across the window
        agg: Dict[str, Optional[float]] = {}
        keys = OBDSensors().model_dump().keys()
        for k in keys:
            vals = []
            for s in snapshots:
                v = getattr(s.sensors, k, None)
                if v is not None:
                    vals.append(float(v))
            agg[k] = median(vals) if vals else None

        # health gate: MIL/DTC if ANY reading in the window reports one
        mil = any(bool(getattr(s.health, "mil_on", False)) for s in snapshots)
        dtcs = sorted({c for s in snapshots for c in (getattr(s.health, "dtc_codes", []) or [])})
        health = OBDHealth(mil_on=mil, dtc_codes=dtcs)

        result = self._diagnose(agg, health, vehicle_id, "window")
        result["window_size"] = len(snapshots)
        return result


detector = Detector()
