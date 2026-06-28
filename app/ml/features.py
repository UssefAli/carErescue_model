"""
Feature engineering for anomaly detection.

Two kinds of features:
  * RAW_FEATURES      — direct sensor readings.
  * RELATIONAL_FEATURES — physics-based relationships (ratios / differences)
                          that hold on ANY healthy engine, so they transfer
                          across vehicles regardless of make/model.

The full, ordered feature list (FEATURE_ORDER) is the contract shared between
training and inference — both must build the vector in exactly this order.

Each feature is mapped to a likely vehicle SYSTEM (aligned with the main
CarErescue `SkillName` request types) so a flagged deviation can be named.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EPS = 1e-6

# Direct sensor signals we trust as informative for this car.
RAW_FEATURES: List[str] = [
    "engine_rpm",
    "vehicle_speed",
    "engine_load",
    "coolant_temperature",
    "short_term_fuel_trim",
    "long_term_fuel_trim",
    "intake_manifold_pressure",
    "intake_air_temp",
    "throttle",
    "timing_advance",
    "control_module_voltage",
    "barometric_pressure",
    "fuel_air_equiv_ratio",
    "catalyst_temp_s1",
    "catalyst_temp_s2",
]

# Derived, car-agnostic relationships.
RELATIONAL_FEATURES: List[str] = [
    "total_fuel_trim",     # |STFT + LTFT| — how hard the ECU is compensating
    "load_vs_throttle",    # engine_load - throttle — intake/efficiency mismatch
    "map_vs_throttle",     # MAP relative to throttle — intake leak / sensor
    "rpm_per_speed",       # engine_rpm / speed — drivetrain / gearing
    "coolant_vs_load",     # coolant relative to load — cooling problem
    "catalyst_delta",      # cat S1 - S2 — exhaust / catalyst health
]

FEATURE_ORDER: List[str] = RAW_FEATURES + RELATIONAL_FEATURES

# Driver / operating-point signals. A high value here is an INPUT, not a fault
# (flooring the throttle is not a malfunction), so these are used only as
# context — never as direct evidence of an anomaly.
CONTEXT_FEATURES: List[str] = [
    "engine_rpm",
    "vehicle_speed",
    "engine_load",
    "throttle",
    "timing_advance",
    "intake_manifold_pressure",
    "load_vs_throttle",
    "map_vs_throttle",
    "rpm_per_speed",
]

# Health-indicator signals: these should stay stable for a healthy engine, so a
# sustained deviation here IS meaningful. The anomaly score is computed over
# these only.
HEALTH_FEATURES: List[str] = [
    "coolant_temperature",
    "short_term_fuel_trim",
    "long_term_fuel_trim",
    "total_fuel_trim",
    "control_module_voltage",
    "fuel_air_equiv_ratio",
    "intake_air_temp",
    "barometric_pressure",
    "catalyst_temp_s1",
    "catalyst_temp_s2",
    "catalyst_delta",
    "coolant_vs_load",
]

# feature -> likely affected system (mirrors CarErescue SkillName values)
FEATURE_TO_SYSTEM: Dict[str, str] = {
    "engine_rpm": "Core Engine Repair",
    "vehicle_speed": "Drivetrain",
    "engine_load": "Core Engine Repair",
    "coolant_temperature": "Cooling System",
    "coolant_vs_load": "Cooling System",
    "short_term_fuel_trim": "Fuel System",
    "long_term_fuel_trim": "Fuel System",
    "total_fuel_trim": "Fuel System",
    "fuel_air_equiv_ratio": "Fuel System",
    "intake_manifold_pressure": "Core Engine Repair",
    "map_vs_throttle": "Core Engine Repair",
    "intake_air_temp": "Core Engine Repair",
    "barometric_pressure": "Computer & Sensors",
    "throttle": "Computer & Sensors",
    "load_vs_throttle": "Computer & Sensors",
    "timing_advance": "Core Engine Repair",
    "control_module_voltage": "Battery & Charging",
    "catalyst_temp_s1": "Exhaust System",
    "catalyst_temp_s2": "Exhaust System",
    "catalyst_delta": "Exhaust System",
    "rpm_per_speed": "Drivetrain",
}


def _g(d: Dict[str, Optional[float]], key: str) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if (isinstance(v, float) and np.isnan(v)) else v


def relational_from_sensors(s: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Compute the derived relational features from a sensor dict."""
    stft = _g(s, "short_term_fuel_trim")
    ltft = _g(s, "long_term_fuel_trim")
    load = _g(s, "engine_load")
    throttle = _g(s, "throttle")
    mapp = _g(s, "intake_manifold_pressure")
    rpm = _g(s, "engine_rpm")
    speed = _g(s, "vehicle_speed")
    coolant = _g(s, "coolant_temperature")
    cat1 = _g(s, "catalyst_temp_s1")
    cat2 = _g(s, "catalyst_temp_s2")

    out: Dict[str, Optional[float]] = {}
    out["total_fuel_trim"] = (abs(stft) + abs(ltft)) if (stft is not None and ltft is not None) else None
    out["load_vs_throttle"] = (load - throttle) if (load is not None and throttle is not None) else None
    out["map_vs_throttle"] = (mapp - throttle) if (mapp is not None and throttle is not None) else None
    out["rpm_per_speed"] = (rpm / (speed + EPS)) if (rpm is not None and speed is not None and speed > 0.5) else None
    out["coolant_vs_load"] = (coolant - load) if (coolant is not None and load is not None) else None
    out["catalyst_delta"] = (cat1 - cat2) if (cat1 is not None and cat2 is not None) else None
    return out


def featurize_sensors(sensors: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """One sensor dict -> full feature dict (raw + relational) in any order."""
    feats: Dict[str, Optional[float]] = {f: _g(sensors, f) for f in RAW_FEATURES}
    feats.update(relational_from_sensors(sensors))
    return feats


def featurize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add relational columns to a dataframe of raw sensor readings."""
    out = pd.DataFrame(index=df.index)
    for f in RAW_FEATURES:
        out[f] = df[f] if f in df.columns else np.nan

    stft, ltft = out["short_term_fuel_trim"], out["long_term_fuel_trim"]
    load, throttle = out["engine_load"], out["throttle"]
    mapp = out["intake_manifold_pressure"]
    rpm, speed = out["engine_rpm"], out["vehicle_speed"]
    coolant = out["coolant_temperature"]
    cat1, cat2 = out["catalyst_temp_s1"], out["catalyst_temp_s2"]

    out["total_fuel_trim"] = stft.abs() + ltft.abs()
    out["load_vs_throttle"] = load - throttle
    out["map_vs_throttle"] = mapp - throttle
    out["rpm_per_speed"] = np.where(speed > 0.5, rpm / (speed + EPS), np.nan)
    out["coolant_vs_load"] = coolant - load
    out["catalyst_delta"] = cat1 - cat2
    return out[FEATURE_ORDER]
