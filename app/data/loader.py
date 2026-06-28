"""
Loader for the Toyota Etios 2014 OBD-II dataset (eron93br/carOBD).

The raw CSVs have 27 columns with quirky names (a trailing ` ()` on every
header and a typo `ENGINE_RUN_TINE`). This module maps those raw headers to
clean snapshot field names, exposes the dataset to the API, and converts a
raw row into an `OBDSnapshot`.

Files come in three kinds: drive*.csv (driving), idle*.csv (engine idling),
live*.csv (mixed live capture). All are 1 Hz.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.config import settings
from app.schemas import (
    ColumnInfo,
    OBDHealth,
    OBDSensors,
    OBDSnapshot,
)

# raw CSV header  ->  (clean field, unit, OBD PID)
COLUMN_MAP: Dict[str, ColumnInfo] = {
    "ENGINE_RUN_TINE": ColumnInfo(raw_name="ENGINE_RUN_TINE", field="engine_run_time", unit="s", pid="0x1F"),
    "ENGINE_RPM": ColumnInfo(raw_name="ENGINE_RPM", field="engine_rpm", unit="rpm", pid="0x0C"),
    "VEHICLE_SPEED": ColumnInfo(raw_name="VEHICLE_SPEED", field="vehicle_speed", unit="km/h", pid="0x0D"),
    "THROTTLE": ColumnInfo(raw_name="THROTTLE", field="throttle", unit="%", pid="0x11"),
    "ENGINE_LOAD": ColumnInfo(raw_name="ENGINE_LOAD", field="engine_load", unit="%", pid="0x04"),
    "COOLANT_TEMPERATURE": ColumnInfo(raw_name="COOLANT_TEMPERATURE", field="coolant_temperature", unit="C", pid="0x05"),
    "LONG_TERM_FUEL_TRIM_BANK_1": ColumnInfo(raw_name="LONG_TERM_FUEL_TRIM_BANK_1", field="long_term_fuel_trim", unit="%", pid="0x07"),
    "SHORT_TERM_FUEL_TRIM_BANK_1": ColumnInfo(raw_name="SHORT_TERM_FUEL_TRIM_BANK_1", field="short_term_fuel_trim", unit="%", pid="0x06"),
    "INTAKE_MANIFOLD_PRESSURE": ColumnInfo(raw_name="INTAKE_MANIFOLD_PRESSURE", field="intake_manifold_pressure", unit="kPa", pid="0x0B"),
    "FUEL_TANK": ColumnInfo(raw_name="FUEL_TANK", field="fuel_tank_level", unit="%", pid="0x2F"),
    "ABSOLUTE_THROTTLE_B": ColumnInfo(raw_name="ABSOLUTE_THROTTLE_B", field="absolute_throttle_b", unit="%", pid="0x47"),
    "PEDAL_D": ColumnInfo(raw_name="PEDAL_D", field="accelerator_pedal_d", unit="%", pid="0x49"),
    "PEDAL_E": ColumnInfo(raw_name="PEDAL_E", field="accelerator_pedal_e", unit="%", pid="0x4A"),
    "COMMANDED_THROTTLE_ACTUATOR": ColumnInfo(raw_name="COMMANDED_THROTTLE_ACTUATOR", field="commanded_throttle_actuator", unit="%", pid="0x4C"),
    "FUEL_AIR_COMMANDED_EQUIV_RATIO": ColumnInfo(raw_name="FUEL_AIR_COMMANDED_EQUIV_RATIO", field="fuel_air_equiv_ratio", unit="ratio", pid="0x44"),
    "ABSOLUTE_BAROMETRIC_PRESSURE": ColumnInfo(raw_name="ABSOLUTE_BAROMETRIC_PRESSURE", field="barometric_pressure", unit="kPa", pid="0x33"),
    "RELATIVE_THROTTLE_POSITION": ColumnInfo(raw_name="RELATIVE_THROTTLE_POSITION", field="relative_throttle_position", unit="%", pid="0x45"),
    "INTAKE_AIR_TEMP": ColumnInfo(raw_name="INTAKE_AIR_TEMP", field="intake_air_temp", unit="C", pid="0x0F"),
    "TIMING_ADVANCE": ColumnInfo(raw_name="TIMING_ADVANCE", field="timing_advance", unit="deg", pid="0x0E"),
    "CATALYST_TEMPERATURE_BANK1_SENSOR1": ColumnInfo(raw_name="CATALYST_TEMPERATURE_BANK1_SENSOR1", field="catalyst_temp_s1", unit="C", pid="0x3C"),
    "CATALYST_TEMPERATURE_BANK1_SENSOR2": ColumnInfo(raw_name="CATALYST_TEMPERATURE_BANK1_SENSOR2", field="catalyst_temp_s2", unit="C", pid="0x3E"),
    "CONTROL_MODULE_VOLTAGE": ColumnInfo(raw_name="CONTROL_MODULE_VOLTAGE", field="control_module_voltage", unit="V", pid="0x42"),
    "COMMANDED_EVAPORATIVE_PURGE": ColumnInfo(raw_name="COMMANDED_EVAPORATIVE_PURGE", field="commanded_evap_purge", unit="%", pid="0x2E"),
    "TIME_RUN_WITH_MIL_ON": ColumnInfo(raw_name="TIME_RUN_WITH_MIL_ON", field="time_run_with_mil_on", unit="min", pid="0x4D"),
    "TIME_SINCE_TROUBLE_CODES_CLEARED": ColumnInfo(raw_name="TIME_SINCE_TROUBLE_CODES_CLEARED", field="time_since_codes_cleared", unit="min", pid="0x4E"),
    "DISTANCE_TRAVELED_WITH_MIL_ON": ColumnInfo(raw_name="DISTANCE_TRAVELED_WITH_MIL_ON", field="distance_with_mil_on", unit="km", pid="0x21"),
    "WARM_UPS_SINCE_CODES_CLEARED": ColumnInfo(raw_name="WARM_UPS_SINCE_CODES_CLEARED", field="warm_ups_since_codes_cleared", unit="count", pid="0x30"),
}

# fields that belong to OBDSensors (everything except the MIL/health columns)
_HEALTH_FIELDS = {
    "time_run_with_mil_on",
    "distance_with_mil_on",
    "time_since_codes_cleared",
}


def _normalize_header(name: str) -> str:
    """Strip the trailing ' ()' and surrounding whitespace from a raw header."""
    return name.replace("()", "").strip()


def list_files() -> List[str]:
    path = settings.dataset_path
    if not path.exists():
        return []
    return sorted(p.name for p in path.glob("*.csv"))


def _resolve_file(filename: Optional[str]) -> Path:
    files = list_files()
    if not files:
        raise FileNotFoundError(f"No CSV files found in {settings.dataset_path}")
    target = filename or files[0]
    full = settings.dataset_path / target
    if full.name != target or not full.exists():
        raise FileNotFoundError(f"Dataset file '{target}' not found")
    return full


@lru_cache(maxsize=16)
def load_dataframe(filename: Optional[str] = None) -> pd.DataFrame:
    """Load one CSV, rename columns to clean field names, coerce to numeric."""
    full = _resolve_file(filename)
    df = pd.read_csv(full, skipinitialspace=True)
    rename = {}
    for col in df.columns:
        key = _normalize_header(col)
        if key in COLUMN_MAP:
            rename[col] = COLUMN_MAP[key].field
    df = df.rename(columns=rename)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def columns_schema() -> List[ColumnInfo]:
    return list(COLUMN_MAP.values())


def total_rows() -> int:
    total = 0
    for f in list_files():
        try:
            total += len(load_dataframe(f))
        except Exception:
            continue
    return total


def row_to_snapshot(row: pd.Series, vehicle_id: str, source: str) -> OBDSnapshot:
    """Convert one DataFrame row into an OBDSnapshot."""
    sensor_kwargs = {}
    health_kwargs = {}
    for info in COLUMN_MAP.values():
        field = info.field
        if field not in row.index:
            continue
        value = row[field]
        if pd.isna(value):
            value = None
        else:
            value = float(value)
        if field in _HEALTH_FIELDS:
            health_kwargs[field] = value
        else:
            sensor_kwargs[field] = value

    # The MIL columns in this dataset are sentinels (constant / 0xFF), not a
    # real malfunction signal, so we report mil_on as None (unknown) rather than
    # falsely flagging every row. Real dongles set this field directly.
    health = OBDHealth(
        mil_on=None,
        dtc_count=None,
        dtc_codes=[],
        time_run_with_mil_on=health_kwargs.get("time_run_with_mil_on"),
        distance_with_mil_on=health_kwargs.get("distance_with_mil_on"),
    )

    return OBDSnapshot(
        vehicle_id=vehicle_id,
        timestamp=datetime.now(timezone.utc),
        source=source,
        health=health,
        sensors=OBDSensors(**sensor_kwargs),
    )


def sample_snapshots(filename: Optional[str], n: int) -> List[OBDSnapshot]:
    df = load_dataframe(filename)
    source = filename or list_files()[0]
    rows = df.head(max(0, n))
    return [
        row_to_snapshot(row, settings.default_vehicle_id, source)
        for _, row in rows.iterrows()
    ]


def column_stats(filename: Optional[str]) -> Dict[str, Dict[str, float]]:
    df = load_dataframe(filename)
    stats: Dict[str, Dict[str, float]] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
        }
    return stats
