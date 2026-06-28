"""
Pydantic schemas for the OBD diagnostics service.

The `OBDSnapshot` is the core data contract: one timestamped reading of the
car's sensors, matching what a real ELM327 / Carloop dongle produces ~1x/sec.
Every sensor field is Optional, because cheap dongles and older cars do not
expose every PID. The diagnostics pipeline (added later) consumes this object.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class OBDHealth(BaseModel):
    """Self-diagnostic signals used to decide whether a car is safe to baseline."""

    mil_on: Optional[bool] = Field(
        None, description="Malfunction Indicator Lamp (check-engine light) is on."
    )
    dtc_count: Optional[int] = Field(
        None, description="Number of stored Diagnostic Trouble Codes."
    )
    dtc_codes: List[str] = Field(
        default_factory=list, description="Stored DTC codes, e.g. ['P0171']."
    )
    time_run_with_mil_on: Optional[float] = Field(
        None, description="Minutes the engine has run with the MIL on."
    )
    distance_with_mil_on: Optional[float] = Field(
        None, description="Distance (km) traveled with the MIL on."
    )


class OBDSensors(BaseModel):
    """Live OBD-II PID readings (standardised SAE J1979 Mode 01 signals)."""

    engine_rpm: Optional[float] = Field(None, description="Engine speed (rpm).")
    vehicle_speed: Optional[float] = Field(None, description="Vehicle speed (km/h).")
    engine_load: Optional[float] = Field(None, description="Calculated engine load (%).")
    coolant_temperature: Optional[float] = Field(None, description="Coolant temp (C).")
    short_term_fuel_trim: Optional[float] = Field(None, description="STFT bank 1 (%).")
    long_term_fuel_trim: Optional[float] = Field(None, description="LTFT bank 1 (%).")
    intake_manifold_pressure: Optional[float] = Field(None, description="MAP (kPa).")
    intake_air_temp: Optional[float] = Field(None, description="Intake air temp (C).")
    throttle: Optional[float] = Field(None, description="Throttle position (%).")
    relative_throttle_position: Optional[float] = Field(None, description="Relative throttle (%).")
    absolute_throttle_b: Optional[float] = Field(None, description="Absolute throttle B (%).")
    commanded_throttle_actuator: Optional[float] = Field(None, description="Commanded throttle (%).")
    accelerator_pedal_d: Optional[float] = Field(None, description="Accelerator pedal D (%).")
    accelerator_pedal_e: Optional[float] = Field(None, description="Accelerator pedal E (%).")
    timing_advance: Optional[float] = Field(None, description="Ignition timing advance (deg).")
    fuel_air_equiv_ratio: Optional[float] = Field(None, description="Commanded air-fuel equiv ratio.")
    barometric_pressure: Optional[float] = Field(None, description="Absolute barometric pressure (kPa).")
    catalyst_temp_s1: Optional[float] = Field(None, description="Catalyst temp bank1 sensor1 (C).")
    catalyst_temp_s2: Optional[float] = Field(None, description="Catalyst temp bank1 sensor2 (C).")
    control_module_voltage: Optional[float] = Field(None, description="Control module / battery voltage (V).")
    commanded_evap_purge: Optional[float] = Field(None, description="Commanded evaporative purge (%).")
    fuel_tank_level: Optional[float] = Field(None, description="Fuel tank level (%).")
    engine_run_time: Optional[float] = Field(None, description="Run time since engine start (s).")
    warm_ups_since_codes_cleared: Optional[float] = Field(None, description="Warm-ups since codes cleared.")
    time_since_codes_cleared: Optional[float] = Field(None, description="Minutes since codes cleared.")


class OBDSnapshot(BaseModel):
    """One full OBD reading at a single instant."""

    vehicle_id: str = Field(..., description="Which car this reading belongs to.")
    timestamp: Optional[datetime] = Field(None, description="When the reading was taken.")
    source: Optional[str] = Field(None, description="Origin, e.g. dataset file or 'live'.")
    health: OBDHealth = Field(default_factory=OBDHealth)
    sensors: OBDSensors = Field(default_factory=OBDSensors)


# ---- responses for the dataset endpoints (non-AI) ----

class ColumnInfo(BaseModel):
    raw_name: str
    field: str
    unit: str
    pid: Optional[str] = None


class DatasetInfo(BaseModel):
    vehicle: str
    source: str
    description: str
    sampling_hz: int
    files: List[str]
    total_rows: int
    columns: int


# ---- diagnostics (anomaly detection) ----

class FeatureDeviation(BaseModel):
    feature: str
    z_score: float = Field(..., description="Robust z vs this vehicle's baseline.")
    value: Optional[float] = None
    system: str


class HealthGate(BaseModel):
    mil_on: bool
    dtc_codes: List[str] = Field(default_factory=list)
    passed: bool


class Diagnosis(BaseModel):
    vehicle_id: str
    is_anomaly: bool
    anomaly_score: float = Field(..., description="0 = normal, 1 = highly anomalous.")
    threshold: float
    likely_system: Optional[str] = Field(None, description="Likely affected system (a SkillName).")
    confidence: str = Field(..., description="low | medium | high")
    baseline_used: str = Field(..., description="'population' or a vehicle id.")
    health_gate: HealthGate
    top_deviations: List[FeatureDeviation] = Field(default_factory=list)
    message: str
    window_size: Optional[int] = None


class WindowAnalyzeRequest(BaseModel):
    vehicle_id: str
    snapshots: List[OBDSnapshot] = Field(..., description="Recent readings for one car.")


class BaselineRegisterResponse(BaseModel):
    vehicle_id: str
    rows_used: int
    features_baselined: int
    message: str
