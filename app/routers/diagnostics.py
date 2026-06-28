"""
Diagnostics endpoints (the AI layer).

  POST /diagnostics/analyze            -> score one OBDSnapshot, return a Diagnosis
  POST /diagnostics/baseline/{vid}     -> register a car's own baseline from samples
                                          (makes detection specific to that vehicle)
  GET  /diagnostics/status             -> model metadata + validation results

The service only RETURNS the diagnosis. It does not call back into the
CarErescue API — the main app decides what to do with the result.
"""
from typing import List

from fastapi import APIRouter, HTTPException, Path

from app.data import loader
from app.ml import baseline as baseline_mod
from app.ml.detector import detector
from app.ml.features import featurize_dataframe
from app.schemas import (
    BaselineRegisterResponse,
    Diagnosis,
    OBDSnapshot,
    WindowAnalyzeRequest,
)
import pandas as pd

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.get("/status", summary="Model status & validation")
async def status():
    detector.ensure_loaded()
    return {
        "ready": detector.ready,
        "meta": detector.meta,
        "registered_vehicle_baselines": list(detector._vehicle_baselines.keys()),
    }


@router.post("/analyze", response_model=Diagnosis, summary="Analyze one OBD snapshot")
async def analyze(snapshot: OBDSnapshot):
    """Quick check on a single reading. Note: instantaneous readings are
    transient-sensitive — prefer /analyze-window for a stable diagnosis."""
    try:
        return detector.analyze(snapshot)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analysis failed: {e}")


@router.post(
    "/analyze-window",
    response_model=Diagnosis,
    summary="Analyze a window of recent readings (recommended)",
)
async def analyze_window(payload: WindowAnalyzeRequest):
    """Median-aggregate a window of recent snapshots, then diagnose. Removes
    transients so only sustained abnormal conditions are flagged."""
    if not payload.snapshots:
        raise HTTPException(status_code=400, detail="Provide at least one snapshot.")
    try:
        return detector.analyze_window(payload.vehicle_id, payload.snapshots)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analysis failed: {e}")


@router.post(
    "/baseline/{vehicle_id}",
    response_model=BaselineRegisterResponse,
    summary="Register a vehicle's own baseline from healthy snapshots",
)
async def register_baseline(
    snapshots: List[OBDSnapshot],
    vehicle_id: str = Path(..., description="Car to baseline."),
):
    if not snapshots:
        raise HTTPException(status_code=400, detail="Provide at least a few snapshots.")
    rows = [s.sensors.model_dump() for s in snapshots]
    df = pd.DataFrame(rows)
    feats = featurize_dataframe(df)
    baseline = baseline_mod.compute_baseline(feats)
    if not baseline:
        raise HTTPException(status_code=400, detail="Could not compute a baseline from these snapshots.")
    detector.register_vehicle_baseline(vehicle_id, baseline)
    return BaselineRegisterResponse(
        vehicle_id=vehicle_id,
        rows_used=len(df),
        features_baselined=len(baseline),
        message="Baseline registered; future analyses for this vehicle use it.",
    )
