"""
Dataset endpoints (NON-AI).

These let you inspect the Toyota Etios OBD data the future model will train on:
list files, see the column schema, pull sample snapshots, and view per-column
statistics. No model / inference here yet.
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.data import loader
from app.schemas import ColumnInfo, DatasetInfo, OBDSnapshot

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/info", response_model=DatasetInfo, summary="Dataset overview")
async def dataset_info():
    files = loader.list_files()
    if not files:
        raise HTTPException(status_code=404, detail="No dataset files found")
    return DatasetInfo(
        vehicle="Toyota Etios 2014",
        source="eron93br/carOBD (Carloop OBD-II logger)",
        description="1 Hz OBD-II recordings: drive*/idle*/live* captures.",
        sampling_hz=1,
        files=files,
        total_rows=loader.total_rows(),
        columns=len(loader.columns_schema()),
    )


@router.get("/files", summary="List dataset files")
async def dataset_files():
    return {"files": loader.list_files()}


@router.get("/columns", response_model=List[ColumnInfo], summary="Column / PID schema")
async def dataset_columns():
    return loader.columns_schema()


@router.get("/sample", response_model=List[OBDSnapshot], summary="Sample snapshots")
async def dataset_sample(
    file: Optional[str] = Query(None, description="CSV file name; defaults to first."),
    n: int = Query(5, ge=1, le=100, description="How many rows to return."),
):
    try:
        return loader.sample_snapshots(file, n)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/stats", summary="Per-column statistics")
async def dataset_stats(
    file: Optional[str] = Query(None, description="CSV file name; defaults to first."),
):
    try:
        return {"file": file or (loader.list_files() or [None])[0], "stats": loader.column_stats(file)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
