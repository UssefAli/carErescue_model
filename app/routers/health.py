from fastapi import APIRouter

from app.config import settings
from app.data import loader

router = APIRouter(tags=["health"])


@router.get("/", summary="Service banner")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "docs": "/docs",
    }


@router.get("/health", summary="Health check (used by Railway)")
async def health():
    files = loader.list_files()
    return {
        "status": "ok",
        "environment": settings.environment,
        "dataset_loaded": len(files) > 0,
        "dataset_files": len(files),
    }
