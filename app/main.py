"""
CarErescue OBD Diagnostics — standalone FastAPI microservice.

Separate from the main CarErescue backend: own config, own env, own port.
Deployed to Railway via the bundled Dockerfile.

NOTE: AI / anomaly-detection endpoints are intentionally NOT included yet —
they will be added after the approach is confirmed. This service currently
exposes only health + dataset-inspection endpoints.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from app.config import settings
from app.routers import dataset, diagnostics, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load (or lazily train) the anomaly-detection model at startup so the first
    # request is fast and failures surface on boot, not mid-request.
    try:
        from app.ml.detector import detector
        detector.ensure_loaded()
    except Exception as e:  # pragma: no cover - don't block boot on ML issues
        print(f"[startup] model load deferred: {e}")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="OBD-II data + anomaly-detection diagnostics for CarErescue.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(diagnostics.router)
