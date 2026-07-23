"""Health-check and version endpoints for operational monitoring."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()


class HealthResponse(BaseModel):
    """Liveness probe response."""

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VersionResponse(BaseModel):
    """Version and environment metadata."""

    model_config = ConfigDict(frozen=True)

    api_version: str
    pipeline_version: str
    python_version: str
    platform: str
    educational_disclaimer: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns HTTP 200 when the application process is alive. Suitable for container health checks.",
)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Version metadata",
    description="Returns the API version, pipeline version, and runtime environment metadata.",
)
async def version(request: Request) -> VersionResponse:
    config = request.app.state.config
    return VersionResponse(
        api_version=config.api_version,
        pipeline_version=config.pipeline_version,
        python_version=sys.version,
        platform=platform.platform(),
        educational_disclaimer=config.educational_disclaimer,
    )
