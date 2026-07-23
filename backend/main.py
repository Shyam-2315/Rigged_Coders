"""TrustVest AI — FastAPI application entry point (Phase 9).

Run locally:
    uvicorn backend.main:app --reload --port 8000

Run in production:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.v1.router import api_v1_router
from backend.config import BackendConfig

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):  # type: ignore[type-arg]
    """Warm up shared resources on startup; release on shutdown."""
    config: BackendConfig = application.state.config  # type: ignore[attr-defined]
    LOGGER.info(
        "TrustVest AI API v%s starting up on pipeline_version=%s",
        config.api_version,
        config.pipeline_version,
    )
    # Pre-import ML modules so the first request is not penalised by cold-start
    try:
        from ml.orchestrator import TrustVestIntelligenceOrchestrator  # noqa: F401
        from ml.investments.retrieval import search_products  # noqa: F401

        LOGGER.info("ML modules pre-loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("ML module pre-load warning (non-fatal): %s", exc)
    yield
    LOGGER.info("TrustVest AI API shutting down.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: BackendConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    cfg = config or BackendConfig()

    application = FastAPI(
        title="TrustVest AI",
        summary="Transparent Credit Scoring & AI-Driven Micro-Investment Advisor",
        description=(
            "REST API exposing TrustVest AI's ML pipeline modules for credit scoring, "
            "behavioral risk profiling, investment product retrieval, portfolio recommendation, "
            "and Monte Carlo financial simulation. "
            "All outputs are educational illustrations only and do not constitute regulated financial advice."
        ),
        version=cfg.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Store config on application state so routes can access it via request.app.state
    application.state.config = cfg  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Request timing middleware
    # ------------------------------------------------------------------
    @application.middleware("http")
    async def add_process_time_header(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response

    # ------------------------------------------------------------------
    # Global exception handlers
    # ------------------------------------------------------------------
    @application.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        LOGGER.warning("Validation error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "type": "validation_error"},
        )

    @application.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal error occurred. Please try again later.",
                "type": "internal_error",
            },
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    application.include_router(api_v1_router, prefix="/api/v1")

    return application


# ---------------------------------------------------------------------------
# Module-level singleton (used by uvicorn)
# ---------------------------------------------------------------------------

app = create_app()
