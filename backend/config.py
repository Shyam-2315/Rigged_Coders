"""Backend application configuration for Phase 9."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

API_VERSION = "0.9.0"
PIPELINE_VERSION = "v0.9.0"


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Runtime configuration for the TrustVest AI FastAPI backend."""

    api_version: str = API_VERSION
    pipeline_version: str = PIPELINE_VERSION

    # CORS — default allows localhost dev origins; override via env in production
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8000",  # Self (Swagger UI)
        "*",                      # Permissive for hackathon demo; tighten for production
    )

    # Orchestrator cache — share one in-process instance across requests
    orchestrator_cache_enabled: bool = True
    orchestrator_cache_max_entries: int = 256

    # Simulation defaults exposed at the API layer
    default_num_simulations: int = 1_000
    max_num_simulations: int = 10_000

    # Products endpoint
    max_products_per_response: int = 25

    educational_disclaimer: str = (
        "TrustVest AI provides educational illustrations only. Outputs from credit scoring, "
        "risk profiling, portfolio recommendations, and financial simulations are based on "
        "statistical models and synthetic or assumed data. They do not constitute regulated "
        "financial advice, lending decisions, or predictions of future market performance. "
        "Consult qualified professionals before making financial decisions."
    )
