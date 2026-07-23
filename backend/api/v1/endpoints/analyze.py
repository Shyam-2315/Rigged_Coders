"""/api/v1/analyze — unified orchestrator endpoint (Phase 8 → Phase 9 API)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ml.orchestrator import TrustVestIntelligenceOrchestrator
from ml.orchestrator.config import OrchestratorConfig
from ml.orchestrator.schemas import UnifiedAnalysisRequest, UnifiedAnalysisResponse

router = APIRouter()

# One shared orchestrator instance (with cache) reused across requests.
_orchestrator: TrustVestIntelligenceOrchestrator | None = None


def _get_orchestrator(request: Request) -> TrustVestIntelligenceOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        cfg = request.app.state.config
        _orchestrator = TrustVestIntelligenceOrchestrator(
            config=OrchestratorConfig(
                cache_enabled=cfg.orchestrator_cache_enabled,
                cache_max_entries=cfg.orchestrator_cache_max_entries,
            )
        )
    return _orchestrator


@router.post(
    "/analyze",
    response_model=UnifiedAnalysisResponse,
    summary="Unified analysis (all phases)",
    description=(
        "Run the full TrustVest AI pipeline: credit scoring (Phase 3), behavioral risk profiling (Phase 4), "
        "knowledge-base retrieval (Phase 5), portfolio recommendation (Phase 6), and Monte Carlo simulation (Phase 7). "
        "Results are educational illustrations only."
    ),
)
async def analyze(body: UnifiedAnalysisRequest, request: Request) -> UnifiedAnalysisResponse:
    orchestrator = _get_orchestrator(request)
    return await orchestrator.analyze_user_async(body)
