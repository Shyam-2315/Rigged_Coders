"""/api/v1/credit-score — Phase 3 credit scoring endpoint.

Accepts a unified request payload (same contract as /analyze) and returns
only the credit analysis portion.  The orchestrator pipeline handles
preprocessing internally, so callers do not need to supply Phase 2 features.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from ml.orchestrator.config import OrchestratorConfig
from ml.orchestrator.pipeline import AnalysisPipeline
from ml.orchestrator.schemas import (
    CreditAnalysisResult,
    UnifiedAnalysisRequest,
)
from ml.orchestrator.telemetry import TelemetryCollector
from ml.orchestrator.validators import validate_analysis_request

router = APIRouter()


class CreditScoreResponse(BaseModel):
    """Credit scoring endpoint response."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    credit_score: float = Field(description="Estimated credit likelihood score (0–100)")
    confidence: float = Field(ge=0, le=1, description="Model confidence proxy (0–1)")
    top_positive_features: list[dict] = Field(description="SHAP-derived positive feature contributions")
    top_negative_features: list[dict] = Field(description="SHAP-derived negative feature contributions")
    source: str = Field(description="'model' when the ML model ran, 'fallback' when a supplied score was used")
    educational_disclaimer: str


_pipeline: AnalysisPipeline | None = None


def _get_pipeline() -> AnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AnalysisPipeline(config=OrchestratorConfig())
    return _pipeline


@router.post(
    "/credit-score",
    response_model=CreditScoreResponse,
    summary="Credit likelihood score (Phase 3)",
    description=(
        "Runs the Phase 3 XGBoost credit model on the supplied user profile and returns an "
        "educational credit likelihood score (0–100), model confidence, and signed SHAP feature contributions. "
        "Accepts the same request contract as /analyze for consistency. "
        "Not a regulated credit score."
    ),
)
async def credit_score(body: UnifiedAnalysisRequest) -> CreditScoreResponse:
    from ml.orchestrator.pipeline import PipelineState

    request = validate_analysis_request(body)
    pipeline = _get_pipeline()
    telemetry = TelemetryCollector()

    # Run only validate + preprocess + credit scoring stages
    state = PipelineState(request=request)
    pipeline._stage_validate_request(state, telemetry)
    pipeline._stage_preprocess_input(state, telemetry)
    pipeline._stage_credit_scoring(state, telemetry)

    analysis: CreditAnalysisResult = state.credit_analysis  # type: ignore[assignment]
    return CreditScoreResponse(
        request_id=request.request_id or "unknown",
        credit_score=analysis.credit_score,
        confidence=analysis.confidence,
        top_positive_features=list(analysis.top_positive_features),
        top_negative_features=list(analysis.top_negative_features),
        source=analysis.source,
        educational_disclaimer=OrchestratorConfig().educational_disclaimer,
    )
