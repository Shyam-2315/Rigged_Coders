"""/api/v1/risk-profile — Phase 4 behavioral risk profiling endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from ml.orchestrator.config import OrchestratorConfig
from ml.orchestrator.pipeline import AnalysisPipeline, PipelineState
from ml.orchestrator.schemas import RiskAnalysisResult, UnifiedAnalysisRequest
from ml.orchestrator.telemetry import TelemetryCollector
from ml.orchestrator.validators import validate_analysis_request

router = APIRouter()


class RiskProfileResponse(BaseModel):
    """Risk profiling endpoint response."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    risk_profile: str = Field(description="Low / Medium / High")
    persona: str = Field(description="Behavioral investor persona name")
    persona_details: dict = Field(description="Persona description, philosophy, strengths, and potential risks")
    confidence: float = Field(ge=0, le=1)
    probabilities: dict = Field(description="{'low': float, 'medium': float, 'high': float}")
    recommendation_summary: str
    top_positive_factors: list[str]
    top_negative_factors: list[str]
    explanation_method: str | None
    educational_disclaimer: str


_pipeline: AnalysisPipeline | None = None


def _get_pipeline() -> AnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AnalysisPipeline(config=OrchestratorConfig())
    return _pipeline


@router.post(
    "/risk-profile",
    response_model=RiskProfileResponse,
    summary="Behavioral risk profile (Phase 4)",
    description=(
        "Classifies the investor's risk tolerance as Low, Medium, or High using the Phase 4 "
        "behavioral questionnaire classifier. Returns the persona, class probabilities, confidence, "
        "and local SHAP explanations. Not personalised financial advice."
    ),
)
async def risk_profile(body: UnifiedAnalysisRequest) -> RiskProfileResponse:
    request = validate_analysis_request(body)
    pipeline = _get_pipeline()
    telemetry = TelemetryCollector()

    state = PipelineState(request=request)
    pipeline._stage_validate_request(state, telemetry)
    pipeline._stage_risk_profiling(state, telemetry)

    analysis: RiskAnalysisResult = state.risk_analysis  # type: ignore[assignment]
    return RiskProfileResponse(
        request_id=request.request_id or "unknown",
        risk_profile=analysis.risk_profile,
        persona=analysis.persona,
        persona_details=analysis.persona_details,
        confidence=analysis.confidence,
        probabilities=analysis.probabilities,
        recommendation_summary=analysis.recommendation_summary,
        top_positive_factors=list(analysis.top_positive_factors),
        top_negative_factors=list(analysis.top_negative_factors),
        explanation_method=analysis.explanation_method,
        educational_disclaimer=OrchestratorConfig().educational_disclaimer,
    )
