"""Input validation and request coercion for Phase 7."""

from __future__ import annotations

from collections.abc import Mapping

from ml.recommendation.schemas import RecommendationResponse

from .schemas import ProjectionRequest


def coerce_projection_request(request: ProjectionRequest | Mapping[str, object]) -> ProjectionRequest:
    """Parse and validate a FastAPI payload or script dictionary."""
    parsed = request if isinstance(request, ProjectionRequest) else ProjectionRequest.model_validate(request)
    if isinstance(parsed.recommendation, Mapping):
        recommendation = RecommendationResponse.model_validate(parsed.recommendation)
        return parsed.model_copy(update={"recommendation": recommendation})
    return parsed


def resolve_monthly_sip(request: ProjectionRequest) -> int:
    """Prefer explicit monthly SIP, otherwise fall back to the recommendation budget."""
    if request.monthly_sip > 0:
        return request.monthly_sip
    return request.recommendation.monthly_budget
