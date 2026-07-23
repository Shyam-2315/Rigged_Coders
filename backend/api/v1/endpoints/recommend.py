"""/api/v1/recommend — Phase 6 portfolio recommendation endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from ml.recommendation import recommend_portfolio
from ml.recommendation.schemas import RecommendationRequest, RecommendationResponse

router = APIRouter()


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Portfolio recommendation (Phase 6)",
    description=(
        "Generates deterministic, diversified educational portfolio illustrations across "
        "conservative, recommended, and growth variants. "
        "Requires a credit score and risk profile from Phases 3–4. "
        "Returns allocation percentages, monthly amounts, health metrics, and plain-language explanations. "
        "Not a regulated investment recommendation."
    ),
)
async def recommend(body: RecommendationRequest) -> RecommendationResponse:
    return recommend_portfolio(body.model_dump(mode="json"))
