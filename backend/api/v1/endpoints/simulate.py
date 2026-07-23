"""/api/v1/simulate — Phase 7 Monte Carlo financial simulation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ml.simulation import project_portfolio
from ml.simulation.schemas import ProjectionRequest, ProjectionResponse

router = APIRouter()


@router.post(
    "/simulate",
    response_model=ProjectionResponse,
    summary="Monte Carlo financial simulation (Phase 7)",
    description=(
        "Runs educational Monte Carlo projections across conservative, recommended, and growth "
        "portfolio scenarios. Requires a Phase 6 RecommendationResponse payload nested under "
        "the 'recommendation' field. Returns terminal-value distribution, goal probability, "
        "inflation-adjusted figures, yearly projections, and portfolio analytics. "
        "Not a forecast of future market performance."
    ),
)
async def simulate(body: ProjectionRequest, request: Request) -> ProjectionResponse:
    cfg = request.app.state.config
    # Clamp num_simulations to the configured maximum
    payload = body.model_dump(mode="json")
    if payload["num_simulations"] > cfg.max_num_simulations:
        payload["num_simulations"] = cfg.max_num_simulations
    return project_portfolio(payload)
