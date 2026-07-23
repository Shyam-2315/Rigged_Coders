"""API v1 router — mounts all Phase 9 endpoint modules."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.v1.endpoints.analyze import router as analyze_router
from backend.api.v1.endpoints.credit_score import router as credit_score_router
from backend.api.v1.endpoints.health import router as health_router
from backend.api.v1.endpoints.products import router as products_router
from backend.api.v1.endpoints.recommend import router as recommend_router
from backend.api.v1.endpoints.risk_profile import router as risk_profile_router
from backend.api.v1.endpoints.simulate import router as simulate_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router, tags=["System"])
api_v1_router.include_router(analyze_router, tags=["Unified Orchestrator"])
api_v1_router.include_router(credit_score_router, tags=["Credit Scoring"])
api_v1_router.include_router(risk_profile_router, tags=["Risk Profiling"])
api_v1_router.include_router(products_router, tags=["Investment Products"])
api_v1_router.include_router(recommend_router, tags=["Portfolio Recommendation"])
api_v1_router.include_router(simulate_router, tags=["Financial Simulation"])
