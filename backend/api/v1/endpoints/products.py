"""/api/v1/products — Phase 5 investment product knowledge base endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from ml.investments.retrieval import RetrievalContext, search_products
from ml.investments.repository import InvestmentProductRepository
from ml.investments.schemas import InvestmentGoal, LiquidityLevel, RiskLevel

router = APIRouter()


class ProductSummary(BaseModel):
    """Compact investment product representation for API responses."""

    model_config = ConfigDict(frozen=True)

    product_id: str
    name: str
    risk_level: str
    expected_return: str
    investment_horizon: str
    liquidity: str
    relevance_score: float
    why_recommended: list[str]
    potential_risks: list[str]
    simple_explanation: str


class ProductsResponse(BaseModel):
    """Investment product retrieval response."""

    model_config = ConfigDict(frozen=True)

    query: str | None
    total_matches: int
    products: list[ProductSummary]
    educational_disclaimer: str


@router.get(
    "/products",
    response_model=ProductsResponse,
    summary="Investment product search (Phase 5)",
    description=(
        "Search and rank investment products from the Phase 5 knowledge base. "
        "Supports free-text query and optional filter parameters. "
        "Results are ranked by goal, risk, persona, horizon, liquidity, and trust match. "
        "Not a recommendation to buy or sell any product."
    ),
)
async def list_products(
    request: Request,
    q: Annotated[str | None, Query(description="Free-text search query (e.g. 'retirement', 'gold', 'safe investment')")] = None,
    risk: Annotated[str | None, Query(description="Filter by risk level: Low / Medium / High")] = None,
    goal: Annotated[str | None, Query(description="Filter by investment goal, e.g. 'Retirement'")] = None,
    persona: Annotated[str | None, Query(description="Filter by persona name, e.g. 'Balanced Builder'")] = None,
    horizon: Annotated[int | None, Query(ge=0, le=60, description="Minimum investment horizon in years")] = None,
    liquidity: Annotated[str | None, Query(description="Minimum liquidity: Low / Moderate / High / Instant")] = None,
    limit: Annotated[int, Query(ge=1, le=25, description="Maximum number of results to return")] = 10,
) -> ProductsResponse:
    cfg = request.app.state.config

    # Build optional retrieval context from query params
    context: RetrievalContext | None = None
    ctx_kwargs: dict = {}
    if risk:
        try:
            ctx_kwargs["risk_profile"] = RiskLevel(risk.title())
        except ValueError:
            pass
    if goal:
        try:
            ctx_kwargs["goals"] = (InvestmentGoal(goal),)
        except ValueError:
            pass
    if persona:
        ctx_kwargs["persona"] = persona  # type: ignore[assignment]
    if horizon is not None:
        ctx_kwargs["investment_horizon_years"] = horizon
    if liquidity:
        try:
            ctx_kwargs["minimum_liquidity"] = LiquidityLevel(liquidity.title())
        except ValueError:
            pass
    if ctx_kwargs:
        context = RetrievalContext(**ctx_kwargs)

    if q:
        matches = search_products(
            q,
            context=context.model_dump(mode="json") if context else None,
            top_k=min(limit, cfg.max_products_per_response),
        )
    else:
        # No query — return all products optionally filtered by context
        repo = InvestmentProductRepository()
        all_products = repo.get_all()
        if q is None and not ctx_kwargs:
            # Return raw listing sorted by trust score descending
            matches_raw = sorted(all_products, key=lambda p: getattr(p, "trust_score", 0), reverse=True)
            products_out = [
                ProductSummary(
                    product_id=p.id,
                    name=p.name,
                    risk_level=p.risk_level.value,
                    expected_return=f"{p.expected_return_min:.1f}%–{p.expected_return_max:.1f}%",
                    investment_horizon=f"{p.min_horizon_years}+ years",
                    liquidity=p.liquidity.value,
                    relevance_score=0.0,
                    why_recommended=[],
                    potential_risks=list(p.potential_risks),
                    simple_explanation=p.simple_explanation,
                )
                for p in matches_raw[: min(limit, cfg.max_products_per_response)]
            ]
            return ProductsResponse(
                query=None,
                total_matches=len(all_products),
                products=products_out,
                educational_disclaimer=cfg.educational_disclaimer,
            )
        matches = search_products(
            "",
            context=context.model_dump(mode="json") if context else None,
            top_k=min(limit, cfg.max_products_per_response),
        )

    products_out = [
        ProductSummary(
            product_id=item.product.id,
            name=item.product.name,
            risk_level=item.product.risk_level.value,
            expected_return=item.expected_return,
            investment_horizon=item.investment_horizon,
            liquidity=item.product.liquidity.value,
            relevance_score=round(item.relevance_score, 2),
            why_recommended=list(item.why_recommended),
            potential_risks=list(item.potential_risks),
            simple_explanation=item.simple_explanation,
        )
        for item in matches
    ]

    return ProductsResponse(
        query=q,
        total_matches=len(products_out),
        products=products_out,
        educational_disclaimer=cfg.educational_disclaimer,
    )
