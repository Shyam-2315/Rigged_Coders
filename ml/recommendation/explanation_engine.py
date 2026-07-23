"""Plain-language explanations, trade-offs, and educational insights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .allocation_engine import AllocationResult
from .config import RecommendationConfig
from .schemas import AssetClass, PortfolioAllocation, PortfolioStyle, RecommendationRequest
from .scoring_engine import ScoredCandidate


@dataclass(frozen=True, slots=True)
class ExplainedPortfolio:
    """Human-readable evidence layered onto a numerical allocation result."""

    allocation: tuple[PortfolioAllocation, ...]
    summary: str
    trade_offs: tuple[str, ...]


class ExplanationEngine:
    """Generate deterministic, non-prescriptive explanations from calculation inputs."""

    def __init__(self, config: RecommendationConfig | None = None) -> None:
        self.config = config or RecommendationConfig()

    def explain_portfolio(
        self,
        allocation_result: AllocationResult,
        candidates: Iterable[ScoredCandidate],
        request: RecommendationRequest,
        style: PortfolioStyle,
    ) -> ExplainedPortfolio:
        candidate_by_id = {candidate.product.id: candidate for candidate in candidates}
        explained = tuple(
            item.model_copy(update={"reason": self._product_reason(item, candidate_by_id[item.product_id], request, style)})
            for item in allocation_result.allocation
        )
        summary = (
            f"This {style.value.lower()} educational illustration spreads ₹{request.monthly_budget:,.0f} across "
            f"{len(explained)} products. Its weighted planning range is {allocation_result.expected_return_range} "
            f"with illustrative volatility of {allocation_result.estimated_risk:.1f}%."
        )
        return ExplainedPortfolio(
            allocation=explained,
            summary=summary,
            trade_offs=self._trade_offs(explained, request),
        )

    def insights(
        self,
        allocation: Iterable[PortfolioAllocation],
        request: RecommendationRequest,
    ) -> tuple[str, ...]:
        items = tuple(allocation)
        insights: list[str] = []
        if request.emergency_fund_months < 3:
            insights.append(
                "Consider building at least three months of emergency reserves before increasing exposure to less-liquid products."
            )
        elif request.emergency_fund_months < self.config.emergency_fund_target_months:
            insights.append(
                "Consider progressing toward six months of emergency reserves to improve financial resilience."
            )
        if request.dependents > 0 and request.existing_savings < request.monthly_income:
            insights.append(
                "With dependent obligations and a limited existing-savings buffer, prioritise accessible reserves when reviewing this illustration."
            )
        savings_headroom = request.monthly_savings - request.monthly_budget
        if savings_headroom >= 1_000:
            insights.append(
                f"If cash flow remains stable, you could review whether up to ₹{savings_headroom:,.0f} of monthly savings is available for a higher SIP."
            )
        elif savings_headroom < 0:
            insights.append(
                "The selected monthly budget is above reported monthly savings; review affordability before treating this as a recurring SIP."
            )
        if len({item.asset_class for item in items}) < 4 and request.investment_horizon >= 5:
            insights.append(
                "Consider reviewing diversification across additional asset classes if it remains consistent with your risk profile and product availability."
            )
        if max(item.allocation for item in items) >= 50:
            insights.append(
                "One product has a large allocation; review concentration risk and the underlying product's terms before implementation."
            )
        if request.investment_goal.value != "Tax Saving" and not any(
            item.score_breakdown.tax_efficiency >= 100 for item in items
        ):
            insights.append(
                "Consider checking whether tax-saving options such as ELSS, PPF, or NPS fit your eligibility, horizon, and current tax rules."
            )
        if request.investment_horizon < 5:
            insights.append(
                "A shorter horizon can make market-linked products less predictable; keep near-term goal money accessible."
            )
        if not insights:
            insights.append("Review this illustration periodically when income, goals, emergency reserves, or risk comfort changes.")
        return tuple(insights)

    @staticmethod
    def _product_reason(
        item: PortfolioAllocation,
        candidate: ScoredCandidate,
        request: RecommendationRequest,
        style: PortfolioStyle,
    ) -> str:
        product = candidate.product
        category_reason = {
            AssetClass.EQUITY: "It provides diversified market-linked growth potential for a longer horizon.",
            AssetClass.DEBT: "It adds a comparatively stable fixed-income component and helps balance equity risk.",
            AssetClass.GOLD: "It adds an asset that can diversify equity exposure and may help with inflation protection.",
            AssetClass.CASH: "It keeps part of the plan accessible for short-term needs and resilience.",
            AssetClass.GOVERNMENT_SAVINGS: "It adds a government-linked savings component with its stated terms and liquidity limits.",
            AssetClass.REAL_ASSETS: "It broadens exposure beyond conventional equity and debt, while retaining market risk.",
        }[item.asset_class]
        goal_reason = (
            f" It directly supports the {request.investment_goal.value.lower()} goal."
            if request.investment_goal in product.supported_goals
            else " It is used as a diversification sleeve rather than the primary goal vehicle."
        )
        allocation_reason = (
            " Its allocation reflects the conservative tilt."
            if style == PortfolioStyle.CONSERVATIVE
            else " Its allocation reflects the growth tilt within the stated risk ceiling."
            if style == PortfolioStyle.GROWTH
            else " Its allocation balances its score, asset-class role, minimum investment, and concentration cap."
        )
        return f"{product.name} is included because {category_reason}{goal_reason}{allocation_reason}"

    @staticmethod
    def _trade_offs(allocation: Iterable[PortfolioAllocation], request: RecommendationRequest) -> tuple[str, ...]:
        items = tuple(allocation)
        trade_offs: list[str] = []
        if any(item.asset_class == AssetClass.EQUITY for item in items):
            trade_offs.append("Equity allocations can fluctuate substantially and may be unsuitable for money needed soon.")
        if any(item.asset_class == AssetClass.GOLD for item in items):
            trade_offs.append("Gold can diversify a portfolio but may be volatile and does not generate regular income.")
        if any(item.asset_class == AssetClass.GOVERNMENT_SAVINGS for item in items):
            trade_offs.append("Government-linked products can carry lock-ins, eligibility criteria, or changing issue availability.")
        if request.emergency_fund_months < 3:
            trade_offs.append("Limited emergency reserves can make even a diversified investment plan harder to sustain during a disruption.")
        return tuple(trade_offs or ["Expected return ranges are illustrative planning assumptions, not forecasts or guarantees."])
