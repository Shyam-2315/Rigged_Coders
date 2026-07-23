"""Allocation, return/risk aggregation, and portfolio-health calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable

from .config import RecommendationConfig
from .portfolio_optimizer import OptimizationResult
from .schemas import AssetClass, PortfolioAllocation, PortfolioHealth, PortfolioStyle, RecommendationRequest
from .scoring_engine import LIQUIDITY_SCORES, TAX_SCORES, ScoredCandidate
from ml.investments.schemas import RiskLevel


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Numerical allocation outputs that can be passed to Phase 7."""

    allocation: tuple[PortfolioAllocation, ...]
    expected_annual_return_min: float
    expected_annual_return_max: float
    estimated_risk: float
    risk_level: RiskLevel
    health: PortfolioHealth

    @property
    def expected_return_range(self) -> str:
        return format_return_range(self.expected_annual_return_min, self.expected_annual_return_max)


class AllocationEngine:
    """Allocate the monthly budget subject to every portfolio policy constraint."""

    def __init__(self, config: RecommendationConfig | None = None) -> None:
        self.config = config or RecommendationConfig()

    def allocate(self, optimized: OptimizationResult, request: RecommendationRequest) -> AllocationResult:
        candidates = list(optimized.selected)
        amounts = self._allocate_amounts(candidates, optimized.asset_class_targets, request)
        allocation = tuple(
            PortfolioAllocation(
                product=candidate.product.name,
                product_id=candidate.product.id,
                asset_class=candidate.asset_class,
                allocation=round(amount / request.monthly_budget * 100, 2),
                monthly_amount=amount,
                expected_annual_return_min=candidate.product.expected_annual_return_min,
                expected_annual_return_max=candidate.product.expected_annual_return_max,
                risk_contribution=round(
                    candidate.product.historical_volatility * amount / request.monthly_budget,
                    3,
                ),
                liquidity_contribution=round(
                    LIQUIDITY_SCORES[candidate.product.liquidity] * amount / request.monthly_budget,
                    3,
                ),
                selection_score=candidate.score,
                score_breakdown=candidate.components,
                reason="",
                potential_risks=candidate.product.potential_risks,
            )
            for candidate, amount in zip(candidates, amounts, strict=True)
        )
        return_min = sum(
            item.expected_annual_return_min * item.allocation / 100 for item in allocation
        )
        return_max = sum(
            item.expected_annual_return_max * item.allocation / 100 for item in allocation
        )
        risk = sum(item.risk_contribution for item in allocation)
        health = PortfolioHealthCalculator(self.config).calculate(
            allocation=allocation,
            candidates=candidates,
            request=request,
            style=optimized.style,
            estimated_risk=risk,
        )
        return AllocationResult(
            allocation=allocation,
            expected_annual_return_min=round(return_min, 2),
            expected_annual_return_max=round(return_max, 2),
            estimated_risk=round(risk, 2),
            risk_level=risk_level_from_volatility(risk),
            health=health,
        )

    def _allocate_amounts(
        self,
        candidates: list[ScoredCandidate],
        targets: dict[AssetClass, float],
        request: RecommendationRequest,
    ) -> list[int]:
        if not candidates:
            raise ValueError("Cannot allocate an empty portfolio")
        grouped: dict[AssetClass, list[tuple[int, ScoredCandidate]]] = {}
        for index, candidate in enumerate(candidates):
            grouped.setdefault(candidate.asset_class, []).append((index, candidate))
        active_target_total = sum(max(0.0, targets.get(asset_class, 0.0)) for asset_class in grouped)
        if active_target_total <= 0:
            group_weights = {asset_class: 100 / len(grouped) for asset_class in grouped}
        else:
            group_weights = {
                asset_class: max(0.0, targets.get(asset_class, 0.0)) / active_target_total * 100
                for asset_class in grouped
            }
        desired_amounts = [0.0] * len(candidates)
        for asset_class, members in grouped.items():
            score_total = sum(max(candidate.score, 1.0) for _, candidate in members)
            for index, candidate in members:
                desired_amounts[index] = (
                    request.monthly_budget
                    * group_weights[asset_class]
                    / 100
                    * max(candidate.score, 1.0)
                    / score_total
                )

        min_percent_amount = ceil(request.monthly_budget * self.config.minimum_allocation_percent / 100)
        maximum_amount = floor(request.monthly_budget * self.config.maximum_single_product_percent / 100)
        floors = [max(min_percent_amount, ceil(candidate.product.minimum_investment)) for candidate in candidates]
        if any(value > maximum_amount for value in floors):
            raise ValueError("A selected product's minimum investment exceeds the single-product allocation cap")
        if sum(floors) > request.monthly_budget:
            raise ValueError("Product minimum investments exceed the supplied monthly budget")
        amounts = [min(maximum_amount, max(floor_value, round(desired))) for floor_value, desired in zip(floors, desired_amounts, strict=True)]
        self._rebalance_to_budget(amounts, floors, maximum_amount, candidates, int(round(request.monthly_budget)))
        return amounts

    @staticmethod
    def _rebalance_to_budget(
        amounts: list[int],
        floors: list[int],
        maximum_amount: int,
        candidates: list[ScoredCandidate],
        budget: int,
    ) -> None:
        difference = budget - sum(amounts)
        if difference > 0:
            ordering = sorted(range(len(amounts)), key=lambda index: (-candidates[index].score, candidates[index].product.id))
            for index in ordering:
                addition = min(difference, maximum_amount - amounts[index])
                amounts[index] += addition
                difference -= addition
                if difference == 0:
                    break
        elif difference < 0:
            remaining_reduction = -difference
            ordering = sorted(
                range(len(amounts)),
                key=lambda index: (amounts[index] - floors[index], candidates[index].score, candidates[index].product.id),
                reverse=True,
            )
            for index in ordering:
                reduction = min(remaining_reduction, amounts[index] - floors[index])
                amounts[index] -= reduction
                remaining_reduction -= reduction
                if remaining_reduction == 0:
                    break
            difference = -remaining_reduction
        if sum(amounts) != budget:
            raise ValueError("Unable to rebalance portfolio to the exact monthly budget within allocation constraints")


class PortfolioHealthCalculator:
    """Compute 0-100 rule-based health metrics without market prediction."""

    def __init__(self, config: RecommendationConfig | None = None) -> None:
        self.config = config or RecommendationConfig()

    def calculate(
        self,
        *,
        allocation: Iterable[PortfolioAllocation],
        candidates: Iterable[ScoredCandidate],
        request: RecommendationRequest,
        style: PortfolioStyle,
        estimated_risk: float,
    ) -> PortfolioHealth:
        items = list(allocation)
        candidate_list = list(candidates)
        classes = {item.asset_class for item in items}
        max_allocation = max(item.allocation for item in items)
        diversification = min(
            100,
            round(20 + 50 * min(len(classes), 5) / 5 + 15 * min(len(items), 6) / 6 + 15 * (1 - max_allocation / 100)),
        )
        style_adjustment = {
            PortfolioStyle.CONSERVATIVE: -3.0,
            PortfolioStyle.RECOMMENDED: 0.0,
            PortfolioStyle.GROWTH: 3.0,
        }[style]
        target_risk = max(0.0, self.config.risk_volatility_targets[request.risk_profile.value] + style_adjustment)
        risk_alignment = round(max(0.0, 100 - abs(estimated_risk - target_risk) * 5.0))
        liquidity = round(sum(item.liquidity_contribution for item in items))
        inflation = round(
            min(
                100.0,
                20.0
                + sum(
                    item.allocation
                    * (
                        1.0
                        if candidate.product.inflation_protection
                        else 0.60
                        if item.asset_class == AssetClass.EQUITY
                        else 0.20
                    )
                    / 100
                    for item, candidate in zip(items, candidate_list, strict=True)
                )
                * 80,
            )
        )
        goal_alignment = round(
            min(
                100.0,
                30.0
                + sum(
                    item.allocation
                    * (1.0 if request.investment_goal in candidate.product.supported_goals else 0.40)
                    for item, candidate in zip(items, candidate_list, strict=True)
                ),
            )
        )
        tax_efficiency = round(
            sum(
                TAX_SCORES[candidate.product.tax_efficiency] * item.allocation / 100
                for item, candidate in zip(items, candidate_list, strict=True)
            )
        )
        stability = round(
            min(
                100.0,
                max(
                    0.0,
                    100
                    - estimated_risk * 2.4
                    + sum(
                        item.allocation
                        * (0.20 if item.asset_class in {AssetClass.CASH, AssetClass.GOVERNMENT_SAVINGS} else 0.0)
                        for item in items
                    ),
                ),
            )
        )
        overall = round(
            0.15 * diversification
            + 0.20 * risk_alignment
            + 0.15 * liquidity
            + 0.15 * inflation
            + 0.20 * goal_alignment
            + 0.10 * tax_efficiency
            + 0.05 * stability
        )
        return PortfolioHealth(
            diversification_score=int(diversification),
            risk_alignment_score=int(risk_alignment),
            liquidity_score=int(liquidity),
            inflation_protection_score=int(inflation),
            goal_alignment_score=int(goal_alignment),
            tax_efficiency_score=int(tax_efficiency),
            portfolio_stability_score=int(stability),
            overall_portfolio_score=int(min(100, max(0, overall))),
        )


def risk_level_from_volatility(volatility: float) -> RiskLevel:
    """Convert the weighted illustrative volatility into the API risk label."""
    if volatility <= 8:
        return RiskLevel.LOW
    if volatility <= 17:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def format_return_range(minimum: float, maximum: float) -> str:
    """Format a compact return range suitable for API and report consumers."""
    return f"{_format_percent(minimum)}–{_format_percent(maximum)}"


def _format_percent(value: float) -> str:
    rounded = round(value, 1)
    label = f"{rounded:.1f}".rstrip("0").rstrip(".")
    return f"{label}%"
