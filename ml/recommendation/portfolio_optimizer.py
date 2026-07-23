"""Strategy-based, constrained product selection for portfolio illustrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from typing import Iterable

from .config import RecommendationConfig
from .schemas import AssetClass, PortfolioStyle, RecommendationRequest
from .scoring_engine import ScoredCandidate


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """The selected products and asset-class targets handed to allocation."""

    style: PortfolioStyle
    selected: tuple[ScoredCandidate, ...]
    asset_class_targets: dict[AssetClass, float]


class PortfolioStrategy(ABC):
    """Strategy interface for conservative, baseline, and growth tilts."""

    style: PortfolioStyle

    @abstractmethod
    def asset_class_targets(self, request: RecommendationRequest) -> dict[AssetClass, float]:
        """Return target weights for the supplied investor risk profile."""


class _ConfiguredPortfolioStrategy(PortfolioStrategy):
    def __init__(self, style: PortfolioStyle, config: RecommendationConfig) -> None:
        self.style = style
        self.config = config

    def asset_class_targets(self, request: RecommendationRequest) -> dict[AssetClass, float]:
        source = self.config.allocation_targets[self.style.value][request.risk_profile.value]
        return {AssetClass(name): float(weight) for name, weight in source.items()}


class ConservativePortfolioStrategy(_ConfiguredPortfolioStrategy):
    def __init__(self, config: RecommendationConfig) -> None:
        super().__init__(PortfolioStyle.CONSERVATIVE, config)


class RecommendedPortfolioStrategy(_ConfiguredPortfolioStrategy):
    def __init__(self, config: RecommendationConfig) -> None:
        super().__init__(PortfolioStyle.RECOMMENDED, config)


class GrowthPortfolioStrategy(_ConfiguredPortfolioStrategy):
    def __init__(self, config: RecommendationConfig) -> None:
        super().__init__(PortfolioStyle.GROWTH, config)


class PortfolioOptimizer:
    """Select 3-6 products while enforcing affordability and concentration limits."""

    def __init__(self, config: RecommendationConfig | None = None) -> None:
        self.config = config or RecommendationConfig()

    def optimize(
        self,
        candidates: Iterable[ScoredCandidate],
        request: RecommendationRequest,
        strategy: PortfolioStrategy,
    ) -> OptimizationResult:
        candidate_list = list(candidates)
        targets = strategy.asset_class_targets(request)
        selected = self._select_diversified_candidates(candidate_list, request, targets)
        return OptimizationResult(style=strategy.style, selected=tuple(selected), asset_class_targets=targets)

    def _select_diversified_candidates(
        self,
        candidates: list[ScoredCandidate],
        request: RecommendationRequest,
        targets: dict[AssetClass, float],
    ) -> list[ScoredCandidate]:
        by_class: dict[AssetClass, list[ScoredCandidate]] = {}
        for candidate in candidates:
            by_class.setdefault(candidate.asset_class, []).append(candidate)
        for values in by_class.values():
            values.sort(key=lambda candidate: (-candidate.score, candidate.product.id))

        selected: list[ScoredCandidate] = []
        required_classes: list[AssetClass] = []
        if request.emergency_fund_months < self.config.emergency_fund_target_months and AssetClass.CASH in by_class:
            required_classes.append(AssetClass.CASH)
        ordered_classes = sorted(targets, key=lambda item: (-targets[item], item.value))
        for asset_class in [*required_classes, *ordered_classes]:
            if asset_class not in by_class or any(item.asset_class == asset_class for item in selected):
                continue
            candidate = by_class[asset_class][0]
            if self._can_add(selected, candidate, request):
                selected.append(candidate)

        target_count = min(
            self.config.maximum_products,
            max(self.config.minimum_products, min(5, len(by_class), len(candidates))),
        )
        remaining = [candidate for candidate in candidates if candidate not in selected]
        remaining.sort(
            key=lambda candidate: (
                candidate.asset_class in {item.asset_class for item in selected},
                -candidate.score,
                candidate.product.id,
            )
        )
        for candidate in remaining:
            if len(selected) >= target_count:
                break
            if self._can_add(selected, candidate, request):
                selected.append(candidate)

        if len(selected) < self.config.minimum_products:
            raise ValueError(
                "The eligible catalog products cannot meet the three-product minimum while respecting each product's "
                "minimum investment and the 60% concentration cap. Increase the monthly budget or revise constraints."
            )
        return selected

    def _can_add(
        self,
        selected: list[ScoredCandidate],
        candidate: ScoredCandidate,
        request: RecommendationRequest,
    ) -> bool:
        minimum_percent_amount = ceil(request.monthly_budget * self.config.minimum_allocation_percent / 100)
        maximum_amount = request.monthly_budget * self.config.maximum_single_product_percent / 100
        candidate_floor = max(minimum_percent_amount, ceil(candidate.product.minimum_investment))
        if candidate_floor > maximum_amount:
            return False
        existing_floors = sum(
            max(minimum_percent_amount, ceil(item.product.minimum_investment)) for item in selected
        )
        return existing_floors + candidate_floor <= request.monthly_budget


def default_strategies(config: RecommendationConfig | None = None) -> tuple[PortfolioStrategy, ...]:
    """Return the stable alternatives generated by every recommendation request."""
    active_config = config or RecommendationConfig()
    return (
        ConservativePortfolioStrategy(active_config),
        RecommendedPortfolioStrategy(active_config),
        GrowthPortfolioStrategy(active_config),
    )
