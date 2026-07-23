"""Candidate retrieval and transparent weighted product scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .config import RecommendationConfig
from .schemas import AssetClass, ProductScoreBreakdown, RecommendationRequest
from .validation import RISK_RANK, meets_minimum_liquidity, minimum_liquidity
from ml.investments.repository import InvestmentProductRepository
from ml.investments.retrieval import LIQUIDITY_RANK
from ml.investments.schemas import InvestmentProduct, LiquidityLevel, ProductCategory, TaxEfficiency


LIQUIDITY_SCORES = {
    LiquidityLevel.INSTANT: 100.0,
    LiquidityLevel.HIGH: 80.0,
    LiquidityLevel.MODERATE: 60.0,
    LiquidityLevel.LOW: 30.0,
}
TAX_SCORES = {TaxEfficiency.LOW: 35.0, TaxEfficiency.MODERATE: 65.0, TaxEfficiency.HIGH: 100.0}


def asset_class_for(product: InvestmentProduct) -> AssetClass:
    """Map Phase 5 categories to the diversification buckets used in Phase 6."""
    if "gold" in product.id or product.category == ProductCategory.COMMODITY:
        return AssetClass.GOLD
    if product.category == ProductCategory.CASH:
        return AssetClass.CASH
    if product.government_backed or product.category == ProductCategory.GOVERNMENT_SAVINGS:
        return AssetClass.GOVERNMENT_SAVINGS
    if product.category in {ProductCategory.DEBT, ProductCategory.FIXED_INCOME}:
        return AssetClass.DEBT
    if product.category in {ProductCategory.EQUITY, ProductCategory.INTERNATIONAL_EQUITY}:
        return AssetClass.EQUITY
    return AssetClass.REAL_ASSETS


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A catalogue product with a reproducible score and selection evidence."""

    product: InvestmentProduct
    asset_class: AssetClass
    score: float
    components: ProductScoreBreakdown
    retrieval_reasons: tuple[str, ...]


class CandidateProductRetriever:
    """Repository-backed candidate filtering before ranking or allocation."""

    def __init__(
        self,
        repository: InvestmentProductRepository | None = None,
        config: RecommendationConfig | None = None,
    ) -> None:
        self.repository = repository or InvestmentProductRepository()
        self.config = config or RecommendationConfig()

    def retrieve(self, request: RecommendationRequest) -> list[InvestmentProduct]:
        """Filter on risk, credit eligibility, horizon, liquidity, and affordability.

        Goal, persona, and government-backed preference remain score inputs rather
        than hard exclusions. This preserves the minimum-diversification policy
        for sparse goals such as tax saving or emergency funding.
        """
        requested_liquidity = minimum_liquidity(request, self.config)
        products = self.repository.get_all()
        candidates = self._filter(products, request, requested_liquidity, require_horizon=True, require_age=True)
        if len(candidates) < self.config.minimum_products:
            candidates = self._filter(products, request, requested_liquidity, require_horizon=False, require_age=True)
        if len(candidates) < self.config.minimum_products:
            candidates = self._filter(products, request, requested_liquidity, require_horizon=False, require_age=False)
        if len(candidates) < self.config.minimum_products and request.liquidity_preference is None:
            candidates = self._filter(products, request, LiquidityLevel.LOW, require_horizon=False, require_age=False)
        if len(candidates) < self.config.minimum_products:
            raise ValueError(
                "The catalog does not contain enough eligible products to meet the three-product diversification policy "
                "for this budget, risk profile, and credit eligibility."
            )
        return candidates

    def _filter(
        self,
        products: Iterable[InvestmentProduct],
        request: RecommendationRequest,
        requested_liquidity: LiquidityLevel,
        *,
        require_horizon: bool,
        require_age: bool,
    ) -> list[InvestmentProduct]:
        max_product_amount = request.monthly_budget * self.config.maximum_single_product_percent / 100
        candidates: list[InvestmentProduct] = []
        for product in products:
            if RISK_RANK[product.risk_level] > RISK_RANK[request.risk_profile]:
                continue
            if product.credit_score_requirement > request.credit_score:
                continue
            if product.minimum_investment > max_product_amount:
                continue
            if require_age and not product.ideal_age_range.minimum <= request.age <= product.ideal_age_range.maximum:
                continue
            if require_horizon and product.category != ProductCategory.CASH and product.recommended_horizon > request.investment_horizon:
                continue
            if not meets_minimum_liquidity(product.liquidity, requested_liquidity):
                continue
            candidates.append(product)
        return candidates


class ProductScoringEngine:
    """Score products from fixed factors that sum to a 0-100 result."""

    def __init__(self, config: RecommendationConfig | None = None) -> None:
        self.config = config or RecommendationConfig()
        rules = self.config.load_rules()
        self.weights = {name: float(value) for name, value in rules["scoring_weights"].items()}

    def score(self, products: Iterable[InvestmentProduct], request: RecommendationRequest) -> list[ScoredCandidate]:
        product_list = list(products)
        class_counts = Counter(asset_class_for(product) for product in product_list)
        candidates = [self._score_product(product, request, class_counts) for product in product_list]
        return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.product.id))

    def _score_product(
        self,
        product: InvestmentProduct,
        request: RecommendationRequest,
        class_counts: Counter[AssetClass],
    ) -> ScoredCandidate:
        asset_class = asset_class_for(product)
        required_liquidity = minimum_liquidity(request, self.config)
        risk_gap = RISK_RANK[request.risk_profile] - RISK_RANK[product.risk_level]
        risk_match = 100.0 if risk_gap == 0 else max(55.0, 90.0 - 15.0 * risk_gap)
        return_midpoint = (product.expected_annual_return_min + product.expected_annual_return_max) / 2
        expected_return = min(100.0, max(0.0, (return_midpoint - 2.0) / 14.0 * 100.0))
        liquidity_gap = max(0, LIQUIDITY_RANK[required_liquidity] - LIQUIDITY_RANK[product.liquidity])
        liquidity = max(0.0, LIQUIDITY_SCORES[product.liquidity] - 25.0 * liquidity_gap)
        rarity = max(class_counts.values(), default=1) / class_counts[asset_class]
        diversification_value = min(100.0, 30.0 + 70.0 * rarity)
        max_product_amount = request.monthly_budget * self.config.maximum_single_product_percent / 100
        budget_compatibility = 100.0 if product.minimum_investment <= request.monthly_budget * 0.20 else 75.0
        if product.minimum_investment > max_product_amount:
            budget_compatibility = 0.0
        trust_score = product.trust_score
        if request.government_backed_preference and product.government_backed:
            trust_score = min(100.0, trust_score + 10.0)
        components = ProductScoreBreakdown(
            goal_match=100.0 if request.investment_goal in product.supported_goals else 25.0,
            risk_match=risk_match,
            persona_match=100.0 if request.behavioral_persona in product.behavioral_personas else 45.0,
            expected_return=expected_return,
            liquidity=liquidity,
            tax_efficiency=TAX_SCORES[product.tax_efficiency],
            inflation_protection=100.0 if product.inflation_protection else 25.0,
            trust_score=trust_score,
            popularity=product.popularity_score,
            diversification_value=diversification_value,
            budget_compatibility=budget_compatibility,
        )
        weighted_score = sum(self.weights[name] * getattr(components, name) / 100 for name in self.weights)
        reasons = [
            f"Meets the {request.risk_profile.value.lower()}-risk ceiling",
            "Meets the supplied credit-score eligibility requirement",
            f"Fits the monthly allocation cap of ₹{max_product_amount:,.0f}",
        ]
        if request.investment_goal in product.supported_goals:
            reasons.append(f"Supports the {request.investment_goal.value.lower()} goal")
        if request.behavioral_persona in product.behavioral_personas:
            reasons.append(f"Matches the {request.behavioral_persona.value} persona")
        if meets_minimum_liquidity(product.liquidity, required_liquidity):
            reasons.append(f"Meets the {required_liquidity.value.lower()} liquidity preference")
        if request.government_backed_preference and product.government_backed:
            reasons.append("Honours the government-backed preference")
        return ScoredCandidate(
            product=product,
            asset_class=asset_class,
            score=round(min(100.0, weighted_score), 2),
            components=components,
            retrieval_reasons=tuple(reasons),
        )
