"""Cross-field validation and deterministic preference derivation for Phase 6."""

from __future__ import annotations

from collections.abc import Mapping

from .config import RecommendationConfig
from .schemas import IncomeStability, RecommendationRequest
from ml.investments.retrieval import LIQUIDITY_RANK
from ml.investments.schemas import BehavioralPersonaName, LiquidityLevel, RiskLevel


RISK_RANK = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
PERSONAS_BY_RISK = {
    RiskLevel.LOW: {BehavioralPersonaName.SECURE_SAVER, BehavioralPersonaName.CONSERVATIVE_PLANNER},
    RiskLevel.MEDIUM: {BehavioralPersonaName.BALANCED_BUILDER, BehavioralPersonaName.STRATEGIC_INVESTOR},
    RiskLevel.HIGH: {BehavioralPersonaName.GROWTH_EXPLORER, BehavioralPersonaName.AGGRESSIVE_WEALTH_SEEKER},
}


def coerce_request(request: RecommendationRequest | Mapping[str, object]) -> RecommendationRequest:
    """Parse an API payload and enforce compatibility with Phase 4 persona output."""
    parsed = request if isinstance(request, RecommendationRequest) else RecommendationRequest.model_validate(request)
    permitted_personas = PERSONAS_BY_RISK[parsed.risk_profile]
    if parsed.behavioral_persona not in permitted_personas:
        names = ", ".join(persona.value for persona in sorted(permitted_personas, key=lambda item: item.value))
        raise ValueError(f"behavioral_persona must match the {parsed.risk_profile.value} risk profile ({names})")
    return parsed


def minimum_liquidity(request: RecommendationRequest, config: RecommendationConfig) -> LiquidityLevel:
    """Infer liquidity needs from resilience inputs unless the caller states one."""
    if request.liquidity_preference is not None:
        return request.liquidity_preference
    savings_to_income_ratio = request.existing_savings / request.monthly_income
    if (
        request.emergency_fund_months < 3
        or request.income_stability == IncomeStability.LOW
        or (request.dependents > 0 and savings_to_income_ratio < 1)
    ):
        return LiquidityLevel.HIGH
    if request.emergency_fund_months < config.emergency_fund_target_months or request.income_stability == IncomeStability.MEDIUM:
        return LiquidityLevel.MODERATE
    return LiquidityLevel.LOW


def meets_minimum_liquidity(product_liquidity: LiquidityLevel, requested_liquidity: LiquidityLevel) -> bool:
    """Compare catalogue liquidity labels using their Phase 5 ordinal semantics."""
    return LIQUIDITY_RANK[product_liquidity] >= LIQUIDITY_RANK[requested_liquidity]
