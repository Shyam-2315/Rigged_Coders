"""Explainable retrieval and ranking for investment-product metadata.

The functions in this module rank catalog matches only. They do not construct
portfolios, set allocations, or decide whether any product is suitable to buy.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .repository import InvestmentProductRepository
from .schemas import (
    BehavioralPersonaName,
    InvestmentGoal,
    InvestmentProduct,
    LiquidityLevel,
    RiskLevel,
)


LIQUIDITY_RANK: dict[LiquidityLevel, int] = {
    LiquidityLevel.LOW: 1,
    LiquidityLevel.MODERATE: 2,
    LiquidityLevel.HIGH: 3,
    LiquidityLevel.INSTANT: 4,
}


class RetrievalContext(BaseModel):
    """Optional explicit metadata filters that inform relevance scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    risk_profile: RiskLevel | None = None
    goals: tuple[InvestmentGoal, ...] = ()
    persona: BehavioralPersonaName | None = None
    investment_horizon_years: int | None = Field(default=None, ge=0, le=60)
    minimum_liquidity: LiquidityLevel | None = None
    credit_score: int | None = Field(default=None, ge=0, le=900)
    beginner: bool | None = None
    government_backed: bool | None = None


class RetrievedProduct(BaseModel):
    """A ranked product match with transparent, non-prescriptive explanation."""

    model_config = ConfigDict(frozen=True)

    product: InvestmentProduct
    relevance_score: float = Field(ge=0, le=100)
    why_recommended: tuple[str, ...]
    suitable_for: tuple[str, ...]
    potential_risks: tuple[str, ...]
    expected_return: str
    investment_horizon: str
    simple_explanation: str

    @property
    def why_retrieved(self) -> tuple[str, ...]:
        """Alias that avoids implying a purchase recommendation."""
        return self.why_recommended


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    product: InvestmentProduct
    score: float
    reasons: tuple[str, ...]


class ProductRelevanceRanker:
    """Score products against transparent, fixed metadata criteria."""

    def score(self, product: InvestmentProduct, context: RetrievalContext, *, query_terms: set[str] | None = None) -> RankedCandidate:
        score = 0.0
        reasons: list[str] = []
        if context.risk_profile is not None and context.risk_profile in product.risk_profile_requirement:
            score += 25.0
            reasons.append(f"Matches the {context.risk_profile.value} risk-profile filter")
        if context.goals:
            goal_matches = [goal for goal in context.goals if goal in product.supported_goals]
            if goal_matches:
                score += 20.0 * len(goal_matches) / len(context.goals)
                reasons.append("Matches goal: " + ", ".join(goal.value for goal in goal_matches))
        if context.persona is not None and context.persona in product.behavioral_personas:
            score += 15.0
            reasons.append(f"Matches behavioral persona: {context.persona.value}")
        if context.investment_horizon_years is not None:
            if product.recommended_horizon <= context.investment_horizon_years:
                score += 15.0
                reasons.append("Fits the requested investment horizon")
            elif product.recommended_horizon <= context.investment_horizon_years + 2:
                score += 6.0
                reasons.append("Needs a slightly longer horizon than requested")
        if context.minimum_liquidity is not None and LIQUIDITY_RANK[product.liquidity] >= LIQUIDITY_RANK[context.minimum_liquidity]:
            score += 10.0
            reasons.append(f"Meets {context.minimum_liquidity.value.lower()}-liquidity requirement")
        if context.credit_score is not None and product.credit_score_requirement <= context.credit_score:
            score += 5.0
            reasons.append("Meets the supplied credit-score eligibility filter")
        if context.beginner is True and product.suitable_for_beginners:
            score += 5.0
            reasons.append("Marked as beginner-suitable")
        if context.government_backed is True and product.government_backed:
            score += 5.0
            reasons.append("Marked as government-backed")
        if query_terms:
            text_score, text_reasons = _text_match(product, query_terms)
            score += text_score
            reasons.extend(text_reasons)
        score += product.popularity_score * 0.025
        score += product.trust_score * 0.025
        if not reasons:
            reasons.append("Listed from the validated investment-product catalog")
        return RankedCandidate(product=product, score=round(min(score, 100.0), 3), reasons=tuple(dict.fromkeys(reasons)))


class InvestmentRetrievalEngine:
    """Repository-backed API for filtering, smart search, and relevance ranking."""

    def __init__(self, repository: InvestmentProductRepository | None = None, ranker: ProductRelevanceRanker | None = None) -> None:
        self.repository = repository or InvestmentProductRepository()
        self.ranker = ranker or ProductRelevanceRanker()

    def get_all_products(self, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
        return self._rank(self.repository.get_all(), _coerce_context(context))

    def get_product_by_id(self, product_id: str) -> RetrievedProduct | None:
        product = self.repository.get_by_id(product_id)
        if product is None:
            return None
        return self._to_result(self.ranker.score(product, RetrievalContext()))

    def filter_by_risk(self, risk_level: RiskLevel | str, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
        selected_risk = _coerce_enum(risk_level, RiskLevel, "risk_level")
        active_context = _coerce_context(context)
        if active_context.risk_profile is None:
            active_context = active_context.model_copy(update={"risk_profile": selected_risk})
        products = [product for product in self.repository.get_all() if product.risk_level == selected_risk]
        return self._rank(products, active_context, fixed_reason=f"Filtered to {selected_risk.value} product risk")

    def filter_by_goal(self, goal: InvestmentGoal | str, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
        selected_goal = _coerce_enum(goal, InvestmentGoal, "goal")
        active_context = _coerce_context(context)
        if selected_goal not in active_context.goals:
            active_context = active_context.model_copy(update={"goals": (*active_context.goals, selected_goal)})
        products = [product for product in self.repository.get_all() if selected_goal in product.supported_goals]
        return self._rank(products, active_context, fixed_reason=f"Supports goal: {selected_goal.value}")

    def filter_by_persona(self, persona: BehavioralPersonaName | str, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
        selected_persona = _coerce_enum(persona, BehavioralPersonaName, "persona")
        active_context = _coerce_context(context).model_copy(update={"persona": selected_persona})
        products = [product for product in self.repository.get_all() if selected_persona in product.behavioral_personas]
        return self._rank(products, active_context, fixed_reason=f"Supports persona: {selected_persona.value}")

    def filter_by_horizon(self, years: int, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
        if years < 0:
            raise ValueError("years cannot be negative")
        active_context = _coerce_context(context).model_copy(update={"investment_horizon_years": years})
        products = [product for product in self.repository.get_all() if product.recommended_horizon <= years]
        return self._rank(products, active_context, fixed_reason=f"Fits a horizon of {years} years or less")

    def filter_by_liquidity(self, liquidity: LiquidityLevel | str, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
        required_liquidity = _coerce_enum(liquidity, LiquidityLevel, "liquidity")
        active_context = _coerce_context(context).model_copy(update={"minimum_liquidity": required_liquidity})
        products = [
            product for product in self.repository.get_all()
            if LIQUIDITY_RANK[product.liquidity] >= LIQUIDITY_RANK[required_liquidity]
        ]
        return self._rank(products, active_context, fixed_reason=f"Meets {required_liquidity.value.lower()} liquidity")

    def filter_by_beginner(self, suitable: bool = True, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
        active_context = _coerce_context(context).model_copy(update={"beginner": suitable})
        products = [product for product in self.repository.get_all() if product.suitable_for_beginners == suitable]
        return self._rank(products, active_context, fixed_reason=f"Beginner suitability is {suitable}")

    def filter_by_government_backed(self, government_backed: bool = True, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
        active_context = _coerce_context(context).model_copy(update={"government_backed": government_backed})
        products = [product for product in self.repository.get_all() if product.government_backed == government_backed]
        return self._rank(products, active_context, fixed_reason=f"Government-backed flag is {government_backed}")

    def filter_by_credit_score(self, credit_score: int, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
        if not 0 <= credit_score <= 900:
            raise ValueError("credit_score must be between 0 and 900")
        active_context = _coerce_context(context).model_copy(update={"credit_score": credit_score})
        products = [product for product in self.repository.get_all() if product.credit_score_requirement <= credit_score]
        return self._rank(products, active_context, fixed_reason="Meets supplied credit-score eligibility")

    def search_products(
        self,
        query: str,
        context: RetrievalContext | Mapping[str, object] | None = None,
        *,
        limit: int | None = 10,
    ) -> list[RetrievedProduct]:
        """Search product metadata, adding transparent intent hints for common phrases."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero when supplied")
        terms = set(re.findall(r"[a-z0-9]+", normalized_query.casefold()))
        active_context = _merge_context(_coerce_context(context), _infer_context(normalized_query))
        candidates = [product for product in self.repository.get_all() if _text_match(product, terms)[0] > 0]
        ranked = self._rank(candidates, active_context, query_terms=terms, fixed_reason=f"Matches search: {normalized_query}")
        return ranked if limit is None else ranked[:limit]

    def _rank(
        self,
        products: Iterable[InvestmentProduct],
        context: RetrievalContext,
        *,
        query_terms: set[str] | None = None,
        fixed_reason: str | None = None,
    ) -> list[RetrievedProduct]:
        candidates = [self.ranker.score(product, context, query_terms=query_terms) for product in products]
        if fixed_reason:
            candidates = [
                RankedCandidate(candidate.product, candidate.score, (fixed_reason, *candidate.reasons))
                for candidate in candidates
            ]
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.product.name))
        return [self._to_result(candidate) for candidate in candidates]

    @staticmethod
    def _to_result(candidate: RankedCandidate) -> RetrievedProduct:
        product = candidate.product
        suitable_for = (
            *(f"Persona: {persona.value}" for persona in product.behavioral_personas),
            *(f"Goal: {goal.value}" for goal in product.supported_goals),
        )
        return RetrievedProduct(
            product=product,
            relevance_score=candidate.score,
            why_recommended=candidate.reasons,
            suitable_for=suitable_for,
            potential_risks=product.potential_risks,
            expected_return=(
                f"Illustrative annual range: {product.expected_annual_return_min:.1f}% to "
                f"{product.expected_annual_return_max:.1f}%"
            ),
            investment_horizon=f"{product.recommended_horizon} years or longer",
            simple_explanation=product.plain_language_description,
        )


def _text_match(product: InvestmentProduct, terms: set[str]) -> tuple[float, list[str]]:
    searchable = " ".join(
        (
            product.name,
            product.category.value,
            product.subcategory,
            product.description,
            product.plain_language_description,
            " ".join(product.search_terms),
            " ".join(goal.value for goal in product.supported_goals),
        )
    ).casefold()
    matched = {term for term in terms if term in searchable}
    if not matched:
        return 0.0, []
    exact_name = " ".join(sorted(terms)) in product.name.casefold()
    score = min(30.0, 8.0 * len(matched) + (10.0 if exact_name else 0.0))
    return score, ["Text match: " + ", ".join(sorted(matched))]


def _infer_context(query: str) -> RetrievalContext:
    normalized = query.casefold()
    updates: dict[str, object] = {}
    if "safe investment" in normalized or "safe" in normalized:
        updates["risk_profile"] = RiskLevel.LOW
    if "retirement" in normalized or "pension" in normalized:
        updates["goals"] = (InvestmentGoal.RETIREMENT,)
    elif "tax" in normalized:
        updates["goals"] = (InvestmentGoal.TAX_SAVING,)
    elif "monthly income" in normalized or "passive income" in normalized:
        updates["goals"] = (InvestmentGoal.PASSIVE_INCOME,)
    elif "emergency" in normalized:
        updates["goals"] = (InvestmentGoal.EMERGENCY_FUND,)
        updates["minimum_liquidity"] = LiquidityLevel.HIGH
    return RetrievalContext(**updates)


def _merge_context(primary: RetrievalContext, inferred: RetrievalContext) -> RetrievalContext:
    updates: dict[str, object] = {}
    for field in RetrievalContext.model_fields:
        primary_value = getattr(primary, field)
        inferred_value = getattr(inferred, field)
        if primary_value not in (None, ()):
            updates[field] = primary_value
        elif inferred_value not in (None, ()):
            updates[field] = inferred_value
    return RetrievalContext(**updates)


def _coerce_context(context: RetrievalContext | Mapping[str, object] | None) -> RetrievalContext:
    if context is None:
        return RetrievalContext()
    if isinstance(context, RetrievalContext):
        return context
    return RetrievalContext.model_validate(context)


def _coerce_enum(value: object, enum_type: type[RiskLevel] | type[InvestmentGoal] | type[BehavioralPersonaName] | type[LiquidityLevel], label: str):
    if isinstance(value, enum_type):
        return value
    normalized = str(value).strip().casefold()
    for candidate in enum_type:
        if candidate.value.casefold() == normalized or candidate.name.casefold() == normalized.replace(" ", "_"):
            return candidate
    valid = ", ".join(candidate.value for candidate in enum_type)
    raise ValueError(f"Unsupported {label}: {value!r}. Valid values: {valid}")


_DEFAULT_ENGINE: InvestmentRetrievalEngine | None = None


def _engine() -> InvestmentRetrievalEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = InvestmentRetrievalEngine()
    return _DEFAULT_ENGINE


def get_all_products(context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
    return _engine().get_all_products(context)


def get_product_by_id(product_id: str) -> RetrievedProduct | None:
    return _engine().get_product_by_id(product_id)


def filter_by_risk(risk_level: RiskLevel | str, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
    return _engine().filter_by_risk(risk_level, context)


def filter_by_goal(goal: InvestmentGoal | str, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
    return _engine().filter_by_goal(goal, context)


def filter_by_persona(persona: BehavioralPersonaName | str, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
    return _engine().filter_by_persona(persona, context)


def filter_by_horizon(years: int, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
    return _engine().filter_by_horizon(years, context)


def filter_by_liquidity(liquidity: LiquidityLevel | str, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
    return _engine().filter_by_liquidity(liquidity, context)


def filter_by_beginner(suitable: bool = True, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
    return _engine().filter_by_beginner(suitable, context)


def filter_by_government_backed(government_backed: bool = True, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
    return _engine().filter_by_government_backed(government_backed, context)


def filter_by_credit_score(credit_score: int, context: RetrievalContext | Mapping[str, object] | None = None) -> list[RetrievedProduct]:
    return _engine().filter_by_credit_score(credit_score, context)


def search_products(
    query: str,
    context: RetrievalContext | Mapping[str, object] | None = None,
    *,
    limit: int | None = 10,
) -> list[RetrievedProduct]:
    return _engine().search_products(query, context, limit=limit)
