"""Typed API contracts for the Phase 6 recommendation engine."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.investments.schemas import (
    BehavioralPersonaName,
    InvestmentGoal,
    LiquidityLevel,
    RiskLevel,
)


class IncomeStability(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class PortfolioStyle(str, Enum):
    CONSERVATIVE = "Conservative"
    RECOMMENDED = "Recommended"
    GROWTH = "Growth"


class AssetClass(str, Enum):
    EQUITY = "Equity"
    DEBT = "Debt"
    GOLD = "Gold"
    CASH = "Cash"
    GOVERNMENT_SAVINGS = "Government Savings"
    REAL_ASSETS = "Real Assets"


class RecommendationRequest(BaseModel):
    """Validated user context supplied by the prior TrustVest phases or FastAPI."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    credit_score: int = Field(ge=0, le=900)
    risk_profile: RiskLevel
    behavioral_persona: BehavioralPersonaName
    monthly_income: float = Field(gt=0)
    monthly_savings: float = Field(ge=0)
    monthly_budget: int = Field(ge=1_000, description="Whole rupees available for a recurring monthly investment")
    existing_savings: float = Field(ge=0)
    investment_goal: InvestmentGoal
    investment_horizon: int = Field(ge=0, le=60, description="Years")
    age: int = Field(ge=18, le=100)
    dependents: int = Field(ge=0, le=30)
    emergency_fund_months: float = Field(ge=0, le=60)
    income_stability: IncomeStability
    liquidity_preference: LiquidityLevel | None = None
    government_backed_preference: bool = False

    @model_validator(mode="after")
    def validate_financial_capacity(self) -> "RecommendationRequest":
        if self.monthly_budget > self.monthly_income:
            raise ValueError("monthly_budget cannot exceed monthly_income")
        return self


class ProductScoreBreakdown(BaseModel):
    """The 0-100 factor values that produced a product's weighted score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal_match: float = Field(ge=0, le=100)
    risk_match: float = Field(ge=0, le=100)
    persona_match: float = Field(ge=0, le=100)
    expected_return: float = Field(ge=0, le=100)
    liquidity: float = Field(ge=0, le=100)
    tax_efficiency: float = Field(ge=0, le=100)
    inflation_protection: float = Field(ge=0, le=100)
    trust_score: float = Field(ge=0, le=100)
    popularity: float = Field(ge=0, le=100)
    diversification_value: float = Field(ge=0, le=100)
    budget_compatibility: float = Field(ge=0, le=100)


class PortfolioAllocation(BaseModel):
    """One product's allocation and the deterministic calculation trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product: str
    product_id: str
    asset_class: AssetClass
    allocation: float = Field(ge=5, le=60, description="Percent of monthly budget")
    monthly_amount: int = Field(ge=0)
    expected_annual_return_min: float
    expected_annual_return_max: float
    risk_contribution: float = Field(ge=0)
    liquidity_contribution: float = Field(ge=0, le=100)
    selection_score: float = Field(ge=0, le=100)
    score_breakdown: ProductScoreBreakdown
    reason: str
    potential_risks: tuple[str, ...]


class PortfolioHealth(BaseModel):
    """Transparent, rule-based health metrics for a portfolio illustration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    diversification_score: int = Field(ge=0, le=100, serialization_alias="Diversification")
    risk_alignment_score: int = Field(ge=0, le=100, serialization_alias="Risk Alignment")
    liquidity_score: int = Field(ge=0, le=100, serialization_alias="Liquidity")
    inflation_protection_score: int = Field(ge=0, le=100, serialization_alias="Inflation Protection")
    goal_alignment_score: int = Field(ge=0, le=100, serialization_alias="Goal Alignment")
    tax_efficiency_score: int = Field(ge=0, le=100, serialization_alias="Tax Efficiency")
    portfolio_stability_score: int = Field(ge=0, le=100, serialization_alias="Portfolio Stability")
    overall_portfolio_score: int = Field(ge=0, le=100, serialization_alias="Overall Portfolio Score")


class PortfolioVariant(BaseModel):
    """An allocation variant at a stated conservative/recommended/growth tilt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    style: PortfolioStyle
    expected_return_range: str
    expected_annual_return_min: float
    expected_annual_return_max: float
    estimated_risk: float = Field(ge=0, le=100, description="Illustrative volatility percentage")
    risk_level: RiskLevel
    allocation: tuple[PortfolioAllocation, ...]
    portfolio_health: PortfolioHealth


class AlternativePortfolios(BaseModel):
    """The three deterministic allocation views generated from the same inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conservative: PortfolioVariant
    recommended: PortfolioVariant
    growth: PortfolioVariant


class RecommendationResponse(BaseModel):
    """FastAPI-ready, Monte-Carlo-compatible Phase 6 response contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    portfolio_score: int = Field(ge=0, le=100)
    expected_return_range: str
    expected_annual_return_min: float
    expected_annual_return_max: float
    estimated_risk: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    monthly_budget: int = Field(ge=0)
    recommended_portfolio: tuple[PortfolioAllocation, ...]
    portfolio_health: PortfolioHealth
    alternative_portfolios: AlternativePortfolios
    insights: tuple[str, ...]
    explanation_summary: str
    trade_offs: tuple[str, ...]
    rules_version: str
    educational_disclaimer: str
