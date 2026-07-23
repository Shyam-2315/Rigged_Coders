"""Strongly typed investment-product schemas used by the knowledge base."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class BehavioralPersonaName(str, Enum):
    SECURE_SAVER = "Secure Saver"
    CONSERVATIVE_PLANNER = "Conservative Planner"
    BALANCED_BUILDER = "Balanced Builder"
    STRATEGIC_INVESTOR = "Strategic Investor"
    GROWTH_EXPLORER = "Growth Explorer"
    AGGRESSIVE_WEALTH_SEEKER = "Aggressive Wealth Seeker"


class InvestmentGoal(str, Enum):
    EMERGENCY_FUND = "Emergency Fund"
    VACATION = "Vacation"
    EDUCATION = "Education"
    MARRIAGE = "Marriage"
    CAR = "Car"
    HOUSE = "House"
    WEALTH_CREATION = "Wealth Creation"
    RETIREMENT = "Retirement"
    PASSIVE_INCOME = "Passive Income"
    TAX_SAVING = "Tax Saving"
    CHILD_EDUCATION = "Child Education"


class ProductCategory(str, Enum):
    CASH = "Cash & Savings"
    DEBT = "Debt Mutual Fund"
    FIXED_INCOME = "Fixed Income"
    GOVERNMENT_SAVINGS = "Government Savings"
    EQUITY = "Indian Equity"
    COMMODITY = "Commodity"
    REAL_ASSET = "Real Asset"
    INTERNATIONAL_EQUITY = "International Equity"


class LiquidityLevel(str, Enum):
    INSTANT = "Instant"
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"


class TaxEfficiency(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class NumericRange(BaseModel):
    """An inclusive, validated range used for age and income suitability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum: float = Field(ge=0)
    maximum: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "NumericRange":
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot be greater than maximum")
        return self


class InvestmentProduct(BaseModel):
    """Validated, explainable metadata for one investment product type.

    Return and volatility figures are illustrative planning assumptions rather
    than quotes, guarantees, or forecasts for a specific fund or institution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    name: str = Field(min_length=3, max_length=120)
    category: ProductCategory
    subcategory: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=20)
    plain_language_description: str = Field(min_length=20)
    risk_level: RiskLevel
    behavioral_personas: tuple[BehavioralPersonaName, ...] = Field(min_length=1)
    supported_goals: tuple[InvestmentGoal, ...] = Field(min_length=1)
    minimum_investment: float = Field(ge=0)
    recommended_monthly_sip: float = Field(ge=0)
    expected_annual_return_min: float = Field(ge=-100, le=100)
    expected_annual_return_max: float = Field(ge=-100, le=100)
    historical_volatility: float = Field(ge=0, le=100)
    liquidity: LiquidityLevel
    withdrawal_time: str = Field(min_length=3, max_length=80)
    lock_in_period: int = Field(ge=0, le=720, description="Months")
    tax_efficiency: TaxEfficiency
    inflation_protection: bool
    expense_ratio: float = Field(ge=0, le=10)
    recommended_horizon: int = Field(ge=0, le=60, description="Years")
    ideal_age_range: NumericRange
    ideal_income_range: NumericRange
    credit_score_requirement: int = Field(ge=0, le=900, description="0 means no requirement")
    risk_profile_requirement: tuple[RiskLevel, ...] = Field(min_length=1)
    suitable_for_beginners: bool
    suitable_for_students: bool
    suitable_for_retirement: bool
    suitable_for_emergency_fund: bool
    shariah_compliant: bool
    government_backed: bool
    popularity_score: float = Field(ge=0, le=100)
    trust_score: float = Field(ge=0, le=100)
    potential_risks: tuple[str, ...] = Field(min_length=1)
    search_terms: tuple[str, ...] = Field(min_length=1)

    @field_validator("behavioral_personas", "supported_goals", "risk_profile_requirement", "search_terms")
    @classmethod
    def validate_unique_collection(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        if len(set(values)) != len(values):
            raise ValueError("Collection values must be unique")
        return values

    @model_validator(mode="after")
    def validate_financial_consistency(self) -> "InvestmentProduct":
        if self.expected_annual_return_min > self.expected_annual_return_max:
            raise ValueError("expected_annual_return_min cannot exceed expected_annual_return_max")
        if self.suitable_for_emergency_fund and (self.lock_in_period > 0 or self.liquidity not in {LiquidityLevel.INSTANT, LiquidityLevel.HIGH}):
            raise ValueError("Emergency-fund products must have no lock-in and high or instant liquidity")
        if self.government_backed and self.trust_score < 70:
            raise ValueError("Government-backed products must have a trust_score of at least 70")
        return self


class MarketAssumptions(BaseModel):
    """Illustrative inputs for future Monte Carlo simulation, not market forecasts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inflation_rate: float = Field(ge=0, le=1)
    risk_free_rate: float = Field(ge=0, le=1)
    expected_equity_premium: float = Field(ge=-1, le=1)
    expected_gold_return: float = Field(ge=-1, le=1)
    debt_return: float = Field(ge=-1, le=1)
    cash_return: float = Field(ge=-1, le=1)
    monte_carlo_default_volatility: float = Field(ge=0, le=1)
    assumption_version: str = Field(default="v1.0")
    note: str = Field(min_length=20)
