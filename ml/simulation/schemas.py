"""Typed API contracts for the Phase 7 projection and Monte Carlo engine."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ml.recommendation.schemas import PortfolioStyle, RecommendationResponse


class ProjectionRequest(BaseModel):
    """Validated simulation inputs consumable from FastAPI or Phase 6 callers."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    initial_investment: float = Field(ge=0, description="Lump-sum amount already invested or allocated")
    monthly_sip: int = Field(ge=0, description="Recurring monthly contribution in whole rupees")
    investment_horizon_years: int = Field(ge=1, le=60)
    goal_amount: float | None = Field(default=None, ge=0)
    random_seed: int = Field(default=2026, ge=0)
    num_simulations: int = Field(default=10_000, ge=100, le=100_000)
    inflation_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Override Phase 5 inflation assumption; decimal form e.g. 0.055",
    )
    recommendation: RecommendationResponse
    generate_charts: bool = True
    generate_reports: bool = False
    active_scenario: PortfolioStyle = PortfolioStyle.RECOMMENDED


class DistributionMetrics(BaseModel):
    """Summary statistics over simulated terminal portfolio values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runs: int = Field(ge=1)
    mean_value: float = Field(ge=0)
    median_value: float = Field(ge=0)
    minimum_value: float = Field(ge=0)
    maximum_value: float = Field(ge=0)
    standard_deviation: float = Field(ge=0)
    best_case: float = Field(ge=0, description="90th percentile terminal value")
    worst_case: float = Field(ge=0, description="10th percentile terminal value")
    percentile_10: float = Field(ge=0)
    percentile_25: float = Field(ge=0)
    percentile_50: float = Field(ge=0)
    percentile_75: float = Field(ge=0)
    percentile_90: float = Field(ge=0)


class GoalProbabilityResult(BaseModel):
    """Educational goal attainment probabilities from simulated paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal_amount: float = Field(ge=0)
    probability_of_success: float = Field(ge=0, le=100)
    probability_of_shortfall: float = Field(ge=0, le=100)
    probability_of_exceeding: float = Field(ge=0, le=100)


class InflationAdjustedResult(BaseModel):
    """Nominal and inflation-adjusted terminal values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nominal: float = Field(ge=0)
    real: float = Field(ge=0)
    inflation_rate_used: float = Field(ge=0, le=1)
    horizon_years: int = Field(ge=1)


class PortfolioAnalyticsResult(BaseModel):
    """Illustrative risk/return analytics derived from simulated paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_cagr: float
    volatility: float = Field(ge=0, description="Annualized volatility percentage")
    sharpe_ratio: float
    max_drawdown: float = Field(ge=0, description="Maximum drawdown percentage on median path")
    downside_risk: float = Field(ge=0, description="Downside deviation percentage")
    upside_capture: float = Field(description="Illustrative upside participation ratio")


class YearlyProjectionPoint(BaseModel):
    """One year on the simulated projection timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int = Field(ge=0)
    median: float = Field(ge=0)
    best_case: float = Field(ge=0, description="90th percentile")
    worst_case: float = Field(ge=0, description="10th percentile")
    expected_range_low: float = Field(ge=0, description="25th percentile")
    expected_range_high: float = Field(ge=0, description="75th percentile")
    nominal_median: float = Field(ge=0)
    real_median: float = Field(ge=0)


class ScenarioSimulationResult(BaseModel):
    """Independent Monte Carlo output for one portfolio style."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    style: PortfolioStyle
    expected_return_range: str
    expected_annual_return_min: float
    expected_annual_return_max: float
    estimated_risk: float = Field(ge=0)
    simulation: DistributionMetrics
    goal_probability: GoalProbabilityResult | None = None
    inflation_adjusted: InflationAdjustedResult
    analytics: PortfolioAnalyticsResult
    yearly_projections: tuple[YearlyProjectionPoint, ...]


class ChartArtifacts(BaseModel):
    """Relative plot paths saved under ml/simulation/plots/."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    portfolio_growth: str | None = None
    simulation_fan: str | None = None
    distribution_histogram: str | None = None
    confidence_interval: str | None = None
    goal_probability: str | None = None
    scenario_comparison: str | None = None
    yearly_projection: str | None = None


class ProjectionResponse(BaseModel):
    """FastAPI-ready Phase 7 response contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation: DistributionMetrics
    goal_probability: GoalProbabilityResult | None = None
    inflation_adjusted: InflationAdjustedResult
    analytics: PortfolioAnalyticsResult
    yearly_projections: tuple[YearlyProjectionPoint, ...]
    scenarios: tuple[ScenarioSimulationResult, ...]
    active_scenario: PortfolioStyle
    explanation_summary: str
    educational_disclaimer: str
    simulation_version: str
    charts: ChartArtifacts | None = None
    reports: dict[str, str] | None = None
