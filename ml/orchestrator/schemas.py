"""Typed API contracts for the Phase 8 TrustVest Intelligence Orchestrator."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from ml.recommendation.schemas import RecommendationResponse
from ml.simulation.schemas import ProjectionResponse


class PipelineStageName(str, Enum):
    VALIDATE_REQUEST = "validate_request"
    PREPROCESS_INPUT = "preprocess_input"
    CREDIT_SCORING = "credit_scoring"
    RISK_PROFILING = "risk_profiling"
    KNOWLEDGE_BASE_RETRIEVAL = "knowledge_base_retrieval"
    PORTFOLIO_RECOMMENDATION = "portfolio_recommendation"
    FINANCIAL_PROJECTION = "financial_projection"
    GENERATE_RESPONSE = "generate_response"


class StageStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class StageSkipped(Exception):
    """Raised when a pipeline stage is intentionally skipped."""


class PersonalInformation(BaseModel):
    """Demographic and identity fields supplied by the client."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    age: int = Field(ge=18, le=100)
    occupation: str = Field(min_length=2, max_length=120)
    gender: str | None = Field(default="Female", max_length=40)
    state: str | None = Field(default="Maharashtra", max_length=80)
    city_tier: str | None = Field(default="Tier 1", max_length=40)
    education: str | None = Field(default="Graduate", max_length=80)
    marital_status: str | None = Field(default="Single", max_length=40)
    employment_type: str | None = Field(default="Salaried", max_length=80)
    dependents: int = Field(default=0, ge=0, le=30)


class FinancialInformation(BaseModel):
    """Income, savings, and resilience inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    monthly_income: float = Field(gt=0)
    monthly_savings: float = Field(ge=0)
    monthly_budget: int = Field(ge=1_000, description="Whole rupees allocated to recurring investment")
    existing_savings: float = Field(ge=0)
    emergency_fund_months: float = Field(ge=0, le=60)
    income_stability: str = Field(description="High/Medium/Low or Stable/Moderate/Variable")
    fallback_credit_score: int | None = Field(
        default=None,
        ge=0,
        le=900,
        description="Optional fallback when the credit model is unavailable",
    )
    fallback_risk_profile: str | None = Field(
        default=None,
        description="Optional Low/Medium/High fallback when the risk model is unavailable",
    )
    fallback_behavioral_persona: str | None = Field(
        default=None,
        description="Optional persona override paired with fallback_risk_profile",
    )

    @model_validator(mode="after")
    def validate_capacity(self) -> "FinancialInformation":
        if self.monthly_budget > self.monthly_income:
            raise ValueError("monthly_budget cannot exceed monthly_income")
        if self.monthly_savings > self.monthly_income:
            raise ValueError("monthly_savings cannot exceed monthly_income")
        return self


class BehavioralQuestionnaire(BaseModel):
    """Phase 4-compatible behavioral questionnaire."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    investment_goal: str
    investment_horizon_years: int = Field(ge=0, le=60)
    expected_annual_return_percent: float = Field(ge=0, le=100)
    reaction_to_20_percent_loss: str
    previous_investment_experience: str
    mutual_fund_knowledge: str
    stock_knowledge: str
    preferred_liquidity: str
    financial_confidence: str
    preferred_investment_frequency: str
    preferred_investment_type: str


class InvestmentPreferences(BaseModel):
    """Optional portfolio preference overrides."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    government_backed_preference: bool = False
    liquidity_preference: str | None = None


class SimulationParameters(BaseModel):
    """Monte Carlo simulation controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    initial_investment: float | None = Field(default=None, ge=0)
    goal_amount: float | None = Field(default=None, ge=0)
    random_seed: int = Field(default=2026, ge=0)
    num_simulations: int = Field(default=10_000, ge=100, le=100_000)
    generate_charts: bool = False
    generate_reports: bool = False
    active_scenario: str = Field(default="Recommended")


class UnifiedAnalysisRequest(BaseModel):
    """Single entry-point request consumed by FastAPI and the frontend."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    personal: PersonalInformation
    financial: FinancialInformation
    behavioral: BehavioralQuestionnaire
    investment_preferences: InvestmentPreferences = Field(default_factory=InvestmentPreferences)
    simulation: SimulationParameters = Field(default_factory=SimulationParameters)
    knowledge_base_query: str | None = Field(default=None, max_length=200)
    use_cache: bool = True
    request_id: str | None = None

    @field_validator("request_id")
    @classmethod
    def default_request_id(cls, value: str | None) -> str:
        return value or str(uuid4())


class PipelineTraceEntry(BaseModel):
    """One auditable pipeline stage execution record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: PipelineStageName
    status: StageStatus
    duration_ms: float = Field(ge=0)
    output_summary: str | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()


class CreditAnalysisResult(BaseModel):
    """Phase 3 credit scoring output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    credit_score: float
    confidence: float
    top_positive_features: tuple[dict[str, Any], ...]
    top_negative_features: tuple[dict[str, Any], ...]
    source: str = Field(description="model or fallback")


class RiskAnalysisResult(BaseModel):
    """Phase 4 risk profiling output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    risk_profile: str
    persona: str
    persona_details: dict[str, Any]
    confidence: float
    probabilities: dict[str, float]
    recommendation_summary: str
    top_positive_factors: tuple[str, ...] = ()
    top_negative_factors: tuple[str, ...] = ()
    explanation_method: str | None = None


class KnowledgeBaseSummary(BaseModel):
    """Compact Phase 5 retrieval summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str | None
    product_count: int = Field(ge=0)
    top_products: tuple[dict[str, Any], ...]


class ModuleTelemetry(BaseModel):
    """Latency and status metrics for one module."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    module: str
    status: StageStatus
    latency_ms: float = Field(ge=0)


class TelemetrySummary(BaseModel):
    """Aggregate telemetry for one orchestration run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_pipeline_time_ms: float = Field(ge=0)
    module_metrics: tuple[ModuleTelemetry, ...]
    success_rate: float = Field(ge=0, le=100)


class UnifiedAnalysisResponse(BaseModel):
    """FastAPI-ready unified response contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    timestamp: datetime
    credit_analysis: CreditAnalysisResult | None = None
    risk_analysis: RiskAnalysisResult | None = None
    knowledge_base: KnowledgeBaseSummary | None = None
    portfolio_recommendation: RecommendationResponse | None = None
    financial_projection: ProjectionResponse | None = None
    pipeline_trace: tuple[PipelineTraceEntry, ...]
    telemetry: TelemetrySummary
    warnings: tuple[str, ...]
    next_steps: tuple[str, ...]
    educational_disclaimer: str
    pipeline_version: str


class AuditRecord(BaseModel):
    """Structured audit log entry for one orchestration run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    request_id: str
    pipeline_version: str
    models_used: tuple[str, ...]
    simulation_seed: int | None
    execution_time_ms: float = Field(ge=0)
    warnings: tuple[str, ...]
    stage_statuses: dict[str, str]
    educational_disclaimer: str
