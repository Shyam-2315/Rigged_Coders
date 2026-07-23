"""Educational next-step suggestions and unified response assembly."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .config import OrchestratorConfig
from .schemas import (
    CreditAnalysisResult,
    KnowledgeBaseSummary,
    PipelineStageName,
    PipelineTraceEntry,
    RiskAnalysisResult,
    StageStatus,
    UnifiedAnalysisRequest,
    UnifiedAnalysisResponse,
)
from .telemetry import TelemetryCollector


def build_unified_response(
    request: UnifiedAnalysisRequest,
    *,
    config: OrchestratorConfig,
    pipeline_trace: tuple[PipelineTraceEntry, ...],
    telemetry: TelemetryCollector,
    credit_analysis: CreditAnalysisResult | None,
    risk_analysis: RiskAnalysisResult | None,
    knowledge_base: KnowledgeBaseSummary | None,
    portfolio_recommendation,
    financial_projection,
    warnings: tuple[str, ...],
) -> UnifiedAnalysisResponse:
    """Assemble the final FastAPI-ready orchestrator response."""
    combined_warnings = tuple(dict.fromkeys(warnings))
    next_steps = generate_next_steps(
        request, credit_analysis, risk_analysis, portfolio_recommendation, financial_projection
    )
    started = time.perf_counter()
    telemetry_summary = telemetry.build_summary(pipeline_trace)
    response_trace = pipeline_trace + (
        PipelineTraceEntry(
            stage=PipelineStageName.GENERATE_RESPONSE,
            status=StageStatus.SUCCESS,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            output_summary=f"next_steps={len(next_steps)} warnings={len(combined_warnings)}",
        ),
    )
    telemetry_summary = telemetry.build_summary(response_trace)
    return UnifiedAnalysisResponse(
        request_id=request.request_id or "unknown",
        timestamp=datetime.now(timezone.utc),
        credit_analysis=credit_analysis,
        risk_analysis=risk_analysis,
        knowledge_base=knowledge_base,
        portfolio_recommendation=portfolio_recommendation,
        financial_projection=financial_projection,
        pipeline_trace=response_trace,
        telemetry=telemetry_summary,
        warnings=combined_warnings,
        next_steps=next_steps,
        educational_disclaimer=config.educational_disclaimer,
        pipeline_version=config.pipeline_version,
    )


def generate_next_steps(
    request: UnifiedAnalysisRequest,
    credit_analysis: CreditAnalysisResult | None,
    risk_analysis: RiskAnalysisResult | None,
    portfolio_recommendation,
    financial_projection,
) -> tuple[str, ...]:
    """Generate educational, non-prescriptive follow-up suggestions."""
    steps: list[str] = []

    if request.financial.emergency_fund_months < 6:
        steps.append(
            "Increase your emergency fund toward at least six months of essential expenses before increasing your recurring investment amount."
        )
    if request.financial.monthly_budget > request.financial.monthly_savings * 0.8:
        steps.append(
            "Keep your monthly investment budget comfortably below monthly savings so recurring contributions remain sustainable."
        )
    if credit_analysis is not None and credit_analysis.credit_score < 650:
        steps.append(
            "Review bill-payment consistency, savings discipline, and existing obligations to strengthen your financial profile over time."
        )
    if risk_analysis is not None and risk_analysis.risk_profile == "High":
        steps.append(
            "Because your illustrated risk profile is growth-oriented, review concentration and volatility trade-offs before committing additional capital."
        )
    if portfolio_recommendation is not None:
        steps.append("Review the recommended educational portfolio annually and rebalance when your goals, income, or risk tolerance change.")
    if financial_projection is not None and financial_projection.goal_probability is not None:
        if financial_projection.goal_probability.probability_of_success < 60:
            steps.append(
                "If your simulated goal probability is lower than desired, consider extending the horizon, increasing SIP gradually with income, or revisiting the goal amount."
            )
    steps.extend(
        [
            "Increase SIP gradually as income grows rather than committing a large jump immediately.",
            "Diversify over time across asset classes rather than concentrating new contributions in one product type.",
            "Treat every TrustVest AI output as an educational illustration, not a transaction recommendation.",
        ]
    )
    return tuple(dict.fromkeys(steps))
