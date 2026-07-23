"""TrustVest Intelligence Orchestrator facade and report generation."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from pathlib import Path

from .audit import append_audit_record, audit_record_to_dict, build_audit_record
from .cache import NullCache, OrchestratorCache
from .config import OrchestratorConfig
from .pipeline import AnalysisPipeline
from .response_builder import build_unified_response
from .schemas import UnifiedAnalysisRequest, UnifiedAnalysisResponse
from .telemetry import TelemetryCollector
from .validators import validate_analysis_request

LOGGER = logging.getLogger(__name__)


class TrustVestIntelligenceOrchestrator:
    """Single entry point that coordinates all TrustVest AI modules."""

    def __init__(self, *, config: OrchestratorConfig | None = None) -> None:
        self.config = config or OrchestratorConfig()
        self.pipeline = AnalysisPipeline(config=self.config)
        self.cache = (
            OrchestratorCache(max_entries=self.config.cache_max_entries)
            if self.config.cache_enabled
            else NullCache()
        )

    def analyze_user(self, request: UnifiedAnalysisRequest | Mapping[str, object]) -> UnifiedAnalysisResponse:
        """Primary synchronous entry point for FastAPI and backend callers."""
        parsed = validate_analysis_request(request)
        if parsed.use_cache and self.config.cache_enabled:
            cached = self.cache.get(parsed)
            if cached is not None:
                LOGGER.info("Returning cached orchestrator response for request_id=%s", parsed.request_id)
                return cached

        telemetry = TelemetryCollector()
        state = self.pipeline.run(parsed, telemetry)
        response = build_unified_response(
            parsed,
            config=self.config,
            pipeline_trace=tuple(state.pipeline_trace),
            telemetry=telemetry,
            credit_analysis=state.credit_analysis,
            risk_analysis=state.risk_analysis,
            knowledge_base=state.knowledge_base,
            portfolio_recommendation=state.portfolio_recommendation,
            financial_projection=state.financial_projection,
            warnings=tuple(state.warnings),
        )
        audit = build_audit_record(
            parsed,
            response,
            config=self.config,
            models_used=tuple(dict.fromkeys(state.models_used)),
            simulation_seed=parsed.simulation.random_seed if parsed.simulation.enabled else None,
        )
        if self.config.audit_log_enabled:
            append_audit_record(audit, path=self.config.audit_log_path)
        if self.config.log_pipeline_trace:
            LOGGER.info(
                "Pipeline trace request_id=%s stages=%s",
                response.request_id,
                [(entry.stage.value, entry.status.value) for entry in response.pipeline_trace],
            )
        LOGGER.info(
            "Completed orchestrator run request_id=%s duration_ms=%.2f success_rate=%.2f",
            response.request_id,
            response.telemetry.total_pipeline_time_ms,
            response.telemetry.success_rate,
        )
        if parsed.use_cache and self.config.cache_enabled:
            self.cache.set(parsed, response)
        return response

    async def analyze_user_async(self, request: UnifiedAnalysisRequest | Mapping[str, object]) -> UnifiedAnalysisResponse:
        """Async wrapper suitable for FastAPI async routes."""
        return await asyncio.to_thread(self.analyze_user, request)


def analyze_user(request: UnifiedAnalysisRequest | Mapping[str, object]) -> UnifiedAnalysisResponse:
    """Convenience function for scripts, FastAPI dependencies, and report generation."""
    return TrustVestIntelligenceOrchestrator().analyze_user(request)


def generate_phase8_reports(config: OrchestratorConfig | None = None) -> dict[str, str]:
    """Generate the four requested Phase 8 reproducible report artifacts."""
    active_config = config or OrchestratorConfig()
    active_config.create_output_directories()
    orchestrator = TrustVestIntelligenceOrchestrator(config=active_config)
    sample_request = _sample_request()
    response = orchestrator.analyze_user(sample_request)
    audit = build_audit_record(
        sample_request,
        response,
        config=active_config,
        models_used=("credit_model", "risk_model", "recommendation_engine", "simulation_engine"),
        simulation_seed=sample_request.simulation.random_seed,
    )
    metrics = {
        "pipeline_version": active_config.pipeline_version,
        "telemetry": response.telemetry.model_dump(mode="json"),
        "pipeline_trace": [entry.model_dump(mode="json") for entry in response.pipeline_trace],
        "warnings": list(response.warnings),
        "next_steps": list(response.next_steps),
        "educational_disclaimer": active_config.educational_disclaimer,
    }
    sample_payload = response.model_dump(mode="json")
    _write_json(active_config.pipeline_metrics_path, metrics)
    _write_json(active_config.sample_responses_path, [sample_payload])
    _write_json(active_config.audit_log_example_path, audit_record_to_dict(audit))
    _write_text(active_config.pipeline_summary_path, _pipeline_summary_markdown(response, active_config))
    report_paths = {
        "pipeline_summary": str(active_config.pipeline_summary_path),
        "pipeline_metrics": str(active_config.pipeline_metrics_path),
        "sample_responses": str(active_config.sample_responses_path),
        "audit_log_example": str(active_config.audit_log_example_path),
    }
    LOGGER.info("Generated Phase 8 reports in %s", active_config.reports_dir)
    return report_paths


def _sample_request() -> UnifiedAnalysisRequest:
    return UnifiedAnalysisRequest.model_validate(
        {
            "personal": {
                "age": 31,
                "occupation": "Salaried professional",
                "gender": "Female",
                "state": "Maharashtra",
                "city_tier": "Tier 1",
                "education": "Graduate",
                "marital_status": "Single",
                "dependents": 1,
            },
            "financial": {
                "monthly_income": 85000,
                "monthly_savings": 22000,
                "monthly_budget": 10000,
                "existing_savings": 350000,
                "emergency_fund_months": 5,
                "income_stability": "High",
                "fallback_credit_score": 760,
            },
            "behavioral": {
                "investment_goal": "Wealth Creation",
                "investment_horizon_years": 10,
                "expected_annual_return_percent": 12,
                "reaction_to_20_percent_loss": "Hold and wait",
                "previous_investment_experience": "Intermediate",
                "mutual_fund_knowledge": "Good",
                "stock_knowledge": "Basic",
                "preferred_liquidity": "Flexible",
                "financial_confidence": "High",
                "preferred_investment_frequency": "Monthly",
                "preferred_investment_type": "Balanced mutual funds",
            },
            "investment_preferences": {"government_backed_preference": False},
            "simulation": {
                "enabled": True,
                "goal_amount": 2500000,
                "random_seed": 2026,
                "num_simulations": 1000,
                "generate_charts": False,
                "generate_reports": False,
            },
            "knowledge_base_query": "retirement",
            "use_cache": False,
        }
    )


def _pipeline_summary_markdown(response: UnifiedAnalysisResponse, config: OrchestratorConfig) -> str:
    trace_rows = "\n".join(
        f"| {entry.stage.value} | {entry.status.value} | {entry.duration_ms:.2f} | {entry.output_summary or '-'} |"
        for entry in response.pipeline_trace
    )
    return f"""# Phase 8 Pipeline Summary

## Scope

The TrustVest Intelligence Orchestrator coordinates credit scoring, risk profiling, knowledge-base retrieval, portfolio recommendation, and financial projection modules. It does not perform financial calculations itself and does not provide regulated investment advice.

## Run Summary

- Request ID: {response.request_id}
- Pipeline version: {config.pipeline_version}
- Total time: {response.telemetry.total_pipeline_time_ms:.2f} ms
- Success rate: {response.telemetry.success_rate:.2f}%

## Pipeline Trace

| Stage | Status | Duration (ms) | Summary |
| --- | --- | ---: | --- |
{trace_rows}

## Next Steps

{chr(10).join(f"- {step}" for step in response.next_steps)}

## Disclaimer

{config.educational_disclaimer}
"""


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: Path, contents: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


if __name__ == "__main__":
    generate_phase8_reports()
