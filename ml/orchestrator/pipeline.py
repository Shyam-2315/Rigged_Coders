"""Fault-tolerant multi-stage pipeline for Phase 8 orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import joblib
import pandas as pd

from ml.investments.retrieval import search_products
from ml.preprocessing.feature_engineering import CreditFeatureEngineer
from ml.recommendation import recommend_portfolio
from ml.risk.inference import predict_with_explanation as predict_risk_with_explanation
from ml.simulation import project_portfolio
from ml.training.inference import predict_with_explanation as predict_credit_with_explanation

from ml.config import FeatureEngineeringConfig

from .config import OrchestratorConfig
from .schemas import (
    CreditAnalysisResult,
    KnowledgeBaseSummary,
    PipelineStageName,
    PipelineTraceEntry,
    RiskAnalysisResult,
    StageSkipped,
    StageStatus,
    UnifiedAnalysisRequest,
)
from .telemetry import TelemetryCollector
from .validators import (
    build_credit_raw_record,
    build_projection_request,
    build_recommendation_request,
    build_retrieval_context,
    build_risk_fallback,
    build_risk_questionnaire,
    validate_analysis_request,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Mutable state carried through the orchestrator pipeline."""

    request: UnifiedAnalysisRequest
    warnings: list[str] = field(default_factory=list)
    credit_analysis: CreditAnalysisResult | None = None
    risk_analysis: RiskAnalysisResult | None = None
    knowledge_base: KnowledgeBaseSummary | None = None
    portfolio_recommendation: Any | None = None
    financial_projection: Any | None = None
    credit_score: float | None = None
    risk_profile: str | None = None
    behavioral_persona: str | None = None
    models_used: list[str] = field(default_factory=list)
    pipeline_trace: list[PipelineTraceEntry] = field(default_factory=list)


class AnalysisPipeline:
    """Coordinates previously built TrustVest modules without performing calculations itself."""

    def __init__(self, *, config: OrchestratorConfig | None = None) -> None:
        self.config = config or OrchestratorConfig()

    def run(self, request: UnifiedAnalysisRequest, telemetry: TelemetryCollector) -> PipelineState:
        state = PipelineState(request=validate_analysis_request(request))
        stages: tuple[tuple[PipelineStageName, Callable[[PipelineState, TelemetryCollector], None]], ...] = (
            (PipelineStageName.VALIDATE_REQUEST, self._stage_validate_request),
            (PipelineStageName.PREPROCESS_INPUT, self._stage_preprocess_input),
            (PipelineStageName.CREDIT_SCORING, self._stage_credit_scoring),
            (PipelineStageName.RISK_PROFILING, self._stage_risk_profiling),
            (PipelineStageName.KNOWLEDGE_BASE_RETRIEVAL, self._stage_knowledge_base),
            (PipelineStageName.PORTFOLIO_RECOMMENDATION, self._stage_portfolio_recommendation),
            (PipelineStageName.FINANCIAL_PROJECTION, self._stage_financial_projection),
        )
        for stage_name, handler in stages:
            self._execute_stage(state, telemetry, stage_name, handler)
        return state

    def _execute_stage(
        self,
        state: PipelineState,
        telemetry: TelemetryCollector,
        stage: PipelineStageName,
        handler: Callable[[PipelineState, TelemetryCollector], None],
    ) -> None:
        started = time.perf_counter()
        status = StageStatus.SUCCESS
        output_summary: str | None = None
        error: str | None = None
        stage_warnings: list[str] = []
        try:
            with telemetry.track_module(stage.value):
                handler(state, telemetry)
            output_summary = self._stage_summary(state, stage)
        except StageSkipped as exc:
            status = StageStatus.SKIPPED
            output_summary = str(exc)
            stage_warnings.append(str(exc))
        except Exception as exc:  # noqa: BLE001 - fault-tolerant pipeline
            status = StageStatus.FAILED
            error = str(exc)
            stage_warnings.append(f"{stage.value} failed: {exc}")
            state.warnings.extend(stage_warnings)
            LOGGER.exception("Pipeline stage failed: %s", stage.value)
        duration_ms = (time.perf_counter() - started) * 1000.0
        state.pipeline_trace.append(
            PipelineTraceEntry(
                stage=stage,
                status=status,
                duration_ms=round(duration_ms, 3),
                output_summary=output_summary,
                error=error,
                warnings=tuple(stage_warnings),
            )
        )

    def _stage_validate_request(self, state: PipelineState, _telemetry: TelemetryCollector) -> None:
        validate_analysis_request(state.request)

    def _stage_preprocess_input(self, state: PipelineState, _telemetry: TelemetryCollector) -> None:
        build_risk_questionnaire(state.request)
        build_credit_raw_record(state.request)

    def _stage_credit_scoring(self, state: PipelineState, _telemetry: TelemetryCollector) -> None:
        try:
            processed = self._prepare_credit_features(state.request)
            result = predict_credit_with_explanation(processed)
            state.models_used.append("credit_model")
            state.credit_analysis = CreditAnalysisResult(
                credit_score=float(result["credit_score"]),
                confidence=float(result["confidence"]),
                top_positive_features=tuple(result["top_positive_features"]),
                top_negative_features=tuple(result["top_negative_features"]),
                source="model",
            )
            state.credit_score = float(result["credit_score"])
        except Exception as exc:
            fallback = state.request.financial.fallback_credit_score
            if fallback is None:
                raise RuntimeError(f"Credit scoring unavailable and no fallback_credit_score supplied: {exc}") from exc
            state.warnings.append(f"Credit model unavailable; using fallback_credit_score={fallback}.")
            state.credit_analysis = CreditAnalysisResult(
                credit_score=float(fallback),
                confidence=0.50,
                top_positive_features=(),
                top_negative_features=(),
                source="fallback",
            )
            state.credit_score = float(fallback)

    def _stage_risk_profiling(self, state: PipelineState, _telemetry: TelemetryCollector) -> None:
        try:
            questionnaire = build_risk_questionnaire(state.request)
            result = predict_risk_with_explanation(questionnaire)
            state.models_used.append("risk_model")
            state.risk_analysis = RiskAnalysisResult(
                risk_profile=result["risk_profile"],
                persona=result["persona"],
                persona_details=result["persona_details"],
                confidence=float(result["confidence"]),
                probabilities=result["probabilities"],
                recommendation_summary=result["recommendation_summary"],
                top_positive_factors=tuple(result.get("top_positive_factors", ())),
                top_negative_factors=tuple(result.get("top_negative_factors", ())),
                explanation_method=result.get("explanation_method"),
            )
        except Exception as exc:
            state.warnings.append(f"Risk model unavailable; using questionnaire heuristic fallback: {exc}")
            state.risk_analysis = build_risk_fallback(state.request)
        state.risk_profile = state.risk_analysis.risk_profile
        state.behavioral_persona = state.risk_analysis.persona

    def _stage_knowledge_base(self, state: PipelineState, _telemetry: TelemetryCollector) -> None:
        query = state.request.knowledge_base_query or state.request.behavioral.investment_goal
        context = build_retrieval_context(
            state.request,
            credit_score=state.credit_score,
            risk_profile=state.risk_profile,
            persona=state.behavioral_persona,
        )
        try:
            matches = search_products(query, context=context.model_dump(mode="json"))
        except Exception as exc:
            state.warnings.append(f"Knowledge base retrieval failed: {exc}")
            state.knowledge_base = KnowledgeBaseSummary(query=query, product_count=0, top_products=())
            return
        top_products = tuple(
            {
                "product_id": item.product.id,
                "name": item.product.name,
                "relevance_score": item.relevance_score,
                "expected_return": item.expected_return,
                "simple_explanation": item.simple_explanation,
            }
            for item in matches[:5]
        )
        state.knowledge_base = KnowledgeBaseSummary(
            query=query,
            product_count=len(matches),
            top_products=top_products,
        )

    def _stage_portfolio_recommendation(self, state: PipelineState, _telemetry: TelemetryCollector) -> None:
        if state.credit_score is None or state.risk_profile is None or state.behavioral_persona is None:
            raise RuntimeError("Portfolio recommendation requires credit score and risk profile outputs")
        recommendation_request = build_recommendation_request(
            state.request,
            credit_score=state.credit_score,
            risk_profile=state.risk_profile,
            behavioral_persona=state.behavioral_persona,
        )
        state.portfolio_recommendation = recommend_portfolio(recommendation_request)
        state.models_used.append("recommendation_engine")

    def _stage_financial_projection(self, state: PipelineState, _telemetry: TelemetryCollector) -> None:
        if not state.request.simulation.enabled:
            raise StageSkipped("Financial projection skipped because simulation.enabled=false")
        if state.portfolio_recommendation is None:
            raise RuntimeError("Financial projection requires a portfolio recommendation")
        projection_request = build_projection_request(
            state.request,
            state.portfolio_recommendation.model_dump(mode="json", by_alias=True),
            simulation_seed=self.config.simulation_seed,
        )
        state.financial_projection = project_portfolio(projection_request)
        state.models_used.append("simulation_engine")

    def _prepare_credit_features(self, request: UnifiedAnalysisRequest) -> dict[str, float]:
        raw = build_credit_raw_record(request)
        engineer = CreditFeatureEngineer(FeatureEngineeringConfig())
        engineered = engineer.transform(raw).frame
        if self.config.scaler_path.exists() and self.config.encoder_path.exists():
            scaler = joblib.load(self.config.scaler_path)
            encoder = joblib.load(self.config.encoder_path)
            numeric_cols = [column for column in engineered.columns if engineered[column].dtype != "object"]
            categorical_cols = [column for column in engineered.columns if engineered[column].dtype == "object"]
            scaled = engineered.copy()
            if numeric_cols:
                scaled.loc[:, numeric_cols] = scaler.transform(engineered[numeric_cols])
            if categorical_cols:
                encoded = encoder.transform(engineered[categorical_cols])
                encoded_frame = pd.DataFrame(
                    encoded,
                    columns=encoder.get_feature_names_out(categorical_cols),
                    index=engineered.index,
                )
                scaled = pd.concat([scaled.drop(columns=categorical_cols), encoded_frame], axis=1)
            feature_map = scaled.iloc[0].to_dict()
        else:
            feature_map = {key: float(value) for key, value in engineered.select_dtypes(include="number").iloc[0].items()}
        return {str(key): float(value) for key, value in feature_map.items()}

    def _stage_summary(self, state: PipelineState, stage: PipelineStageName) -> str | None:
        if stage == PipelineStageName.CREDIT_SCORING and state.credit_analysis:
            return f"credit_score={state.credit_analysis.credit_score:.2f} source={state.credit_analysis.source}"
        if stage == PipelineStageName.RISK_PROFILING and state.risk_analysis:
            return f"risk_profile={state.risk_analysis.risk_profile} persona={state.risk_analysis.persona}"
        if stage == PipelineStageName.KNOWLEDGE_BASE_RETRIEVAL and state.knowledge_base:
            return f"products={state.knowledge_base.product_count}"
        if stage == PipelineStageName.PORTFOLIO_RECOMMENDATION and state.portfolio_recommendation:
            return f"portfolio_score={state.portfolio_recommendation.portfolio_score}"
        if stage == PipelineStageName.FINANCIAL_PROJECTION and state.financial_projection:
            return f"median_value={state.financial_projection.simulation.median_value:.0f}"
        if stage == PipelineStageName.PREPROCESS_INPUT:
            return "mapped unified request to module contracts"
        if stage == PipelineStageName.VALIDATE_REQUEST:
            return "request validated"
        return None
