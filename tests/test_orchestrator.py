"""Integration tests for the Phase 8 TrustVest Intelligence Orchestrator."""

from __future__ import annotations

import unittest

from ml.orchestrator import TrustVestIntelligenceOrchestrator, UnifiedAnalysisRequest, analyze_user
from ml.orchestrator.cache import OrchestratorCache
from ml.orchestrator.config import OrchestratorConfig
from ml.orchestrator.schemas import PipelineStageName, StageStatus


SAMPLE_REQUEST = {
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
    "simulation": {
        "enabled": True,
        "goal_amount": 2500000,
        "random_seed": 2026,
        "num_simulations": 500,
        "generate_charts": False,
        "generate_reports": False,
    },
    "use_cache": False,
}


class TrustVestIntelligenceOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OrchestratorConfig(cache_enabled=False, log_pipeline_trace=False)
        self.orchestrator = TrustVestIntelligenceOrchestrator(config=self.config)

    def test_analyze_user_returns_unified_contract(self) -> None:
        response = self.orchestrator.analyze_user(SAMPLE_REQUEST)

        self.assertTrue(response.request_id)
        self.assertIn("educational", response.educational_disclaimer.lower())
        self.assertEqual(response.pipeline_version, "v0.8.0")
        self.assertIsNotNone(response.credit_analysis)
        self.assertIsNotNone(response.risk_analysis)
        self.assertIsNotNone(response.portfolio_recommendation)
        self.assertIsNotNone(response.financial_projection)
        self.assertGreater(len(response.next_steps), 0)
        self.assertGreater(len(response.pipeline_trace), 0)
        stage_names = {entry.stage for entry in response.pipeline_trace}
        self.assertIn(PipelineStageName.GENERATE_RESPONSE, stage_names)

    def test_fault_tolerance_when_simulation_disabled(self) -> None:
        payload = {
            **SAMPLE_REQUEST,
            "simulation": {**SAMPLE_REQUEST["simulation"], "enabled": False},
        }
        response = self.orchestrator.analyze_user(payload)

        self.assertIsNone(response.financial_projection)
        self.assertIsNotNone(response.credit_analysis)
        self.assertIsNotNone(response.portfolio_recommendation)
        projection_trace = next(
            entry for entry in response.pipeline_trace if entry.stage == PipelineStageName.FINANCIAL_PROJECTION
        )
        self.assertEqual(projection_trace.status, StageStatus.SKIPPED)

    def test_cache_reuses_identical_requests(self) -> None:
        cached_config = OrchestratorConfig(cache_enabled=True, cache_max_entries=8, log_pipeline_trace=False)
        orchestrator = TrustVestIntelligenceOrchestrator(config=cached_config)
        request = UnifiedAnalysisRequest.model_validate({**SAMPLE_REQUEST, "use_cache": True})

        first = orchestrator.analyze_user(request)
        second = orchestrator.analyze_user(request)

        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))

    def test_cache_key_ignores_request_id(self) -> None:
        cache = OrchestratorCache(max_entries=4)
        first = UnifiedAnalysisRequest.model_validate({**SAMPLE_REQUEST, "request_id": "alpha"})
        second = UnifiedAnalysisRequest.model_validate({**SAMPLE_REQUEST, "request_id": "beta"})
        self.assertEqual(cache._cache_key(first), cache._cache_key(second))

    def test_module_entry_point(self) -> None:
        response = analyze_user({**SAMPLE_REQUEST, "use_cache": False})
        self.assertIsNotNone(response.risk_analysis)
        self.assertGreater(response.telemetry.total_pipeline_time_ms, 0)


if __name__ == "__main__":
    unittest.main()
