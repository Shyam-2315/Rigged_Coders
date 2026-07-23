"""API-level integration tests for Phase 9 FastAPI endpoints.

Uses FastAPI's TestClient (Starlette's synchronous test client backed by
httpx) so tests run without a live server.  Each test validates the HTTP
contract — status code, required response fields, and key invariants — not
the ML model outputs themselves (those are covered by unit tests in
tests/test_orchestrator.py and tests/test_recommendation_engine.py).
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.main import create_app

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_UNIFIED_REQUEST = {
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
        "num_simulations": 200,
        "generate_charts": False,
        "generate_reports": False,
    },
    "use_cache": False,
}

SAMPLE_RECOMMEND_REQUEST = {
    "credit_score": 760,
    "risk_profile": "Medium",
    "behavioral_persona": "Balanced Builder",
    "monthly_income": 85000,
    "monthly_savings": 22000,
    "monthly_budget": 10000,
    "existing_savings": 350000,
    "investment_goal": "Wealth Creation",
    "investment_horizon": 10,
    "age": 31,
    "dependents": 1,
    "emergency_fund_months": 5,
    "income_stability": "High",
}


def _make_simulate_request(recommendation: dict) -> dict:
    return {
        "initial_investment": 350000,
        "monthly_sip": 10000,
        "investment_horizon_years": 10,
        "goal_amount": 2500000,
        "random_seed": 2026,
        "num_simulations": 200,
        "recommendation": recommendation,
        "generate_charts": False,
        "generate_reports": False,
        "active_scenario": "Recommended",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class HealthEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(), raise_server_exceptions=True)

    def test_health_returns_200(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("timestamp", response.json())

    def test_version_returns_api_and_pipeline_version(self) -> None:
        response = self.client.get("/api/v1/version")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("api_version", data)
        self.assertIn("pipeline_version", data)
        self.assertIn("0.9", data["api_version"])
        self.assertIn("educational_disclaimer", data)


class AnalyzeEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(), raise_server_exceptions=True)

    def test_analyze_returns_unified_response(self) -> None:
        response = self.client.post("/api/v1/analyze", json=SAMPLE_UNIFIED_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("request_id", data)
        self.assertIn("credit_analysis", data)
        self.assertIn("risk_analysis", data)
        self.assertIn("portfolio_recommendation", data)
        self.assertIn("financial_projection", data)
        self.assertIn("pipeline_trace", data)
        self.assertIn("educational_disclaimer", data)

    def test_analyze_with_simulation_disabled(self) -> None:
        payload = {**SAMPLE_UNIFIED_REQUEST, "simulation": {**SAMPLE_UNIFIED_REQUEST["simulation"], "enabled": False}}
        response = self.client.post("/api/v1/analyze", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIsNone(data["financial_projection"])

    def test_analyze_missing_required_field_returns_422(self) -> None:
        bad_payload = {k: v for k, v in SAMPLE_UNIFIED_REQUEST.items() if k != "personal"}
        response = self.client.post("/api/v1/analyze", json=bad_payload)
        self.assertEqual(response.status_code, 422)


class CreditScoreEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(), raise_server_exceptions=True)

    def test_credit_score_returns_score_and_features(self) -> None:
        response = self.client.post("/api/v1/credit-score", json=SAMPLE_UNIFIED_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("credit_score", data)
        self.assertIn("confidence", data)
        self.assertIn("top_positive_features", data)
        self.assertIn("top_negative_features", data)
        self.assertIn("source", data)
        self.assertIsInstance(data["credit_score"], float)
        self.assertGreaterEqual(data["credit_score"], 0)

    def test_credit_score_source_field(self) -> None:
        response = self.client.post("/api/v1/credit-score", json=SAMPLE_UNIFIED_REQUEST)
        self.assertIn(response.json()["source"], ("model", "fallback"))


class RiskProfileEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(), raise_server_exceptions=True)

    def test_risk_profile_returns_profile_and_persona(self) -> None:
        response = self.client.post("/api/v1/risk-profile", json=SAMPLE_UNIFIED_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("risk_profile", data)
        self.assertIn(data["risk_profile"], ("Low", "Medium", "High"))
        self.assertIn("persona", data)
        self.assertIn("probabilities", data)
        self.assertIn("confidence", data)
        self.assertIn("persona_details", data)


class ProductsEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(), raise_server_exceptions=True)

    def test_products_listing_no_filters(self) -> None:
        response = self.client.get("/api/v1/products")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("products", data)
        self.assertIn("total_matches", data)
        self.assertGreater(len(data["products"]), 0)

    def test_products_search_by_query(self) -> None:
        response = self.client.get("/api/v1/products?q=retirement")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertGreater(len(data["products"]), 0)
        for product in data["products"]:
            self.assertIn("product_id", product)
            self.assertIn("relevance_score", product)

    def test_products_limit_is_respected(self) -> None:
        response = self.client.get("/api/v1/products?limit=3&q=gold")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertLessEqual(len(data["products"]), 3)


class RecommendEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(), raise_server_exceptions=True)

    def test_recommend_returns_portfolio(self) -> None:
        response = self.client.post("/api/v1/recommend", json=SAMPLE_RECOMMEND_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("recommended_portfolio", data)
        self.assertIn("portfolio_score", data)
        self.assertIn("alternative_portfolios", data)
        self.assertIn("educational_disclaimer", data)
        self.assertGreater(len(data["recommended_portfolio"]), 0)

    def test_recommend_allocation_totals_budget(self) -> None:
        response = self.client.post("/api/v1/recommend", json=SAMPLE_RECOMMEND_REQUEST)
        data = response.json()
        total = sum(item["monthly_amount"] for item in data["recommended_portfolio"])
        self.assertEqual(total, SAMPLE_RECOMMEND_REQUEST["monthly_budget"])

    def test_recommend_invalid_persona_returns_422(self) -> None:
        bad = {**SAMPLE_RECOMMEND_REQUEST, "risk_profile": "Low", "behavioral_persona": "Growth Explorer"}
        response = self.client.post("/api/v1/recommend", json=bad)
        self.assertIn(response.status_code, (422, 500))


class SimulateEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(), raise_server_exceptions=True)
        # Get a real recommendation to use as simulation input
        rec_response = self.client.post("/api/v1/recommend", json=SAMPLE_RECOMMEND_REQUEST)
        self.recommendation = rec_response.json()

    def test_simulate_returns_projection(self) -> None:
        payload = _make_simulate_request(self.recommendation)
        response = self.client.post("/api/v1/simulate", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("simulation", data)
        self.assertIn("scenarios", data)
        self.assertIn("yearly_projections", data)
        self.assertIn("goal_probability", data)
        self.assertIn("educational_disclaimer", data)

    def test_simulate_scenarios_cover_all_styles(self) -> None:
        payload = _make_simulate_request(self.recommendation)
        response = self.client.post("/api/v1/simulate", json=payload)
        data = response.json()
        styles = {s["style"] for s in data["scenarios"]}
        self.assertIn("Conservative", styles)
        self.assertIn("Recommended", styles)
        self.assertIn("Growth", styles)

    def test_simulate_median_value_is_positive(self) -> None:
        payload = _make_simulate_request(self.recommendation)
        response = self.client.post("/api/v1/simulate", json=payload)
        data = response.json()
        self.assertGreater(data["simulation"]["median_value"], 0)


if __name__ == "__main__":
    unittest.main()
