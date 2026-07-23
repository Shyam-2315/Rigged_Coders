"""Deterministic integration tests for the Phase 7 projection engine."""

from __future__ import annotations

import unittest

from ml.recommendation import recommend_portfolio
from ml.simulation import project_portfolio


BASE_RECOMMENDATION_REQUEST = {
    "credit_score": 760,
    "risk_profile": "Medium",
    "behavioral_persona": "Balanced Builder",
    "monthly_income": 85_000,
    "monthly_savings": 22_000,
    "monthly_budget": 10_000,
    "existing_savings": 350_000,
    "investment_goal": "Wealth Creation",
    "investment_horizon": 10,
    "age": 31,
    "dependents": 1,
    "emergency_fund_months": 5,
    "income_stability": "High",
}


class FinancialProjectionEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recommendation = recommend_portfolio(BASE_RECOMMENDATION_REQUEST)

    def test_projection_is_deterministic_and_structured(self) -> None:
        request = {
            "initial_investment": 350_000,
            "monthly_sip": 10_000,
            "investment_horizon_years": 10,
            "goal_amount": 2_500_000,
            "random_seed": 2026,
            "num_simulations": 1_000,
            "recommendation": self.recommendation.model_dump(mode="json", by_alias=True),
            "generate_charts": False,
            "generate_reports": False,
        }
        first = project_portfolio(request)
        second = project_portfolio(request)

        self.assertEqual(first, second)
        self.assertEqual(first.simulation.runs, 1_000)
        self.assertEqual(len(first.scenarios), 3)
        self.assertIn("educational purposes only", first.educational_disclaimer.lower())
        self.assertIn("simulated market paths", first.explanation_summary.lower())
        self.assertIsNotNone(first.goal_probability)
        assert first.goal_probability is not None
        self.assertGreaterEqual(first.goal_probability.probability_of_success, 0.0)
        self.assertLessEqual(first.goal_probability.probability_of_success, 100.0)
        self.assertAlmostEqual(
            first.goal_probability.probability_of_success + first.goal_probability.probability_of_shortfall,
            100.0,
            places=1,
        )
        self.assertEqual(len(first.yearly_projections), 11)
        self.assertGreater(first.inflation_adjusted.real, 0)
        self.assertLess(first.inflation_adjusted.real, first.inflation_adjusted.nominal)

    def test_scenarios_have_distinct_medians(self) -> None:
        response = project_portfolio(
            {
                "initial_investment": 350_000,
                "monthly_sip": 10_000,
                "investment_horizon_years": 10,
                "goal_amount": 2_500_000,
                "random_seed": 2026,
                "num_simulations": 500,
                "recommendation": self.recommendation.model_dump(mode="json", by_alias=True),
                "generate_charts": False,
            }
        )
        medians = {scenario.style.value: scenario.simulation.median_value for scenario in response.scenarios}
        self.assertGreater(medians["Growth"], medians["Conservative"])


if __name__ == "__main__":
    unittest.main()
