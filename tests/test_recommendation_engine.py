"""Deterministic integration tests for the Phase 6 recommendation contract."""

from __future__ import annotations

import unittest

from ml.recommendation import recommend_portfolio


BASE_REQUEST = {
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


class PortfolioRecommendationEngineTests(unittest.TestCase):
    def test_recommendation_is_deterministic_and_constrained(self) -> None:
        first = recommend_portfolio(BASE_REQUEST)
        second = recommend_portfolio(BASE_REQUEST)

        self.assertEqual(first, second)
        self.assertEqual(first.rules_version, "v0.6.0")
        self.assertIn("Educational illustration only", first.educational_disclaimer)
        for variant in (
            first.alternative_portfolios.conservative,
            first.alternative_portfolios.recommended,
            first.alternative_portfolios.growth,
        ):
            self.assertGreaterEqual(len(variant.allocation), 3)
            self.assertLessEqual(len(variant.allocation), 6)
            self.assertEqual(sum(item.monthly_amount for item in variant.allocation), BASE_REQUEST["monthly_budget"])
            self.assertTrue(all(5 <= item.allocation <= 60 for item in variant.allocation))
            self.assertTrue(all(item.reason and item.potential_risks for item in variant.allocation))

    def test_persona_must_be_consistent_with_risk_profile(self) -> None:
        invalid_request = {**BASE_REQUEST, "risk_profile": "Low", "behavioral_persona": "Growth Explorer"}
        with self.assertRaisesRegex(ValueError, "behavioral_persona must match"):
            recommend_portfolio(invalid_request)


if __name__ == "__main__":
    unittest.main()
