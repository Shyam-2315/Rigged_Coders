"""Domain features for the TrustVest alternative-credit training dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.config import FeatureEngineeringConfig


WALLET_USAGE_SCORES = {
    "No usage": 0.0,
    "Occasional": 0.33,
    "Regular": 0.67,
    "Heavy": 1.0,
}

EMPLOYMENT_STABILITY_SCORES = {
    "Not employed": 0.28,
    "Gig": 0.48,
    "Contract": 0.58,
    "Freelance": 0.58,
    "Self-employed": 0.65,
    "Salaried": 0.80,
    "Government": 0.92,
}


@dataclass(frozen=True, slots=True)
class FeatureEngineeringResult:
    """The transformed frame and the names of features added to it."""

    frame: pd.DataFrame
    created_features: tuple[str, ...]


class CreditFeatureEngineer:
    """Create transparent financial, digital, behavioural, and interaction signals."""

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        self.config = config

    def transform(self, frame: pd.DataFrame) -> FeatureEngineeringResult:
        """Return a copy with deterministic, target-independent feature additions."""
        engineered = frame.copy()
        income = engineered["monthly_income"].clip(lower=self.config.min_income)
        savings_rate = self._bounded(engineered["monthly_savings"] / income)
        expense_ratio = self._bounded(engineered["monthly_expenses"] / income)
        bank_balance_ratio = self._bounded(
            engineered["bank_balance"] / (income * self.config.annualization_months)
        )
        emergency_fund_ratio = self._bounded(
            engineered["emergency_fund_months"] / self.config.emergency_fund_target_months
        )
        wallet_score = engineered["wallet_usage"].map(WALLET_USAGE_SCORES).fillna(0.0)
        employment_score = (
            engineered["employment_type"].map(EMPLOYMENT_STABILITY_SCORES).fillna(0.0)
        )

        repayment_score = self._bounded(engineered["repayment_consistency"] / 100)
        digital_literacy = self._bounded(engineered["digital_literacy_score"] / 100)
        device_trust = self._bounded(engineered["device_trust_score"] / 100)
        upi_activity = self._bounded(
            engineered["upi_transactions_per_month"] / self.config.upi_transaction_cap
        )
        ecommerce_activity = self._bounded(
            engineered["ecommerce_transactions"] / self.config.ecommerce_transaction_cap
        )
        subscription_activity = self._bounded(
            engineered["online_subscription_count"] / self.config.subscription_count_cap
        )
        loan_history = self._bounded(
            engineered["loan_history_length"] / self.config.loan_history_year_cap
        )
        years_employed = self._bounded(
            engineered["years_employed"] / self.config.years_employed_cap
        )
        sim_age = self._bounded(engineered["sim_age"] / self.config.sim_age_month_cap)
        inverse_device_age = 1 - self._bounded(
            engineered["device_age"] / self.config.device_age_year_cap
        )
        income_capacity = self._bounded(income / self.config.income_normalization_cap)
        loan_count = self._bounded(
            engineered["existing_small_loans"] / self.config.loan_count_cap
        )

        features = {
            "expense_to_income_ratio": expense_ratio,
            "savings_to_income_ratio": savings_rate,
            "bank_balance_to_annual_income_ratio": bank_balance_ratio,
            "emergency_fund_adequacy_ratio": emergency_fund_ratio,
            "debt_burden_ratio": engineered["existing_small_loans"] / income,
            "financial_health_score": 100
            * (
                0.25 * income_capacity
                + 0.30 * savings_rate
                + 0.25 * bank_balance_ratio
                + 0.20 * emergency_fund_ratio
            ),
            "digital_payment_score": 100
            * (
                0.30 * upi_activity
                + 0.25 * wallet_score
                + 0.20 * (1 - engineered["cash_transaction_ratio"])
                + 0.25 * digital_literacy
            ),
            "credit_behavior_score": 100
            * (
                0.30 * (1 - engineered["late_payment_ratio"])
                + 0.20
                * (
                    1
                    - self._bounded(
                        engineered["missed_utility_payments"] / self.config.missed_payment_cap
                    )
                )
                + 0.30 * repayment_score
                + 0.20 * engineered["utility_bill_payment_rate"]
            ),
            "employment_stability_feature_score": 100
            * (
                0.45 * employment_score
                + 0.30 * years_employed
                + 0.25 * income_capacity
            ),
            "financial_cushion_score": 100
            * (
                0.35 * savings_rate
                + 0.35 * emergency_fund_ratio
                + 0.30 * bank_balance_ratio
            ),
            "device_reliability_score": 100
            * (
                0.25 * sim_age
                + 0.20 * inverse_device_age
                + 0.25 * engineered["location_consistency"]
                + 0.30 * device_trust
            ),
            "digital_engagement_score": 100
            * (
                0.25 * subscription_activity
                + 0.25 * ecommerce_activity
                + 0.25 * wallet_score
                + 0.25 * upi_activity
            ),
            "banking_maturity_score": 100
            * (
                0.35
                * self._bounded(
                    engineered["bank_account_age"] / self.config.bank_account_age_year_cap
                )
                + 0.25 * loan_history
                + 0.40 * repayment_score
            ),
            "risk_indicator": 100
            * (
                0.35 * engineered["late_payment_ratio"]
                + 0.25
                * self._bounded(
                    engineered["missed_utility_payments"] / self.config.missed_payment_cap
                )
                + 0.20 * engineered["cash_transaction_ratio"]
                + 0.20 * loan_count
            ),
            "income_x_savings_rate": income * savings_rate,
            "income_x_payment_consistency": income * repayment_score,
            "digital_activity_x_device_trust": engineered["digital_activity_score"]
            * engineered["device_trust_score"],
            "bank_balance_x_emergency_fund": engineered["bank_balance"]
            * engineered["emergency_fund_months"],
            "repayment_x_loan_history": engineered["repayment_consistency"]
            * engineered["loan_history_length"],
            "age_x_income": engineered["age"] * income,
            "upi_x_digital_literacy": engineered["upi_transactions_per_month"]
            * engineered["digital_literacy_score"],
        }
        for name, values in features.items():
            engineered[name] = values.astype(float)

        return FeatureEngineeringResult(
            frame=engineered,
            created_features=tuple(features),
        )

    @staticmethod
    def _bounded(values: pd.Series) -> pd.Series:
        """Limit component ratios to a transparent 0--1 scoring range."""
        return values.astype(float).clip(lower=0.0, upper=1.0)
