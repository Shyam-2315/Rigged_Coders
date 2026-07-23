"""Request validation and cross-phase input mapping for Phase 8."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ml.investments.retrieval import RetrievalContext
from ml.investments.schemas import InvestmentGoal, LiquidityLevel
from ml.recommendation.schemas import IncomeStability
from ml.risk.feature_engineering import engineer_features
from ml.risk.personas import recommendation_summary, select_persona

from .schemas import RiskAnalysisResult, UnifiedAnalysisRequest


PHASE4_GOAL_MAP = {
    "wealth creation": "Wealth creation",
    "retirement": "Retirement",
    "emergency fund": "Emergency reserve",
    "emergency reserve": "Emergency reserve",
    "education": "Children education",
    "child education": "Children education",
    "children education": "Children education",
    "house": "Major purchase",
    "car": "Major purchase",
    "major purchase": "Major purchase",
    "capital preservation": "Capital preservation",
    "passive income": "Wealth creation",
    "tax saving": "Capital preservation",
    "vacation": "Major purchase",
    "marriage": "Major purchase",
}

PHASE6_GOAL_MAP = {
    "wealth creation": InvestmentGoal.WEALTH_CREATION,
    "retirement": InvestmentGoal.RETIREMENT,
    "emergency fund": InvestmentGoal.EMERGENCY_FUND,
    "emergency reserve": InvestmentGoal.EMERGENCY_FUND,
    "education": InvestmentGoal.EDUCATION,
    "child education": InvestmentGoal.CHILD_EDUCATION,
    "children education": InvestmentGoal.CHILD_EDUCATION,
    "house": InvestmentGoal.HOUSE,
    "car": InvestmentGoal.CAR,
    "major purchase": InvestmentGoal.WEALTH_CREATION,
    "capital preservation": InvestmentGoal.EMERGENCY_FUND,
    "passive income": InvestmentGoal.PASSIVE_INCOME,
    "tax saving": InvestmentGoal.TAX_SAVING,
    "vacation": InvestmentGoal.VACATION,
    "marriage": InvestmentGoal.MARRIAGE,
}

INCOME_STABILITY_TO_PHASE4 = {
    "high": "Stable",
    "medium": "Moderate",
    "low": "Variable",
    "stable": "Stable",
    "moderate": "Moderate",
    "variable": "Variable",
}

INCOME_STABILITY_TO_PHASE6 = {
    "high": IncomeStability.HIGH,
    "medium": IncomeStability.MEDIUM,
    "low": IncomeStability.LOW,
    "stable": IncomeStability.HIGH,
    "moderate": IncomeStability.MEDIUM,
    "variable": IncomeStability.LOW,
}


def validate_analysis_request(request: UnifiedAnalysisRequest | Mapping[str, object]) -> UnifiedAnalysisRequest:
    """Parse and validate a unified orchestrator request."""
    return request if isinstance(request, UnifiedAnalysisRequest) else UnifiedAnalysisRequest.model_validate(request)


def build_risk_questionnaire(request: UnifiedAnalysisRequest) -> dict[str, object]:
    """Map the unified request to the Phase 4 questionnaire contract."""
    financial = request.financial
    personal = request.personal
    behavioral = request.behavioral
    return {
        "age": personal.age,
        "occupation": personal.occupation,
        "dependents": personal.dependents,
        "monthly_income": financial.monthly_income,
        "monthly_savings": financial.monthly_savings,
        "monthly_investment_budget": financial.monthly_budget,
        "emergency_fund_months": financial.emergency_fund_months,
        "investment_horizon_years": behavioral.investment_horizon_years,
        "expected_annual_return_percent": behavioral.expected_annual_return_percent,
        "income_stability": map_income_stability_phase4(financial.income_stability),
        "investment_goal": map_investment_goal_phase4(behavioral.investment_goal),
        "reaction_to_20_percent_loss": behavioral.reaction_to_20_percent_loss,
        "previous_investment_experience": behavioral.previous_investment_experience,
        "mutual_fund_knowledge": behavioral.mutual_fund_knowledge,
        "stock_knowledge": behavioral.stock_knowledge,
        "preferred_liquidity": behavioral.preferred_liquidity,
        "financial_confidence": behavioral.financial_confidence,
        "preferred_investment_frequency": behavioral.preferred_investment_frequency,
        "preferred_investment_type": behavioral.preferred_investment_type,
    }


def build_recommendation_request(
    request: UnifiedAnalysisRequest,
    *,
    credit_score: float,
    risk_profile: str,
    behavioral_persona: str,
) -> dict[str, object]:
    """Map orchestrator context to the Phase 6 recommendation contract."""
    financial = request.financial
    personal = request.personal
    behavioral = request.behavioral
    preferences = request.investment_preferences
    payload: dict[str, object] = {
        "credit_score": int(round(credit_score)),
        "risk_profile": risk_profile,
        "behavioral_persona": behavioral_persona,
        "monthly_income": financial.monthly_income,
        "monthly_savings": financial.monthly_savings,
        "monthly_budget": financial.monthly_budget,
        "existing_savings": financial.existing_savings,
        "investment_goal": map_investment_goal_phase6(behavioral.investment_goal).value,
        "investment_horizon": behavioral.investment_horizon_years,
        "age": personal.age,
        "dependents": personal.dependents,
        "emergency_fund_months": financial.emergency_fund_months,
        "income_stability": map_income_stability_phase6(financial.income_stability).value,
        "government_backed_preference": preferences.government_backed_preference,
    }
    if preferences.liquidity_preference:
        payload["liquidity_preference"] = preferences.liquidity_preference
    return payload


def build_projection_request(
    request: UnifiedAnalysisRequest,
    recommendation: Any,
    *,
    simulation_seed: int,
) -> dict[str, object]:
    """Map orchestrator context to the Phase 7 projection contract."""
    simulation = request.simulation
    initial_investment = simulation.initial_investment
    if initial_investment is None:
        initial_investment = request.financial.existing_savings
    return {
        "initial_investment": initial_investment,
        "monthly_sip": request.financial.monthly_budget,
        "investment_horizon_years": max(request.behavioral.investment_horizon_years, 1),
        "goal_amount": simulation.goal_amount,
        "random_seed": simulation.random_seed or simulation_seed,
        "num_simulations": simulation.num_simulations,
        "recommendation": recommendation,
        "generate_charts": simulation.generate_charts,
        "generate_reports": simulation.generate_reports,
        "active_scenario": simulation.active_scenario,
    }


def build_retrieval_context(
    request: UnifiedAnalysisRequest,
    *,
    credit_score: float | None,
    risk_profile: str | None,
    persona: str | None,
) -> RetrievalContext:
    """Build a Phase 5 retrieval context from orchestrator inputs."""
    goal = map_investment_goal_phase6(request.behavioral.investment_goal)
    liquidity = request.investment_preferences.liquidity_preference
    return RetrievalContext(
        risk_profile=risk_profile,  # type: ignore[arg-type]
        goals=(goal,),
        persona=persona,  # type: ignore[arg-type]
        investment_horizon_years=request.behavioral.investment_horizon_years,
        minimum_liquidity=liquidity,  # type: ignore[arg-type]
        credit_score=int(round(credit_score)) if credit_score is not None else None,
        government_backed=request.investment_preferences.government_backed_preference or None,
    )


def build_credit_raw_record(request: UnifiedAnalysisRequest) -> pd.DataFrame:
    """Build a single-row raw synthetic-user frame for Phase 2 feature engineering."""
    financial = request.financial
    personal = request.personal
    savings_rate = financial.monthly_savings / financial.monthly_income if financial.monthly_income else 0.0
    monthly_expenses = max(financial.monthly_income - financial.monthly_savings, 0.0)
    income_stability_score = {"Stable": 88.0, "Moderate": 62.0, "Variable": 35.0}[
        map_income_stability_phase4(financial.income_stability)
    ]
    record = {
        "age": personal.age,
        "dependents": personal.dependents,
        "years_employed": max(personal.age - 22, 1),
        "monthly_income": financial.monthly_income,
        "monthly_expenses": monthly_expenses,
        "monthly_savings": financial.monthly_savings,
        "bank_account_age": 48,
        "mobile_number_age": 36,
        "smartphone_years": 4,
        "digital_literacy_score": 72.0,
        "upi_transactions_per_month": 28,
        "upi_average_transaction": 650.0,
        "utility_bill_payment_rate": 96.0,
        "mobile_recharge_frequency": 1.0,
        "wallet_average_balance": min(financial.existing_savings * 0.05, financial.monthly_income),
        "ecommerce_transactions": 12,
        "online_subscription_count": 3,
        "bank_balance": financial.existing_savings,
        "atm_withdrawals": 4,
        "cash_transaction_ratio": 0.18,
        "missed_utility_payments": 0,
        "late_payment_ratio": 0.02,
        "savings_rate": savings_rate * 100.0,
        "investment_frequency": 1.0,
        "emergency_fund_months": financial.emergency_fund_months,
        "loan_history_length": 2,
        "existing_small_loans": 0,
        "repayment_consistency": 92.0,
        "device_trust_score": 84.0,
        "device_age": 2,
        "sim_age": 24,
        "number_of_devices": 1,
        "location_consistency": 0.92,
        "financial_discipline_index": min(savings_rate * 100.0 + 20.0, 100.0),
        "digital_activity_score": 68.0,
        "income_stability_score": income_stability_score,
        "payment_consistency_score": 90.0,
        "digital_trust_index": 74.0,
        "gender": personal.gender or "Female",
        "state": personal.state or "Maharashtra",
        "city_tier": personal.city_tier or "Tier 1",
        "education": personal.education or "Graduate",
        "marital_status": personal.marital_status or "Single",
        "occupation": personal.occupation,
        "employment_type": personal.employment_type or "Salaried",
        "wallet_usage": "Regular",
    }
    return pd.DataFrame([record])


def map_income_stability_phase4(value: str) -> str:
    mapped = INCOME_STABILITY_TO_PHASE4.get(value.strip().lower())
    if mapped is None:
        raise ValueError(f"Unsupported income_stability value: {value}")
    return mapped


def map_income_stability_phase6(value: str) -> IncomeStability:
    mapped = INCOME_STABILITY_TO_PHASE6.get(value.strip().lower())
    if mapped is None:
        raise ValueError(f"Unsupported income_stability value: {value}")
    return mapped


def map_investment_goal_phase4(value: str) -> str:
    mapped = PHASE4_GOAL_MAP.get(value.strip().lower())
    if mapped is None:
        raise ValueError(f"Unsupported investment_goal value: {value}")
    return mapped


def map_investment_goal_phase6(value: str) -> InvestmentGoal:
    mapped = PHASE6_GOAL_MAP.get(value.strip().lower())
    if mapped is None:
        raise ValueError(f"Unsupported investment_goal value: {value}")
    return mapped


def build_risk_fallback(request: UnifiedAnalysisRequest) -> RiskAnalysisResult:
    """Derive an educational risk profile when the ML classifier is unavailable."""
    financial = request.financial
    questionnaire = build_risk_questionnaire(request)
    features = engineer_features(questionnaire).iloc[0]

    if financial.fallback_risk_profile is not None:
        risk_profile = financial.fallback_risk_profile.strip().title()
        if risk_profile not in {"Low", "Medium", "High"}:
            raise ValueError("fallback_risk_profile must be Low, Medium, or High")
    else:
        tolerance = float(features["risk_tolerance_score"])
        if tolerance < 40.0:
            risk_profile = "Low"
        elif tolerance < 70.0:
            risk_profile = "Medium"
        else:
            risk_profile = "High"

    persona = select_persona(risk_profile, features)
    persona_name = financial.fallback_behavioral_persona or persona.name
    persona_details = persona.to_dict()
    if financial.fallback_behavioral_persona:
        persona_details = {**persona_details, "name": persona_name, "source": "fallback_override"}

    probability_key = risk_profile.lower()
    probabilities = {"low": 0.0, "medium": 0.0, "high": 0.0}
    probabilities[probability_key] = 0.50
    return RiskAnalysisResult(
        risk_profile=risk_profile,
        persona=persona_name,
        persona_details=persona_details,
        confidence=0.50,
        probabilities=probabilities,
        recommendation_summary=recommendation_summary(risk_profile, persona),
        top_positive_factors=(),
        top_negative_factors=(),
        explanation_method="questionnaire heuristic fallback",
    )
