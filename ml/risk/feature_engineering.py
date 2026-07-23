"""Validated feature engineering for behavioral-investor questionnaire data."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ENGINEERED_FEATURE_COLUMNS, QUESTIONNAIRE_CATEGORICAL_COLUMNS, QUESTIONNAIRE_NUMERIC_COLUMNS


INCOME_STABILITY_SCORES = {"Stable": 90.0, "Moderate": 60.0, "Variable": 28.0}
EXPERIENCE_SCORES = {"None": 5.0, "Beginner": 28.0, "Intermediate": 62.0, "Advanced": 90.0}
KNOWLEDGE_SCORES = {"None": 5.0, "Basic": 30.0, "Good": 62.0, "Advanced": 90.0}
LIQUIDITY_SCORES = {
    "Immediate access": 10.0,
    "Within 3 months": 38.0,
    "Flexible": 65.0,
    "Can lock in for years": 90.0,
}
CONFIDENCE_SCORES = {"Low": 20.0, "Moderate": 52.0, "High": 78.0, "Very high": 94.0}
LOSS_RECOVERY_SCORES = {
    "Sell immediately": 5.0,
    "Reduce investments": 28.0,
    "Hold and wait": 64.0,
    "Invest more": 92.0,
}
GOAL_SCORES = {
    "Capital preservation": 15.0,
    "Emergency reserve": 20.0,
    "Major purchase": 40.0,
    "Children education": 64.0,
    "Retirement": 82.0,
    "Wealth creation": 92.0,
}
FREQUENCY_SCORES = {"On demand": 25.0, "Monthly": 58.0, "Quarterly": 68.0, "Annually": 48.0}
INVESTMENT_TYPE_SCORES = {
    "Savings / fixed deposits": 12.0,
    "Debt funds": 30.0,
    "Balanced mutual funds": 55.0,
    "Equity mutual funds": 75.0,
    "Direct stocks": 92.0,
}


def engineer_features(
    records: pd.DataFrame | Mapping[str, object] | Sequence[Mapping[str, object]],
    *,
    require_complete_schema: bool = True,
) -> pd.DataFrame:
    """Build transparent behavioral-finance features from questionnaire answers.

    The function is intentionally deterministic: the same valid questionnaire
    answer always yields the same engineered feature values.
    """
    frame = _as_frame(records)
    if require_complete_schema:
        _validate_questionnaire(frame)
    else:
        frame = frame.copy()

    numeric = frame.loc[:, QUESTIONNAIRE_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="raise")
    result = frame.copy()
    result.loc[:, QUESTIONNAIRE_NUMERIC_COLUMNS] = numeric

    income = numeric["monthly_income"].clip(lower=1.0)
    savings_rate = (numeric["monthly_savings"] / income).clip(0.0, 1.0)
    budget_rate = (numeric["monthly_investment_budget"] / income).clip(0.0, 1.0)
    emergency_score = (numeric["emergency_fund_months"].clip(0.0, 12.0) / 12.0) * 100.0
    horizon_score = (numeric["investment_horizon_years"].clip(0.0, 20.0) / 20.0) * 100.0
    return_score = ((numeric["expected_annual_return_percent"].clip(4.0, 24.0) - 4.0) / 20.0) * 100.0
    dependent_pressure = (numeric["dependents"].clip(0.0, 6.0) / 6.0) * 100.0

    income_stability = _map_scores(result["income_stability"], INCOME_STABILITY_SCORES, "income_stability")
    experience = _map_scores(result["previous_investment_experience"], EXPERIENCE_SCORES, "previous_investment_experience")
    mutual_fund_knowledge = _map_scores(result["mutual_fund_knowledge"], KNOWLEDGE_SCORES, "mutual_fund_knowledge")
    stock_knowledge = _map_scores(result["stock_knowledge"], KNOWLEDGE_SCORES, "stock_knowledge")
    liquidity = _map_scores(result["preferred_liquidity"], LIQUIDITY_SCORES, "preferred_liquidity")
    confidence = _map_scores(result["financial_confidence"], CONFIDENCE_SCORES, "financial_confidence")
    loss_recovery = _map_scores(result["reaction_to_20_percent_loss"], LOSS_RECOVERY_SCORES, "reaction_to_20_percent_loss")
    goal_orientation = _map_scores(result["investment_goal"], GOAL_SCORES, "investment_goal")
    frequency = _map_scores(result["preferred_investment_frequency"], FREQUENCY_SCORES, "preferred_investment_frequency")
    investment_type = _map_scores(result["preferred_investment_type"], INVESTMENT_TYPE_SCORES, "preferred_investment_type")

    result["risk_tolerance_score"] = _clip_score(
        0.32 * loss_recovery
        + 0.20 * return_score
        + 0.16 * stock_knowledge
        + 0.16 * investment_type
        + 0.08 * confidence
        + 0.08 * horizon_score
    )
    result["financial_preparedness"] = _clip_score(
        0.42 * emergency_score + 0.28 * savings_rate * 100.0 + 0.18 * income_stability + 0.12 * (100.0 - dependent_pressure)
    )
    result["income_stability_index"] = _clip_score(income_stability)
    result["investment_experience_index"] = _clip_score(0.58 * experience + 0.22 * mutual_fund_knowledge + 0.20 * stock_knowledge)
    result["liquidity_preference_index"] = _clip_score(liquidity)
    result["behavioral_confidence_score"] = _clip_score(0.72 * confidence + 0.18 * experience + 0.10 * mutual_fund_knowledge)
    result["loss_recovery_score"] = _clip_score(loss_recovery)
    result["investment_readiness_score"] = _clip_score(
        0.38 * emergency_score
        + 0.26 * savings_rate * 100.0
        + 0.22 * budget_rate * 100.0
        + 0.14 * income_stability
    )
    result["long_term_orientation_score"] = _clip_score(
        0.54 * horizon_score + 0.25 * goal_orientation + 0.11 * frequency + 0.10 * liquidity
    )
    return result


def model_input_frame(records: pd.DataFrame | Mapping[str, object] | Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Return raw and engineered columns that feed the risk-model preprocessor."""
    engineered = engineer_features(records)
    return engineered.loc[:, (*QUESTIONNAIRE_NUMERIC_COLUMNS, *QUESTIONNAIRE_CATEGORICAL_COLUMNS, *ENGINEERED_FEATURE_COLUMNS)]


def _as_frame(records: pd.DataFrame | Mapping[str, object] | Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, Mapping):
        return pd.DataFrame([records])
    return pd.DataFrame(list(records))


def _validate_questionnaire(frame: pd.DataFrame) -> None:
    expected = set((*QUESTIONNAIRE_NUMERIC_COLUMNS, *QUESTIONNAIRE_CATEGORICAL_COLUMNS))
    missing = sorted(expected.difference(frame.columns))
    unexpected = sorted(set(frame.columns).difference(expected).difference({"risk_profile", "respondent_id"}))
    if missing or unexpected:
        problems: list[str] = []
        if missing:
            problems.append(f"missing answers: {missing}")
        if unexpected:
            problems.append(f"unexpected fields: {unexpected}")
        raise ValueError("Questionnaire schema mismatch (" + "; ".join(problems) + ")")
    if frame.loc[:, list(expected)].isna().any().any():
        raise ValueError("Questionnaire contains missing answers")
    if (pd.to_numeric(frame["age"], errors="raise") < 18).any() or (pd.to_numeric(frame["age"], errors="raise") > 100).any():
        raise ValueError("age must be between 18 and 100")
    for column in ("monthly_income", "monthly_savings", "monthly_investment_budget", "emergency_fund_months", "investment_horizon_years", "dependents"):
        if (pd.to_numeric(frame[column], errors="raise") < 0).any():
            raise ValueError(f"{column} cannot be negative")
    annual_return = pd.to_numeric(frame["expected_annual_return_percent"], errors="raise")
    if ((annual_return < 0) | (annual_return > 100)).any():
        raise ValueError("expected_annual_return_percent must be between 0 and 100")
    mappings = {
        "income_stability": INCOME_STABILITY_SCORES,
        "previous_investment_experience": EXPERIENCE_SCORES,
        "mutual_fund_knowledge": KNOWLEDGE_SCORES,
        "stock_knowledge": KNOWLEDGE_SCORES,
        "preferred_liquidity": LIQUIDITY_SCORES,
        "financial_confidence": CONFIDENCE_SCORES,
        "reaction_to_20_percent_loss": LOSS_RECOVERY_SCORES,
        "investment_goal": GOAL_SCORES,
        "preferred_investment_frequency": FREQUENCY_SCORES,
        "preferred_investment_type": INVESTMENT_TYPE_SCORES,
    }
    for column, mapping in mappings.items():
        unknown = sorted(set(frame[column].astype(str)).difference(mapping))
        if unknown:
            raise ValueError(f"Unsupported value(s) for {column}: {unknown}")
    if (frame["monthly_savings"].astype(float) > frame["monthly_income"].astype(float)).any():
        raise ValueError("monthly_savings cannot exceed monthly_income")
    if (frame["monthly_investment_budget"].astype(float) > frame["monthly_savings"].astype(float)).any():
        raise ValueError("monthly_investment_budget cannot exceed monthly_savings")


def _map_scores(values: pd.Series, mapping: Mapping[str, float], column: str) -> pd.Series:
    scores = values.map(mapping)
    if scores.isna().any():
        unknown = sorted(values[scores.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported value(s) for {column}: {unknown}")
    return scores.astype(float)


def _clip_score(values: pd.Series | np.ndarray | float) -> pd.Series:
    return pd.Series(np.clip(values, 0.0, 100.0), index=getattr(values, "index", None), dtype=float).round(3)
