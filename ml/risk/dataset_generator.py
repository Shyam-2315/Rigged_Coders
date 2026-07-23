"""Create correlated synthetic investor questionnaires for behavioral-risk research."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import RANDOM_SEED, RiskConfig
from .feature_engineering import (
    CONFIDENCE_SCORES,
    EXPERIENCE_SCORES,
    GOAL_SCORES,
    INVESTMENT_TYPE_SCORES,
    KNOWLEDGE_SCORES,
    LIQUIDITY_SCORES,
    LOSS_RECOVERY_SCORES,
    engineer_features,
)


LOGGER = logging.getLogger(__name__)


def generate_investor_profiles(num_profiles: int = 50_000, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate realistic, correlated questionnaire answers and derived risk labels.

    ``risk_profile`` is a deterministic result of behavioral-finance signals;
    no profile receives a randomly assigned class label.
    """
    if num_profiles <= 0:
        raise ValueError("num_profiles must be greater than zero")
    rng = np.random.default_rng(random_seed)
    profile = pd.DataFrame({"respondent_id": [f"RISK-{index:06d}" for index in range(1, num_profiles + 1)]})

    age = np.rint(np.clip(rng.normal(37, 11, num_profiles), 21, 67)).astype(int)
    profile["age"] = age
    occupation = _sample_occupation(age, rng)
    profile["occupation"] = occupation
    stability = _income_stability(occupation, rng)
    profile["income_stability"] = stability

    occupation_income = {
        "Student": 22_000, "Salaried professional": 72_000, "Government employee": 65_000,
        "Business owner": 95_000, "Freelancer": 52_000, "Homemaker": 25_000, "Retired": 43_000,
    }
    age_factor = 0.78 + (age - 21) * 0.016
    income = np.array([occupation_income[value] for value in occupation], dtype=float)
    income *= age_factor * rng.lognormal(mean=0.0, sigma=0.34, size=num_profiles)
    profile["monthly_income"] = np.rint(np.clip(income, 12_000, 500_000)).astype(int)

    dependents = np.clip(np.rint((age - 25) / 11 + rng.normal(0.7, 0.9, num_profiles)), 0, 5).astype(int)
    dependents[occupation == "Student"] = 0
    profile["dependents"] = dependents

    stability_score = pd.Series(stability).map({"Stable": 0.90, "Moderate": 0.60, "Variable": 0.28}).to_numpy()
    financial_discipline = np.clip(
        0.30 + 0.40 * stability_score + 0.20 * np.tanh((age - 26) / 20) + rng.normal(0, 0.13, num_profiles),
        0.08,
        0.95,
    )
    risk_appetite = np.clip(
        0.12 + 0.55 * rng.beta(2.2, 2.0, num_profiles) + 0.18 * financial_discipline - 0.10 * (age > 55),
        0.03,
        0.97,
    )
    literacy = np.clip(
        0.12 + 0.34 * risk_appetite + 0.25 * financial_discipline + 0.16 * np.tanh((age - 24) / 18) + rng.normal(0, 0.12, num_profiles),
        0.02,
        0.98,
    )

    savings_rate = np.clip(0.04 + 0.36 * financial_discipline + 0.10 * stability_score - 0.025 * dependents + rng.normal(0, 0.06, num_profiles), 0.02, 0.62)
    savings = np.rint(profile["monthly_income"].to_numpy() * savings_rate).astype(int)
    profile["monthly_savings"] = savings
    investment_share = np.clip(0.12 + 0.68 * risk_appetite + 0.10 * literacy + rng.normal(0, 0.09, num_profiles), 0.05, 0.92)
    profile["monthly_investment_budget"] = np.minimum(savings, np.rint(savings * investment_share).astype(int))
    profile["emergency_fund_months"] = np.round(
        np.clip(0.5 + 9.5 * financial_discipline + 1.5 * stability_score - 0.45 * dependents + rng.normal(0, 1.7, num_profiles), 0, 24),
        1,
    )

    horizon = np.rint(np.clip(1.0 + 17.0 * risk_appetite + 2.5 * literacy + rng.normal(0, 2.4, num_profiles), 1, 25)).astype(int)
    profile["investment_horizon_years"] = horizon
    profile["expected_annual_return_percent"] = np.round(
        np.clip(4.5 + 17.0 * risk_appetite + 2.0 * literacy + rng.normal(0, 1.8, num_profiles), 4, 25), 1
    )
    profile["reaction_to_20_percent_loss"] = _bucket(risk_appetite + rng.normal(0, 0.08, num_profiles), [0.18, 0.43, 0.74], list(LOSS_RECOVERY_SCORES))
    profile["previous_investment_experience"] = _bucket(
        np.clip(0.32 * (age - 20) / 45 + 0.38 * literacy + 0.30 * risk_appetite + rng.normal(0, 0.10, num_profiles), 0, 1),
        [0.23, 0.48, 0.75],
        list(EXPERIENCE_SCORES),
    )
    profile["mutual_fund_knowledge"] = _bucket(literacy + rng.normal(0, 0.10, num_profiles), [0.21, 0.52, 0.78], list(KNOWLEDGE_SCORES))
    profile["stock_knowledge"] = _bucket(
        np.clip(0.72 * literacy + 0.28 * risk_appetite + rng.normal(0, 0.11, num_profiles), 0, 1),
        [0.23, 0.54, 0.79],
        list(KNOWLEDGE_SCORES),
    )
    profile["preferred_liquidity"] = _bucket(
        np.clip(0.75 * risk_appetite + 0.12 * financial_discipline + rng.normal(0, 0.11, num_profiles), 0, 1),
        [0.22, 0.46, 0.73],
        list(LIQUIDITY_SCORES),
    )
    profile["financial_confidence"] = _bucket(
        np.clip(0.52 * literacy + 0.31 * risk_appetite + 0.17 * financial_discipline + rng.normal(0, 0.10, num_profiles), 0, 1),
        [0.28, 0.56, 0.80],
        list(CONFIDENCE_SCORES),
    )
    profile["investment_goal"] = _goals(risk_appetite, horizon, financial_discipline, rng)
    profile["preferred_investment_frequency"] = _frequency(financial_discipline, risk_appetite, rng)
    profile["preferred_investment_type"] = _bucket(
        np.clip(0.67 * risk_appetite + 0.19 * literacy + rng.normal(0, 0.08, num_profiles), 0, 1),
        [0.19, 0.37, 0.64, 0.83],
        list(INVESTMENT_TYPE_SCORES),
    )

    engineered = engineer_features(profile)
    behavioral_risk_score = (
        0.28 * engineered["loss_recovery_score"]
        + 0.19 * engineered["long_term_orientation_score"]
        + 0.16 * engineered["investment_experience_index"]
        + 0.13 * engineered["risk_tolerance_score"]
        + 0.10 * engineered["liquidity_preference_index"]
        + 0.08 * engineered["financial_preparedness"]
        + 0.06 * engineered["behavioral_confidence_score"]
    )
    # Labels are deterministically derived from behavioural capacity and
    # preference signals. Income is only represented through savings and
    # preparedness ratios, never as a direct risk-class driver.
    profile["risk_profile"] = pd.cut(
        behavioral_risk_score,
        bins=[-np.inf, 39.0, 65.0, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return profile


def generate_and_save(config: RiskConfig | None = None) -> pd.DataFrame:
    """Generate the configured dataset and persist a CSV suitable for Phase 4."""
    active_config = config or RiskConfig()
    active_config.create_output_directories()
    dataset = generate_investor_profiles(active_config.num_profiles, active_config.random_seed)
    dataset.to_csv(active_config.dataset_path, index=False)
    LOGGER.info("Generated %s behavioral investor profiles at %s", len(dataset), active_config.dataset_path)
    return dataset


def _sample_occupation(age: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(len(age), dtype=object)
    for index, current_age in enumerate(age):
        if current_age < 25:
            choices, weights = ["Student", "Salaried professional", "Freelancer"], [0.34, 0.48, 0.18]
        elif current_age > 59:
            choices, weights = ["Retired", "Business owner", "Government employee", "Salaried professional"], [0.45, 0.20, 0.18, 0.17]
        else:
            choices = ["Salaried professional", "Government employee", "Business owner", "Freelancer", "Homemaker"]
            weights = [0.46, 0.17, 0.16, 0.14, 0.07]
        values[index] = rng.choice(choices, p=weights)
    return values


def _income_stability(occupation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lookup = {
        "Student": (["Variable", "Moderate"], [0.82, 0.18]),
        "Salaried professional": (["Stable", "Moderate", "Variable"], [0.67, 0.28, 0.05]),
        "Government employee": (["Stable", "Moderate"], [0.88, 0.12]),
        "Business owner": (["Variable", "Moderate", "Stable"], [0.40, 0.43, 0.17]),
        "Freelancer": (["Variable", "Moderate", "Stable"], [0.54, 0.38, 0.08]),
        "Homemaker": (["Variable", "Moderate"], [0.66, 0.34]),
        "Retired": (["Stable", "Moderate"], [0.63, 0.37]),
    }
    return np.array(
        [rng.choice(lookup[value][0], p=lookup[value][1]) for value in occupation],
        dtype=object,
    )


def _bucket(values: np.ndarray, thresholds: list[float], labels: list[str]) -> np.ndarray:
    return np.asarray(labels, dtype=object)[np.digitize(values, thresholds, right=False)]


def _goals(risk_appetite: np.ndarray, horizon: np.ndarray, discipline: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    score = np.clip(0.58 * risk_appetite + 0.27 * horizon / 25.0 + 0.15 * discipline + rng.normal(0, 0.11, len(horizon)), 0, 1)
    return _bucket(score, [0.16, 0.29, 0.46, 0.72, 0.86], list(GOAL_SCORES))


def _frequency(discipline: np.ndarray, risk_appetite: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    score = np.clip(0.68 * discipline + 0.16 * risk_appetite + rng.normal(0, 0.13, len(discipline)), 0, 1)
    return _bucket(score, [0.18, 0.54, 0.80], ["On demand", "Monthly", "Quarterly", "Annually"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TrustVest behavioral investor profiles.")
    parser.add_argument("--num-profiles", type=int, default=50_000, help="Number of profiles to generate.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducible generation.")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_and_save(replace(RiskConfig(), num_profiles=arguments.num_profiles, random_seed=arguments.seed))


if __name__ == "__main__":
    main()
