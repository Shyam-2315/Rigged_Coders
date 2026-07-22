"""Generate a realistic, reproducible alternative-credit population for India.

The module deliberately models relationships between demographic, financial,
digital, and device signals.  It is intended for educational and hackathon
experimentation only; it must not be used for real lending decisions.
"""

from __future__ import annotations

import argparse
import binascii
import logging
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from faker import Faker

try:
    from ml.config import GeneratorConfig
except ModuleNotFoundError:  # Supports: python ml/generators/synthetic_generator.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import GeneratorConfig  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Location:
    """A representative Indian city and its city-tier classification."""

    state: str
    city_tier: str


LOCATIONS: Final[tuple[Location, ...]] = (
    Location("Maharashtra", "Tier 1"),
    Location("Karnataka", "Tier 1"),
    Location("Delhi", "Tier 1"),
    Location("Tamil Nadu", "Tier 1"),
    Location("Telangana", "Tier 1"),
    Location("West Bengal", "Tier 1"),
    Location("Gujarat", "Tier 1"),
    Location("Uttar Pradesh", "Tier 2"),
    Location("Rajasthan", "Tier 2"),
    Location("Madhya Pradesh", "Tier 2"),
    Location("Kerala", "Tier 2"),
    Location("Punjab", "Tier 2"),
    Location("Bihar", "Tier 3"),
    Location("Odisha", "Tier 3"),
    Location("Assam", "Tier 3"),
    Location("Jharkhand", "Tier 3"),
    Location("Chhattisgarh", "Tier 3"),
    Location("Andhra Pradesh", "Tier 3"),
)

OCCUPATIONS: Final[tuple[str, ...]] = (
    "Student",
    "Software Engineer",
    "Teacher",
    "Farmer",
    "Driver",
    "Shop Owner",
    "Delivery Partner",
    "Freelancer",
    "Doctor",
    "Government Employee",
    "Business Owner",
    "Cashier",
    "Sales Executive",
    "Electrician",
    "Plumber",
    "Homemaker",
)

# Broad occupational mix, intentionally weighted toward the low- and
# middle-income occupations that form most of the target population.
CORE_OCCUPATION_WEIGHTS: Final[np.ndarray] = np.array(
    [0.045, 0.080, 0.140, 0.090, 0.100, 0.070, 0.055, 0.010, 0.055, 0.060, 0.070, 0.090, 0.075, 0.060]
)

INCOME_MEDIANS: Final[dict[str, float]] = {
    "Student": 10_000,
    "Software Engineer": 75_000,
    "Teacher": 32_000,
    "Farmer": 20_000,
    "Driver": 18_000,
    "Shop Owner": 35_000,
    "Delivery Partner": 16_000,
    "Freelancer": 32_000,
    "Doctor": 95_000,
    "Government Employee": 48_000,
    "Business Owner": 55_000,
    "Cashier": 17_000,
    "Sales Executive": 25_000,
    "Electrician": 25_000,
    "Plumber": 22_000,
    "Homemaker": 9_000,
}

EMPLOYMENT_TYPES: Final[dict[str, str]] = {
    "Student": "Not employed",
    "Software Engineer": "Salaried",
    "Teacher": "Salaried",
    "Farmer": "Self-employed",
    "Driver": "Contract",
    "Shop Owner": "Self-employed",
    "Delivery Partner": "Gig",
    "Freelancer": "Freelance",
    "Doctor": "Self-employed",
    "Government Employee": "Government",
    "Business Owner": "Self-employed",
    "Cashier": "Salaried",
    "Sales Executive": "Salaried",
    "Electrician": "Self-employed",
    "Plumber": "Self-employed",
    "Homemaker": "Not employed",
}

STABLE_EMPLOYMENT: Final[dict[str, float]] = {
    "Not employed": 0.28,
    "Gig": 0.48,
    "Contract": 0.58,
    "Freelance": 0.58,
    "Self-employed": 0.65,
    "Salaried": 0.80,
    "Government": 0.92,
}

WALLET_LEVELS: Final[tuple[str, ...]] = ("No usage", "Occasional", "Regular", "Heavy")
WALLET_SCORES: Final[dict[str, float]] = {
    "No usage": 0.0,
    "Occasional": 0.33,
    "Regular": 0.67,
    "Heavy": 1.0,
}
EDUCATION_LEVELS: Final[tuple[str, ...]] = (
    "Secondary",
    "Higher Secondary",
    "Diploma",
    "Graduate",
    "Postgraduate",
)
EDUCATION_SCORES: Final[dict[str, float]] = {
    "Secondary": 0.15,
    "Higher Secondary": 0.35,
    "Diploma": 0.50,
    "Graduate": 0.72,
    "Postgraduate": 0.92,
}

# Compact bitmap alphabet used only by the zero-dependency PNG fallback.
FONT_3X5: Final[dict[str, tuple[str, ...]]] = {
    "A": ("010", "101", "111", "101", "101"), "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"), "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"), "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"), "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"), "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"), "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"), "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"), "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "011", "001"), "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"), "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"), "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"), "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"), "Z": ("111", "001", "010", "100", "111"),
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("110", "001", "010", "100", "111"), "3": ("110", "001", "010", "001", "110"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "110", "001", "110"),
    "6": ("011", "100", "110", "101", "010"), "7": ("111", "001", "010", "010", "010"),
    "8": ("010", "101", "010", "101", "010"), "9": ("010", "101", "011", "001", "110"),
    "-": ("000", "000", "111", "000", "000"), "(": ("001", "010", "010", "010", "001"),
    ")": ("100", "010", "010", "010", "100"), "/": ("001", "001", "010", "100", "100"),
    " ": ("000", "000", "000", "000", "000"),
}


class DatasetValidationError(ValueError):
    """Raised when generated data violates a business constraint."""


class SyntheticPopulationGenerator:
    """Build a correlated synthetic population using a seeded NumPy generator."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.faker = Faker("en_IN")
        self.faker.seed_instance(self.config.random_seed)

    def generate(self) -> pd.DataFrame:
        """Generate, engineer, and validate the configured population."""
        n = self.config.num_users
        rng = self.rng
        LOGGER.info("Generating %s synthetic user records with seed=%s", n, self.config.random_seed)

        age = np.clip(np.rint(18 + rng.beta(2.1, 2.9, n) * 47), 18, 65).astype(int)
        gender = rng.choice(["Male", "Female", "Non-binary"], n, p=[0.51, 0.485, 0.005])
        location_index = rng.choice(len(LOCATIONS), n, p=self._location_weights())
        states = np.array([LOCATIONS[i].state for i in location_index], dtype=object)
        city_tiers = np.array([LOCATIONS[i].city_tier for i in location_index], dtype=object)
        tier_score = np.select(
            [city_tiers == "Tier 1", city_tiers == "Tier 2"], [1.0, 0.55], default=0.20
        )

        education = self._education(age, tier_score)
        education_score = np.array([EDUCATION_SCORES[value] for value in education])
        marital_status = self._marital_status(age)
        dependents = self._dependents(age, marital_status)
        occupation = self._occupation(age, education_score, tier_score)
        employment_type = np.array([EMPLOYMENT_TYPES[value] for value in occupation], dtype=object)
        employment_stability = np.array(
            [STABLE_EMPLOYMENT[value] for value in employment_type], dtype=float
        )
        years_employed = self._years_employed(age, occupation)

        monthly_income = self._monthly_income(
            occupation, years_employed, education_score, tier_score
        )
        savings_rate = self._savings_rate(
            monthly_income, dependents, employment_stability, tier_score
        )
        monthly_savings = np.rint(monthly_income * savings_rate).astype(int)
        monthly_expenses = (monthly_income - monthly_savings).astype(int)

        bank_account_age = self._bank_account_age(age, years_employed)
        smartphone_years = self._smartphone_years(age, monthly_income, tier_score)
        mobile_number_age = np.clip(
            smartphone_years + rng.normal(1.7, 1.8, n), 0.2, np.minimum(age - 12, 22)
        )
        digital_literacy = np.clip(
            28
            + 37 * education_score
            + 14 * tier_score
            + 1.9 * smartphone_years
            - 0.32 * np.maximum(age - 38, 0)
            + rng.normal(0, 8, n),
            5,
            100,
        )

        upi_transactions = self._upi_transactions(digital_literacy, monthly_income, tier_score)
        cash_ratio = np.clip(
            0.83
            - 0.0052 * upi_transactions
            - 0.0024 * digital_literacy
            - 0.09 * tier_score
            + rng.normal(0, 0.065, n),
            0.02,
            0.95,
        )
        wallet_usage = self._wallet_usage(digital_literacy, upi_transactions)
        wallet_score = np.array([WALLET_SCORES[value] for value in wallet_usage])
        upi_average_transaction = np.rint(
            np.clip(
                160
                + 0.022 * monthly_income
                + 2.8 * digital_literacy
                + rng.normal(0, 260, n),
                50,
                20_000,
            )
        ).astype(int)
        wallet_average_balance = np.rint(
            np.clip(
                monthly_income * (0.015 + 0.09 * wallet_score) * rng.lognormal(0, 0.35, n),
                0,
                40_000,
            )
        ).astype(int)
        ecommerce_transactions = np.minimum(
            rng.poisson(np.clip(0.5 + digital_literacy / 11 + tier_score * 2, 0.2, 20)), 45
        ).astype(int)
        online_subscription_count = np.minimum(
            rng.poisson(np.clip((digital_literacy - 18) / 19 + tier_score, 0.1, 6)), 12
        ).astype(int)
        mobile_recharge_frequency = np.clip(
            np.rint(1 + digital_literacy / 38 + rng.normal(0, 0.75, n)), 1, 6
        ).astype(int)
        atm_withdrawals = np.clip(
            rng.poisson(np.clip(7.0 - 0.04 * upi_transactions - 1.8 * wallet_score, 0.4, 9)), 0, 25
        ).astype(int)

        emergency_fund_months = np.clip(
            savings_rate * 20
            + bank_account_age * 0.23
            + employment_stability * 2.0
            + rng.normal(0, 2.0, n),
            0,
            24,
        )
        bank_balance = np.rint(
            np.clip(
                monthly_expenses * emergency_fund_months
                + monthly_savings * (1.5 + bank_account_age * 0.35)
                + rng.normal(0, np.maximum(monthly_income * 0.5, 1_000), n),
                0,
                2_500_000,
            )
        ).astype(int)

        repayment_consistency = np.clip(
            48
            + 30 * savings_rate
            + 18 * employment_stability
            + 0.8 * emergency_fund_months
            + 0.14 * bank_account_age
            + rng.normal(0, 8, n),
            20,
            100,
        )
        late_payment_ratio = np.clip(
            0.85
            - repayment_consistency / 110
            - 0.14 * savings_rate
            - 0.12 * employment_stability
            + rng.normal(0, 0.045, n),
            0,
            0.50,
        )
        utility_bill_payment_rate = np.clip(
            1.0
            - late_payment_ratio * 0.82
            - rng.normal(0.015, 0.018, n),
            0.50,
            1.0,
        )
        missed_utility_payments = rng.binomial(
            12, np.clip((1 - utility_bill_payment_rate) * 0.72, 0, 0.48)
        ).astype(int)
        loan_history_length = np.clip(
            np.minimum(years_employed, bank_account_age)
            - rng.exponential(1.6, n)
            + (monthly_income > 30_000) * 1.8,
            0,
            30,
        )
        existing_small_loans = np.clip(
            rng.poisson(np.clip(0.35 + late_payment_ratio * 3.2 - savings_rate * 0.8, 0.05, 2.7)),
            0,
            6,
        ).astype(int)
        investment_frequency = np.clip(
            np.rint(
                np.maximum(
                    0,
                    savings_rate * 9
                    + (monthly_income / 60_000)
                    + digital_literacy / 55
                    + rng.normal(0, 1.2, n),
                )
            ),
            0,
            15,
        ).astype(int)

        device_age = np.clip(
            smartphone_years * 0.65 + rng.normal(0.8, 1.1, n), 0.1, 8
        )
        sim_age = np.clip(
            mobile_number_age + rng.normal(0, 0.9, n), 0.1, 22
        )
        number_of_devices = np.clip(
            1
            + rng.poisson(np.clip((monthly_income - 12_000) / 75_000 + digital_literacy / 110, 0.05, 2.0)),
            1,
            5,
        ).astype(int)
        location_consistency = np.clip(
            0.47
            + 0.020 * sim_age
            + 0.013 * bank_account_age
            + 0.08 * employment_stability
            + rng.normal(0, 0.09, n),
            0.20,
            1.0,
        )
        device_trust_score = np.clip(
            36
            + 22 * location_consistency
            + 1.1 * sim_age
            + 0.6 * bank_account_age
            + 6 * (number_of_devices == 1)
            - 1.8 * device_age
            + rng.normal(0, 4, n),
            40,
            100,
        )

        data = pd.DataFrame(
            {
                "user_id": [f"TV{index:06d}" for index in range(1, n + 1)],
                "age": age,
                "gender": gender,
                "state": states,
                "city_tier": city_tiers,
                "education": education,
                "marital_status": marital_status,
                "dependents": dependents,
                "occupation": occupation,
                "employment_type": employment_type,
                "years_employed": np.round(years_employed, 1),
                "monthly_income": monthly_income,
                "monthly_expenses": monthly_expenses,
                "monthly_savings": monthly_savings,
                "bank_account_age": np.round(bank_account_age, 1),
                "mobile_number_age": np.round(mobile_number_age, 1),
                "smartphone_years": np.round(smartphone_years, 1),
                "digital_literacy_score": np.round(digital_literacy, 1),
                "upi_transactions_per_month": upi_transactions,
                "upi_average_transaction": upi_average_transaction,
                "utility_bill_payment_rate": np.round(utility_bill_payment_rate, 3),
                "mobile_recharge_frequency": mobile_recharge_frequency,
                "wallet_usage": wallet_usage,
                "wallet_average_balance": wallet_average_balance,
                "ecommerce_transactions": ecommerce_transactions,
                "online_subscription_count": online_subscription_count,
                "bank_balance": bank_balance,
                "atm_withdrawals": atm_withdrawals,
                "cash_transaction_ratio": np.round(cash_ratio, 3),
                "missed_utility_payments": missed_utility_payments,
                "late_payment_ratio": np.round(late_payment_ratio, 3),
                "savings_rate": np.round(savings_rate, 3),
                "investment_frequency": investment_frequency,
                "emergency_fund_months": np.round(emergency_fund_months, 1),
                "loan_history_length": np.round(loan_history_length, 1),
                "existing_small_loans": existing_small_loans,
                "repayment_consistency": np.round(repayment_consistency, 1),
                "device_trust_score": np.round(device_trust_score, 1),
                "device_age": np.round(device_age, 1),
                "sim_age": np.round(sim_age, 1),
                "number_of_devices": number_of_devices,
                "location_consistency": np.round(location_consistency, 3),
            }
        )
        data = self._add_engineered_features(data)
        data["credit_likelihood"] = self._credit_likelihood(data)
        self.validate(data)
        return data

    def _location_weights(self) -> np.ndarray:
        """Weight locations by broad urban and population representation."""
        weights = np.array(
            [0.075, 0.065, 0.060, 0.060, 0.045, 0.050, 0.045, 0.105, 0.070, 0.065,
             0.040, 0.040, 0.075, 0.050, 0.030, 0.045, 0.030, 0.065]
        )
        return weights / weights.sum()

    def _education(self, age: np.ndarray, tier_score: np.ndarray) -> np.ndarray:
        probabilities = np.column_stack(
            (
                0.32 - 0.10 * tier_score,
                0.29 - 0.05 * tier_score,
                np.full(len(age), 0.15),
                0.18 + 0.14 * tier_score,
                0.06 + 0.06 * tier_score,
            )
        )
        probabilities[:, 3] += (age >= 25) * 0.025
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        draws = self.rng.random(len(age))
        return np.array(
            [EDUCATION_LEVELS[np.searchsorted(np.cumsum(row), draw)] for row, draw in zip(probabilities, draws)],
            dtype=object,
        )

    def _marital_status(self, age: np.ndarray) -> np.ndarray:
        married_probability = np.clip((age - 21) / 26, 0.02, 0.86)
        married = self.rng.random(len(age)) < married_probability
        status = np.where(married, "Married", "Single").astype(object)
        status[(age > 40) & (self.rng.random(len(age)) < 0.035)] = "Divorced/Widowed"
        return status

    def _dependents(self, age: np.ndarray, marital_status: np.ndarray) -> np.ndarray:
        expected = np.where(marital_status == "Married", 0.75 + (age > 30) * 0.75, 0.12)
        expected += (marital_status == "Divorced/Widowed") * 0.35
        return np.minimum(self.rng.poisson(expected), 5).astype(int)

    def _occupation(
        self, age: np.ndarray, education_score: np.ndarray, tier_score: np.ndarray
    ) -> np.ndarray:
        n = len(age)
        occupation = self.rng.choice(
            OCCUPATIONS[1:-1], size=n, p=CORE_OCCUPATION_WEIGHTS
        ).astype(object)
        student_probability = np.clip(0.65 - (age - 18) * 0.10, 0, 0.62)
        homemaker_probability = np.where(age >= 23, 0.055, 0.012)
        student_mask = self.rng.random(n) < student_probability
        homemaker_mask = (~student_mask) & (self.rng.random(n) < homemaker_probability)
        occupation[student_mask] = "Student"
        occupation[homemaker_mask] = "Homemaker"
        technical_mask = (~student_mask) & (education_score > 0.7) & (tier_score > 0.5)
        occupation[technical_mask & (self.rng.random(n) < 0.29)] = "Software Engineer"
        occupation[technical_mask & (self.rng.random(n) < 0.08)] = "Doctor"
        return occupation

    def _years_employed(self, age: np.ndarray, occupation: np.ndarray) -> np.ndarray:
        maximum = np.maximum(age - 18, 0).astype(float)
        years = np.clip(maximum * self.rng.beta(1.8, 2.1, len(age)), 0, maximum)
        years[(occupation == "Student") | (occupation == "Homemaker")] = 0
        return years

    def _monthly_income(
        self,
        occupation: np.ndarray,
        years_employed: np.ndarray,
        education_score: np.ndarray,
        tier_score: np.ndarray,
    ) -> np.ndarray:
        base = np.array([INCOME_MEDIANS[value] for value in occupation])
        experience_factor = 1 + np.minimum(years_employed, 20) * 0.027
        education_factor = 0.87 + education_score * 0.26
        city_factor = 0.91 + tier_score * 0.20
        variation = self.rng.lognormal(mean=0, sigma=0.30, size=len(occupation))
        income = base * experience_factor * education_factor * city_factor * variation
        return np.rint(np.clip(income, 8_000, 250_000)).astype(int)

    def _savings_rate(
        self,
        monthly_income: np.ndarray,
        dependents: np.ndarray,
        employment_stability: np.ndarray,
        tier_score: np.ndarray,
    ) -> np.ndarray:
        income_effect = 0.055 * np.log1p(monthly_income / 8_000)
        return np.clip(
            0.025
            + income_effect
            + 0.075 * employment_stability
            + 0.025 * tier_score
            - 0.023 * dependents
            + self.rng.normal(0, 0.045, len(monthly_income)),
            0,
            0.50,
        )

    def _bank_account_age(self, age: np.ndarray, years_employed: np.ndarray) -> np.ndarray:
        candidate = np.maximum(years_employed + self.rng.normal(3.5, 3, len(age)), 0.1)
        return np.clip(np.minimum(candidate, age - 16), 0.1, 25)

    def _smartphone_years(
        self, age: np.ndarray, monthly_income: np.ndarray, tier_score: np.ndarray
    ) -> np.ndarray:
        possible_years = np.maximum(age - 14, 0.5)
        adoption = 0.22 + 0.37 * tier_score + 0.08 * np.log1p(monthly_income / 10_000)
        return np.clip(possible_years * adoption + self.rng.normal(0, 1.8, len(age)), 0.2, 15)

    def _upi_transactions(
        self, digital_literacy: np.ndarray, monthly_income: np.ndarray, tier_score: np.ndarray
    ) -> np.ndarray:
        mean = np.clip(
            2 + digital_literacy * 1.55 + tier_score * 28 + monthly_income / 3_800,
            0.5,
            250,
        )
        zero_mask = self.rng.random(len(mean)) < np.clip(0.20 - digital_literacy / 600, 0.01, 0.18)
        transactions = self.rng.poisson(mean)
        transactions[zero_mask] = 0
        return np.minimum(transactions, 400).astype(int)

    def _wallet_usage(self, digital_literacy: np.ndarray, upi_transactions: np.ndarray) -> np.ndarray:
        propensity = np.clip(0.50 * (digital_literacy / 100) + 0.50 * (upi_transactions / 250), 0, 1)
        return np.select(
            [propensity < 0.22, propensity < 0.46, propensity < 0.72],
            ["No usage", "Occasional", "Regular"],
            default="Heavy",
        ).astype(object)

    def _add_engineered_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic, interpretable aggregates from source signals."""
        result = data.copy()
        wallet_score = result["wallet_usage"].map(WALLET_SCORES).astype(float)
        employment_score = result["employment_type"].map(STABLE_EMPLOYMENT).astype(float)

        payment_consistency = np.clip(
            100
            * (
                0.43 * result["utility_bill_payment_rate"]
                + 0.34 * (result["repayment_consistency"] / 100)
                + 0.13 * (1 - result["late_payment_ratio"] / 0.50)
                + 0.10 * (1 - result["missed_utility_payments"] / 12)
            ),
            0,
            100,
        )
        financial_discipline = np.clip(
            100
            * (
                0.34 * (result["savings_rate"] / 0.50)
                + 0.28 * (payment_consistency / 100)
                + 0.21 * (result["emergency_fund_months"] / 24)
                + 0.10 * (result["investment_frequency"] / 15)
                + 0.07 * (1 - result["existing_small_loans"] / 6)
            ),
            0,
            100,
        )
        digital_activity = np.clip(
            100
            * (
                0.39 * (result["upi_transactions_per_month"] / 400)
                + 0.20 * (result["digital_literacy_score"] / 100)
                + 0.17 * wallet_score
                + 0.15 * (result["ecommerce_transactions"] / 45)
                + 0.09 * (result["online_subscription_count"] / 12)
            ),
            0,
            100,
        )
        income_stability = np.clip(
            100
            * (
                0.44 * employment_score
                + 0.30 * (np.minimum(result["years_employed"], 20) / 20)
                + 0.16 * (np.minimum(result["bank_account_age"], 25) / 25)
                + 0.10 * (result["monthly_income"] / 250_000)
            ),
            0,
            100,
        )
        digital_trust = np.clip(
            100
            * (
                0.48 * (result["device_trust_score"] / 100)
                + 0.26 * result["location_consistency"]
                + 0.16 * (np.minimum(result["sim_age"], 22) / 22)
                + 0.10 * (np.minimum(result["bank_account_age"], 25) / 25)
            ),
            0,
            100,
        )
        result["financial_discipline_index"] = np.round(financial_discipline, 1)
        result["digital_activity_score"] = np.round(digital_activity, 1)
        result["income_stability_score"] = np.round(income_stability, 1)
        result["payment_consistency_score"] = np.round(payment_consistency, 1)
        result["digital_trust_index"] = np.round(digital_trust, 1)
        return result

    def _credit_likelihood(self, data: pd.DataFrame) -> pd.Series:
        """Calculate a deterministic alternative-credit likelihood in the 0-100 range."""
        income_score = np.clip(np.log1p(data["monthly_income"] / 8_000) / np.log1p(250_000 / 8_000), 0, 1)
        bank_score = np.clip(np.log1p(data["bank_balance"] / 5_000) / np.log1p(2_500_000 / 5_000), 0, 1)
        upi_score = np.minimum(data["upi_transactions_per_month"], 250) / 250
        fund_score = data["emergency_fund_months"] / 24
        missed_penalty = data["missed_utility_payments"] / 12
        late_penalty = data["late_payment_ratio"] / 0.50

        likelihood = 100 * (
            0.12 * income_score
            + 0.12 * (data["savings_rate"] / 0.50)
            + 0.23 * (data["payment_consistency_score"] / 100)
            + 0.07 * upi_score
            + 0.11 * (data["device_trust_score"] / 100)
            + 0.10 * bank_score
            + 0.08 * fund_score
            + 0.06 * (data["digital_literacy_score"] / 100)
            + 0.06 * (data["income_stability_score"] / 100)
            + 0.05 * (data["digital_trust_index"] / 100)
            - 0.12 * missed_penalty
            - 0.12 * late_penalty
        )
        return pd.Series(np.round(np.clip(likelihood, 0, 100), 1), index=data.index)

    @staticmethod
    def validate(data: pd.DataFrame) -> None:
        """Fail early if arithmetic, bounds, or completeness constraints are broken."""
        required_columns = {
            "user_id", "monthly_income", "monthly_expenses", "monthly_savings", "savings_rate",
            "utility_bill_payment_rate", "late_payment_ratio", "device_trust_score",
            "credit_likelihood", "financial_discipline_index", "digital_activity_score",
            "income_stability_score", "payment_consistency_score", "digital_trust_index",
        }
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise DatasetValidationError(f"Missing required columns: {sorted(missing_columns)}")
        if data.isna().any().any():
            raise DatasetValidationError("Generated dataset contains missing values")
        if not data["user_id"].is_unique:
            raise DatasetValidationError("user_id values must be unique")
        if (data["monthly_income"] < 0).any() or (data["monthly_savings"] < 0).any():
            raise DatasetValidationError("Income and savings cannot be negative")
        if (data["monthly_savings"] > data["monthly_income"]).any():
            raise DatasetValidationError("Savings cannot exceed income")
        if (data["monthly_expenses"] > data["monthly_income"] * 1.05).any():
            raise DatasetValidationError("Expenses exceed the allowed income threshold")
        if not np.allclose(
            data["monthly_income"], data["monthly_expenses"] + data["monthly_savings"], atol=1
        ):
            raise DatasetValidationError("Income must equal expenses plus savings")
        bounds = {
            "age": (18, 65),
            "monthly_income": (8_000, 250_000),
            "bank_account_age": (0, 25),
            "upi_transactions_per_month": (0, 400),
            "utility_bill_payment_rate": (0.5, 1.0),
            "savings_rate": (0, 0.5),
            "device_trust_score": (40, 100),
            "credit_likelihood": (0, 100),
            "financial_discipline_index": (0, 100),
            "digital_activity_score": (0, 100),
            "income_stability_score": (0, 100),
            "payment_consistency_score": (0, 100),
            "digital_trust_index": (0, 100),
        }
        for column, (minimum, maximum) in bounds.items():
            if ((data[column] < minimum) | (data[column] > maximum)).any():
                raise DatasetValidationError(f"{column} falls outside [{minimum}, {maximum}]")

    def save(self, data: pd.DataFrame) -> Path:
        """Persist the dataset in the configured output directory."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.output_dir / "synthetic_users.csv"
        data.to_csv(output_path, index=False)
        LOGGER.info("Saved dataset to %s", output_path)
        return output_path

    def create_visualizations(self, data: pd.DataFrame) -> list[Path]:
        """Create the requested distribution and correlation plots."""
        self.config.plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._create_matplotlib_visualizations(data)
        except (ImportError, OSError) as error:
            LOGGER.warning(
                "Matplotlib is unavailable (%s); creating portable PNG charts instead.", error
            )
            return self._create_portable_visualizations(data)

    def _create_matplotlib_visualizations(self, data: pd.DataFrame) -> list[Path]:
        """Create annotated charts with Matplotlib when its binary backend is available."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        paths: list[Path] = []
        plt.style.use("seaborn-v0_8-whitegrid")

        paths.append(self._histogram(plt, data["monthly_income"], "Monthly income distribution", "Monthly income (INR)", "income_distribution.png", bins=70))
        paths.append(self._histogram(plt, data["credit_likelihood"], "Credit likelihood distribution", "Credit likelihood (0-100)", "credit_distribution.png", bins=50, color="#1f77b4"))
        paths.append(self._histogram(plt, data["age"], "Age distribution", "Age", "age_distribution.png", bins=np.arange(17.5, 66.5, 1), color="#2ca02c"))

        correlation_columns = [
            "monthly_income", "monthly_savings", "upi_transactions_per_month",
            "utility_bill_payment_rate", "late_payment_ratio", "bank_balance",
            "emergency_fund_months", "repayment_consistency", "device_trust_score",
            "digital_literacy_score", "payment_consistency_score", "credit_likelihood",
        ]
        correlation = data[correlation_columns].corr(numeric_only=True)
        figure, axis = plt.subplots(figsize=(12, 9))
        image = axis.imshow(correlation, cmap="RdYlGn", vmin=-1, vmax=1)
        axis.set_xticks(range(len(correlation_columns)), correlation_columns, rotation=55, ha="right", fontsize=8)
        axis.set_yticks(range(len(correlation_columns)), correlation_columns, fontsize=8)
        for row in range(len(correlation_columns)):
            for column in range(len(correlation_columns)):
                axis.text(column, row, f"{correlation.iat[row, column]:.2f}", ha="center", va="center", fontsize=6)
        figure.colorbar(image, ax=axis, label="Pearson correlation")
        axis.set_title("Alternative-credit feature correlations")
        figure.tight_layout()
        heatmap_path = self.config.plots_dir / "correlation_heatmap.png"
        figure.savefig(heatmap_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(heatmap_path)
        LOGGER.info("Saved %s visualizations to %s", len(paths), self.config.plots_dir)
        return paths

    def _create_portable_visualizations(self, data: pd.DataFrame) -> list[Path]:
        """Render chart data with the standard-library PNG fallback.

        This avoids native GUI/image dependencies in constrained CI or sandboxed
        environments. Normal installations use the annotated Matplotlib charts.
        """
        paths = [
            self._portable_histogram(data["monthly_income"], "MONTHLY INCOME DISTRIBUTION", "INR PER MONTH", "income_distribution.png", 70, (255, 127, 14)),
            self._portable_histogram(data["credit_likelihood"], "CREDIT LIKELIHOOD DISTRIBUTION", "CREDIT SCORE (0-100)", "credit_distribution.png", 50, (31, 119, 180)),
            self._portable_heatmap(data),
            self._portable_histogram(data["age"], "AGE DISTRIBUTION", "AGE", "age_distribution.png", np.arange(17.5, 66.5, 1), (44, 160, 44)),
        ]
        LOGGER.info("Saved %s portable PNG visualizations to %s", len(paths), self.config.plots_dir)
        return paths

    def _histogram(
        self,
        plt: object,
        series: pd.Series,
        title: str,
        x_label: str,
        filename: str,
        bins: int | np.ndarray,
        color: str = "#ff7f0e",
    ) -> Path:
        # Pyplot is intentionally passed in so importing this module does not
        # require a native Matplotlib backend when plots are not requested.
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.hist(series, bins=bins, color=color, edgecolor="white", alpha=0.9)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Users")
        figure.tight_layout()
        output_path = self.config.plots_dir / filename
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _portable_histogram(
        self,
        series: pd.Series,
        title: str,
        x_label: str,
        filename: str,
        bins: int | np.ndarray,
        color: tuple[int, int, int],
    ) -> Path:
        """Render a simple distribution plot using only NumPy and the stdlib."""
        canvas, left, top, right, bottom = self._plot_canvas()
        counts, _ = np.histogram(series.to_numpy(), bins=bins)
        maximum = max(int(counts.max()), 1)
        width = right - left
        bar_width = max(1, width // len(counts))
        for index, count in enumerate(counts):
            x0 = left + index * bar_width
            x1 = min(x0 + bar_width - 1, right)
            height = int((count / maximum) * (bottom - top))
            self._fill_rectangle(canvas, x0, bottom - height, x1, bottom, color)
        self._draw_text(canvas, title, left, 14, scale=3, color=(35, 35, 35))
        self._draw_text(canvas, "USER COUNT", left, top - 22, scale=2, color=(80, 80, 80))
        self._draw_text(canvas, x_label, left, bottom + 26, scale=2, color=(80, 80, 80))
        output_path = self.config.plots_dir / filename
        self._write_png(output_path, canvas)
        return output_path

    def _portable_heatmap(self, data: pd.DataFrame) -> Path:
        """Render the requested correlation heatmap without external image libraries."""
        columns = [
            "monthly_income", "monthly_savings", "upi_transactions_per_month",
            "utility_bill_payment_rate", "late_payment_ratio", "bank_balance",
            "emergency_fund_months", "repayment_consistency", "device_trust_score",
            "digital_literacy_score", "payment_consistency_score", "credit_likelihood",
        ]
        correlation = data[columns].corr(numeric_only=True).to_numpy()
        canvas = np.full((930, 900, 3), 255, dtype=np.uint8)
        grid_size = 52
        start_x, start_y = 138, 145
        self._draw_text(canvas, "CORRELATION HEATMAP", 138, 22, scale=3, color=(35, 35, 35))
        self._draw_text(canvas, "FEATURE INDEX", 138, 95, scale=2, color=(80, 80, 80))
        for index in range(len(columns)):
            self._draw_text(canvas, str(index + 1), start_x + index * grid_size + 20, 125, scale=2, color=(70, 70, 70))
            self._draw_text(canvas, str(index + 1), 112, start_y + index * grid_size + 20, scale=2, color=(70, 70, 70))
        for row in range(correlation.shape[0]):
            for column in range(correlation.shape[1]):
                value = float(correlation[row, column])
                # Red (-1) through yellow (0) to green (+1), matching the
                # Matplotlib RdYlGn palette's interpretation.
                if value < 0:
                    amount = int(255 * (value + 1))
                    color = (255, amount, 35)
                else:
                    amount = int(255 * (1 - value))
                    color = (amount, 190, 35)
                x0 = start_x + column * grid_size
                y0 = start_y + row * grid_size
                self._fill_rectangle(canvas, x0, y0, x0 + grid_size - 2, y0 + grid_size - 2, color)
        legends = (
            "1 INCOME  2 SAVINGS  3 UPI  4 UTILITY",
            "5 LATE  6 BALANCE  7 FUND  8 REPAY",
            "9 DEVICE  10 LITERACY  11 PAYMENT  12 CREDIT",
        )
        for index, legend in enumerate(legends):
            self._draw_text(canvas, legend, 52, 798 + index * 28, scale=2, color=(55, 55, 55))
        output_path = self.config.plots_dir / "correlation_heatmap.png"
        self._write_png(output_path, canvas)
        return output_path

    @staticmethod
    def _plot_canvas() -> tuple[np.ndarray, int, int, int, int]:
        """Create a white chart surface with neutral gridlines and axes."""
        canvas = np.full((720, 1200, 3), 255, dtype=np.uint8)
        left, top, right, bottom = 90, 50, 1140, 650
        for y in np.linspace(top, bottom, 7, dtype=int):
            canvas[y : y + 1, left:right] = (228, 228, 228)
        canvas[top:bottom + 2, left:left + 2] = (65, 65, 65)
        canvas[bottom:bottom + 2, left:right + 2] = (65, 65, 65)
        return canvas, left, top, right, bottom

    @staticmethod
    def _fill_rectangle(
        canvas: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        """Fill a safely clipped rectangle on an RGB NumPy canvas."""
        height, width, _ = canvas.shape
        canvas[max(0, y0) : min(height, y1 + 1), max(0, x0) : min(width, x1 + 1)] = color

    @staticmethod
    def _draw_text(
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        scale: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a small uppercase label using the portable bitmap alphabet."""
        cursor = x
        for character in text.upper():
            glyph = FONT_3X5.get(character, FONT_3X5[" "])
            for row, pixels in enumerate(glyph):
                for column, pixel in enumerate(pixels):
                    if pixel == "1":
                        SyntheticPopulationGenerator._fill_rectangle(
                            canvas,
                            cursor + column * scale,
                            y + row * scale,
                            cursor + (column + 1) * scale - 1,
                            y + (row + 1) * scale - 1,
                            color,
                        )
            cursor += 4 * scale

    @staticmethod
    def _write_png(output_path: Path, canvas: np.ndarray) -> None:
        """Write an RGB array as a standards-compliant PNG using the stdlib."""
        height, width, channels = canvas.shape
        if channels != 3:
            raise ValueError("Portable PNG writer accepts only RGB canvases")

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
            )

        raw_rows = b"".join(b"\x00" + row.tobytes() for row in canvas)
        payload = b"\x89PNG\r\n\x1a\n"
        payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        payload += chunk(b"IDAT", zlib.compress(raw_rows, level=9))
        payload += chunk(b"IEND", b"")
        output_path.write_bytes(payload)

    @staticmethod
    def print_report(data: pd.DataFrame) -> None:
        """Print compact diagnostics requested for every generation run."""
        report_columns = [
            "age", "monthly_income", "monthly_expenses", "monthly_savings", "savings_rate",
            "upi_transactions_per_month", "utility_bill_payment_rate", "bank_balance",
            "repayment_consistency", "device_trust_score", "credit_likelihood",
        ]
        correlation_columns = [
            "monthly_income", "monthly_savings", "upi_transactions_per_month",
            "payment_consistency_score", "device_trust_score", "credit_likelihood",
        ]
        print(f"Dataset Shape: {data.shape}")
        print("\nMissing Values:\n", data.isna().sum().to_string())
        print("\nSummary Statistics:\n", data[report_columns].describe().round(2).to_string())
        print("\nCorrelation Matrix:\n", data[correlation_columns].corr().round(3).to_string())
        print("\nTarget Distribution:\n", data["credit_likelihood"].describe().round(2).to_string())


def parse_arguments() -> argparse.Namespace:
    """Parse optional size and seed overrides for command-line operation."""
    parser = argparse.ArgumentParser(description="Generate TrustVest AI synthetic users.")
    parser.add_argument("--num-users", type=int, default=GeneratorConfig().num_users)
    parser.add_argument("--seed", type=int, default=GeneratorConfig().random_seed)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG visualization output.")
    return parser.parse_args()


def main() -> None:
    """Generate, validate, save, report, and visualize the synthetic population."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    arguments = parse_arguments()
    config = GeneratorConfig(num_users=arguments.num_users, random_seed=arguments.seed)
    generator = SyntheticPopulationGenerator(config)
    data = generator.generate()
    output_path = generator.save(data)
    if not arguments.no_plots:
        generator.create_visualizations(data)
    generator.print_report(data)
    print(f"\nDataset written to: {output_path}")


if __name__ == "__main__":
    main()
