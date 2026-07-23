"""Configuration and deterministic policy constants for Phase 6."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


RECOMMENDATION_ROOT = Path(__file__).resolve().parent
ML_ROOT = RECOMMENDATION_ROOT.parent

SCORING_FACTORS = (
    "goal_match",
    "risk_match",
    "persona_match",
    "expected_return",
    "liquidity",
    "tax_efficiency",
    "inflation_protection",
    "trust_score",
    "popularity",
    "diversification_value",
    "budget_compatibility",
)

DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "goal_match": 20.0,
    "risk_match": 15.0,
    "persona_match": 10.0,
    "expected_return": 10.0,
    "liquidity": 8.0,
    "tax_efficiency": 7.0,
    "inflation_protection": 7.0,
    "trust_score": 7.0,
    "popularity": 4.0,
    "diversification_value": 7.0,
    "budget_compatibility": 5.0,
}

DEFAULT_ALLOCATION_TARGETS: dict[str, dict[str, dict[str, float]]] = {
    "Conservative": {
        "Low": {"Cash": 25, "Debt": 35, "Government Savings": 30, "Gold": 10, "Equity": 0},
        "Medium": {"Cash": 20, "Debt": 30, "Government Savings": 25, "Gold": 10, "Equity": 15},
        "High": {"Cash": 15, "Debt": 30, "Government Savings": 20, "Gold": 10, "Equity": 25},
    },
    "Recommended": {
        "Low": {"Cash": 20, "Debt": 35, "Government Savings": 35, "Gold": 10, "Equity": 0},
        "Medium": {"Cash": 10, "Debt": 25, "Government Savings": 15, "Gold": 10, "Equity": 40},
        "High": {"Cash": 5, "Debt": 15, "Government Savings": 10, "Gold": 10, "Equity": 60},
    },
    "Growth": {
        "Low": {"Cash": 15, "Debt": 35, "Government Savings": 35, "Gold": 15, "Equity": 0},
        "Medium": {"Cash": 5, "Debt": 15, "Government Savings": 10, "Gold": 10, "Equity": 60},
        "High": {"Cash": 5, "Debt": 10, "Government Savings": 5, "Gold": 10, "Equity": 70},
    },
}

DEFAULT_RISK_VOLATILITY_TARGETS = {"Low": 6.0, "Medium": 13.0, "High": 19.0}


@dataclass(frozen=True, slots=True)
class RecommendationConfig:
    """Runtime locations and transparent policy limits for the engine."""

    models_dir: Path = ML_ROOT / "models"
    reports_dir: Path = ML_ROOT / "reports"
    rules_filename: str = "recommendation_rules.json"
    rules_version: str = "v0.6.0"
    minimum_monthly_budget: int = 1_000
    minimum_products: int = 3
    maximum_products: int = 6
    minimum_allocation_percent: float = 5.0
    maximum_single_product_percent: float = 60.0
    emergency_fund_target_months: float = 6.0
    scoring_weights: Mapping[str, float] = field(default_factory=lambda: DEFAULT_SCORING_WEIGHTS.copy())
    allocation_targets: Mapping[str, Mapping[str, Mapping[str, float]]] = field(
        default_factory=lambda: DEFAULT_ALLOCATION_TARGETS.copy()
    )
    risk_volatility_targets: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_RISK_VOLATILITY_TARGETS.copy()
    )
    educational_disclaimer: str = (
        "Educational illustration only; this deterministic tool does not provide regulated financial advice, "
        "predict market returns, or recommend a transaction. Review suitability, costs, tax rules, and product "
        "terms with a qualified professional before investing."
    )

    @property
    def rules_path(self) -> Path:
        return self.models_dir / self.rules_filename

    def create_output_directories(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_rules(self) -> dict[str, Any]:
        """Load and validate the versioned rules artifact used by this engine."""
        if not self.rules_path.exists():
            return self.default_rules()
        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Recommendation rules must be a JSON object")
        weights = payload.get("scoring_weights")
        if not isinstance(weights, dict) or set(weights) != set(SCORING_FACTORS):
            raise ValueError("Recommendation rules contain an invalid scoring_weights schema")
        total = sum(float(value) for value in weights.values())
        if round(total, 6) != 100.0 or any(float(value) < 0 for value in weights.values()):
            raise ValueError("Recommendation scoring weights must be non-negative and sum to 100")
        return payload

    def default_rules(self) -> dict[str, Any]:
        """Return the built-in policy representation when the JSON artifact is absent."""
        return {
            "version": self.rules_version,
            "scoring_weights": dict(self.scoring_weights),
            "portfolio_constraints": {
                "minimum_products": self.minimum_products,
                "maximum_products": self.maximum_products,
                "minimum_allocation_percent": self.minimum_allocation_percent,
                "maximum_single_product_percent": self.maximum_single_product_percent,
            },
            "emergency_fund_target_months": self.emergency_fund_target_months,
            "allocation_targets": self.allocation_targets,
            "risk_volatility_targets": self.risk_volatility_targets,
        }

    def __post_init__(self) -> None:
        if self.minimum_monthly_budget <= 0:
            raise ValueError("minimum_monthly_budget must be greater than zero")
        if not 3 <= self.minimum_products <= self.maximum_products <= 6:
            raise ValueError("Portfolio product limits must satisfy 3 <= minimum <= maximum <= 6")
        if not 0 < self.minimum_allocation_percent <= self.maximum_single_product_percent <= 100:
            raise ValueError("Allocation limits must be ordered percentages in (0, 100]")
