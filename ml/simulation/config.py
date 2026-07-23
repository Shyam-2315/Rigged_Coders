"""Configuration for the Phase 7 financial projection and Monte Carlo engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.investments.config import InvestmentKnowledgeConfig
from ml.investments.schemas import MarketAssumptions


SIMULATION_ROOT = Path(__file__).resolve().parent
ML_ROOT = SIMULATION_ROOT.parent

DEFAULT_NUM_SIMULATIONS = 10_000
DEFAULT_RANDOM_SEED = 2026
DEFAULT_GOAL_AMOUNT = 2_500_000

EDUCATIONAL_DISCLAIMER = (
    "This simulation is based on statistical assumptions and historical-style modeling. "
    "It is for educational purposes only and does not constitute investment advice or a "
    "prediction of future market performance."
)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Runtime locations and Monte Carlo defaults for Phase 7."""

    reports_dir: Path = ML_ROOT / "reports"
    plots_dir: Path = SIMULATION_ROOT / "plots"
    simulation_version: str = "v0.7.0"
    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    random_seed: int = DEFAULT_RANDOM_SEED
    percentile_levels: tuple[int, ...] = (10, 25, 50, 75, 90)
    investment_config: InvestmentKnowledgeConfig = field(default_factory=InvestmentKnowledgeConfig)
    educational_disclaimer: str = EDUCATIONAL_DISCLAIMER

    simulation_summary_filename: str = "simulation_summary.json"
    projection_report_filename: str = "projection_report.md"
    scenario_comparison_filename: str = "scenario_comparison.md"
    yearly_projection_filename: str = "yearly_projection.csv"

    def create_output_directories(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @property
    def simulation_summary_path(self) -> Path:
        return self.reports_dir / self.simulation_summary_filename

    @property
    def projection_report_path(self) -> Path:
        return self.reports_dir / self.projection_report_filename

    @property
    def scenario_comparison_path(self) -> Path:
        return self.reports_dir / self.scenario_comparison_filename

    @property
    def yearly_projection_path(self) -> Path:
        return self.reports_dir / self.yearly_projection_filename

    def load_market_assumptions(self) -> MarketAssumptions:
        """Load Phase 5 illustrative planning assumptions for inflation and risk-free rate."""
        path = self.investment_config.market_assumptions_path
        if not path.exists():
            return MarketAssumptions(
                inflation_rate=0.055,
                risk_free_rate=0.065,
                expected_equity_premium=0.055,
                expected_gold_return=0.07,
                debt_return=0.065,
                cash_return=0.035,
                monte_carlo_default_volatility=0.15,
                assumption_version="v1.0",
                note="Built-in fallback assumptions for simulation only; not live market data.",
            )
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return MarketAssumptions.model_validate(payload)

    def __post_init__(self) -> None:
        if self.num_simulations < 100:
            raise ValueError("num_simulations must be at least 100")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
