"""Educational chart generation for Phase 7 simulation outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml.recommendation.schemas import PortfolioStyle

from .config import SimulationConfig, ML_ROOT
from .monte_carlo import decimal_volatility, midpoint_return, simulate_portfolio_paths
from .schemas import ChartArtifacts, ProjectionResponse, ScenarioSimulationResult

LOGGER = logging.getLogger(__name__)


def _style_slug(style: PortfolioStyle) -> str:
    return style.value.lower().replace(" ", "_")


def generate_projection_charts(
    response: ProjectionResponse,
    *,
    config: SimulationConfig,
    initial_investment: float,
    monthly_sip: int,
    horizon_years: int,
    random_seed: int,
    num_simulations: int,
) -> ChartArtifacts:
    """Render and save the requested Phase 7 chart set."""
    config.create_output_directories()
    active = next(item for item in response.scenarios if item.style == response.active_scenario)
    yearly_paths = _active_yearly_paths(
        active,
        initial_investment=initial_investment,
        monthly_sip=monthly_sip,
        horizon_years=horizon_years,
        random_seed=random_seed,
        num_simulations=num_simulations,
    )
    slug = _style_slug(response.active_scenario)
    artifacts = ChartArtifacts(
        portfolio_growth=_save_portfolio_growth(yearly_paths, config.plots_dir / f"{slug}_portfolio_growth.png"),
        simulation_fan=_save_simulation_fan(yearly_paths, config.plots_dir / f"{slug}_simulation_fan.png"),
        distribution_histogram=_save_distribution_histogram(
            yearly_paths[:, -1],
            config.plots_dir / f"{slug}_distribution_histogram.png",
        ),
        confidence_interval=_save_confidence_interval(
            active.yearly_projections,
            config.plots_dir / f"{slug}_confidence_interval.png",
        ),
        goal_probability=_save_goal_probability_chart(
            response,
            config.plots_dir / f"{slug}_goal_probability.png",
        ),
        scenario_comparison=_save_scenario_comparison(
            response.scenarios,
            config.plots_dir / "scenario_comparison.png",
        ),
        yearly_projection=_save_yearly_projection(
            active.yearly_projections,
            config.plots_dir / f"{slug}_yearly_projection.png",
        ),
    )
    plt.close("all")
    LOGGER.info("Saved Phase 7 charts to %s", config.plots_dir)
    return artifacts


def _active_yearly_paths(
    active: ScenarioSimulationResult,
    *,
    initial_investment: float,
    monthly_sip: int,
    horizon_years: int,
    random_seed: int,
    num_simulations: int,
) -> np.ndarray:
    expected_return = midpoint_return(active.expected_annual_return_min, active.expected_annual_return_max)
    volatility = decimal_volatility(active.estimated_risk)
    seed_offset = {PortfolioStyle.CONSERVATIVE: 11, PortfolioStyle.RECOMMENDED: 23, PortfolioStyle.GROWTH: 37}[
        active.style
    ]
    return simulate_portfolio_paths(
        initial_investment=initial_investment,
        monthly_contribution=float(monthly_sip),
        horizon_years=horizon_years,
        expected_annual_return=expected_return,
        annual_volatility=volatility,
        num_simulations=num_simulations,
        random_seed=random_seed + seed_offset,
    )


def _save_portfolio_growth(yearly_paths: np.ndarray, path: Path) -> str:
    years = np.arange(yearly_paths.shape[1])
    median = np.median(yearly_paths, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, median, color="#1f77b4", linewidth=2.5, label="Median simulated path")
    ax.fill_between(
        years,
        np.percentile(yearly_paths, 25, axis=0),
        np.percentile(yearly_paths, 75, axis=0),
        color="#1f77b4",
        alpha=0.2,
        label="25th–75th percentile range",
    )
    ax.set_title("Portfolio Growth Projection (Educational Simulation)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return _relative_plot_path(path)


def _save_simulation_fan(yearly_paths: np.ndarray, path: Path) -> str:
    years = np.arange(yearly_paths.shape[1])
    sample_count = min(120, yearly_paths.shape[0])
    sample = yearly_paths[:sample_count]
    fig, ax = plt.subplots(figsize=(10, 6))
    for path_values in sample:
        ax.plot(years, path_values, color="#888888", alpha=0.08, linewidth=0.8)
    ax.plot(years, np.median(yearly_paths, axis=0), color="#d62728", linewidth=2.5, label="Median path")
    ax.set_title("Monte Carlo Simulation Fan Chart")
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return _relative_plot_path(path)


def _save_distribution_histogram(final_values: np.ndarray, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_values, bins=40, color="#2ca02c", alpha=0.85, edgecolor="white")
    ax.axvline(float(np.median(final_values)), color="#d62728", linestyle="--", linewidth=2, label="Median")
    ax.set_title("Terminal Portfolio Value Distribution")
    ax.set_xlabel("Terminal Value (₹)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return _relative_plot_path(path)


def _save_confidence_interval(yearly_projections, path: Path) -> str:
    years = [point.year for point in yearly_projections]
    median = [point.median for point in yearly_projections]
    low = [point.expected_range_low for point in yearly_projections]
    high = [point.expected_range_high for point in yearly_projections]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(years, low, high, color="#ff7f0e", alpha=0.25, label="25th–75th percentile band")
    ax.plot(years, median, color="#ff7f0e", linewidth=2.5, label="Median")
    ax.set_title("Confidence Interval Projection")
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return _relative_plot_path(path)


def _save_goal_probability_chart(response: ProjectionResponse, path: Path) -> str | None:
    if response.goal_probability is None:
        return None
    labels = ["Success", "Shortfall"]
    values = [
        response.goal_probability.probability_of_success,
        response.goal_probability.probability_of_shortfall,
    ]
    colors = ["#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Goal Attainment Probability (Educational Simulation)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return _relative_plot_path(path)


def _save_scenario_comparison(scenarios: tuple[ScenarioSimulationResult, ...], path: Path) -> str:
    labels = [item.style.value for item in scenarios]
    medians = [item.simulation.median_value for item in scenarios]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, medians, color=["#8c564b", "#1f77b4", "#9467bd"])
    ax.set_title("Scenario Comparison — Median Terminal Value")
    ax.set_ylabel("Median Value (₹)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return _relative_plot_path(path)


def _save_yearly_projection(yearly_projections, path: Path) -> str:
    years = [point.year for point in yearly_projections]
    median = [point.median for point in yearly_projections]
    best = [point.best_case for point in yearly_projections]
    worst = [point.worst_case for point in yearly_projections]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, median, label="Median", linewidth=2.5)
    ax.plot(years, best, label="Best case (90th pct)", linestyle="--")
    ax.plot(years, worst, label="Worst case (10th pct)", linestyle="--")
    ax.set_title("Year-wise Projection Timeline")
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return _relative_plot_path(path)


def _relative_plot_path(path: Path) -> str:
    return str(Path("ml") / path.relative_to(ML_ROOT))
