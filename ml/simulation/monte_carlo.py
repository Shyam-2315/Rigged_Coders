"""Log-normal Monte Carlo engine for educational portfolio projections."""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def midpoint_return(expected_annual_return_min: float, expected_annual_return_max: float) -> float:
    """Convert Phase 6 percentage bounds to a decimal midpoint return."""
    return ((expected_annual_return_min + expected_annual_return_max) / 2.0) / 100.0


def decimal_volatility(estimated_risk_percent: float) -> float:
    """Convert Phase 6 illustrative risk percentage to decimal volatility."""
    return max(estimated_risk_percent, 0.01) / 100.0


def simulate_portfolio_paths(
    *,
    initial_investment: float,
    monthly_contribution: float,
    horizon_years: int,
    expected_annual_return: float,
    annual_volatility: float,
    num_simulations: int,
    random_seed: int,
) -> np.ndarray:
    """Simulate portfolio values using monthly geometric Brownian motion with SIP.

    Returns an array with shape ``(num_simulations, horizon_years + 1)`` where
    column 0 is the starting value and subsequent columns are year-end snapshots.
    """
    if horizon_years < 1:
        raise ValueError("horizon_years must be at least 1")
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")

    rng = np.random.default_rng(random_seed)
    months = horizon_years * 12
    dt = 1.0 / 12.0
    drift = (expected_annual_return - 0.5 * annual_volatility**2) * dt
    diffusion = annual_volatility * np.sqrt(dt)

    values = np.full((num_simulations, months + 1), float(initial_investment), dtype=np.float64)
    shocks = rng.normal(loc=drift, scale=diffusion, size=(num_simulations, months))

    for month in range(months):
        growth = np.exp(shocks[:, month])
        values[:, month + 1] = values[:, month] * growth + monthly_contribution

    yearly_indices = np.arange(0, months + 1, 12)
    yearly_paths = values[:, yearly_indices]
    LOGGER.debug(
        "Simulated %s paths over %s years (mu=%.4f, sigma=%.4f, seed=%s)",
        num_simulations,
        horizon_years,
        expected_annual_return,
        annual_volatility,
        random_seed,
    )
    return yearly_paths


def distribution_metrics(final_values: np.ndarray, percentile_levels: tuple[int, ...]) -> dict[str, float]:
    """Compute terminal-value distribution statistics."""
    if final_values.size == 0:
        raise ValueError("final_values cannot be empty")
    percentiles = np.percentile(final_values, percentile_levels)
    percentile_map = {f"percentile_{level}": float(value) for level, value in zip(percentile_levels, percentiles)}
    return {
        "runs": int(final_values.size),
        "mean_value": float(np.mean(final_values)),
        "median_value": float(np.median(final_values)),
        "minimum_value": float(np.min(final_values)),
        "maximum_value": float(np.max(final_values)),
        "standard_deviation": float(np.std(final_values)),
        "best_case": float(np.percentile(final_values, 90)),
        "worst_case": float(np.percentile(final_values, 10)),
        **percentile_map,
    }
