"""Portfolio analytics derived from simulated projection paths."""

from __future__ import annotations

import numpy as np

from .schemas import PortfolioAnalyticsResult


def _annualized_returns(yearly_paths: np.ndarray) -> np.ndarray:
    """Compute annualized return series for each simulation path."""
    starting = yearly_paths[:, :-1]
    ending = yearly_paths[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(starting > 0, ending / starting - 1.0, 0.0)
    return returns


def calculate_portfolio_analytics(
    yearly_paths: np.ndarray,
    *,
    horizon_years: int,
    risk_free_rate: float,
    initial_investment: float,
    monthly_contribution: float,
) -> PortfolioAnalyticsResult:
    """Compute illustrative risk/return analytics from simulated yearly paths."""
    if yearly_paths.ndim != 2 or yearly_paths.shape[1] < 2:
        raise ValueError("yearly_paths must contain at least two year snapshots")

    final_values = yearly_paths[:, -1]
    total_invested = initial_investment + monthly_contribution * 12 * horizon_years
    invested_base = max(total_invested, 1.0)
    median_final = float(np.median(final_values))
    expected_cagr = ((median_final / invested_base) ** (1.0 / horizon_years) - 1.0) * 100.0

    annual_returns = _annualized_returns(yearly_paths)
    volatility = float(np.std(annual_returns) * np.sqrt(1.0) * 100.0)
    mean_return = float(np.mean(annual_returns))
    excess_return = mean_return - risk_free_rate
    sharpe_ratio = excess_return / (volatility / 100.0) if volatility > 0 else 0.0

    path_drawdowns = []
    for path in yearly_paths:
        running_max = np.maximum.accumulate(path)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdowns = np.where(running_max > 0, (running_max - path) / running_max, 0.0)
        path_drawdowns.append(float(np.max(drawdowns)))
    max_drawdown = float(np.median(path_drawdowns) * 100.0)

    negative_returns = annual_returns[annual_returns < 0]
    downside_risk = float(np.std(negative_returns) * 100.0) if negative_returns.size else 0.0

    positive_mask = annual_returns > 0
    upside_capture = float(np.mean(annual_returns[positive_mask]) * 100.0) if np.any(positive_mask) else 0.0

    return PortfolioAnalyticsResult(
        expected_cagr=round(expected_cagr, 2),
        volatility=round(volatility, 2),
        sharpe_ratio=round(sharpe_ratio, 2),
        max_drawdown=round(max_drawdown, 2),
        downside_risk=round(downside_risk, 2),
        upside_capture=round(upside_capture, 2),
    )
