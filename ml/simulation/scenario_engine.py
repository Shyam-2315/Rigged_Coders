"""Scenario orchestration for conservative, recommended, and growth portfolios."""

from __future__ import annotations

import logging

import numpy as np

from ml.recommendation.schemas import PortfolioStyle, PortfolioVariant, RecommendationResponse

from .analytics import calculate_portfolio_analytics
from .config import SimulationConfig
from .goal_probability import calculate_goal_probability
from .monte_carlo import decimal_volatility, distribution_metrics, midpoint_return, simulate_portfolio_paths
from .schemas import (
    DistributionMetrics,
    GoalProbabilityResult,
    InflationAdjustedResult,
    PortfolioAnalyticsResult,
    ScenarioSimulationResult,
    YearlyProjectionPoint,
)

LOGGER = logging.getLogger(__name__)


def _variant_for_style(recommendation: RecommendationResponse, style: PortfolioStyle) -> PortfolioVariant:
    alternatives = recommendation.alternative_portfolios
    mapping = {
        PortfolioStyle.CONSERVATIVE: alternatives.conservative,
        PortfolioStyle.RECOMMENDED: alternatives.recommended,
        PortfolioStyle.GROWTH: alternatives.growth,
    }
    return mapping[style]


def _inflation_adjusted(
    nominal_median: float,
    *,
    inflation_rate: float,
    horizon_years: int,
) -> InflationAdjustedResult:
    real_value = nominal_median / ((1.0 + inflation_rate) ** horizon_years)
    return InflationAdjustedResult(
        nominal=round(nominal_median, 2),
        real=round(real_value, 2),
        inflation_rate_used=inflation_rate,
        horizon_years=horizon_years,
    )


def _yearly_projections(
    yearly_paths: np.ndarray,
    *,
    inflation_rate: float,
) -> tuple[YearlyProjectionPoint, ...]:
    projections: list[YearlyProjectionPoint] = []
    for year_index in range(yearly_paths.shape[1]):
        column = yearly_paths[:, year_index]
        nominal_median = float(np.median(column))
        real_median = nominal_median / ((1.0 + inflation_rate) ** max(year_index, 0))
        projections.append(
            YearlyProjectionPoint(
                year=year_index,
                median=round(nominal_median, 2),
                best_case=round(float(np.percentile(column, 90)), 2),
                worst_case=round(float(np.percentile(column, 10)), 2),
                expected_range_low=round(float(np.percentile(column, 25)), 2),
                expected_range_high=round(float(np.percentile(column, 75)), 2),
                nominal_median=round(nominal_median, 2),
                real_median=round(real_median, 2),
            )
        )
    return tuple(projections)


def run_scenario_simulation(
    *,
    recommendation: RecommendationResponse,
    style: PortfolioStyle,
    initial_investment: float,
    monthly_sip: int,
    horizon_years: int,
    goal_amount: float | None,
    inflation_rate: float,
    risk_free_rate: float,
    config: SimulationConfig,
    random_seed: int,
    num_simulations: int,
) -> ScenarioSimulationResult:
    """Run an independent Monte Carlo simulation for one portfolio style."""
    variant = _variant_for_style(recommendation, style)
    expected_return = midpoint_return(variant.expected_annual_return_min, variant.expected_annual_return_max)
    volatility = decimal_volatility(variant.estimated_risk)

    yearly_paths = simulate_portfolio_paths(
        initial_investment=initial_investment,
        monthly_contribution=float(monthly_sip),
        horizon_years=horizon_years,
        expected_annual_return=expected_return,
        annual_volatility=volatility,
        num_simulations=num_simulations,
        random_seed=random_seed + _seed_offset(style),
    )
    final_values = yearly_paths[:, -1]
    metrics = DistributionMetrics(**distribution_metrics(final_values, config.percentile_levels))
    goal_probability: GoalProbabilityResult | None = None
    if goal_amount is not None:
        goal_probability = calculate_goal_probability(final_values, goal_amount)

    inflation_adjusted = _inflation_adjusted(metrics.median_value, inflation_rate=inflation_rate, horizon_years=horizon_years)
    analytics: PortfolioAnalyticsResult = calculate_portfolio_analytics(
        yearly_paths,
        horizon_years=horizon_years,
        risk_free_rate=risk_free_rate,
        initial_investment=initial_investment,
        monthly_contribution=float(monthly_sip),
    )
    yearly = _yearly_projections(yearly_paths, inflation_rate=inflation_rate)
    LOGGER.info(
        "Simulated %s scenario: median=%.2f goal_success=%s",
        style.value,
        metrics.median_value,
        None if goal_probability is None else goal_probability.probability_of_success,
    )
    return ScenarioSimulationResult(
        style=style,
        expected_return_range=variant.expected_return_range,
        expected_annual_return_min=variant.expected_annual_return_min,
        expected_annual_return_max=variant.expected_annual_return_max,
        estimated_risk=variant.estimated_risk,
        simulation=metrics,
        goal_probability=goal_probability,
        inflation_adjusted=inflation_adjusted,
        analytics=analytics,
        yearly_projections=yearly,
    )


def run_all_scenarios(
    *,
    recommendation: RecommendationResponse,
    initial_investment: float,
    monthly_sip: int,
    horizon_years: int,
    goal_amount: float | None,
    inflation_rate: float,
    risk_free_rate: float,
    config: SimulationConfig,
    random_seed: int,
    num_simulations: int,
) -> tuple[ScenarioSimulationResult, ...]:
    """Run independent simulations for conservative, recommended, and growth portfolios."""
    results = [
        run_scenario_simulation(
            recommendation=recommendation,
            style=style,
            initial_investment=initial_investment,
            monthly_sip=monthly_sip,
            horizon_years=horizon_years,
            goal_amount=goal_amount,
            inflation_rate=inflation_rate,
            risk_free_rate=risk_free_rate,
            config=config,
            random_seed=random_seed,
            num_simulations=num_simulations,
        )
        for style in (PortfolioStyle.CONSERVATIVE, PortfolioStyle.RECOMMENDED, PortfolioStyle.GROWTH)
    ]
    return tuple(results)


def _seed_offset(style: PortfolioStyle) -> int:
    return {
        PortfolioStyle.CONSERVATIVE: 11,
        PortfolioStyle.RECOMMENDED: 23,
        PortfolioStyle.GROWTH: 37,
    }[style]
