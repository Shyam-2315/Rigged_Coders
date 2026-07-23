"""Facade and report generator for the Phase 7 projection engine."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from pathlib import Path

from ml.recommendation.schemas import PortfolioStyle

from .charts import generate_projection_charts
from .config import SimulationConfig
from .scenario_engine import run_all_scenarios
from .schemas import ProjectionRequest, ProjectionResponse
from .validation import coerce_projection_request, resolve_monthly_sip

LOGGER = logging.getLogger(__name__)


class FinancialProjectionEngine:
    """Monte Carlo orchestration entry point suitable for a FastAPI route."""

    def __init__(self, *, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()

    def project(self, request: ProjectionRequest | Mapping[str, object]) -> ProjectionResponse:
        """Generate educational long-term projections from a Phase 6 recommendation."""
        parsed = coerce_projection_request(request)
        assumptions = self.config.load_market_assumptions()
        inflation_rate = parsed.inflation_rate if parsed.inflation_rate is not None else assumptions.inflation_rate
        monthly_sip = resolve_monthly_sip(parsed)
        scenarios = run_all_scenarios(
            recommendation=parsed.recommendation,
            initial_investment=parsed.initial_investment,
            monthly_sip=monthly_sip,
            horizon_years=parsed.investment_horizon_years,
            goal_amount=parsed.goal_amount,
            inflation_rate=inflation_rate,
            risk_free_rate=assumptions.risk_free_rate,
            config=self.config,
            random_seed=parsed.random_seed,
            num_simulations=parsed.num_simulations,
        )
        active = next(item for item in scenarios if item.style == parsed.active_scenario)
        goal_probability = active.goal_probability
        summary = _build_summary(active, parsed, monthly_sip)
        charts = None
        reports = None
        if parsed.generate_charts:
            charts = generate_projection_charts(
                ProjectionResponse(
                    simulation=active.simulation,
                    goal_probability=goal_probability,
                    inflation_adjusted=active.inflation_adjusted,
                    analytics=active.analytics,
                    yearly_projections=active.yearly_projections,
                    scenarios=scenarios,
                    active_scenario=parsed.active_scenario,
                    explanation_summary=summary,
                    educational_disclaimer=self.config.educational_disclaimer,
                    simulation_version=self.config.simulation_version,
                ),
                config=self.config,
                initial_investment=parsed.initial_investment,
                monthly_sip=monthly_sip,
                horizon_years=parsed.investment_horizon_years,
                random_seed=parsed.random_seed,
                num_simulations=parsed.num_simulations,
            )
        response = ProjectionResponse(
            simulation=active.simulation,
            goal_probability=goal_probability,
            inflation_adjusted=active.inflation_adjusted,
            analytics=active.analytics,
            yearly_projections=active.yearly_projections,
            scenarios=scenarios,
            active_scenario=parsed.active_scenario,
            explanation_summary=summary,
            educational_disclaimer=self.config.educational_disclaimer,
            simulation_version=self.config.simulation_version,
            charts=charts,
            reports=reports,
        )
        if parsed.generate_reports:
            report_paths = generate_phase7_reports(response, config=self.config)
            response = response.model_copy(update={"reports": report_paths})
        LOGGER.info(
            "Generated Phase 7 projection for %s scenario: median=%.2f runs=%s",
            parsed.active_scenario.value,
            active.simulation.median_value,
            active.simulation.runs,
        )
        return response


def project_portfolio(request: ProjectionRequest | Mapping[str, object]) -> ProjectionResponse:
    """Convenience function for scripts, FastAPI dependencies, and report generation."""
    return FinancialProjectionEngine().project(request)


def generate_phase7_reports(
    response: ProjectionResponse,
    *,
    config: SimulationConfig | None = None,
) -> dict[str, str]:
    """Generate the four requested Phase 7 reproducible report artifacts."""
    active_config = config or SimulationConfig()
    active_config.create_output_directories()
    summary_payload = {
        "simulation_version": response.simulation_version,
        "active_scenario": response.active_scenario.value,
        "simulation": response.simulation.model_dump(mode="json"),
        "goal_probability": None if response.goal_probability is None else response.goal_probability.model_dump(mode="json"),
        "inflation_adjusted": response.inflation_adjusted.model_dump(mode="json"),
        "analytics": response.analytics.model_dump(mode="json"),
        "scenarios": [item.model_dump(mode="json") for item in response.scenarios],
        "explanation_summary": response.explanation_summary,
        "educational_disclaimer": response.educational_disclaimer,
    }
    _write_json(active_config.simulation_summary_path, summary_payload)
    _write_text(active_config.projection_report_path, _projection_report_markdown(response, active_config))
    _write_text(active_config.scenario_comparison_path, _scenario_comparison_markdown(response, active_config))
    _write_yearly_csv(active_config.yearly_projection_path, response)
    report_paths = {
        "simulation_summary": str(active_config.simulation_summary_path),
        "projection_report": str(active_config.projection_report_path),
        "scenario_comparison": str(active_config.scenario_comparison_path),
        "yearly_projection": str(active_config.yearly_projection_path),
    }
    LOGGER.info("Generated Phase 7 reports in %s", active_config.reports_dir)
    return report_paths


def generate_sample_phase7_reports(config: SimulationConfig | None = None) -> dict[str, str]:
    """Generate reproducible Phase 7 reports from the canonical Phase 6 example."""
    from ml.recommendation import recommend_portfolio

    active_config = config or SimulationConfig()
    recommendation = recommend_portfolio(
        {
            "credit_score": 760,
            "risk_profile": "Medium",
            "behavioral_persona": "Balanced Builder",
            "monthly_income": 85000,
            "monthly_savings": 22000,
            "monthly_budget": 10000,
            "existing_savings": 350000,
            "investment_goal": "Wealth Creation",
            "investment_horizon": 10,
            "age": 31,
            "dependents": 1,
            "emergency_fund_months": 5,
            "income_stability": "High",
        }
    )
    response = FinancialProjectionEngine(config=active_config).project(
        {
            "initial_investment": 350000,
            "monthly_sip": 10000,
            "investment_horizon_years": 10,
            "goal_amount": 2_500_000,
            "recommendation": recommendation.model_dump(mode="json", by_alias=True),
            "generate_charts": True,
            "generate_reports": False,
        }
    )
    return generate_phase7_reports(response, config=active_config)


def _build_summary(active, request: ProjectionRequest, monthly_sip: int) -> str:
    goal_text = ""
    if active.goal_probability is not None:
        goal_text = (
            f"your {request.active_scenario.value.lower()} portfolio has approximately a "
            f"{active.goal_probability.probability_of_success:.0f}% probability of reaching your goal amount "
            f"of ₹{active.goal_probability.goal_amount:,.0f} over the selected {request.investment_horizon_years}-year horizon. "
        )
    return (
        f"Based on {active.simulation.runs:,} simulated market paths using the assumptions defined in TrustVest AI, "
        f"{goal_text}"
        f"The median projected portfolio value is ₹{active.simulation.median_value:,.0f}, with an illustrative "
        f"10th–90th percentile range of ₹{active.simulation.percentile_10:,.0f} to ₹{active.simulation.percentile_90:,.0f}. "
        f"After adjusting for inflation, the median real value is approximately ₹{active.inflation_adjusted.real:,.0f}. "
        f"These are simulated outcomes intended for educational purposes and should not be interpreted as predictions. "
        f"The illustration assumes an initial investment of ₹{request.initial_investment:,.0f} and a monthly contribution of ₹{monthly_sip:,}."
    )


def _projection_report_markdown(response: ProjectionResponse, config: SimulationConfig) -> str:
    goal_section = "No goal amount was supplied for this illustration."
    if response.goal_probability is not None:
        goal = response.goal_probability
        goal_section = (
            f"- Goal amount: ₹{goal.goal_amount:,.0f}\n"
            f"- Probability of success: {goal.probability_of_success:.2f}%\n"
            f"- Probability of shortfall: {goal.probability_of_shortfall:.2f}%\n"
            f"- Probability of exceeding goal: {goal.probability_of_exceeding:.2f}%"
        )
    return f"""# Phase 7 Projection Report

## Scope

This report summarises an educational Monte Carlo simulation built from Phase 6 portfolio recommendations and Phase 5 planning assumptions. It does **not** predict future market performance or provide regulated investment advice.

## Active Scenario

- Scenario: {response.active_scenario.value}
- Simulation runs: {response.simulation.runs:,}
- Median terminal value: ₹{response.simulation.median_value:,.0f}
- Mean terminal value: ₹{response.simulation.mean_value:,.0f}
- 10th percentile: ₹{response.simulation.percentile_10:,.0f}
- 90th percentile: ₹{response.simulation.percentile_90:,.0f}

## Goal Probability

{goal_section}

## Inflation Adjustment

- Nominal median: ₹{response.inflation_adjusted.nominal:,.0f}
- Real median: ₹{response.inflation_adjusted.real:,.0f}
- Inflation assumption: {response.inflation_adjusted.inflation_rate_used * 100:.2f}%

## Portfolio Analytics

- Expected CAGR: {response.analytics.expected_cagr:.2f}%
- Volatility: {response.analytics.volatility:.2f}%
- Sharpe ratio (illustrative): {response.analytics.sharpe_ratio:.2f}
- Maximum drawdown (simulated median path): {response.analytics.max_drawdown:.2f}%
- Downside risk: {response.analytics.downside_risk:.2f}%
- Upside capture (illustrative): {response.analytics.upside_capture:.2f}%

## Summary

{response.explanation_summary}

## Disclaimer

{config.educational_disclaimer}
"""


def _scenario_comparison_markdown(response: ProjectionResponse, config: SimulationConfig) -> str:
    lines = [
        "# Phase 7 Scenario Comparison",
        "",
        config.educational_disclaimer,
        "",
        "| Scenario | Return Range | Risk | Median Value | Goal Success | CAGR |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario in response.scenarios:
        goal_success = "N/A"
        if scenario.goal_probability is not None:
            goal_success = f"{scenario.goal_probability.probability_of_success:.1f}%"
        lines.append(
            f"| {scenario.style.value} | {scenario.expected_return_range} | {scenario.estimated_risk:.2f}% | "
            f"₹{scenario.simulation.median_value:,.0f} | {goal_success} | {scenario.analytics.expected_cagr:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def _write_yearly_csv(path: Path, response: ProjectionResponse) -> None:
    active = next(item for item in response.scenarios if item.style == response.active_scenario)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "year",
                "median",
                "best_case",
                "worst_case",
                "expected_range_low",
                "expected_range_high",
                "nominal_median",
                "real_median",
            ],
        )
        writer.writeheader()
        for point in active.yearly_projections:
            writer.writerow(point.model_dump(mode="json"))


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: Path, contents: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


if __name__ == "__main__":
    generate_sample_phase7_reports()
