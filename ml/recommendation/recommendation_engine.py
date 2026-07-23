"""Facade and report generator for the Phase 6 portfolio recommendation engine."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

from .allocation_engine import AllocationEngine
from .config import RecommendationConfig
from .explanation_engine import ExplanationEngine
from .portfolio_optimizer import PortfolioOptimizer, default_strategies
from .schemas import (
    AlternativePortfolios,
    PortfolioStyle,
    PortfolioVariant,
    RecommendationRequest,
    RecommendationResponse,
)
from .scoring_engine import CandidateProductRetriever, ProductScoringEngine
from .validation import coerce_request
from ml.investments.repository import InvestmentProductRepository


LOGGER = logging.getLogger(__name__)


class PortfolioRecommendationEngine:
    """Repository-backed orchestration entry point suitable for a FastAPI route."""

    def __init__(
        self,
        *,
        config: RecommendationConfig | None = None,
        repository: InvestmentProductRepository | None = None,
    ) -> None:
        self.config = config or RecommendationConfig()
        self.repository = repository or InvestmentProductRepository()
        self.retriever = CandidateProductRetriever(self.repository, self.config)
        self.scorer = ProductScoringEngine(self.config)
        self.optimizer = PortfolioOptimizer(self.config)
        self.allocator = AllocationEngine(self.config)
        self.explainer = ExplanationEngine(self.config)

    def recommend(self, request: RecommendationRequest | Mapping[str, object]) -> RecommendationResponse:
        """Create an explainable portfolio illustration without forecasting markets."""
        parsed_request = coerce_request(request)
        products = self.retriever.retrieve(parsed_request)
        candidates = self.scorer.score(products, parsed_request)
        variants: dict[PortfolioStyle, PortfolioVariant] = {}
        explanations = {}
        for strategy in default_strategies(self.config):
            optimized = self.optimizer.optimize(candidates, parsed_request, strategy)
            allocated = self.allocator.allocate(optimized, parsed_request)
            explained = self.explainer.explain_portfolio(allocated, optimized.selected, parsed_request, strategy.style)
            explanations[strategy.style] = explained
            variants[strategy.style] = PortfolioVariant(
                style=strategy.style,
                expected_return_range=allocated.expected_return_range,
                expected_annual_return_min=allocated.expected_annual_return_min,
                expected_annual_return_max=allocated.expected_annual_return_max,
                estimated_risk=allocated.estimated_risk,
                risk_level=allocated.risk_level,
                allocation=explained.allocation,
                portfolio_health=allocated.health,
            )
        recommended = variants[PortfolioStyle.RECOMMENDED]
        LOGGER.info(
            "Generated Phase 6 portfolio: score=%s products=%s risk=%s",
            recommended.portfolio_health.overall_portfolio_score,
            len(recommended.allocation),
            recommended.risk_level.value,
        )
        return RecommendationResponse(
            portfolio_score=recommended.portfolio_health.overall_portfolio_score,
            expected_return_range=recommended.expected_return_range,
            expected_annual_return_min=recommended.expected_annual_return_min,
            expected_annual_return_max=recommended.expected_annual_return_max,
            estimated_risk=recommended.estimated_risk,
            risk_level=recommended.risk_level,
            monthly_budget=int(round(parsed_request.monthly_budget)),
            recommended_portfolio=recommended.allocation,
            portfolio_health=recommended.portfolio_health,
            alternative_portfolios=AlternativePortfolios(
                conservative=variants[PortfolioStyle.CONSERVATIVE],
                recommended=recommended,
                growth=variants[PortfolioStyle.GROWTH],
            ),
            insights=self.explainer.insights(recommended.allocation, parsed_request),
            explanation_summary=explanations[PortfolioStyle.RECOMMENDED].summary,
            trade_offs=explanations[PortfolioStyle.RECOMMENDED].trade_offs,
            rules_version=self.config.load_rules().get("version", self.config.rules_version),
            educational_disclaimer=self.config.educational_disclaimer,
        )


def recommend_portfolio(request: RecommendationRequest | Mapping[str, object]) -> RecommendationResponse:
    """Convenience function for scripts, FastAPI dependencies, and Phase 7 callers."""
    return PortfolioRecommendationEngine().recommend(request)


def generate_phase6_reports(config: RecommendationConfig | None = None) -> tuple[Path, Path, Path, Path]:
    """Generate the four requested Phase 6 reproducible report artifacts."""
    active_config = config or RecommendationConfig()
    active_config.create_output_directories()
    engine = PortfolioRecommendationEngine(config=active_config)
    examples = [
        {
            "name": "Balanced long-term wealth illustration",
            "input": {
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
                "income_stability": "High"
            },
        },
        {
            "name": "Conservative near-term illustration",
            "input": {
                "credit_score": 690,
                "risk_profile": "Low",
                "behavioral_persona": "Conservative Planner",
                "monthly_income": 60000,
                "monthly_savings": 12000,
                "monthly_budget": 6000,
                "existing_savings": 70000,
                "investment_goal": "Car",
                "investment_horizon": 3,
                "age": 38,
                "dependents": 2,
                "emergency_fund_months": 2,
                "income_stability": "Medium",
                "government_backed_preference": True
            },
        },
        {
            "name": "Growth-oriented retirement illustration",
            "input": {
                "credit_score": 790,
                "risk_profile": "High",
                "behavioral_persona": "Growth Explorer",
                "monthly_income": 140000,
                "monthly_savings": 45000,
                "monthly_budget": 25000,
                "existing_savings": 900000,
                "investment_goal": "Retirement",
                "investment_horizon": 20,
                "age": 29,
                "dependents": 0,
                "emergency_fund_months": 8,
                "income_stability": "High"
            },
        },
    ]
    rendered_examples = [
        {**example, "output": engine.recommend(example["input"]).model_dump(mode="json", by_alias=True)}
        for example in examples
    ]
    portfolio_examples_path = active_config.reports_dir / "portfolio_examples.json"
    portfolio_rules_path = active_config.reports_dir / "portfolio_rules.md"
    portfolio_metrics_path = active_config.reports_dir / "portfolio_metrics.json"
    recommendation_examples_path = active_config.reports_dir / "recommendation_examples.md"
    _write_json(portfolio_examples_path, rendered_examples)
    _write_text(portfolio_rules_path, _rules_markdown(active_config))
    _write_json(
        portfolio_metrics_path,
        {
            "version": active_config.rules_version,
            "catalog_product_count": len(engine.repository.get_all()),
            "rules": active_config.load_rules(),
            "sample_portfolio_scores": {
                example["name"]: example["output"]["portfolio_score"] for example in rendered_examples
            },
            "disclaimer": active_config.educational_disclaimer,
        },
    )
    _write_text(recommendation_examples_path, _examples_markdown(rendered_examples, active_config))
    LOGGER.info("Generated Phase 6 reports in %s", active_config.reports_dir)
    return portfolio_examples_path, portfolio_rules_path, portfolio_metrics_path, recommendation_examples_path


def _rules_markdown(config: RecommendationConfig) -> str:
    rules = config.load_rules()
    weights = "\n".join(f"| {factor.replace('_', ' ').title()} | {weight}% |" for factor, weight in rules["scoring_weights"].items())
    return f"""# Phase 6 Portfolio Rules

## Scope

This engine creates deterministic, educational portfolio illustrations from the Phase 5 product catalog. It does not forecast markets, provide regulated financial advice, or execute investments.

## Product retrieval

Products must meet the investor's risk ceiling, credit-score requirement, monthly allocation cap, and a liquidity/horizon screen. Goal, behavioral persona, and government-backed preference are scored so sparse goals can still meet the three-product diversification policy.

## Product scoring

| Factor | Weight |
| --- | ---: |
{weights}

## Portfolio constraints

- {config.minimum_products} to {config.maximum_products} products
- Minimum {config.minimum_allocation_percent:.0f}% allocation per selected product
- Maximum {config.maximum_single_product_percent:.0f}% allocation to one product
- Integer monthly amounts must sum exactly to the supplied budget
- Asset-class diversification is preferred across equity, debt, gold, cash, and government savings where catalog suitability permits

## Health metrics

Diversification, risk alignment, liquidity, inflation protection, goal alignment, tax efficiency, portfolio stability, and the weighted overall score are all rule-based 0–100 measures.

## Disclaimer

{config.educational_disclaimer}
"""


def _examples_markdown(examples: list[dict[str, object]], config: RecommendationConfig) -> str:
    sections = ["# Phase 6 Recommendation Examples", "", config.educational_disclaimer]
    for example in examples:
        output = example["output"]
        assert isinstance(output, dict)
        sections.extend(
            [
                "",
                f"## {example['name']}",
                "",
                f"Portfolio score: {output['portfolio_score']}/100  ",
                f"Planning return range: {output['expected_return_range']}  ",
                f"Illustrative risk: {output['estimated_risk']}% ({output['risk_level']})",
                "",
                "| Product | Allocation | Monthly amount |",
                "| --- | ---: | ---: |",
            ]
        )
        for item in output["recommended_portfolio"]:
            assert isinstance(item, dict)
            sections.append(f"| {item['product']} | {item['allocation']}% | ₹{item['monthly_amount']:,} |")
        sections.extend(["", "Insights:"])
        sections.extend(f"- {insight}" for insight in output["insights"])
    return "\n".join(sections) + "\n"


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: Path, contents: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


if __name__ == "__main__":
    generate_phase6_reports()
