"""Seed data and artifact generation for the TrustVest investment knowledge base.

All product returns and risk statistics in this module are illustrative
planning ranges. They are not live market data, quotes, guarantees, or advice.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from typing import Iterable

from .config import InvestmentKnowledgeConfig
from .schemas import (
    BehavioralPersonaName,
    InvestmentGoal,
    InvestmentProduct,
    LiquidityLevel,
    MarketAssumptions,
    NumericRange,
    ProductCategory,
    RiskLevel,
    TaxEfficiency,
)
from .validation import validate_catalog


LOGGER = logging.getLogger(__name__)


def build_investment_products() -> list[InvestmentProduct]:
    """Return the curated catalog of 25 Indian-market product categories."""
    low_personas = (BehavioralPersonaName.SECURE_SAVER, BehavioralPersonaName.CONSERVATIVE_PLANNER)
    medium_personas = (BehavioralPersonaName.BALANCED_BUILDER, BehavioralPersonaName.STRATEGIC_INVESTOR)
    high_personas = (BehavioralPersonaName.GROWTH_EXPLORER, BehavioralPersonaName.AGGRESSIVE_WEALTH_SEEKER)
    all_personas = (*low_personas, *medium_personas, *high_personas)
    emergency_goals = (InvestmentGoal.EMERGENCY_FUND, InvestmentGoal.VACATION, InvestmentGoal.CAR)
    wealth_goals = (InvestmentGoal.WEALTH_CREATION, InvestmentGoal.RETIREMENT, InvestmentGoal.CHILD_EDUCATION)

    products = [
        _product(
            id="liquid-fund", name="Liquid Fund", category=ProductCategory.DEBT, subcategory="Liquid Fund", risk=RiskLevel.LOW,
            goals=emergency_goals, personas=low_personas, returns=(5.0, 7.0), volatility=0.8, liquidity=LiquidityLevel.HIGH,
            withdrawal="1 business day", horizon=1, minimum=500, sip=500, expense=0.25, emergency=True,
            summary="A debt mutual fund investing in very short-term instruments with a focus on capital stability and quick access.",
            risks=("Returns can change with short-term interest rates", "Not covered by a bank deposit guarantee"),
            terms=("safe investment", "liquid", "parking money", "emergency"), popularity=78, trust=82,
        ),
        _product(
            id="overnight-fund", name="Overnight Fund", category=ProductCategory.DEBT, subcategory="Overnight Fund", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.EMERGENCY_FUND, InvestmentGoal.VACATION), personas=low_personas, returns=(4.5, 6.5), volatility=0.2,
            liquidity=LiquidityLevel.HIGH, withdrawal="1 business day", horizon=1, minimum=500, sip=500, expense=0.18, emergency=True,
            summary="A debt fund that holds securities maturing overnight, designed to minimise interest-rate sensitivity.",
            risks=("Returns are usually lower than longer-duration debt funds", "Not covered by a bank deposit guarantee"),
            terms=("safe investment", "overnight", "cash parking", "emergency"), popularity=65, trust=84,
        ),
        _product(
            id="ultra-short-duration-fund", name="Ultra Short Duration Fund", category=ProductCategory.DEBT, subcategory="Ultra Short Duration Fund", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.VACATION, InvestmentGoal.CAR, InvestmentGoal.EMERGENCY_FUND), personas=low_personas,
            returns=(5.5, 7.5), volatility=1.5, liquidity=LiquidityLevel.HIGH, withdrawal="1-2 business days", horizon=1,
            minimum=500, sip=500, expense=0.35, emergency=True,
            summary="A short-maturity debt fund intended for money that may be needed within roughly a year.",
            risks=("Credit quality differs by fund", "Net asset value can fluctuate modestly"),
            terms=("safe investment", "short term", "ultra short", "liquid"), popularity=69, trust=80,
        ),
        _product(
            id="money-market-fund", name="Money Market Fund", category=ProductCategory.DEBT, subcategory="Money Market Fund", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.VACATION, InvestmentGoal.CAR, InvestmentGoal.HOUSE), personas=low_personas,
            returns=(5.5, 7.5), volatility=1.8, liquidity=LiquidityLevel.HIGH, withdrawal="1-2 business days", horizon=1,
            minimum=500, sip=500, expense=0.32,
            summary="A debt fund investing in money-market instruments with maturities typically under one year.",
            risks=("Interest-rate and credit changes can affect returns", "Not a fixed-return product"),
            terms=("safe investment", "money market", "short term", "cash"), popularity=67, trust=81,
        ),
        _product(
            id="corporate-bond-fund", name="Corporate Bond Fund", category=ProductCategory.DEBT, subcategory="Corporate Bond Fund", risk=RiskLevel.MEDIUM,
            goals=(InvestmentGoal.EDUCATION, InvestmentGoal.HOUSE, InvestmentGoal.PASSIVE_INCOME), personas=(*low_personas, *medium_personas),
            returns=(6.0, 8.5), volatility=3.5, liquidity=LiquidityLevel.MODERATE, withdrawal="2-3 business days", horizon=3,
            minimum=500, sip=1_000, expense=0.45,
            summary="A debt fund mainly holding higher-rated company bonds for investors seeking income with moderate fluctuation.",
            risks=("Issuer credit risk", "Bond prices can fall when interest rates rise"),
            terms=("bond", "debt income", "corporate bond", "monthly income"), popularity=63, trust=78,
        ),
        _product(
            id="banking-psu-fund", name="Banking & PSU Fund", category=ProductCategory.DEBT, subcategory="Banking and PSU Debt Fund", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.EDUCATION, InvestmentGoal.HOUSE, InvestmentGoal.PASSIVE_INCOME), personas=(*low_personas, BehavioralPersonaName.BALANCED_BUILDER),
            returns=(6.0, 8.0), volatility=3.0, liquidity=LiquidityLevel.MODERATE, withdrawal="2-3 business days", horizon=3,
            minimum=500, sip=1_000, expense=0.40,
            summary="A debt fund investing primarily in bonds issued by banks and public-sector entities.",
            risks=("Interest-rate sensitivity", "Sector concentration in financial and public-sector issuers"),
            terms=("banking debt", "psu bond", "safe investment", "income"), popularity=58, trust=83,
        ),
        _product(
            id="gilt-fund", name="Gilt Fund", category=ProductCategory.DEBT, subcategory="Government Securities Fund", risk=RiskLevel.MEDIUM,
            goals=(InvestmentGoal.EDUCATION, InvestmentGoal.RETIREMENT, InvestmentGoal.PASSIVE_INCOME), personas=(*low_personas, *medium_personas),
            returns=(5.5, 8.5), volatility=6.5, liquidity=LiquidityLevel.MODERATE, withdrawal="2-3 business days", horizon=5,
            minimum=500, sip=1_000, expense=0.35,
            summary="A debt fund investing in government securities, with low default risk but meaningful interest-rate sensitivity.",
            risks=("Long-duration gilts can fluctuate sharply when rates change", "Returns are not fixed"),
            terms=("gilt", "government bond", "debt", "retirement"), popularity=55, trust=86,
        ),
        _product(
            id="fixed-deposit", name="Fixed Deposit", category=ProductCategory.FIXED_INCOME, subcategory="Bank Fixed Deposit", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.VACATION, InvestmentGoal.EDUCATION, InvestmentGoal.MARRIAGE, InvestmentGoal.CAR), personas=low_personas,
            returns=(5.5, 7.5), volatility=0.0, liquidity=LiquidityLevel.MODERATE, withdrawal="Premature withdrawal with penalty", horizon=2,
            minimum=1_000, sip=0, expense=0.0, tax=TaxEfficiency.LOW, inflation=False,
            summary="A bank deposit with a stated interest rate and maturity date, generally used for predictable short- to medium-term goals.",
            risks=("Interest is commonly taxable", "Premature withdrawal may reduce returns", "Inflation can reduce real purchasing power"),
            terms=("safe investment", "fd", "fixed deposit", "guaranteed"), popularity=88, trust=88,
        ),
        _product(
            id="recurring-deposit", name="Recurring Deposit", category=ProductCategory.FIXED_INCOME, subcategory="Bank Recurring Deposit", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.VACATION, InvestmentGoal.EDUCATION, InvestmentGoal.MARRIAGE, InvestmentGoal.CAR), personas=low_personas,
            returns=(5.0, 7.0), volatility=0.0, liquidity=LiquidityLevel.MODERATE, withdrawal="Premature withdrawal with penalty", horizon=2,
            minimum=100, sip=500, expense=0.0, tax=TaxEfficiency.LOW, inflation=False, students=True,
            summary="A scheduled bank deposit where a fixed amount is contributed each month for a chosen tenure.",
            risks=("Interest is commonly taxable", "Returns may not outpace inflation", "Early closure can attract a penalty"),
            terms=("safe investment", "rd", "recurring deposit", "monthly saving"), popularity=80, trust=87,
        ),
        _product(
            id="ppf", name="PPF", category=ProductCategory.GOVERNMENT_SAVINGS, subcategory="Public Provident Fund", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.RETIREMENT, InvestmentGoal.TAX_SAVING, InvestmentGoal.CHILD_EDUCATION), personas=low_personas,
            returns=(6.5, 7.5), volatility=0.0, liquidity=LiquidityLevel.LOW, withdrawal="Partial withdrawal subject to scheme rules", horizon=15,
            minimum=500, sip=500, expense=0.0, tax=TaxEfficiency.HIGH, inflation=True, government=True,
            summary="A long-term government savings scheme with a statutory lock-in and tax-aware retirement focus.",
            risks=("Long lock-in reduces flexibility", "Rates can be revised by the government", "Contribution limits apply"),
            terms=("ppf", "tax", "retirement", "government saving"), popularity=84, trust=96,
        ),
        _product(
            id="epf", name="EPF", category=ProductCategory.GOVERNMENT_SAVINGS, subcategory="Employees Provident Fund", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.RETIREMENT, InvestmentGoal.TAX_SAVING), personas=low_personas,
            returns=(7.0, 8.5), volatility=0.0, liquidity=LiquidityLevel.LOW, withdrawal="Subject to employment and scheme rules", horizon=15,
            minimum=0, sip=0, expense=0.0, tax=TaxEfficiency.HIGH, inflation=True, government=True,
            summary="A regulated retirement savings arrangement for eligible employees, typically funded through salary contributions.",
            risks=("Access is restricted by scheme rules", "Eligibility depends on employment", "Rates can change over time"),
            terms=("epf", "provident fund", "retirement", "tax"), popularity=86, trust=95,
        ),
        _product(
            id="nps", name="NPS", category=ProductCategory.GOVERNMENT_SAVINGS, subcategory="National Pension System", risk=RiskLevel.MEDIUM,
            goals=(InvestmentGoal.RETIREMENT, InvestmentGoal.TAX_SAVING), personas=(*low_personas, *medium_personas),
            returns=(7.0, 11.0), volatility=10.0, liquidity=LiquidityLevel.LOW, withdrawal="Restricted until retirement subject to scheme rules", horizon=15,
            minimum=500, sip=500, expense=0.12, tax=TaxEfficiency.HIGH, inflation=True, government=True,
            summary="A regulated long-term pension account that can combine equity, corporate debt, and government securities.",
            risks=("Withdrawals are restricted", "Market-linked choices can fluctuate", "Annuity rules may apply"),
            terms=("nps", "pension", "retirement", "tax"), popularity=74, trust=91,
        ),
        _product(
            id="index-fund", name="Index Fund", category=ProductCategory.EQUITY, subcategory="Broad Market Index Fund", risk=RiskLevel.MEDIUM,
            goals=wealth_goals, personas=(*medium_personas, BehavioralPersonaName.GROWTH_EXPLORER), returns=(9.0, 13.0), volatility=16.0,
            liquidity=LiquidityLevel.MODERATE, withdrawal="2-3 business days", horizon=7, minimum=500, sip=500, expense=0.20,
            summary="An equity mutual fund designed to track a market index at a relatively low cost.",
            risks=("Market value can fall substantially in the short term", "Index concentration and tracking error are possible"),
            terms=("index", "passive", "equity", "wealth creation", "retirement"), popularity=91, trust=86,
        ),
        _product(
            id="large-cap-fund", name="Large Cap Fund", category=ProductCategory.EQUITY, subcategory="Large Cap Equity Fund", risk=RiskLevel.MEDIUM,
            goals=wealth_goals, personas=medium_personas, returns=(8.5, 12.5), volatility=15.0, liquidity=LiquidityLevel.MODERATE,
            withdrawal="2-3 business days", horizon=6, minimum=500, sip=500, expense=0.75,
            summary="An equity mutual fund investing primarily in larger, established companies.",
            risks=("Equity markets are volatile", "Active manager choices may lag the benchmark"),
            terms=("large cap", "equity", "retirement", "wealth creation"), popularity=82, trust=82,
        ),
        _product(
            id="flexi-cap-fund", name="Flexi Cap Fund", category=ProductCategory.EQUITY, subcategory="Flexi Cap Equity Fund", risk=RiskLevel.MEDIUM,
            goals=wealth_goals, personas=medium_personas, returns=(9.0, 14.0), volatility=17.0, liquidity=LiquidityLevel.MODERATE,
            withdrawal="2-3 business days", horizon=7, minimum=500, sip=500, expense=0.85,
            summary="An equity fund whose manager can invest across large-, mid-, and small-company segments.",
            risks=("Broad equity drawdowns", "Portfolio style can change with manager decisions"),
            terms=("flexi cap", "equity", "wealth creation", "long term"), popularity=86, trust=81,
        ),
        _product(
            id="mid-cap-fund", name="Mid Cap Fund", category=ProductCategory.EQUITY, subcategory="Mid Cap Equity Fund", risk=RiskLevel.HIGH,
            goals=(InvestmentGoal.WEALTH_CREATION, InvestmentGoal.RETIREMENT, InvestmentGoal.CHILD_EDUCATION), personas=(*medium_personas, *high_personas),
            returns=(10.0, 16.0), volatility=22.0, liquidity=LiquidityLevel.MODERATE, withdrawal="2-3 business days", horizon=8,
            minimum=500, sip=1_000, expense=0.90,
            summary="An equity fund focused on mid-sized companies with higher growth potential and higher volatility.",
            risks=("Large and prolonged drawdowns", "Mid-cap valuations can change quickly"),
            terms=("mid cap", "growth", "equity", "wealth creation"), popularity=77, trust=78,
        ),
        _product(
            id="small-cap-fund", name="Small Cap Fund", category=ProductCategory.EQUITY, subcategory="Small Cap Equity Fund", risk=RiskLevel.HIGH,
            goals=(InvestmentGoal.WEALTH_CREATION, InvestmentGoal.CHILD_EDUCATION), personas=high_personas, returns=(11.0, 18.0), volatility=28.0,
            liquidity=LiquidityLevel.MODERATE, withdrawal="2-3 business days", horizon=10, minimum=500, sip=1_000, expense=0.95,
            summary="An equity fund investing in smaller companies, intended only for long horizons and high volatility tolerance.",
            risks=("Very high volatility", "Liquidity and valuation risk during market stress", "May underperform for extended periods"),
            terms=("small cap", "high growth", "equity", "aggressive"), popularity=70, trust=74,
        ),
        _product(
            id="elss", name="ELSS", category=ProductCategory.EQUITY, subcategory="Equity Linked Savings Scheme", risk=RiskLevel.HIGH,
            goals=(InvestmentGoal.TAX_SAVING, InvestmentGoal.WEALTH_CREATION, InvestmentGoal.RETIREMENT), personas=(*medium_personas, *high_personas),
            returns=(9.0, 15.0), volatility=20.0, liquidity=LiquidityLevel.LOW, withdrawal="After 3-year statutory lock-in", horizon=7,
            minimum=500, sip=500, expense=0.90, lock_in=36, tax=TaxEfficiency.HIGH, inflation=True,
            summary="A diversified equity mutual fund with a statutory three-year lock-in and tax-saving eligibility subject to law.",
            risks=("Equity losses can occur during the lock-in", "Tax treatment can change", "Cannot be redeemed before lock-in ends"),
            terms=("elss", "tax", "tax saving", "equity"), popularity=82, trust=80,
        ),
        _product(
            id="gold-etf", name="Gold ETF", category=ProductCategory.COMMODITY, subcategory="Gold Exchange Traded Fund", risk=RiskLevel.MEDIUM,
            goals=(InvestmentGoal.WEALTH_CREATION, InvestmentGoal.RETIREMENT),
            personas=(*medium_personas, BehavioralPersonaName.GROWTH_EXPLORER), returns=(5.0, 10.0), volatility=18.0, liquidity=LiquidityLevel.HIGH,
            withdrawal="Market settlement cycle", horizon=5, minimum=100, sip=0, expense=0.45, inflation=True,
            summary="An exchange-traded fund designed to track domestic gold prices without storing physical gold.",
            risks=("Gold prices can be volatile", "No cash income", "Trading requires market access and may involve tracking error"),
            terms=("gold", "gold etf", "inflation", "commodity"), popularity=83, trust=82,
        ),
        _product(
            id="silver-etf", name="Silver ETF", category=ProductCategory.COMMODITY, subcategory="Silver Exchange Traded Fund", risk=RiskLevel.HIGH,
            goals=(InvestmentGoal.WEALTH_CREATION, InvestmentGoal.RETIREMENT), personas=(*medium_personas, *high_personas), returns=(4.0, 12.0), volatility=25.0,
            liquidity=LiquidityLevel.HIGH, withdrawal="Market settlement cycle", horizon=5, minimum=100, sip=0, expense=0.55, inflation=True,
            summary="An exchange-traded fund offering market-linked exposure to silver prices.",
            risks=("Higher commodity-price volatility", "No cash income", "Industrial-demand changes can affect prices"),
            terms=("silver", "silver etf", "commodity", "inflation"), popularity=56, trust=75,
        ),
        _product(
            id="sovereign-gold-bond", name="Sovereign Gold Bond", category=ProductCategory.GOVERNMENT_SAVINGS, subcategory="Sovereign Gold Bond", risk=RiskLevel.MEDIUM,
            goals=(InvestmentGoal.WEALTH_CREATION, InvestmentGoal.RETIREMENT), personas=(*low_personas, *medium_personas), returns=(4.0, 9.0), volatility=16.0,
            liquidity=LiquidityLevel.LOW, withdrawal="Tradable subject to market liquidity; scheduled maturity", horizon=8, minimum=1, sip=0,
            expense=0.0, inflation=True, government=True,
            summary="A government-issued gold-linked security with a long maturity and market-linked value.",
            risks=("Long holding period", "Secondary-market price and liquidity can differ from gold value", "Issue availability varies"),
            terms=("gold", "sovereign gold bond", "sgb", "government"), popularity=73, trust=92,
        ),
        _product(
            id="reit", name="REIT", category=ProductCategory.REAL_ASSET, subcategory="Real Estate Investment Trust", risk=RiskLevel.MEDIUM,
            goals=(InvestmentGoal.PASSIVE_INCOME, InvestmentGoal.WEALTH_CREATION, InvestmentGoal.RETIREMENT), personas=(*medium_personas, BehavioralPersonaName.GROWTH_EXPLORER),
            returns=(7.0, 11.0), volatility=18.0, liquidity=LiquidityLevel.HIGH, withdrawal="Market settlement cycle", horizon=5,
            minimum=100, sip=0, expense=0.50,
            summary="A listed trust that gives market-linked exposure to income-producing real estate assets.",
            risks=("Unit prices can be volatile", "Rental income and property valuations can change", "Market liquidity varies"),
            terms=("reit", "real estate", "monthly income", "passive income"), popularity=66, trust=77,
        ),
        _product(
            id="international-index-fund", name="International Index Fund", category=ProductCategory.INTERNATIONAL_EQUITY, subcategory="International Equity Index Fund", risk=RiskLevel.HIGH,
            goals=wealth_goals, personas=(*medium_personas, *high_personas), returns=(8.0, 14.0), volatility=21.0, liquidity=LiquidityLevel.MODERATE,
            withdrawal="2-4 business days", horizon=8, minimum=500, sip=500, expense=0.65,
            summary="A fund tracking an overseas equity index, adding geographic and currency diversification to a portfolio.",
            risks=("Foreign markets and currency can both fluctuate", "Tax and investment limits may change", "Tracking difference is possible"),
            terms=("international", "global index", "us equity", "diversification"), popularity=72, trust=79,
        ),
        _product(
            id="cash-reserve", name="Cash Reserve", category=ProductCategory.CASH, subcategory="Cash Reserve", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.EMERGENCY_FUND, InvestmentGoal.VACATION, InvestmentGoal.CAR), personas=all_personas, returns=(2.0, 4.0), volatility=0.0,
            liquidity=LiquidityLevel.INSTANT, withdrawal="Immediate", horizon=0, minimum=0, sip=0, expense=0.0, tax=TaxEfficiency.LOW,
            emergency=True, students=True,
            summary="Cash set aside for immediate needs and near-term bills rather than long-term investment growth.",
            risks=("Low return may not outpace inflation", "Physical cash can be lost or stolen", "Large balances may earn little"),
            terms=("cash", "safe investment", "emergency", "liquid"), popularity=90, trust=85,
        ),
        _product(
            id="emergency-savings-account", name="Emergency Savings Account", category=ProductCategory.CASH, subcategory="Savings Bank Account", risk=RiskLevel.LOW,
            goals=(InvestmentGoal.EMERGENCY_FUND, InvestmentGoal.VACATION, InvestmentGoal.CAR), personas=all_personas, returns=(2.5, 5.0), volatility=0.0,
            liquidity=LiquidityLevel.INSTANT, withdrawal="Immediate", horizon=0, minimum=0, sip=0, expense=0.0, tax=TaxEfficiency.LOW,
            emergency=True, students=True,
            summary="A bank savings account used to keep emergency money accessible while earning a variable deposit rate.",
            risks=("Interest may not keep up with inflation", "Deposit insurance limits and bank terms apply", "Rates can change"),
            terms=("emergency savings", "safe investment", "bank account", "liquid"), popularity=95, trust=89,
        ),
    ]
    validate_catalog(products)
    return products


def build_market_assumptions() -> MarketAssumptions:
    """Return transparent, illustrative inputs for a future growth simulator."""
    return MarketAssumptions(
        inflation_rate=0.055,
        risk_free_rate=0.065,
        expected_equity_premium=0.055,
        expected_gold_return=0.070,
        debt_return=0.065,
        cash_return=0.035,
        monte_carlo_default_volatility=0.150,
        note="Illustrative long-term planning assumptions for simulation only; they are not live forecasts or investment advice.",
    )


def bootstrap_knowledge_base(config: InvestmentKnowledgeConfig | None = None, *, overwrite: bool = True) -> list[InvestmentProduct]:
    """Persist the curated catalog, market assumptions, CSV, reports, and docs."""
    from .repository import InvestmentProductRepository

    active_config = config or InvestmentKnowledgeConfig()
    repository = InvestmentProductRepository(active_config)
    products = build_investment_products()
    repository.save_catalog(products, overwrite=overwrite)
    repository.save_market_assumptions(build_market_assumptions(), overwrite=overwrite)
    generate_reports(products, active_config)
    LOGGER.info("Investment knowledge base generated with %s products", len(products))
    return products


def generate_reports(products: Iterable[InvestmentProduct], config: InvestmentKnowledgeConfig | None = None) -> None:
    """Generate catalog, statistics, distributions, and module documentation."""
    active_config = config or InvestmentKnowledgeConfig()
    active_config.create_output_directories()
    catalog = list(products)
    risk_distribution = dict(sorted(Counter(product.risk_level.value for product in catalog).items()))
    goal_distribution = dict(sorted(Counter(goal.value for product in catalog for goal in product.supported_goals).items()))
    active_config.risk_distribution_path.write_text(json.dumps(risk_distribution, indent=2), encoding="utf-8")
    active_config.goal_distribution_path.write_text(json.dumps(goal_distribution, indent=2), encoding="utf-8")
    active_config.catalog_report_path.write_text(_catalog_markdown(catalog), encoding="utf-8")
    active_config.statistics_report_path.write_text(_statistics_markdown(catalog, risk_distribution, goal_distribution), encoding="utf-8")
    active_config.documentation_path.write_text(_documentation_markdown(), encoding="utf-8")


def _product(
    *,
    id: str,
    name: str,
    category: ProductCategory,
    subcategory: str,
    risk: RiskLevel,
    goals: tuple[InvestmentGoal, ...],
    personas: tuple[BehavioralPersonaName, ...],
    returns: tuple[float, float],
    volatility: float,
    liquidity: LiquidityLevel,
    withdrawal: str,
    horizon: int,
    summary: str,
    risks: tuple[str, ...],
    terms: tuple[str, ...],
    popularity: float,
    trust: float,
    minimum: float = 500,
    sip: float = 500,
    lock_in: int = 0,
    tax: TaxEfficiency = TaxEfficiency.MODERATE,
    inflation: bool = False,
    expense: float = 0.50,
    profiles: tuple[RiskLevel, ...] | None = None,
    beginners: bool = True,
    students: bool = False,
    retirement: bool | None = None,
    emergency: bool = False,
    shariah: bool = False,
    government: bool = False,
    age_range: tuple[float, float] = (18, 75),
    income_range: tuple[float, float] = (0, 10_000_000),
) -> InvestmentProduct:
    """Build one complete product schema while keeping seed data readable."""
    default_profiles = {
        RiskLevel.LOW: (RiskLevel.LOW, RiskLevel.MEDIUM),
        RiskLevel.MEDIUM: (RiskLevel.MEDIUM, RiskLevel.HIGH),
        RiskLevel.HIGH: (RiskLevel.HIGH,),
    }
    return InvestmentProduct(
        id=id,
        name=name,
        category=category,
        subcategory=subcategory,
        description=summary,
        plain_language_description=f"{name}: {summary}",
        risk_level=risk,
        behavioral_personas=personas,
        supported_goals=goals,
        minimum_investment=minimum,
        recommended_monthly_sip=sip,
        expected_annual_return_min=returns[0],
        expected_annual_return_max=returns[1],
        historical_volatility=volatility,
        liquidity=liquidity,
        withdrawal_time=withdrawal,
        lock_in_period=lock_in,
        tax_efficiency=tax,
        inflation_protection=inflation,
        expense_ratio=expense,
        recommended_horizon=horizon,
        ideal_age_range=NumericRange(minimum=age_range[0], maximum=age_range[1]),
        ideal_income_range=NumericRange(minimum=income_range[0], maximum=income_range[1]),
        credit_score_requirement=0,
        risk_profile_requirement=profiles or default_profiles[risk],
        suitable_for_beginners=beginners,
        suitable_for_students=students,
        suitable_for_retirement=InvestmentGoal.RETIREMENT in goals if retirement is None else retirement,
        suitable_for_emergency_fund=emergency,
        shariah_compliant=shariah,
        government_backed=government,
        popularity_score=popularity,
        trust_score=trust,
        potential_risks=risks,
        search_terms=terms,
    )


def _catalog_markdown(products: list[InvestmentProduct]) -> str:
    rows = "\n".join(
        f"| {product.name} | {product.category.value} | {product.risk_level.value} | "
        f"{product.expected_annual_return_min:.1f}%–{product.expected_annual_return_max:.1f}% | "
        f"{product.liquidity.value} | {product.recommended_horizon} years |"
        for product in products
    )
    return f"""# TrustVest Investment Product Catalog

This catalog contains structured, illustrative metadata for **{len(products)}** investment product categories. It supports retrieval and ranking only; it does not recommend a portfolio or provide personalised financial advice.

| Product | Category | Risk | Illustrative return range | Liquidity | Suggested horizon |
| --- | --- | --- | ---: | --- | ---: |
{rows}

Return ranges, volatility, and horizons are planning metadata, not live product quotes, guarantees, or forecasts.
"""


def _statistics_markdown(products: list[InvestmentProduct], risk: dict[str, int], goals: dict[str, int]) -> str:
    category_counts = dict(sorted(Counter(product.category.value for product in products).items()))
    category_rows = "\n".join(f"| {name} | {count} |" for name, count in category_counts.items())
    risk_rows = "\n".join(f"| {name} | {count} |" for name, count in risk.items())
    goal_rows = "\n".join(f"| {name} | {count} |" for name, count in goals.items())
    return f"""# Investment Knowledge Base Statistics

## Catalog Summary

- Products: **{len(products)}**
- Government-backed metadata entries: **{sum(product.government_backed for product in products)}**
- Beginner-suitable entries: **{sum(product.suitable_for_beginners for product in products)}**
- Emergency-fund-suitable entries: **{sum(product.suitable_for_emergency_fund for product in products)}**

## Category Distribution

| Category | Products |
| --- | ---: |
{category_rows}

## Risk Distribution

| Risk level | Products |
| --- | ---: |
{risk_rows}

## Goal Coverage

| Goal | Products supporting goal |
| --- | ---: |
{goal_rows}
"""


def _documentation_markdown() -> str:
    return """# Investment Intelligence Knowledge Base

Phase 5 is TrustVest AI's structured repository of investment-product metadata. It contains static, illustrative product characteristics and market assumptions for later system phases.

## Scope

- Stores and validates product metadata.
- Retrieves and ranks products against explicit query criteria.
- Explains matches, product risks, expected-return ranges, and horizons.

## Out of Scope

- Portfolio construction, allocation percentages, product purchase instructions, or suitability decisions.
- Live prices, real-time returns, tax advice, or guarantees.

The future Recommendation Engine may consume this package, subject to its own governance and suitability controls.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the TrustVest investment knowledge-base artifacts.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if product data already exists.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _parse_args()
    bootstrap_knowledge_base(overwrite=not arguments.no_overwrite)


if __name__ == "__main__":
    main()
