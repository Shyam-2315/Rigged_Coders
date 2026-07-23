"""Configuration for the Phase 5 investment knowledge base."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


INVESTMENTS_ROOT = Path(__file__).resolve().parent
ML_ROOT = INVESTMENTS_ROOT.parent


@dataclass(frozen=True, slots=True)
class InvestmentKnowledgeConfig:
    """Filesystem locations and catalog constraints for Phase 5."""

    data_dir: Path = ML_ROOT / "data"
    reports_dir: Path = ML_ROOT / "reports"
    docs_dir: Path = ML_ROOT / "docs"

    products_json_filename: str = "investment_products.json"
    products_csv_filename: str = "investment_products.csv"
    market_assumptions_filename: str = "market_assumptions.json"
    catalog_report_filename: str = "investment_catalog.md"
    statistics_report_filename: str = "investment_statistics.md"
    risk_distribution_filename: str = "risk_distribution.json"
    goal_distribution_filename: str = "goal_distribution.json"
    documentation_filename: str = "investment_intelligence_knowledge_base.md"

    minimum_product_count: int = 25

    @property
    def products_json_path(self) -> Path:
        return self.data_dir / self.products_json_filename

    @property
    def products_csv_path(self) -> Path:
        return self.data_dir / self.products_csv_filename

    @property
    def market_assumptions_path(self) -> Path:
        return self.data_dir / self.market_assumptions_filename

    @property
    def catalog_report_path(self) -> Path:
        return self.reports_dir / self.catalog_report_filename

    @property
    def statistics_report_path(self) -> Path:
        return self.reports_dir / self.statistics_report_filename

    @property
    def risk_distribution_path(self) -> Path:
        return self.reports_dir / self.risk_distribution_filename

    @property
    def goal_distribution_path(self) -> Path:
        return self.reports_dir / self.goal_distribution_filename

    @property
    def documentation_path(self) -> Path:
        return self.docs_dir / self.documentation_filename

    def create_output_directories(self) -> None:
        for directory in (self.data_dir, self.reports_dir, self.docs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        if self.minimum_product_count < 25:
            raise ValueError("minimum_product_count cannot be less than the required 25 products")
