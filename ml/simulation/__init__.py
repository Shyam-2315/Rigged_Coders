"""Phase 7 financial projection and Monte Carlo simulation engine.

The package consumes Phase 6 portfolio recommendations and Phase 5 market
assumptions to produce educational, deterministic long-term projections. It
does not forecast markets or provide regulated investment advice.
"""

from .projection_engine import (
    FinancialProjectionEngine,
    generate_phase7_reports,
    generate_sample_phase7_reports,
    project_portfolio,
)
from .schemas import ProjectionRequest, ProjectionResponse

__all__ = [
    "FinancialProjectionEngine",
    "ProjectionRequest",
    "ProjectionResponse",
    "generate_phase7_reports",
    "generate_sample_phase7_reports",
    "project_portfolio",
]
