"""Phase 6 deterministic portfolio recommendation engine.

The package turns validated profile inputs and the Phase 5 product catalog into
educational portfolio illustrations.  It is deliberately deterministic and
does not forecast markets or execute transactions.
"""

from .recommendation_engine import PortfolioRecommendationEngine, recommend_portfolio
from .schemas import RecommendationRequest, RecommendationResponse

__all__ = [
    "PortfolioRecommendationEngine",
    "RecommendationRequest",
    "RecommendationResponse",
    "recommend_portfolio",
]
