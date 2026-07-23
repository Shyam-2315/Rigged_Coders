"""Phase 5 investment intelligence knowledge base for TrustVest AI.

This package stores and retrieves product metadata.  It deliberately does not
construct portfolios, allocate capital, or provide personalised advice.
"""

from .retrieval import (
    filter_by_beginner,
    filter_by_credit_score,
    filter_by_goal,
    filter_by_government_backed,
    filter_by_horizon,
    filter_by_liquidity,
    filter_by_persona,
    filter_by_risk,
    get_all_products,
    get_product_by_id,
    search_products,
)

__all__ = [
    "get_all_products",
    "get_product_by_id",
    "filter_by_risk",
    "filter_by_goal",
    "filter_by_persona",
    "filter_by_horizon",
    "filter_by_liquidity",
    "filter_by_beginner",
    "filter_by_government_backed",
    "filter_by_credit_score",
    "search_products",
]
