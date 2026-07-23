"""Validation layer for investment product metadata and persisted catalog data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from pydantic import ValidationError

from .config import InvestmentKnowledgeConfig
from .schemas import InvestmentProduct, MarketAssumptions


@dataclass(frozen=True, slots=True)
class CatalogValidationResult:
    """A successful catalog-validation summary."""

    product_count: int
    product_ids: tuple[str, ...]


def validate_product(product: InvestmentProduct) -> InvestmentProduct:
    """Apply cross-field rules that complement Pydantic's field validation."""
    if product.suitable_for_students and product.minimum_investment > 10_000:
        raise ValueError(f"Student-suitable product {product.id} has an excessive minimum investment")
    if product.lock_in_period > 0 and product.liquidity.value == "Instant":
        raise ValueError(f"Product {product.id} cannot be instant-liquid with a lock-in period")
    return product


def validate_catalog(
    products: Sequence[InvestmentProduct],
    config: InvestmentKnowledgeConfig | None = None,
) -> CatalogValidationResult:
    """Validate cardinality, identity uniqueness, and every product's metadata."""
    active_config = config or InvestmentKnowledgeConfig()
    if len(products) < active_config.minimum_product_count:
        raise ValueError(f"Catalog needs at least {active_config.minimum_product_count} products; received {len(products)}")
    identifiers = [product.id for product in products]
    names = [product.name.casefold() for product in products]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Catalog product ids must be unique")
    if len(names) != len(set(names)):
        raise ValueError("Catalog product names must be unique")
    for product in products:
        validate_product(product)
    return CatalogValidationResult(product_count=len(products), product_ids=tuple(identifiers))


def parse_catalog(records: Iterable[Mapping[str, object]], config: InvestmentKnowledgeConfig | None = None) -> list[InvestmentProduct]:
    """Parse persisted JSON-safe records into validated product models."""
    try:
        products = [InvestmentProduct.model_validate(record) for record in records]
    except ValidationError as error:
        raise ValueError(f"Invalid investment product metadata: {error}") from error
    validate_catalog(products, config)
    return products


def validate_market_assumptions(record: Mapping[str, object]) -> MarketAssumptions:
    """Parse and validate the market assumptions used by future simulators."""
    try:
        return MarketAssumptions.model_validate(record)
    except ValidationError as error:
        raise ValueError(f"Invalid market assumptions: {error}") from error
