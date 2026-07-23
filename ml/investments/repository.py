"""Repository pattern for persisted investment knowledge-base artifacts."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable

from .config import InvestmentKnowledgeConfig
from .schemas import InvestmentProduct, MarketAssumptions
from .validation import parse_catalog, validate_catalog, validate_market_assumptions


LOGGER = logging.getLogger(__name__)


class InvestmentProductRepository:
    """Load and persist catalog data without exposing filesystem details to callers."""

    def __init__(self, config: InvestmentKnowledgeConfig | None = None) -> None:
        self.config = config or InvestmentKnowledgeConfig()
        self._catalog_cache: tuple[InvestmentProduct, ...] | None = None
        self._market_assumptions_cache: MarketAssumptions | None = None

    def get_all(self) -> list[InvestmentProduct]:
        """Return a copy of every validated product in catalog order."""
        if self._catalog_cache is None:
            self._catalog_cache = tuple(self._load_catalog())
        return list(self._catalog_cache)

    def get_by_id(self, product_id: str) -> InvestmentProduct | None:
        """Return one product by its stable identifier, or ``None`` when absent."""
        normalized = product_id.strip().casefold()
        return next((product for product in self.get_all() if product.id == normalized), None)

    def save_catalog(self, products: Iterable[InvestmentProduct], *, overwrite: bool = True) -> None:
        """Validate and save JSON and CSV representations of the catalog."""
        product_list = list(products)
        validate_catalog(product_list, self.config)
        self.config.create_output_directories()
        if not overwrite and self.config.products_json_path.exists():
            raise FileExistsError(f"Catalog already exists: {self.config.products_json_path}")
        records = [product.model_dump(mode="json") for product in product_list]
        _write_json(self.config.products_json_path, records)
        self._write_csv(product_list)
        self._catalog_cache = tuple(product_list)
        LOGGER.info("Saved %s investment products to %s", len(product_list), self.config.products_json_path)

    def get_market_assumptions(self) -> MarketAssumptions:
        """Load the persisted market-assumptions model, generating it when absent."""
        if self._market_assumptions_cache is None:
            if not self.config.market_assumptions_path.exists():
                from .knowledge_base import build_market_assumptions

                self.save_market_assumptions(build_market_assumptions())
            payload = json.loads(self.config.market_assumptions_path.read_text(encoding="utf-8"))
            self._market_assumptions_cache = validate_market_assumptions(payload)
        return self._market_assumptions_cache

    def save_market_assumptions(self, assumptions: MarketAssumptions, *, overwrite: bool = True) -> None:
        """Validate and persist assumptions consumed by a future simulator."""
        self.config.create_output_directories()
        if not overwrite and self.config.market_assumptions_path.exists():
            raise FileExistsError(f"Market assumptions already exist: {self.config.market_assumptions_path}")
        _write_json(self.config.market_assumptions_path, assumptions.model_dump(mode="json"))
        self._market_assumptions_cache = assumptions

    def _load_catalog(self) -> list[InvestmentProduct]:
        if not self.config.products_json_path.exists():
            from .knowledge_base import build_investment_products

            self.save_catalog(build_investment_products())
        payload = json.loads(self.config.products_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Investment product JSON must contain a list of product objects")
        return parse_catalog(payload, self.config)

    def _write_csv(self, products: list[InvestmentProduct]) -> None:
        records = [_flatten_for_csv(product) for product in products]
        if not records:
            raise ValueError("Cannot save an empty investment catalog")
        with self.config.products_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def _flatten_for_csv(product: InvestmentProduct) -> dict[str, object]:
    """Create an analyst-friendly flat record while keeping JSON canonical."""
    record = product.model_dump(mode="json")
    flattened: dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            flattened[f"{key}_minimum"] = value["minimum"]
            flattened[f"{key}_maximum"] = value["maximum"]
        elif isinstance(value, list):
            flattened[key] = " | ".join(str(item) for item in value)
        else:
            flattened[key] = value
    return flattened


def _write_json(path: Path, payload: object) -> None:
    """Write a UTF-8 JSON document through a sibling temporary file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)
