"""Reusable validation and feature-engineering utilities for TrustVest AI."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .preprocessing import CreditPreprocessingPipeline, run_pipeline

__all__ = ["CreditPreprocessingPipeline", "run_pipeline"]


def __getattr__(name: str) -> object:
    """Lazily expose pipeline entry points without preloading module execution."""
    if name in __all__:
        from .preprocessing import CreditPreprocessingPipeline, run_pipeline

        return {
            "CreditPreprocessingPipeline": CreditPreprocessingPipeline,
            "run_pipeline": run_pipeline,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
