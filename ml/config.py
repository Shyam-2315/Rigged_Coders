"""Configuration for TrustVest AI synthetic data generation."""

from dataclasses import dataclass
from pathlib import Path


NUM_USERS = 100_000
RANDOM_SEED = 2026


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Runtime settings for a reproducible synthetic-population run."""

    num_users: int = NUM_USERS
    random_seed: int = RANDOM_SEED
    output_dir: Path = Path(__file__).resolve().parent / "datasets"
    plots_dir: Path = Path(__file__).resolve().parent / "plots"

    def __post_init__(self) -> None:
        if self.num_users <= 0:
            raise ValueError("num_users must be greater than zero")
