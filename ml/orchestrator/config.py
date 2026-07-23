"""Configuration for the Phase 8 TrustVest Intelligence Orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ORCHESTRATOR_ROOT = Path(__file__).resolve().parent
ML_ROOT = ORCHESTRATOR_ROOT.parent

DEFAULT_SIMULATION_SEED = 2026
DEFAULT_PIPELINE_VERSION = "v0.8.0"

EDUCATIONAL_DISCLAIMER = (
    "TrustVest AI provides educational illustrations only. Outputs from credit scoring, risk profiling, "
    "portfolio recommendations, and financial simulations are based on statistical models and synthetic or "
    "assumed data. They do not constitute regulated financial advice, lending decisions, or predictions of "
    "future market performance. Consult qualified professionals before making financial decisions."
)


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Runtime settings for the unified orchestration layer."""

    reports_dir: Path = ML_ROOT / "reports"
    pipeline_version: str = DEFAULT_PIPELINE_VERSION
    simulation_seed: int = DEFAULT_SIMULATION_SEED
    cache_enabled: bool = True
    cache_max_entries: int = 256
    stage_timeout_seconds: float | None = None
    log_pipeline_trace: bool = True
    generate_charts_by_default: bool = False
    generate_reports_by_default: bool = False
    educational_disclaimer: str = EDUCATIONAL_DISCLAIMER

    pipeline_summary_filename: str = "pipeline_summary.md"
    pipeline_metrics_filename: str = "pipeline_metrics.json"
    sample_responses_filename: str = "sample_responses.json"
    audit_log_example_filename: str = "audit_log_example.json"
    audit_log_filename: str = "orchestrator_audit.jsonl"
    audit_log_enabled: bool = False

    credit_model_path: Path = ML_ROOT / "models" / "credit_model.pkl"
    risk_model_path: Path = ML_ROOT / "models" / "risk_model.pkl"
    scaler_path: Path = ML_ROOT / "models" / "scaler.pkl"
    encoder_path: Path = ML_ROOT / "models" / "encoder.pkl"

    @property
    def pipeline_summary_path(self) -> Path:
        return self.reports_dir / self.pipeline_summary_filename

    @property
    def pipeline_metrics_path(self) -> Path:
        return self.reports_dir / self.pipeline_metrics_filename

    @property
    def sample_responses_path(self) -> Path:
        return self.reports_dir / self.sample_responses_filename

    @property
    def audit_log_example_path(self) -> Path:
        return self.reports_dir / self.audit_log_example_filename

    @property
    def audit_log_path(self) -> Path:
        return self.reports_dir / self.audit_log_filename

    def create_output_directories(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        if self.cache_max_entries <= 0:
            raise ValueError("cache_max_entries must be greater than zero")
        if self.simulation_seed < 0:
            raise ValueError("simulation_seed must be non-negative")
