"""Configuration and schema definitions for Phase 4 risk profiling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RISK_ROOT = Path(__file__).resolve().parent
ML_ROOT = RISK_ROOT.parent
RANDOM_SEED = 2026


QUESTIONNAIRE_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "occupation",
    "income_stability",
    "investment_goal",
    "reaction_to_20_percent_loss",
    "previous_investment_experience",
    "mutual_fund_knowledge",
    "stock_knowledge",
    "preferred_liquidity",
    "financial_confidence",
    "preferred_investment_frequency",
    "preferred_investment_type",
)

QUESTIONNAIRE_NUMERIC_COLUMNS: tuple[str, ...] = (
    "age",
    "monthly_income",
    "monthly_savings",
    "monthly_investment_budget",
    "emergency_fund_months",
    "investment_horizon_years",
    "expected_annual_return_percent",
    "dependents",
)

ENGINEERED_FEATURE_COLUMNS: tuple[str, ...] = (
    "risk_tolerance_score",
    "financial_preparedness",
    "income_stability_index",
    "investment_experience_index",
    "liquidity_preference_index",
    "behavioral_confidence_score",
    "loss_recovery_score",
    "investment_readiness_score",
    "long_term_orientation_score",
)


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """All paths and training choices for the behavioral risk engine."""

    random_seed: int = RANDOM_SEED
    num_profiles: int = 50_000
    data_dir: Path = RISK_ROOT / "data"
    reports_dir: Path = RISK_ROOT / "reports"
    plots_dir: Path = RISK_ROOT / "plots"
    models_dir: Path = ML_ROOT / "models"

    dataset_filename: str = "behavioral_risk_profiles.csv"
    feature_dataset_filename: str = "behavioral_risk_features.csv"
    model_filename: str = "risk_model.pkl"
    encoder_filename: str = "risk_encoder.pkl"
    scaler_filename: str = "risk_scaler.pkl"
    metadata_filename: str = "risk_model_metadata.json"
    classification_report_filename: str = "classification_report.md"
    metrics_filename: str = "training_metrics.json"
    feature_importance_filename: str = "feature_importance.csv"
    confusion_matrix_filename: str = "confusion_matrix.png"
    roc_curve_filename: str = "roc_curve.png"
    shap_summary_filename: str = "shap_summary.png"
    shap_bar_filename: str = "shap_bar.png"
    shap_local_filename: str = "shap_local_explanation.png"
    shap_top_features_filename: str = "shap_top_features.csv"

    target_column: str = "risk_profile"
    test_size: float = 0.20
    comparison_estimators: int = 250
    xgb_estimators: int = 300
    max_shap_samples: int = 1_000
    max_shap_background_rows: int = 250
    n_jobs: int = -1

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.dataset_filename

    @property
    def feature_dataset_path(self) -> Path:
        return self.data_dir / self.feature_dataset_filename

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def encoder_path(self) -> Path:
        return self.models_dir / self.encoder_filename

    @property
    def scaler_path(self) -> Path:
        return self.models_dir / self.scaler_filename

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / self.metadata_filename

    @property
    def classification_report_path(self) -> Path:
        return self.reports_dir / self.classification_report_filename

    @property
    def training_metrics_path(self) -> Path:
        return self.reports_dir / self.metrics_filename

    @property
    def feature_importance_path(self) -> Path:
        return self.reports_dir / self.feature_importance_filename

    @property
    def confusion_matrix_path(self) -> Path:
        return self.plots_dir / self.confusion_matrix_filename

    @property
    def roc_curve_path(self) -> Path:
        return self.plots_dir / self.roc_curve_filename

    @property
    def shap_summary_path(self) -> Path:
        return self.plots_dir / self.shap_summary_filename

    @property
    def shap_bar_path(self) -> Path:
        return self.plots_dir / self.shap_bar_filename

    @property
    def shap_local_path(self) -> Path:
        return self.plots_dir / self.shap_local_filename

    @property
    def shap_top_features_path(self) -> Path:
        return self.reports_dir / self.shap_top_features_filename

    @property
    def questionnaire_columns(self) -> tuple[str, ...]:
        return (*QUESTIONNAIRE_NUMERIC_COLUMNS, *QUESTIONNAIRE_CATEGORICAL_COLUMNS)

    @property
    def model_numeric_columns(self) -> tuple[str, ...]:
        return (*QUESTIONNAIRE_NUMERIC_COLUMNS, *ENGINEERED_FEATURE_COLUMNS)

    def create_output_directories(self) -> None:
        for directory in (self.data_dir, self.reports_dir, self.plots_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        if self.num_profiles <= 0:
            raise ValueError("num_profiles must be greater than zero")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be in the interval (0, 1)")
        if self.comparison_estimators <= 0 or self.xgb_estimators <= 0:
            raise ValueError("Estimator counts must be greater than zero")
        if self.max_shap_samples <= 0 or self.max_shap_background_rows <= 0:
            raise ValueError("SHAP sample sizes must be greater than zero")
