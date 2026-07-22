"""Configuration for TrustVest AI data generation, preprocessing, and modelling."""

from dataclasses import dataclass, field
from pathlib import Path


NUM_USERS = 100_000
RANDOM_SEED = 2026
ML_ROOT = Path(__file__).resolve().parent


SOURCE_CATEGORICAL_COLUMNS = (
    "gender",
    "state",
    "city_tier",
    "education",
    "marital_status",
    "occupation",
    "employment_type",
    "wallet_usage",
)

SOURCE_NUMERIC_COLUMNS = (
    "age",
    "dependents",
    "years_employed",
    "monthly_income",
    "monthly_expenses",
    "monthly_savings",
    "bank_account_age",
    "mobile_number_age",
    "smartphone_years",
    "digital_literacy_score",
    "upi_transactions_per_month",
    "upi_average_transaction",
    "utility_bill_payment_rate",
    "mobile_recharge_frequency",
    "wallet_average_balance",
    "ecommerce_transactions",
    "online_subscription_count",
    "bank_balance",
    "atm_withdrawals",
    "cash_transaction_ratio",
    "missed_utility_payments",
    "late_payment_ratio",
    "savings_rate",
    "investment_frequency",
    "emergency_fund_months",
    "loan_history_length",
    "existing_small_loans",
    "repayment_consistency",
    "device_trust_score",
    "device_age",
    "sim_age",
    "number_of_devices",
    "location_consistency",
    "financial_discipline_index",
    "digital_activity_score",
    "income_stability_score",
    "payment_consistency_score",
    "digital_trust_index",
)


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Runtime settings for a reproducible synthetic-population run."""

    num_users: int = NUM_USERS
    random_seed: int = RANDOM_SEED
    output_dir: Path = ML_ROOT / "datasets"
    plots_dir: Path = ML_ROOT / "plots"

    def __post_init__(self) -> None:
        if self.num_users <= 0:
            raise ValueError("num_users must be greater than zero")


@dataclass(frozen=True, slots=True)
class FeatureEngineeringConfig:
    """All runtime settings for the credit-scoring feature pipeline."""

    random_seed: int = RANDOM_SEED
    source_path: Path = ML_ROOT / "datasets" / "synthetic_users.csv"
    processed_dataset_path: Path = ML_ROOT / "datasets" / "processed_credit_dataset.csv"
    models_dir: Path = ML_ROOT / "models"
    reports_dir: Path = ML_ROOT / "reports"
    plots_dir: Path = ML_ROOT / "plots"

    user_id_column: str = "user_id"
    target_column: str = "credit_likelihood"
    categorical_columns: tuple[str, ...] = SOURCE_CATEGORICAL_COLUMNS
    numeric_columns: tuple[str, ...] = SOURCE_NUMERIC_COLUMNS
    payment_ratio_columns: tuple[str, ...] = (
        "utility_bill_payment_rate",
        "cash_transaction_ratio",
        "late_payment_ratio",
        "savings_rate",
    )

    min_age: int = 18
    max_age: int = 65
    min_income: float = 0.01
    min_expenses: float = 0.0
    min_savings: float = 0.0
    min_credit_likelihood: float = 0.0
    max_credit_likelihood: float = 100.0
    min_ratio: float = 0.0
    max_ratio: float = 1.0

    # Domain-normalisation caps used by transparent engineered scores.
    income_normalization_cap: float = 200_000.0
    annualization_months: float = 12.0
    emergency_fund_target_months: float = 12.0
    upi_transaction_cap: float = 300.0
    ecommerce_transaction_cap: float = 30.0
    subscription_count_cap: float = 10.0
    loan_history_year_cap: float = 15.0
    years_employed_cap: float = 15.0
    sim_age_month_cap: float = 24.0
    device_age_year_cap: float = 10.0
    bank_account_age_year_cap: float = 15.0
    missed_payment_cap: float = 5.0
    loan_count_cap: float = 5.0

    scaling_method: str = "standard"
    correlation_threshold: float = 0.95
    near_zero_variance_threshold: float = 1e-8
    random_forest_estimators: int = 150
    random_forest_max_depth: int = 18
    feature_importance_sample_size: int = 30_000
    top_feature_count: int = 30
    random_forest_n_jobs: int = -1

    scaler_filename: str = "scaler.pkl"
    encoder_filename: str = "encoder.pkl"
    validation_report_filename: str = "validation_report.json"
    corrections_report_filename: str = "data_cleaning_corrections.csv"
    feature_selection_report_filename: str = "feature_selection_report.json"
    feature_importance_filename: str = "feature_importance.csv"
    correlation_report_filename: str = "correlation_report.md"
    pipeline_log_filename: str = "preprocessing_pipeline.jsonl"
    feature_importance_plot_filename: str = "feature_importance.png"
    correlation_heatmap_filename: str = "correlation_heatmap_processed.png"
    target_distribution_filename: str = "target_distribution_processed.png"
    feature_distribution_filename: str = "feature_distribution_summary.png"

    @property
    def scaler_path(self) -> Path:
        return self.models_dir / self.scaler_filename

    @property
    def encoder_path(self) -> Path:
        return self.models_dir / self.encoder_filename

    @property
    def validation_report_path(self) -> Path:
        return self.reports_dir / self.validation_report_filename

    @property
    def corrections_report_path(self) -> Path:
        return self.reports_dir / self.corrections_report_filename

    @property
    def feature_selection_report_path(self) -> Path:
        return self.reports_dir / self.feature_selection_report_filename

    @property
    def feature_importance_path(self) -> Path:
        return self.reports_dir / self.feature_importance_filename

    @property
    def correlation_report_path(self) -> Path:
        return self.reports_dir / self.correlation_report_filename

    @property
    def pipeline_log_path(self) -> Path:
        return self.reports_dir / self.pipeline_log_filename

    @property
    def feature_importance_plot_path(self) -> Path:
        return self.plots_dir / self.feature_importance_plot_filename

    @property
    def correlation_heatmap_path(self) -> Path:
        return self.plots_dir / self.correlation_heatmap_filename

    @property
    def target_distribution_path(self) -> Path:
        return self.plots_dir / self.target_distribution_filename

    @property
    def feature_distribution_path(self) -> Path:
        return self.plots_dir / self.feature_distribution_filename

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.user_id_column,
            *self.categorical_columns,
            *self.numeric_columns,
            self.target_column,
        )

    def create_output_directories(self) -> None:
        """Create only pipeline output directories; never touch the source data."""
        for directory in (self.processed_dataset_path.parent, self.models_dir, self.reports_dir, self.plots_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in the interval (0, 1]")
        if self.scaling_method != "standard":
            raise ValueError("Only StandardScaler ('standard') is currently supported")
        if self.feature_importance_sample_size <= 0:
            raise ValueError("feature_importance_sample_size must be greater than zero")
        configurable_caps = (
            self.income_normalization_cap,
            self.annualization_months,
            self.emergency_fund_target_months,
            self.upi_transaction_cap,
            self.ecommerce_transaction_cap,
            self.subscription_count_cap,
            self.loan_history_year_cap,
            self.years_employed_cap,
            self.sim_age_month_cap,
            self.device_age_year_cap,
            self.bank_account_age_year_cap,
            self.missed_payment_cap,
            self.loan_count_cap,
        )
        if any(cap <= 0 for cap in configurable_caps):
            raise ValueError("Feature-engineering normalisation caps must be greater than zero")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Runtime settings for the reproducible Phase 3 credit-model pipeline.

    The processed dataset is already numeric and scaled, so this configuration
    deliberately describes modelling only.  Every output path and expensive
    training choice is exposed here for CLI callers and later orchestration.
    """

    random_seed: int = RANDOM_SEED
    dataset_path: Path = ML_ROOT / "datasets" / "processed_credit_dataset.csv"
    models_dir: Path = ML_ROOT / "models"
    reports_dir: Path = ML_ROOT / "reports"
    plots_dir: Path = ML_ROOT / "plots"

    target_column: str = "credit_likelihood"
    test_size: float = 0.20
    cv_folds: int = 5
    randomized_search_iterations: int = 8
    n_jobs: int = -1
    search_n_jobs: int = 1
    xgb_n_jobs: int = -1

    comparison_estimators: int = 150
    xgb_base_estimators: int = 300
    shap_sample_size: int = 1_000
    learning_curve_fractions: tuple[float, ...] = (0.2, 0.5, 1.0)
    top_feature_count: int = 20
    confidence_lower_bound: float = 0.50
    confidence_upper_bound: float = 0.99

    credit_model_filename: str = "credit_model.pkl"
    best_model_filename: str = "best_model.pkl"
    metadata_filename: str = "model_metadata.json"
    comparison_filename: str = "model_comparison.csv"
    feature_ranking_filename: str = "feature_ranking.json"
    feature_importance_filename: str = "feature_importance.csv"
    training_report_filename: str = "training_report.md"
    evaluation_report_filename: str = "evaluation_report.md"
    model_card_filename: str = "model_card.md"
    training_log_filename: str = "training_pipeline.jsonl"

    @property
    def credit_model_path(self) -> Path:
        return self.models_dir / self.credit_model_filename

    @property
    def best_model_path(self) -> Path:
        return self.models_dir / self.best_model_filename

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / self.metadata_filename

    @property
    def comparison_path(self) -> Path:
        return self.reports_dir / self.comparison_filename

    @property
    def feature_ranking_path(self) -> Path:
        return self.reports_dir / self.feature_ranking_filename

    @property
    def feature_importance_path(self) -> Path:
        return self.reports_dir / self.feature_importance_filename

    @property
    def training_report_path(self) -> Path:
        return self.reports_dir / self.training_report_filename

    @property
    def evaluation_report_path(self) -> Path:
        return self.reports_dir / self.evaluation_report_filename

    @property
    def model_card_path(self) -> Path:
        return self.reports_dir / self.model_card_filename

    @property
    def training_log_path(self) -> Path:
        return self.reports_dir / self.training_log_filename

    def create_output_directories(self) -> None:
        for directory in (self.models_dir, self.reports_dir, self.plots_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {self.dataset_path}")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be in the interval (0, 1)")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.randomized_search_iterations <= 0:
            raise ValueError("randomized_search_iterations must be greater than zero")
        if self.comparison_estimators <= 0 or self.xgb_base_estimators <= 0:
            raise ValueError("Estimator counts must be greater than zero")
        if self.shap_sample_size <= 0 or self.top_feature_count <= 0:
            raise ValueError("Sample and feature counts must be greater than zero")
        if not 0 <= self.confidence_lower_bound <= self.confidence_upper_bound <= 1:
            raise ValueError("Confidence bounds must be ordered values in [0, 1]")
        if not self.learning_curve_fractions or any(
            fraction <= 0 or fraction > 1 for fraction in self.learning_curve_fractions
        ):
            raise ValueError("learning_curve_fractions must contain values in (0, 1]")
