"""Production-oriented preprocessing pipeline for TrustVest AI credit modelling.

The pipeline never modifies the synthetic source CSV.  It writes a clean,
fully numeric training dataset plus the fitted preprocessing artifacts needed
to reproduce its transformations for future inference data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from ml.config import FeatureEngineeringConfig
    from ml.preprocessing.feature_engineering import CreditFeatureEngineer
    from ml.preprocessing.validation import (
        ValidationEngine,
        ValidationReport,
        write_validation_report,
    )
except ModuleNotFoundError:  # Supports direct execution from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ml.config import FeatureEngineeringConfig
    from ml.preprocessing.feature_engineering import CreditFeatureEngineer
    from ml.preprocessing.validation import (
        ValidationEngine,
        ValidationReport,
        write_validation_report,
    )


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    """An auditable repair applied to one source field."""

    row_index: int
    user_id: str
    column: str
    original_value: str
    corrected_value: str
    reason: str


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Summary returned by an end-to-end preprocessing execution."""

    processed_dataset_path: Path
    rows_loaded: int
    rows_cleaned: int
    created_feature_count: int
    removed_feature_count: int
    final_feature_count: int
    dataset_shape: tuple[int, int]


class JsonFormatter(logging.Formatter):
    """Emit compact JSON lines so progress can be collected by log tooling."""

    _STANDARD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in self._STANDARD_FIELDS and not key.startswith("_")
            }
        )
        return json.dumps(payload, default=str)


class CreditPreprocessingPipeline:
    """Validate, repair, transform, select, and document ML-ready features."""

    def __init__(self, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.validation_engine = ValidationEngine(self.config)
        self.feature_engineer = CreditFeatureEngineer(self.config)
        self.encoder: OneHotEncoder | None = None
        self.scaler: StandardScaler | None = None
        self.corrections: list[CorrectionRecord] = []
        self.created_features: tuple[str, ...] = ()
        self.removed_features: dict[str, list[str]] = {}
        self._variance_reference: pd.Series | None = None
        self._numeric_feature_columns: list[str] = []
        self.logger = LOGGER

    def run(self) -> PipelineRunResult:
        """Execute the complete deterministic pipeline and persist all artifacts."""
        self.config.create_output_directories()
        self._configure_logging()

        source = self.load_data()
        source_report = self.validate(source, stage="source")
        cleaned = self.clean(source)
        cleaned_report = self.validate(cleaned, stage="post_cleaning")
        write_validation_report(
            self.config.validation_report_path,
            {"source": source_report, "post_cleaning": cleaned_report},
        )
        if not cleaned_report.passed:
            raise ValueError(
                "Data cleaning could not satisfy all validation rules. "
                f"See {self.config.validation_report_path}."
            )

        engineered = self.engineer_features(cleaned)
        encoded = self.encode(engineered)
        scaled = self.scale(encoded)
        processed = self.feature_selection(scaled)
        self.save_dataset(processed)
        feature_importance = self.generate_feature_importance(processed)
        self.generate_correlation_report(processed)
        self.generate_visualizations(processed, feature_importance)

        result = PipelineRunResult(
            processed_dataset_path=self.config.processed_dataset_path,
            rows_loaded=len(source),
            rows_cleaned=len(cleaned),
            created_feature_count=len(self.created_features),
            removed_feature_count=sum(len(values) for values in self.removed_features.values()),
            final_feature_count=processed.shape[1] - 1,
            dataset_shape=(int(processed.shape[0]), int(processed.shape[1])),
        )
        self._log("pipeline_complete", **asdict(result))
        return result

    def load_data(self) -> pd.DataFrame:
        """Load the immutable synthetic source dataset from the configured path."""
        if not self.config.source_path.exists():
            raise FileNotFoundError(f"Source dataset not found: {self.config.source_path}")
        frame = pd.read_csv(self.config.source_path)
        self._log("rows_loaded", rows=int(len(frame)), columns=int(frame.shape[1]))
        return frame

    def validate(self, frame: pd.DataFrame, stage: str = "validation") -> ValidationReport:
        """Validate a dataframe and log a structured summary without mutation."""
        report = self.validation_engine.validate(frame, stage)
        failed_checks = [check.name for check in report.checks if not check.passed]
        self._log(
            "data_validated",
            stage=stage,
            passed=report.passed,
            failed_checks=failed_checks,
        )
        return report

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Repair data issues in-place on a copy and write an audit CSV of repairs."""
        missing_columns = sorted(set(self.config.required_columns) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Cannot repair missing required columns: {missing_columns}")

        cleaned = frame.copy()
        self.corrections = []
        self._clean_identifiers(cleaned)
        self._clean_categorical_columns(cleaned)
        self._coerce_and_impute_numeric_columns(cleaned)
        self._apply_business_rule_repairs(cleaned)
        self._write_corrections_report()
        self._log(
            "rows_cleaned",
            rows=int(len(cleaned)),
            corrections=int(len(self.corrections)),
        )
        return cleaned

    def engineer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add transparent domain and interaction features without using the target."""
        result = self.feature_engineer.transform(frame)
        self.created_features = result.created_features
        self._log("new_features_created", count=len(self.created_features), features=list(self.created_features))
        return result.frame

    def encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode configured categoricals and persist the fitted encoder."""
        self._require_columns(frame, [*self.config.categorical_columns, self.config.target_column])
        feature_frame = frame.drop(
            columns=[self.config.user_id_column, self.config.target_column], errors="ignore"
        )
        categorical = list(self.config.categorical_columns)
        numeric = feature_frame.drop(columns=categorical)
        self._numeric_feature_columns = list(numeric.columns)

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        encoded_values = self.encoder.fit_transform(feature_frame[categorical])
        encoded_columns = self.encoder.get_feature_names_out(categorical).tolist()
        encoded = pd.DataFrame(encoded_values, columns=encoded_columns, index=frame.index)
        result = pd.concat([numeric.astype(float), encoded], axis=1)
        result[self.config.target_column] = frame[self.config.target_column].astype(float)
        joblib.dump(self.encoder, self.config.encoder_path)
        self._log("categoricals_encoded", encoded_columns=len(encoded_columns), artifact=str(self.config.encoder_path))
        return result

    def scale(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply StandardScaler to raw numeric and engineered features only."""
        if not self._numeric_feature_columns:
            raise RuntimeError("encode() must be called before scale()")
        self._require_columns(frame, self._numeric_feature_columns)
        self._variance_reference = frame.drop(columns=[self.config.target_column]).var(ddof=0)
        scaled = frame.copy()
        self.scaler = StandardScaler()
        scaled.loc[:, self._numeric_feature_columns] = self.scaler.fit_transform(
            frame[self._numeric_feature_columns]
        )
        joblib.dump(self.scaler, self.config.scaler_path)
        self._log("numeric_features_scaled", columns=len(self._numeric_feature_columns), artifact=str(self.config.scaler_path))
        return scaled

    def feature_selection(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove constant, duplicate, low-variance, and highly correlated features."""
        target = self.config.target_column
        features = [column for column in frame.columns if column != target]
        constants = [column for column in features if frame[column].nunique(dropna=False) <= 1]
        remaining = [column for column in features if column not in constants]

        duplicates = self._duplicate_columns(frame[remaining])
        remaining = [column for column in remaining if column not in duplicates]

        if self._variance_reference is None:
            raise RuntimeError("scale() must be called before feature_selection()")
        near_zero = [
            column
            for column in remaining
            if float(self._variance_reference.get(column, 0.0)) <= self.config.near_zero_variance_threshold
        ]
        remaining = [column for column in remaining if column not in near_zero]

        correlated = self._highly_correlated_columns(frame[remaining])
        remaining = [column for column in remaining if column not in correlated]
        self.removed_features = {
            "constant": constants,
            "duplicate": duplicates,
            "near_zero_variance": near_zero,
            "highly_correlated": correlated,
        }
        self._write_feature_selection_report(len(features), len(remaining))
        self._log(
            "features_removed",
            total=sum(len(values) for values in self.removed_features.values()),
            details={name: len(values) for name, values in self.removed_features.items()},
        )
        return frame.loc[:, [*remaining, target]].copy()

    def save_dataset(self, frame: pd.DataFrame) -> None:
        """Persist a target-last, fully numeric training dataset."""
        if frame[self.config.target_column].isna().any():
            raise ValueError("The target column contains missing values after preprocessing")
        non_numeric = frame.drop(columns=[self.config.target_column]).select_dtypes(exclude=[np.number])
        if not non_numeric.empty:
            raise TypeError(f"Processed dataset contains non-numeric columns: {list(non_numeric.columns)}")
        frame.to_csv(self.config.processed_dataset_path, index=False)
        self._log(
            "dataset_saved",
            path=str(self.config.processed_dataset_path),
            shape=[int(frame.shape[0]), int(frame.shape[1])],
            final_feature_count=int(frame.shape[1] - 1),
        )

    def generate_feature_importance(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit a temporary random forest and save rank-ordered importances."""
        target = self.config.target_column
        features = frame.drop(columns=[target])
        sample = frame
        if len(frame) > self.config.feature_importance_sample_size:
            sample = frame.sample(
                n=self.config.feature_importance_sample_size,
                random_state=self.config.random_seed,
            )
        model = RandomForestRegressor(
            n_estimators=self.config.random_forest_estimators,
            max_depth=self.config.random_forest_max_depth,
            min_samples_leaf=3,
            max_features=0.7,
            random_state=self.config.random_seed,
            n_jobs=self.config.random_forest_n_jobs,
        )
        model.fit(sample.drop(columns=[target]), sample[target])
        importance = pd.DataFrame(
            {"feature": features.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)
        importance.index = importance.index + 1
        importance.index.name = "rank"
        importance.to_csv(self.config.feature_importance_path)
        self._plot_feature_importance(importance)
        self._log(
            "feature_importance_generated",
            rows=int(len(importance)),
            artifact=str(self.config.feature_importance_path),
        )
        return importance

    def generate_correlation_report(self, frame: pd.DataFrame) -> None:
        """Report target correlations and create a readable processed-data heatmap."""
        target = self.config.target_column
        correlation = frame.corr(numeric_only=True)
        target_correlations = correlation[target].drop(target).sort_values(ascending=False)
        positive = target_correlations[target_correlations > 0].head(10)
        negative = target_correlations[target_correlations < 0].sort_values().head(10)
        report = [
            "# Processed Dataset Correlation Report",
            "",
            "Correlations are Pearson coefficients calculated after numeric scaling and feature selection.",
            "",
            "## Strongest Positive Correlations with `credit_likelihood`",
            "",
            *self._correlation_table(positive),
            "",
            "## Strongest Negative Correlations with `credit_likelihood`",
            "",
            *self._correlation_table(negative),
            "",
            f"Correlation-pruning threshold: `{self.config.correlation_threshold}`.",
        ]
        self.config.correlation_report_path.write_text("\n".join(report), encoding="utf-8")

        display_features = target_correlations.abs().head(self.config.top_feature_count).index.tolist()
        display_columns = [target, *display_features]
        display_correlation = correlation.loc[display_columns, display_columns]
        figure, axis = plt.subplots(figsize=(14, 12))
        image = axis.imshow(display_correlation, cmap="coolwarm", vmin=-1, vmax=1)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pearson correlation")
        axis.set_xticks(range(len(display_columns)), labels=display_columns, rotation=90, fontsize=8)
        axis.set_yticks(range(len(display_columns)), labels=display_columns, fontsize=8)
        axis.set_title("Processed Dataset Correlation Heatmap (Target + Top 30 Features)")
        figure.tight_layout()
        figure.savefig(self.config.correlation_heatmap_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        self._log("correlation_report_generated", artifact=str(self.config.correlation_report_path))

    def generate_visualizations(
        self,
        frame: pd.DataFrame,
        feature_importance: pd.DataFrame,
    ) -> None:
        """Create target and representative-feature distribution plots."""
        target = self.config.target_column
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.hist(frame[target], bins=35, color="#2563eb", edgecolor="white", alpha=0.9)
        axis.set_title("Processed Credit Likelihood Distribution")
        axis.set_xlabel("credit_likelihood")
        axis.set_ylabel("Users")
        figure.tight_layout()
        figure.savefig(self.config.target_distribution_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

        top_features = feature_importance["feature"].head(8).tolist()
        figure, axes = plt.subplots(2, 4, figsize=(16, 8))
        for axis, column in zip(axes.flat, top_features, strict=True):
            axis.hist(frame[column], bins=30, color="#0f766e", edgecolor="white", alpha=0.9)
            axis.set_title(column, fontsize=9)
            axis.tick_params(axis="both", labelsize=8)
        for axis in axes.flat[len(top_features) :]:
            axis.set_visible(False)
        figure.suptitle("Distribution Summary: Top Random-Forest Features", y=1.02)
        figure.tight_layout()
        figure.savefig(self.config.feature_distribution_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        self._log(
            "visualizations_generated",
            artifacts=[
                str(self.config.target_distribution_path),
                str(self.config.feature_distribution_path),
            ],
        )

    def _clean_identifiers(self, frame: pd.DataFrame) -> None:
        column = self.config.user_id_column
        identifiers = frame[column].astype("string").str.strip()
        missing = identifiers.isna() | identifiers.eq("")
        for index in frame.index[missing]:
            repaired = self._unique_repaired_identifier("REPAIRED", identifiers)
            self._record_correction(frame, index, column, frame.at[index, column], repaired, "missing_user_id")
            identifiers.at[index] = repaired

        used: set[str] = set()
        for index, identifier in identifiers.items():
            identifier_text = str(identifier)
            if identifier_text in used:
                repaired = self._unique_repaired_identifier(f"{identifier_text}__DUP", identifiers, used)
                self._record_correction(frame, index, column, identifier_text, repaired, "duplicate_user_id")
                identifiers.at[index] = repaired
                identifier_text = repaired
            used.add(identifier_text)
        frame[column] = identifiers.astype(str)

    def _clean_categorical_columns(self, frame: pd.DataFrame) -> None:
        for column in self.config.categorical_columns:
            original = frame[column].copy()
            normalized = original.astype("string").str.strip()
            missing = normalized.isna() | normalized.eq("")
            mode = normalized.loc[~missing].mode(dropna=True)
            replacement = str(mode.iloc[0]) if not mode.empty else "Unknown"
            normalized.loc[missing] = replacement
            changed = missing | (normalized.astype(str) != original.astype(str))
            for index in frame.index[changed]:
                self._record_correction(
                    frame,
                    index,
                    column,
                    original.at[index],
                    normalized.at[index],
                    "missing_or_whitespace_categorical_value",
                )
            frame[column] = normalized.astype(str)

    def _coerce_and_impute_numeric_columns(self, frame: pd.DataFrame) -> None:
        for column in [*self.config.numeric_columns, self.config.target_column]:
            original = frame[column].copy()
            numeric = pd.to_numeric(original, errors="coerce")
            invalid_text = original.notna() & numeric.isna()
            for index in frame.index[invalid_text]:
                self._record_correction(
                    frame,
                    index,
                    column,
                    original.at[index],
                    "NaN",
                    "non_numeric_value_coerced",
                )
            missing = numeric.isna()
            if missing.any():
                replacement = self._numeric_imputation_value(column, numeric)
                for index in frame.index[missing & ~invalid_text]:
                    self._record_correction(
                        frame,
                        index,
                        column,
                        original.at[index],
                        replacement,
                        "missing_numeric_value_imputed",
                    )
                for index in frame.index[invalid_text]:
                    self._record_correction(
                        frame,
                        index,
                        column,
                        "NaN",
                        replacement,
                        "non_numeric_value_imputed",
                    )
                numeric.loc[missing] = replacement
            frame[column] = numeric.astype(float)

    def _apply_business_rule_repairs(self, frame: pd.DataFrame) -> None:
        positive_income = self._positive_income_replacement(frame["monthly_income"])
        self._replace_values(
            frame,
            "monthly_income",
            frame["monthly_income"] <= 0,
            positive_income,
            "income_must_be_positive",
        )
        self._replace_values(
            frame,
            "monthly_expenses",
            frame["monthly_expenses"] < self.config.min_expenses,
            self.config.min_expenses,
            "expenses_must_be_non_negative",
        )
        self._replace_values(
            frame,
            "monthly_savings",
            frame["monthly_savings"] < self.config.min_savings,
            self.config.min_savings,
            "savings_must_be_non_negative",
        )
        self._replace_values(
            frame,
            "monthly_savings",
            frame["monthly_savings"] > frame["monthly_income"],
            frame["monthly_income"],
            "savings_capped_at_income",
        )
        self._replace_values(
            frame,
            "age",
            (frame["age"] < self.config.min_age) | (frame["age"] > self.config.max_age),
            frame["age"].clip(lower=self.config.min_age, upper=self.config.max_age),
            "age_clipped_to_configured_range",
        )
        target = self.config.target_column
        self._replace_values(
            frame,
            target,
            (frame[target] < self.config.min_credit_likelihood)
            | (frame[target] > self.config.max_credit_likelihood),
            frame[target].clip(
                lower=self.config.min_credit_likelihood,
                upper=self.config.max_credit_likelihood,
            ),
            "credit_likelihood_clipped_to_configured_range",
        )
        for column in self.config.payment_ratio_columns:
            self._replace_values(
                frame,
                column,
                (frame[column] < self.config.min_ratio) | (frame[column] > self.config.max_ratio),
                frame[column].clip(lower=self.config.min_ratio, upper=self.config.max_ratio),
                "ratio_clipped_to_configured_range",
            )

    def _replace_values(
        self,
        frame: pd.DataFrame,
        column: str,
        mask: pd.Series,
        replacement: float | pd.Series,
        reason: str,
    ) -> None:
        for index in frame.index[mask]:
            value = replacement.at[index] if isinstance(replacement, pd.Series) else replacement
            self._record_correction(frame, index, column, frame.at[index, column], value, reason)
            frame.at[index, column] = value

    def _numeric_imputation_value(self, column: str, values: pd.Series) -> float:
        valid_values = values.dropna()
        if not valid_values.empty:
            median = float(valid_values.median())
            if column != "monthly_income" or median > 0:
                return median
        if column == "monthly_income":
            return self.config.min_income
        if column == "age":
            return float((self.config.min_age + self.config.max_age) / 2)
        if column == self.config.target_column:
            return float((self.config.min_credit_likelihood + self.config.max_credit_likelihood) / 2)
        if column in self.config.payment_ratio_columns:
            return float((self.config.min_ratio + self.config.max_ratio) / 2)
        return 0.0

    def _positive_income_replacement(self, values: pd.Series) -> float:
        positive_values = values[values > 0]
        return float(positive_values.median()) if not positive_values.empty else self.config.min_income

    def _unique_repaired_identifier(
        self,
        prefix: str,
        identifiers: pd.Series,
        used: set[str] | None = None,
    ) -> str:
        occupied = used or set(identifiers.dropna().astype(str))
        counter = 1
        while f"{prefix}_{counter:06d}" in occupied:
            counter += 1
        return f"{prefix}_{counter:06d}"

    def _record_correction(
        self,
        frame: pd.DataFrame,
        index: int,
        column: str,
        original: object,
        corrected: object,
        reason: str,
    ) -> None:
        identifier = frame.at[index, self.config.user_id_column]
        self.corrections.append(
            CorrectionRecord(
                row_index=int(index),
                user_id=self._display_value(identifier),
                column=column,
                original_value=self._display_value(original),
                corrected_value=self._display_value(corrected),
                reason=reason,
            )
        )

    def _write_corrections_report(self) -> None:
        report = pd.DataFrame(
            [asdict(record) for record in self.corrections],
            columns=[
                "row_index",
                "user_id",
                "column",
                "original_value",
                "corrected_value",
                "reason",
            ],
        )
        report.to_csv(self.config.corrections_report_path, index=False)

    def _write_feature_selection_report(self, input_count: int, output_count: int) -> None:
        report = {
            "input_feature_count": input_count,
            "output_feature_count": output_count,
            "correlation_threshold": self.config.correlation_threshold,
            "near_zero_variance_threshold": self.config.near_zero_variance_threshold,
            "removed_features": self.removed_features,
        }
        self.config.feature_selection_report_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    def _duplicate_columns(self, frame: pd.DataFrame) -> list[str]:
        duplicate_columns: list[str] = []
        seen: dict[int, list[str]] = {}
        for column in frame.columns:
            column_hash = int(pd.util.hash_pandas_object(frame[column], index=False).sum())
            candidates = seen.setdefault(column_hash, [])
            if any(frame[column].equals(frame[candidate]) for candidate in candidates):
                duplicate_columns.append(column)
            else:
                candidates.append(column)
        return duplicate_columns

    def _highly_correlated_columns(self, frame: pd.DataFrame) -> list[str]:
        if frame.shape[1] < 2:
            return []
        correlations = frame.corr().abs()
        upper_triangle = correlations.where(
            np.triu(np.ones(correlations.shape, dtype=bool), k=1)
        )
        return [
            column
            for column in upper_triangle.columns
            if (upper_triangle[column] > self.config.correlation_threshold).any()
        ]

    def _plot_feature_importance(self, importance: pd.DataFrame) -> None:
        top = importance.head(self.config.top_feature_count).sort_values("importance")
        figure, axis = plt.subplots(figsize=(11, 9))
        axis.barh(top["feature"], top["importance"], color="#7c3aed")
        axis.set_title("Top 30 Random Forest Feature Importances")
        axis.set_xlabel("Importance")
        figure.tight_layout()
        figure.savefig(self.config.feature_importance_plot_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _correlation_table(values: pd.Series) -> list[str]:
        if values.empty:
            return ["No features with this correlation direction were retained."]
        rows = ["| Rank | Feature | Correlation |", "| ---: | --- | ---: |"]
        rows.extend(
            f"| {rank} | `{feature}` | {coefficient:.4f} |"
            for rank, (feature, coefficient) in enumerate(values.items(), start=1)
        )
        return rows

    def _configure_logging(self) -> None:
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = JsonFormatter()
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        file_paths = {
            getattr(handler, "baseFilename", None) for handler in self.logger.handlers
        }
        if str(self.config.pipeline_log_path) not in file_paths:
            file_handler = logging.FileHandler(self.config.pipeline_log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _log(self, event: str, **details: object) -> None:
        self.logger.info(event, extra=details)

    @staticmethod
    def _display_value(value: object) -> str:
        return "" if pd.isna(value) else str(value)

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise KeyError(f"Required columns are missing: {missing}")


def run_pipeline(config: FeatureEngineeringConfig | None = None) -> PipelineRunResult:
    """Convenience function for applications and command-line use."""
    return CreditPreprocessingPipeline(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the TrustVest ML-ready credit dataset.")
    parser.add_argument("--source", type=Path, help="Optional source CSV override")
    parser.add_argument("--output", type=Path, help="Optional processed CSV override")
    return parser.parse_args()


def main() -> None:
    """Run the pipeline as ``python -m ml.preprocessing.preprocessing``."""
    arguments = _parse_args()
    config = FeatureEngineeringConfig(
        source_path=arguments.source or FeatureEngineeringConfig().source_path,
        processed_dataset_path=arguments.output or FeatureEngineeringConfig().processed_dataset_path,
    )
    result = run_pipeline(config)
    print(json.dumps(asdict(result), default=str, indent=2))


if __name__ == "__main__":
    main()
