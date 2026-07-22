"""End-to-end Phase 3 training pipeline for TrustVest credit likelihood.

Run from the repository root:
``python -m ml.training.train_credit_model``
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

try:
    from ml.config import TrainingConfig
    from ml.explainability.shap_engine import generate_shap_artifacts
    from ml.training.evaluate import calculate_regression_metrics, generate_evaluation_artifacts
except ModuleNotFoundError:  # Supports direct execution from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ml.config import TrainingConfig
    from ml.explainability.shap_engine import generate_shap_artifacts
    from ml.training.evaluate import calculate_regression_metrics, generate_evaluation_artifacts


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Serializable summary of a completed model-training run."""

    model_path: Path
    metadata_path: Path
    training_samples: int
    validation_samples: int
    feature_count: int
    validation_metrics: dict[str, float]
    best_parameters: dict[str, Any]
    cv_rmse: float


class JsonFormatter(logging.Formatter):
    """Emit portable JSON lines for CI and MLOps log collection."""

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


class CreditModelTrainer:
    """Compare, tune, explain, evaluate, and export the XGBoost credit model."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.logger = LOGGER

    def run(self) -> TrainingRunResult:
        """Execute the complete deterministic training workflow."""
        self.config.create_output_directories()
        self._configure_logging()
        self._log("training_start", config=asdict(self.config))
        features, target = self._load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
        )
        self._log(
            "train_test_split_complete",
            training_samples=len(X_train),
            validation_samples=len(X_test),
            feature_count=X_train.shape[1],
        )

        comparison = self._compare_models(X_train, X_test, y_train, y_test)
        comparison.to_csv(self.config.comparison_path, index=False)
        self._plot_model_comparison(comparison)
        self._log("model_comparison_complete", artifact=str(self.config.comparison_path))

        search = self._tune_xgboost(X_train, y_train)
        best_parameters = _json_safe(search.best_params_)
        cv_rmse = float(-search.best_score_)
        self._log("best_parameters", parameters=best_parameters, cv_rmse=cv_rmse)

        final_model = clone(search.best_estimator_)
        final_model.fit(X_train, y_train)
        predictions = np.asarray(final_model.predict(X_test), dtype=float)
        metrics = calculate_regression_metrics(y_test.to_numpy(), predictions)
        self._log("evaluation_metrics", **metrics)
        generate_evaluation_artifacts(
            y_test.to_numpy(), predictions, self.config.plots_dir, self.config.reports_dir, metrics
        )
        self._generate_learning_curve(final_model, X_train, y_train, X_test, y_test)

        importance = generate_shap_artifacts(
            final_model,
            X_train,
            self.config.plots_dir,
            self.config.reports_dir,
            self.config.shap_sample_size,
            self.config.top_feature_count,
            self.config.random_seed,
        )
        self._log(
            "shap_explainability_complete",
            backend=importance.attrs.get("explanation_backend"),
            top_features=importance["feature"].head(self.config.top_feature_count).tolist(),
        )

        metadata = self._build_metadata(
            X_train,
            X_test,
            metrics,
            best_parameters,
            cv_rmse,
            comparison,
            importance,
        )
        joblib.dump(final_model, self.config.credit_model_path)
        joblib.dump(final_model, self.config.best_model_path)
        self.config.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._log(
            "model_export_complete",
            credit_model=str(self.config.credit_model_path),
            best_model=str(self.config.best_model_path),
            metadata=str(self.config.metadata_path),
        )
        self._write_training_report(comparison, best_parameters, cv_rmse, metrics)
        self._write_model_card(metrics, comparison, importance)

        result = TrainingRunResult(
            model_path=self.config.credit_model_path,
            metadata_path=self.config.metadata_path,
            training_samples=len(X_train),
            validation_samples=len(X_test),
            feature_count=X_train.shape[1],
            validation_metrics=metrics,
            best_parameters=best_parameters,
            cv_rmse=cv_rmse,
        )
        self._log("training_end", **asdict(result))
        return result

    def _load_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        frame = pd.read_csv(self.config.dataset_path)
        if self.config.target_column not in frame.columns:
            raise KeyError(f"Target column '{self.config.target_column}' is not present in {self.config.dataset_path}")
        features = frame.drop(columns=[self.config.target_column])
        target = pd.to_numeric(frame[self.config.target_column], errors="raise").astype(float)
        non_numeric = features.select_dtypes(exclude=[np.number])
        if not non_numeric.empty:
            raise TypeError(f"Processed dataset has non-numeric features: {list(non_numeric.columns)}")
        if features.empty:
            raise ValueError("Processed dataset does not contain any features")
        if features.isna().any().any() or target.isna().any():
            raise ValueError("Processed dataset contains missing values")
        self._log("dataset_loaded", rows=len(frame), features=features.shape[1], path=str(self.config.dataset_path))
        return features.astype(float), target

    def _compare_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for name, model in self._comparison_models().items():
            self._log("model_comparison_started", model=name)
            started = perf_counter()
            model.fit(X_train, y_train)
            training_time = perf_counter() - started
            started = perf_counter()
            predictions = model.predict(X_test)
            prediction_time = perf_counter() - started
            metrics = calculate_regression_metrics(y_test.to_numpy(), np.asarray(predictions, dtype=float))
            rows.append(
                {
                    "model": name,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "mape_percent": metrics["mape_percent"],
                    "training_time_seconds": training_time,
                    "prediction_time_seconds": prediction_time,
                }
            )
            self._log("model_comparison_finished", model=name, **metrics, training_time_seconds=training_time)
            del model
            gc.collect()
        return pd.DataFrame(rows).sort_values("rmse", ignore_index=True)

    def _comparison_models(self) -> dict[str, object]:
        estimators = self.config.comparison_estimators
        return {
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=estimators,
                max_features=0.8,
                min_samples_leaf=2,
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
            ),
            "Gradient Boosting Regressor": GradientBoostingRegressor(
                n_estimators=estimators,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=3,
                random_state=self.config.random_seed,
            ),
            "XGBoost Regressor": self._xgb_regressor(n_estimators=self.config.xgb_base_estimators),
            "Extra Trees Regressor": ExtraTreesRegressor(
                n_estimators=estimators,
                max_features=0.8,
                min_samples_leaf=2,
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
            ),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
                max_iter=estimators,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=self.config.random_seed,
            ),
        }

    def _tune_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
        self._log(
            "cross_validation_started",
            cv_folds=self.config.cv_folds,
            randomized_search_iterations=self.config.randomized_search_iterations,
        )
        search = RandomizedSearchCV(
            estimator=self._xgb_regressor(n_estimators=self.config.xgb_base_estimators),
            param_distributions={
                "max_depth": [3, 4, 5, 6, 8, 10],
                "learning_rate": [0.02, 0.03, 0.05, 0.075, 0.1],
                "n_estimators": [200, 300, 450, 600],
                "subsample": [0.65, 0.75, 0.85, 1.0],
                "colsample_bytree": [0.60, 0.75, 0.85, 1.0],
                "min_child_weight": [1, 3, 5, 8],
                "gamma": [0.0, 0.05, 0.1, 0.25, 0.5],
            },
            n_iter=self.config.randomized_search_iterations,
            scoring="neg_root_mean_squared_error",
            n_jobs=self.config.search_n_jobs,
            cv=KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_seed,
            ),
            random_state=self.config.random_seed,
            refit=True,
            verbose=1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        self._log("cross_validation_complete", best_cv_rmse=float(-search.best_score_))
        return search

    def _xgb_regressor(self, n_estimators: int) -> object:
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "XGBoost is required for Phase 3. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from error
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.05,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=self.config.random_seed,
            n_jobs=self.config.xgb_n_jobs,
            verbosity=0,
        )

    def _plot_model_comparison(self, comparison: pd.DataFrame) -> None:
        displayed = comparison.sort_values("rmse", ascending=False)
        figure, axis = plt.subplots(figsize=(10, 6))
        bars = axis.barh(displayed["model"], displayed["rmse"], color="#2563eb")
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in displayed["rmse"]], padding=3)
        axis.set_title("Model Comparison: Hold-out RMSE (lower is better)")
        axis.set_xlabel("RMSE")
        figure.tight_layout()
        figure.savefig(self.config.plots_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _generate_learning_curve(
        self,
        model: object,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        rows: list[dict[str, float]] = []
        for fraction in sorted(set(self.config.learning_curve_fractions)):
            sample_size = max(2, int(len(X_train) * fraction))
            sample = X_train.sample(n=sample_size, random_state=self.config.random_seed)
            sample_target = y_train.loc[sample.index]
            curve_model = clone(model)
            curve_model.fit(sample, sample_target)
            rows.append(
                {
                    "training_samples": float(sample_size),
                    "training_rmse": calculate_regression_metrics(
                        sample_target.to_numpy(), curve_model.predict(sample)
                    )["rmse"],
                    "validation_rmse": calculate_regression_metrics(
                        y_test.to_numpy(), curve_model.predict(X_test)
                    )["rmse"],
                }
            )
            del curve_model
            gc.collect()
        curve = pd.DataFrame(rows)
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(curve["training_samples"], curve["training_rmse"], marker="o", label="Training RMSE")
        axis.plot(curve["training_samples"], curve["validation_rmse"], marker="o", label="Validation RMSE")
        axis.set_title("XGBoost Learning Curve")
        axis.set_xlabel("Training samples")
        axis.set_ylabel("RMSE")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(self.config.plots_dir / "learning_curve.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _build_metadata(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        metrics: dict[str, float],
        best_parameters: dict[str, Any],
        cv_rmse: float,
        comparison: pd.DataFrame,
        importance: pd.DataFrame,
    ) -> dict[str, Any]:
        target_span = 100.0
        baseline_confidence = float(
            np.clip(
                1 - metrics["rmse"] / target_span,
                self.config.confidence_lower_bound,
                self.config.confidence_upper_bound,
            )
        )
        quantiles = {
            feature: {"p01": float(X_train[feature].quantile(0.01)), "p99": float(X_train[feature].quantile(0.99))}
            for feature in X_train.columns
        }
        return {
            "model_name": "XGBRegressor",
            "model_version": "v0.3.0",
            "training_date_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_version": f"processed_credit_dataset.csv:sha256:{_file_sha256(self.config.dataset_path)[:12]}",
            "git_commit": _git_commit(),
            "random_seed": self.config.random_seed,
            "training_samples": int(len(X_train)),
            "validation_samples": int(len(X_test)),
            "feature_count": int(X_train.shape[1]),
            "feature_names": X_train.columns.tolist(),
            "target_column": self.config.target_column,
            "metrics": metrics,
            "cross_validation": {"folds": self.config.cv_folds, "best_rmse": cv_rmse},
            "hyperparameters": best_parameters,
            "comparison_results": _records_for_json(comparison),
            "top_features": _records_for_json(importance.head(self.config.top_feature_count).reset_index()),
            "explainability": {
                "engine": importance.attrs.get("explanation_backend", "unknown"),
                "note": "TreeSHAP explanations are calculated for the final XGBoost model.",
            },
            "confidence_calibration": {
                "method": "holdout_rmse_with_feature_quantile_ood_penalty",
                "baseline_confidence": baseline_confidence,
                "lower_bound": self.config.confidence_lower_bound,
                "upper_bound": self.config.confidence_upper_bound,
                "note": "Confidence is a heuristic reliability proxy, not a calibrated probability.",
            },
            "feature_quantiles": quantiles,
        }

    def _write_training_report(
        self,
        comparison: pd.DataFrame,
        best_parameters: dict[str, Any],
        cv_rmse: float,
        metrics: dict[str, float],
    ) -> None:
        comparison_rows = "\n".join(
            "| {model} | {rmse:.4f} | {mae:.4f} | {r2:.4f} | {mape_percent:.2f}% | {training_time_seconds:.2f} | {prediction_time_seconds:.4f} |".format(**row)
            for row in comparison.to_dict(orient="records")
        )
        parameter_rows = "\n".join(f"| `{name}` | `{value}` |" for name, value in best_parameters.items())
        report = f"""# Credit Model Training Report

## Data Split

- Dataset: `{self.config.dataset_path.name}`
- Train/test split: `{1 - self.config.test_size:.0%}` / `{self.config.test_size:.0%}`
- Random seed: `{self.config.random_seed}`
- Cross-validation: `{self.config.cv_folds}`-fold KFold

## Model Comparison

| Model | RMSE | MAE | R² | MAPE | Train time (s) | Predict time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{comparison_rows}

XGBoost is the final model because it is a high-capacity, regularised gradient-boosted tree model that supports exact tree contributions for local explanations. The comparison table is retained as empirical evidence; final selection also prioritises the requested explainable XGBoost deployment path.

## Hyperparameter Optimisation

RandomizedSearchCV used negative RMSE across {self.config.cv_folds} folds. Best cross-validation RMSE: **{cv_rmse:.4f}**.

| Parameter | Selected value |
| --- | --- |
{parameter_rows}

## Hold-out Metrics

| RMSE | MAE | R² | MAPE |
| ---: | ---: | ---: | ---: |
| {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['r2']:.4f} | {metrics['mape_percent']:.2f}% |

## Outputs

- `ml/models/credit_model.pkl` and `ml/models/best_model.pkl`
- `ml/models/model_metadata.json`
- `ml/reports/model_comparison.csv`, `feature_importance.csv`, and `feature_ranking.json`
- SHAP, diagnostic, comparison, and learning-curve plots in `ml/plots/`
"""
        self.config.training_report_path.write_text(report, encoding="utf-8")

    def _write_model_card(
        self,
        metrics: dict[str, float],
        comparison: pd.DataFrame,
        importance: pd.DataFrame,
    ) -> None:
        top_features = ", ".join(f"`{name}`" for name in importance["feature"].head(10))
        comparison_winner = str(comparison.iloc[0]["model"])
        report = f"""# TrustVest AI Credit Model Card

## Purpose

This model predicts the synthetic `credit_likelihood` score used for TrustVest educational and hackathon experimentation. It supports product prototyping and explainability demonstrations only.

## Target Variable

`credit_likelihood`, a continuous synthetic score ranging approximately from 0 to 100.

## Training Data

The model was trained on the processed synthetic credit dataset generated by this repository. The training split contains {int((1 - self.config.test_size) * 100)}% of the dataset; the remaining {int(self.config.test_size * 100)}% is a fixed-seed hold-out validation set. The dataset is numeric, scaled, and one-hot encoded before this phase.

## Model Type

`XGBRegressor` trained after {self.config.cv_folds}-fold randomized hyperparameter search. The lowest-RMSE comparison model was `{comparison_winner}`. XGBoost remains the exported model because it combines strong nonlinear tree performance, regularisation controls, and TreeSHAP-compatible local explanations.

## Validation Metrics

| RMSE | MAE | R² | MAPE |
| ---: | ---: | ---: | ---: |
| {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['r2']:.4f} | {metrics['mape_percent']:.2f}% |

Top global SHAP features: {top_features}.

## Limitations

- Performance is measured only on synthetic, processed data and may not transfer to real users, lenders, geographies, or time periods.
- The model expects the exact saved processed-feature schema; it does not perform raw-data cleaning or preprocessing.
- The reported inference confidence is a heuristic reliability proxy based on hold-out error and feature-quantile coverage, not a probability of correctness.
- Feature attribution describes model behaviour; it does not establish causal effects or financial advice.

## Ethical Considerations and Known Biases

The source schema includes demographic and location-related attributes. Even when modelled as synthetic signals, these can encode disparate treatment or proxy discrimination. Never use this model to approve, deny, price, rank, or otherwise make decisions about real people. Any future real-data use would require legal review, governance, documented consent, fairness evaluation across protected groups, human oversight, appeal processes, and ongoing monitoring for drift and harm.

## Educational Disclaimer

TrustVest AI is an educational and hackathon project. This artifact is not a credit bureau score, lending recommendation, financial product, or production decision system.
"""
        self.config.model_card_path.write_text(report, encoding="utf-8")

    def _configure_logging(self) -> None:
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = JsonFormatter()
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        file_paths = {getattr(handler, "baseFilename", None) for handler in self.logger.handlers}
        if str(self.config.training_log_path) not in file_paths:
            file_handler = logging.FileHandler(self.config.training_log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _log(self, event: str, **details: object) -> None:
        self.logger.info(event, extra=details)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_safe(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_training(config: TrainingConfig | None = None) -> TrainingRunResult:
    """Convenience function for applications and orchestration code."""
    return CreditModelTrainer(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TrustVest explainable credit model.")
    parser.add_argument("--dataset", type=Path, help="Optional processed dataset override")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--test-size", type=float, help="Hold-out test fraction override")
    parser.add_argument("--cv-folds", type=int, help="Cross-validation folds override")
    parser.add_argument("--search-iterations", type=int, help="RandomizedSearchCV iterations override")
    parser.add_argument("--shap-sample-size", type=int, help="Maximum SHAP explanation rows")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster 5-fold smoke-training configuration (two search candidates and 250 SHAP rows).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the pipeline as ``python -m ml.training.train_credit_model``."""
    arguments = _parse_args()
    defaults = TrainingConfig()
    config = TrainingConfig(
        dataset_path=arguments.dataset or defaults.dataset_path,
        random_seed=arguments.seed if arguments.seed is not None else defaults.random_seed,
        test_size=arguments.test_size if arguments.test_size is not None else defaults.test_size,
        cv_folds=arguments.cv_folds if arguments.cv_folds is not None else defaults.cv_folds,
        randomized_search_iterations=(2 if arguments.quick else arguments.search_iterations)
        or defaults.randomized_search_iterations,
        shap_sample_size=(250 if arguments.quick else arguments.shap_sample_size) or defaults.shap_sample_size,
        comparison_estimators=20 if arguments.quick else defaults.comparison_estimators,
        xgb_base_estimators=75 if arguments.quick else defaults.xgb_base_estimators,
    )
    result = run_training(config)
    print(json.dumps(asdict(result), default=str, indent=2))


if __name__ == "__main__":
    main()
