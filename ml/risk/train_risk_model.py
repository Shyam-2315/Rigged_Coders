"""Train, compare, and export the TrustVest behavioral risk classifier.

Run from the repository root with ``python -m ml.risk.train_risk_model``.
The command creates the synthetic data automatically when it is unavailable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

from .config import ENGINEERED_FEATURE_COLUMNS, QUESTIONNAIRE_CATEGORICAL_COLUMNS, QUESTIONNAIRE_NUMERIC_COLUMNS, RiskConfig
from .dataset_generator import generate_and_save
from .feature_engineering import model_input_frame


LOGGER = logging.getLogger(__name__)
CLASS_LABELS: tuple[str, ...] = ("Low", "Medium", "High")


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """A compact summary suitable for automation or a model registry."""

    selected_model: str
    test_accuracy: float
    test_weighted_f1: float
    test_roc_auc_ovr_weighted: float
    model_path: Path
    dataset_path: Path


class BehavioralRiskTrainer:
    """Own the deterministic Phase 4 training and reporting workflow."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def run(self) -> TrainingRunResult:
        """Generate inputs when needed, compare models, and export the winner."""
        self.config.create_output_directories()
        self._ensure_dataset()
        # ``None`` is a valid questionnaire answer for investment knowledge
        # and experience, so pandas must not coerce it to a missing value.
        dataset = pd.read_csv(self.config.dataset_path, keep_default_na=False)
        if self.config.target_column not in dataset:
            raise ValueError(f"Dataset is missing target column: {self.config.target_column}")

        engineered = model_input_frame(dataset)
        engineered_with_target = engineered.copy()
        engineered_with_target[self.config.target_column] = dataset[self.config.target_column].astype(str)
        engineered_with_target.to_csv(self.config.feature_dataset_path, index=False)
        target = _encode_targets(dataset[self.config.target_column])

        train_raw, test_raw, y_train, y_test = train_test_split(
            engineered,
            target,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=target,
        )
        scaler, encoder = self._fit_preprocessors(train_raw)
        X_train, feature_names = _transform_features(train_raw, scaler, encoder)
        X_test, _ = _transform_features(test_raw, scaler, encoder)

        models = self._build_models()
        comparison_rows: list[dict[str, float | str]] = []
        fitted_models: dict[str, object] = {}
        for name, model in models.items():
            LOGGER.info("Training %s", name)
            started = perf_counter()
            model.fit(X_train, y_train)
            training_time = perf_counter() - started
            started = perf_counter()
            probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
            predictions = np.asarray(model.predict(X_test), dtype=int)
            prediction_time = perf_counter() - started
            metrics = _classification_metrics(y_test, predictions, probabilities)
            comparison_rows.append({"model": name, **metrics, "training_time_seconds": training_time, "prediction_time_seconds": prediction_time})
            fitted_models[name] = model

        comparison = pd.DataFrame(comparison_rows).sort_values(
            ["f1_weighted", "roc_auc_ovr_weighted", "accuracy", "prediction_time_seconds"],
            ascending=[False, False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        selected_name = str(comparison.iloc[0]["model"])
        selected_model = fitted_models[selected_name]
        final_probabilities = np.asarray(selected_model.predict_proba(X_test), dtype=float)
        final_predictions = np.asarray(selected_model.predict(X_test), dtype=int)
        final_metrics = _classification_metrics(y_test, final_predictions, final_probabilities)

        joblib.dump(selected_model, self.config.model_path)
        joblib.dump(scaler, self.config.scaler_path)
        joblib.dump(encoder, self.config.encoder_path)
        metadata = self._build_metadata(selected_name, feature_names, comparison, final_metrics)
        self.config.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        importance = self._calculate_feature_importance(selected_model, X_test, y_test, feature_names)
        importance.to_csv(self.config.feature_importance_path, index=False)
        self._plot_confusion_matrix(y_test, final_predictions)
        self._plot_roc_curve(y_test, final_probabilities)
        shap_status = self._generate_shap_artifacts(selected_model, X_train, X_test, final_predictions, feature_names)
        self._write_reports(comparison, final_metrics, y_test, final_predictions, selected_name, shap_status)

        LOGGER.info("Selected %s with weighted F1 %.4f", selected_name, final_metrics["f1_weighted"])
        return TrainingRunResult(
            selected_model=selected_name,
            test_accuracy=final_metrics["accuracy"],
            test_weighted_f1=final_metrics["f1_weighted"],
            test_roc_auc_ovr_weighted=final_metrics["roc_auc_ovr_weighted"],
            model_path=self.config.model_path,
            dataset_path=self.config.dataset_path,
        )

    def _ensure_dataset(self) -> None:
        if not self.config.dataset_path.exists():
            LOGGER.info("Dataset not found; generating %s profiles", self.config.num_profiles)
            generate_and_save(self.config)

    def _fit_preprocessors(self, frame: pd.DataFrame) -> tuple[StandardScaler, OneHotEncoder]:
        scaler = StandardScaler()
        scaler.fit(frame.loc[:, self.config.model_numeric_columns])
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        encoder.fit(frame.loc[:, QUESTIONNAIRE_CATEGORICAL_COLUMNS])
        return scaler, encoder

    def _build_models(self) -> dict[str, object]:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:  # pragma: no cover - dependency is declared in requirements.txt
            raise RuntimeError("XGBoost is required for the Phase 4 model comparison") from error
        return {
            "Logistic Regression": LogisticRegression(max_iter=2_000, C=1.0, random_state=self.config.random_seed),
            "Random Forest": RandomForestClassifier(
                n_estimators=self.config.comparison_estimators,
                max_depth=18,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
            ),
            "XGBoost Classifier": XGBClassifier(
                n_estimators=self.config.xgb_estimators,
                max_depth=6,
                learning_rate=0.06,
                subsample=0.88,
                colsample_bytree=0.88,
                min_child_weight=3,
                reg_lambda=1.2,
                objective="multi:softprob",
                num_class=len(CLASS_LABELS),
                eval_metric="mlogloss",
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
                tree_method="hist",
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=self.config.comparison_estimators,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=self.config.xgb_estimators,
                max_leaf_nodes=31,
                l2_regularization=0.25,
                random_state=self.config.random_seed,
            ),
        }

    def _build_metadata(
        self,
        selected_name: str,
        feature_names: list[str],
        comparison: pd.DataFrame,
        final_metrics: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "model_name": selected_name,
            "model_version": "v0.4.0",
            "training_date_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": self.config.random_seed,
            "dataset_path": str(self.config.dataset_path),
            "dataset_rows": self._dataset_rows(),
            "target_column": self.config.target_column,
            "class_labels": list(CLASS_LABELS),
            "questionnaire_numeric_columns": list(QUESTIONNAIRE_NUMERIC_COLUMNS),
            "questionnaire_categorical_columns": list(QUESTIONNAIRE_CATEGORICAL_COLUMNS),
            "engineered_feature_columns": list(ENGINEERED_FEATURE_COLUMNS),
            "model_feature_names": feature_names,
            "metrics": final_metrics,
            "model_comparison": _json_records(comparison),
            "disclaimer": "Trained only on synthetic data for research and product prototyping; not personalised investment advice.",
        }

    def _dataset_rows(self) -> int:
        return int(sum(1 for _ in self.config.dataset_path.open("rb")) - 1)

    def _calculate_feature_importance(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        if hasattr(model, "feature_importances_"):
            values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
            method = "native_feature_importance"
        elif hasattr(model, "coef_"):
            values = np.mean(np.abs(np.asarray(getattr(model, "coef_"), dtype=float)), axis=0)
            method = "mean_absolute_coefficient"
        else:
            from sklearn.inspection import permutation_importance

            sample_size = min(4_000, len(X_test))
            sample_indices = np.arange(sample_size)
            permutation = permutation_importance(
                model,
                X_test[sample_indices],
                y_test[sample_indices],
                n_repeats=4,
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
                scoring="f1_weighted",
            )
            values = permutation.importances_mean
            method = "permutation_importance_weighted_f1"
        return (
            pd.DataFrame({"feature": feature_names, "importance": values})
            .assign(importance_method=method)
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _plot_confusion_matrix(self, y_test: np.ndarray, predictions: np.ndarray) -> None:
        figure, axis = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=CLASS_LABELS,
            cmap="Blues",
            values_format="d",
            ax=axis,
        )
        axis.set_title("Behavioral Risk Profile Confusion Matrix")
        figure.tight_layout()
        figure.savefig(self.config.confusion_matrix_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _plot_roc_curve(self, y_test: np.ndarray, probabilities: np.ndarray) -> None:
        figure, axis = plt.subplots(figsize=(7, 6))
        binary = label_binarize(y_test, classes=np.arange(len(CLASS_LABELS)))
        for index, label in enumerate(CLASS_LABELS):
            false_positive_rate, true_positive_rate, _ = roc_curve(binary[:, index], probabilities[:, index])
            auc = roc_auc_score(binary[:, index], probabilities[:, index])
            axis.plot(false_positive_rate, true_positive_rate, label=f"{label} (AUC {auc:.3f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-Rest ROC Curves")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(self.config.roc_curve_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _generate_shap_artifacts(
        self,
        model: object,
        X_train: np.ndarray,
        X_test: np.ndarray,
        predictions: np.ndarray,
        feature_names: list[str],
    ) -> str:
        """Create SHAP plots for a selected compatible model, without blocking deployment."""
        if not _supports_shap(model):
            return "Skipped: selected model does not expose a supported SHAP interface."
        try:
            import shap

            explain_rows = min(self.config.max_shap_samples, len(X_test))
            sample = pd.DataFrame(X_test[:explain_rows], columns=feature_names)
            if hasattr(model, "coef_"):
                background_rows = min(self.config.max_shap_background_rows, len(X_train))
                background = X_train[:background_rows]
                raw_values = shap.LinearExplainer(model, background).shap_values(sample)
            else:
                raw_values = shap.TreeExplainer(model).shap_values(sample)
            tensor = _normalise_shap_values(raw_values, sample.shape[1])
            selected_class = int(pd.Series(predictions[:explain_rows]).mode().iloc[0])
            class_values = tensor[min(selected_class, tensor.shape[0] - 1)]
            global_importance = np.mean(np.abs(tensor), axis=(0, 1))
            top_indices = np.argsort(global_importance)[-20:][::-1]

            self._plot_shap_summary(sample, class_values, top_indices, CLASS_LABELS[selected_class])
            self._plot_shap_bar(global_importance, feature_names, top_indices)
            self._plot_shap_local(sample.iloc[0], class_values[0], feature_names)
            pd.DataFrame(
                {"feature": np.asarray(feature_names)[top_indices], "mean_absolute_shap": global_importance[top_indices]}
            ).to_csv(self.config.shap_top_features_path, index=False)
            return "Generated global summary, bar plot, local explanation, and top-feature report."
        except (ImportError, ValueError, TypeError, AttributeError, IndexError) as error:
            LOGGER.warning("SHAP generation skipped: %s", error)
            return f"Skipped after SHAP compatibility error: {error}"

    def _plot_shap_summary(self, sample: pd.DataFrame, values: np.ndarray, top_indices: np.ndarray, label: str) -> None:
        figure, axis = plt.subplots(figsize=(10, 7))
        colorbar = None
        for display_index, feature_index in enumerate(top_indices[::-1]):
            jitter = np.random.default_rng(self.config.random_seed + int(feature_index)).normal(0, 0.10, len(sample))
            colorbar = axis.scatter(
                values[:, feature_index],
                display_index + jitter,
                c=sample.iloc[:, feature_index],
                cmap="coolwarm",
                alpha=0.55,
                s=10,
                edgecolors="none",
            )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(range(len(top_indices)), np.asarray(sample.columns)[top_indices[::-1]])
        axis.set_xlabel("SHAP value (impact on selected class)")
        axis.set_title(f"SHAP Global Summary — {label} profile")
        if colorbar is not None:
            figure.colorbar(colorbar, ax=axis, label="Feature value")
        figure.tight_layout()
        figure.savefig(self.config.shap_summary_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _plot_shap_bar(self, importance: np.ndarray, feature_names: list[str], top_indices: np.ndarray) -> None:
        figure, axis = plt.subplots(figsize=(9, 6))
        names = np.asarray(feature_names)[top_indices][::-1]
        values = importance[top_indices][::-1]
        axis.barh(names, values, color="#2563eb")
        axis.set_xlabel("Mean |SHAP value| across risk classes")
        axis.set_title("SHAP Global Feature Importance")
        figure.tight_layout()
        figure.savefig(self.config.shap_bar_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _plot_shap_local(self, row: pd.Series, values: np.ndarray, feature_names: list[str]) -> None:
        top_indices = np.argsort(np.abs(values))[-12:][::-1]
        displayed_values = values[top_indices][::-1]
        labels = [f"{feature_names[index]} = {row.iloc[index]:.2f}" for index in top_indices[::-1]]
        figure, axis = plt.subplots(figsize=(10, 6))
        colors = ["#16a34a" if value > 0 else "#dc2626" for value in displayed_values]
        axis.barh(labels, displayed_values, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("SHAP value")
        axis.set_title("Local SHAP Explanation — representative hold-out profile")
        figure.tight_layout()
        figure.savefig(self.config.shap_local_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _write_reports(
        self,
        comparison: pd.DataFrame,
        final_metrics: dict[str, float],
        y_test: np.ndarray,
        final_predictions: np.ndarray,
        selected_name: str,
        shap_status: str,
    ) -> None:
        metrics_payload = {
            "model_version": "v0.4.0",
            "selected_model": selected_name,
            "class_labels": list(CLASS_LABELS),
            "holdout_metrics": final_metrics,
            "model_comparison": _json_records(comparison),
            "shap": shap_status,
        }
        self.config.training_metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        table_rows = "\n".join(
            "| {model} | {accuracy:.4f} | {precision_weighted:.4f} | {recall_weighted:.4f} | {f1_weighted:.4f} | {roc_auc_ovr_weighted:.4f} | {training_time_seconds:.3f} | {prediction_time_seconds:.4f} |".format(**row)
            for row in comparison.to_dict(orient="records")
        )
        detailed_report = classification_report(
            y_test,
            final_predictions,
            labels=np.arange(len(CLASS_LABELS)),
            target_names=CLASS_LABELS,
            zero_division=0,
        )
        content = f"""# Behavioral Risk Classifier Report

## Selected Model

`{selected_name}` was selected automatically using weighted F1, then weighted one-vs-rest ROC AUC, accuracy, and lower prediction latency as tie-breakers.

## Hold-out Model Comparison

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC AUC OvR (weighted) | Train (s) | Predict (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table_rows}

## Per-Class Report

```
{detailed_report}
```

## Explainability

{shap_status}

The classifier is trained only on synthetic questionnaire data. It supports product research and is not personalised financial advice or a suitability determination.
"""
        self.config.classification_report_path.write_text(content, encoding="utf-8")


def _transform_features(frame: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder) -> tuple[np.ndarray, list[str]]:
    numeric = scaler.transform(frame.loc[:, (*QUESTIONNAIRE_NUMERIC_COLUMNS, *ENGINEERED_FEATURE_COLUMNS)])
    categorical = encoder.transform(frame.loc[:, QUESTIONNAIRE_CATEGORICAL_COLUMNS])
    names = [*QUESTIONNAIRE_NUMERIC_COLUMNS, *ENGINEERED_FEATURE_COLUMNS, *encoder.get_feature_names_out(QUESTIONNAIRE_CATEGORICAL_COLUMNS).tolist()]
    return np.hstack((numeric, categorical)), names


def _encode_targets(target: pd.Series) -> np.ndarray:
    values = target.astype(str)
    unsupported = sorted(set(values).difference(CLASS_LABELS))
    if unsupported:
        raise ValueError(f"Unsupported risk profile labels: {unsupported}")
    encoded = values.map({label: index for index, label in enumerate(CLASS_LABELS)})
    if encoded.isna().any() or encoded.nunique() != len(CLASS_LABELS):
        raise ValueError("Training data must contain Low, Medium, and High risk profiles")
    return encoded.to_numpy(dtype=int)


def _classification_metrics(y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_weighted": float(precision_score(y_true, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "roc_auc_ovr_weighted": float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted", labels=np.arange(len(CLASS_LABELS)))),
    }


def _supports_shap(model: object) -> bool:
    return (
        hasattr(model, "coef_")
        or hasattr(model, "feature_importances_")
        or model.__class__.__module__.startswith("xgboost")
    )


def _normalise_shap_values(values: Any, feature_count: int) -> np.ndarray:
    """Return SHAP values as ``(classes, samples, features)`` across SHAP APIs."""
    if isinstance(values, list):
        result = np.stack([np.asarray(item, dtype=float) for item in values], axis=0)
    else:
        result = np.asarray(values, dtype=float)
        if result.ndim == 2:
            result = result[np.newaxis, :, :]
        elif result.ndim == 3:
            if result.shape[1] == feature_count:  # (samples, features, classes)
                result = np.moveaxis(result, -1, 0)
            elif result.shape[2] == feature_count:  # (classes, samples, features)
                pass
            else:
                raise ValueError(f"Unexpected SHAP tensor shape: {result.shape}")
        else:
            raise ValueError(f"Unexpected SHAP tensor shape: {result.shape}")
    if result.ndim != 3 or result.shape[2] != feature_count:
        raise ValueError(f"Could not normalize SHAP values with shape {result.shape}")
    return result


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_safe(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def run_training(config: RiskConfig | None = None) -> TrainingRunResult:
    """Convenience entry point for applications, automation, and tests."""
    return BehavioralRiskTrainer(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TrustVest's behavioral risk classifier.")
    parser.add_argument("--dataset", type=Path, help="Existing questionnaire dataset to train from.")
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--quick", action="store_true", help="Use fewer estimators and SHAP rows for a smoke run.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = _parse_args()
    config = RiskConfig()
    if arguments.dataset is not None:
        config = replace(config, data_dir=arguments.dataset.parent, dataset_filename=arguments.dataset.name)
    if arguments.seed is not None:
        config = replace(config, random_seed=arguments.seed)
    if arguments.quick:
        config = replace(config, comparison_estimators=50, xgb_estimators=75, max_shap_samples=250)
    result = run_training(config)
    print(json.dumps(asdict(result), default=str, indent=2))


if __name__ == "__main__":
    main()
