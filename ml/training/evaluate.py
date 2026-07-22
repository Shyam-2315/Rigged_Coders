"""Evaluation metrics, plots, and reports for credit-likelihood regressors."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return consistently named, JSON-safe regression metrics.

    MAPE is expressed as a percentage, rather than as the 0--1 fraction
    returned by scikit-learn, because this is the convention used in reports.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "mape_percent": float(mean_absolute_percentage_error(actual, predicted) * 100),
    }


def generate_evaluation_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plots_dir: Path,
    reports_dir: Path,
    metrics: Mapping[str, float],
) -> None:
    """Save validation diagnostics and a readable Markdown evaluation report."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = actual - predicted

    _prediction_vs_actual(actual, predicted, plots_dir / "prediction_vs_actual.png")
    _residual_distribution(predicted, residuals, plots_dir / "residual_distribution.png")
    _residual_histogram(residuals, plots_dir / "residual_histogram.png")
    _write_evaluation_report(reports_dir / "evaluation_report.md", metrics, residuals)


def _prediction_vs_actual(actual: np.ndarray, predicted: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(actual, predicted, s=10, alpha=0.25, color="#2563eb", edgecolors="none")
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    axis.plot([lower, upper], [lower, upper], "--", color="#dc2626", linewidth=1.5, label="Perfect prediction")
    axis.set_title("Actual vs Predicted Credit Likelihood")
    axis.set_xlabel("Actual credit_likelihood")
    axis.set_ylabel("Predicted credit_likelihood")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _residual_distribution(predicted: np.ndarray, residuals: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(predicted, residuals, s=10, alpha=0.25, color="#0f766e", edgecolors="none")
    axis.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    axis.set_title("Prediction Error Plot")
    axis.set_xlabel("Predicted credit_likelihood")
    axis.set_ylabel("Residual (actual - predicted)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _residual_histogram(residuals: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.hist(residuals, bins=40, color="#7c3aed", alpha=0.88, edgecolor="white")
    axis.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    axis.set_title("Residual Distribution")
    axis.set_xlabel("Residual (actual - predicted)")
    axis.set_ylabel("Validation samples")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_evaluation_report(
    output_path: Path,
    metrics: Mapping[str, float],
    residuals: np.ndarray,
) -> None:
    report = f"""# Credit Model Evaluation Report

## Hold-out Validation Metrics

| Metric | Value |
| --- | ---: |
| RMSE | {metrics['rmse']:.4f} |
| MAE | {metrics['mae']:.4f} |
| R² | {metrics['r2']:.4f} |
| MAPE | {metrics['mape_percent']:.2f}% |

## Residual Diagnostics

| Statistic | Value |
| --- | ---: |
| Mean residual | {float(np.mean(residuals)):.4f} |
| Median residual | {float(np.median(residuals)):.4f} |
| Residual standard deviation | {float(np.std(residuals, ddof=0)):.4f} |
| 95th percentile absolute residual | {float(np.percentile(np.abs(residuals), 95)):.4f} |

Residuals are calculated as actual minus predicted credit likelihood. These
figures are for the synthetic hold-out data only and do not establish
performance for real people or lending decisions.

## Generated Visuals

- `ml/plots/prediction_vs_actual.png`
- `ml/plots/residual_distribution.png`
- `ml/plots/residual_histogram.png`
"""
    output_path.write_text(report, encoding="utf-8")
