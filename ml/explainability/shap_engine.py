"""SHAP explainability artifacts for tree-based credit-likelihood models."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_shap_artifacts(
    model: object,
    features: pd.DataFrame,
    plots_dir: Path,
    reports_dir: Path,
    sample_size: int,
    top_feature_count: int,
    random_seed: int,
    generate_force_plot: bool = True,
) -> pd.DataFrame:
    """Create global and local SHAP explanations, returning ranked features.

    Explanation is sampled solely to keep artifact generation predictable for
    the 100k-row synthetic dataset. The model itself is trained on all of the
    training data supplied by the caller.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    sample = _sample_features(features, sample_size, random_seed)
    shap, explainer, shap_values, expected_value, backend = _tree_shap_values(model, sample)
    importance = pd.DataFrame(
        {
            "feature": sample.columns,
            "mean_abs_shap_value": np.abs(shap_values).mean(axis=0),
            "mean_shap_value": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap_value", ascending=False, ignore_index=True)
    importance.index = importance.index + 1
    importance.index.name = "rank"
    importance.to_csv(reports_dir / "feature_importance.csv")

    top_names = importance["feature"].head(min(3, len(importance))).tolist()
    if shap is not None:
        _save_summary_plot(shap, shap_values, sample, plots_dir / "shap_summary.png")
        _save_bar_plot(shap, shap_values, sample, plots_dir / "shap_bar.png")
        _save_bar_plot(shap, shap_values, sample, plots_dir / "feature_importance.png")
        _save_dependence_plots(shap, shap_values, sample, top_names, plots_dir)
        _save_waterfall_plot(
            shap, expected_value, shap_values, sample, plots_dir / "shap_waterfall.png"
        )
        if generate_force_plot:
            _save_force_plot(shap, expected_value, shap_values, sample, plots_dir / "shap_force_plot.html")
    else:
        _save_native_summary_plot(shap_values, sample, importance, plots_dir / "shap_summary.png")
        _save_native_bar_plot(importance, plots_dir / "shap_bar.png")
        _save_native_bar_plot(importance, plots_dir / "feature_importance.png")
        _save_native_dependence_plots(shap_values, sample, top_names, plots_dir)
        _save_native_waterfall_plot(
            expected_value, shap_values[0], sample.iloc[0], plots_dir / "shap_waterfall.png"
        )
        if generate_force_plot:
            _save_native_force_plot(
                expected_value, shap_values[0], sample.iloc[0], plots_dir / "shap_force_plot.html"
            )

    top_records = [
        {
            "rank": int(rank),
            "feature": str(row.feature),
            "mean_abs_shap_value": float(row.mean_abs_shap_value),
            "mean_shap_value": float(row.mean_shap_value),
        }
        for rank, row in importance.head(top_feature_count).iterrows()
    ]
    (reports_dir / "feature_ranking.json").write_text(
        json.dumps(top_records, indent=2), encoding="utf-8"
    )
    importance.attrs["explanation_backend"] = backend
    return importance


def _sample_features(features: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    if len(features) <= sample_size:
        return features.copy()
    return features.sample(n=sample_size, random_state=random_seed).copy()


def _normalise_shap_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        values = values[0]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {array.shape}")
    return array


def _tree_shap_values(
    model: object,
    features: pd.DataFrame,
) -> tuple[object | None, object | None, np.ndarray, float, str]:
    """Use SHAP TreeExplainer when importable, otherwise native TreeSHAP.

    Newer locked-down Windows hosts can block Numba's compiled extension, which
    prevents the external ``shap`` package from importing even though XGBoost
    is available. XGBoost's ``pred_contribs`` API computes the same exact
    TreeSHAP contributions for an XGBoost tree ensemble, so it is a robust
    production fallback that still produces the required artifacts.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = _normalise_shap_values(explainer.shap_values(features))
        expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        return shap, explainer, values, expected, "shap.TreeExplainer"
    except (ImportError, ModuleNotFoundError):
        if not hasattr(model, "get_booster"):
            raise RuntimeError("Native TreeSHAP fallback is available only for XGBoost models")
        try:
            import xgboost as xgb
        except ModuleNotFoundError as error:  # pragma: no cover - guarded by training dependency
            raise RuntimeError("XGBoost is required for TreeSHAP explanations") from error
        matrix = xgb.DMatrix(features, feature_names=features.columns.tolist())
        contributions = np.asarray(model.get_booster().predict(matrix, pred_contribs=True), dtype=float)
        if contributions.ndim != 2 or contributions.shape[1] != features.shape[1] + 1:
            raise ValueError(f"Unexpected XGBoost contribution shape: {contributions.shape}")
        return None, None, contributions[:, :-1], float(np.mean(contributions[:, -1])), "xgboost.pred_contribs"


def _save_summary_plot(shap: object, values: np.ndarray, features: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(11, 8))
    shap.summary_plot(values, features, show=False, max_display=20)
    plt.title("SHAP Summary: Global Feature Effects")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_bar_plot(shap: object, values: np.ndarray, features: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(11, 8))
    shap.summary_plot(values, features, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_dependence_plots(
    shap: object,
    values: np.ndarray,
    features: pd.DataFrame,
    feature_names: list[str],
    plots_dir: Path,
) -> None:
    for feature_name in feature_names:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature_name, values, features, show=False, interaction_index=None)
        plt.title(f"SHAP Dependence: {feature_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"shap_dependence_{_safe_filename(feature_name)}.png", dpi=180, bbox_inches="tight")
        plt.close()


def _save_waterfall_plot(
    shap: object,
    expected: float,
    values: np.ndarray,
    features: pd.DataFrame,
    output_path: Path,
) -> None:
    explanation = shap.Explanation(
        values=values[0],
        base_values=expected,
        data=features.iloc[0].to_numpy(),
        feature_names=features.columns.tolist(),
    )
    plt.figure(figsize=(11, 8))
    shap.plots.waterfall(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_force_plot(
    shap: object,
    expected: float,
    values: np.ndarray,
    features: pd.DataFrame,
    output_path: Path,
) -> None:
    force_plot = shap.force_plot(expected, values[0], features.iloc[0], matplotlib=False)
    shap.save_html(str(output_path), force_plot)


def _save_native_summary_plot(
    values: np.ndarray,
    features: pd.DataFrame,
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render a SHAP-style beeswarm from native XGBoost TreeSHAP values."""
    top_features = importance["feature"].head(20).tolist()[::-1]
    figure, axis = plt.subplots(figsize=(11, 8))
    random = np.random.default_rng(2026)
    for position, feature in enumerate(top_features):
        index = features.columns.get_loc(feature)
        jitter = random.normal(0, 0.11, len(features))
        scatter = axis.scatter(
            values[:, index],
            np.full(len(features), position) + jitter,
            c=features[feature].to_numpy(),
            cmap="coolwarm",
            s=12,
            alpha=0.65,
            linewidths=0,
        )
    figure.colorbar(scatter, ax=axis, label="Feature value")
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_yticks(range(len(top_features)), labels=top_features)
    axis.set_xlabel("TreeSHAP value (impact on model output)")
    axis.set_title("SHAP Summary: Global Feature Effects")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_native_bar_plot(importance: pd.DataFrame, output_path: Path) -> None:
    top = importance.head(20).sort_values("mean_abs_shap_value")
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.barh(top["feature"], top["mean_abs_shap_value"], color="#7c3aed")
    axis.set_xlabel("Mean absolute TreeSHAP value")
    axis.set_title("SHAP Global Feature Importance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_native_dependence_plots(
    values: np.ndarray,
    features: pd.DataFrame,
    feature_names: list[str],
    plots_dir: Path,
) -> None:
    for feature_name in feature_names:
        index = features.columns.get_loc(feature_name)
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(features[feature_name], values[:, index], s=14, alpha=0.6, color="#2563eb", edgecolors="none")
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xlabel(feature_name)
        axis.set_ylabel("TreeSHAP value")
        axis.set_title(f"SHAP Dependence: {feature_name}")
        figure.tight_layout()
        figure.savefig(plots_dir / f"shap_dependence_{_safe_filename(feature_name)}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def _save_native_waterfall_plot(
    expected: float,
    values: np.ndarray,
    features: pd.Series,
    output_path: Path,
) -> None:
    top = pd.Series(values, index=features.index).reindex(
        pd.Series(values, index=features.index).abs().sort_values(ascending=False).head(20).index
    ).sort_values()
    colours = ["#dc2626" if value < 0 else "#16a34a" for value in top]
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.barh(top.index, top.values, color=colours)
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("TreeSHAP contribution")
    axis.set_title(f"SHAP Waterfall: Local Explanation (base value {expected:.2f})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_native_force_plot(
    expected: float,
    values: np.ndarray,
    features: pd.Series,
    output_path: Path,
) -> None:
    impacts = pd.Series(values, index=features.index)
    rows = "\n".join(
        f"<tr><td>{escape(str(feature))}</td><td>{float(value):.4f}</td><td>{float(features[feature]):.4f}</td></tr>"
        for feature, value in impacts.reindex(impacts.abs().sort_values(ascending=False).head(20).index).items()
    )
    output_path.write_text(
        f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>TrustVest TreeSHAP force plot</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse}}td,th{{padding:.4rem .7rem;border:1px solid #ddd}}.neg{{color:#b91c1c}}.pos{{color:#15803d}}</style>
</head><body><h1>TrustVest local TreeSHAP explanation</h1><p>Base value: {expected:.4f}. Positive impacts raise the predicted score; negative impacts lower it.</p>
<table><thead><tr><th>Feature</th><th>TreeSHAP impact</th><th>Processed value</th></tr></thead><tbody>{rows}</tbody></table></body></html>""",
        encoding="utf-8",
    )


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
