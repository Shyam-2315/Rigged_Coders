"""Inference API for the exported TrustVest XGBoost credit model.

Inputs must already conform to the Phase 2 processed feature schema.  This
keeps the model endpoint deterministic; a later FastAPI service can compose
the saved preprocessing artifacts before calling this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ML_ROOT / "models" / "credit_model.pkl"
DEFAULT_METADATA_PATH = ML_ROOT / "models" / "model_metadata.json"


def load_model(model_path: str | Path | None = None) -> object:
    """Load the exported model without retraining it."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Credit model not found: {path}")
    return joblib.load(path)


def predict(
    records: pd.DataFrame | Mapping[str, float] | Sequence[Mapping[str, float]],
    model: object | None = None,
    metadata_path: str | Path | None = None,
) -> np.ndarray:
    """Return one prediction per processed input record."""
    metadata = _load_metadata(metadata_path)
    frame = _prepare_features(records, metadata["feature_names"])
    active_model = model if model is not None else load_model()
    return np.asarray(active_model.predict(frame), dtype=float)


def predict_with_explanation(
    record: pd.DataFrame | Mapping[str, float],
    model: object | None = None,
    metadata_path: str | Path | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Return a score, calibrated confidence, and signed local feature effects."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")
    metadata = _load_metadata(metadata_path)
    frame = _prepare_features(record, metadata["feature_names"])
    if len(frame) != 1:
        raise ValueError("predict_with_explanation accepts exactly one record")
    active_model = model if model is not None else load_model()
    score = float(np.asarray(active_model.predict(frame), dtype=float)[0])
    impacts = _local_feature_impacts(active_model, frame)
    return {
        "credit_score": round(score, 2),
        "confidence": round(_confidence(frame, metadata), 4),
        "top_positive_features": _top_impacts(impacts, positive=True, top_n=top_n),
        "top_negative_features": _top_impacts(impacts, positive=False, top_n=top_n),
    }


def _load_metadata(metadata_path: str | Path | None) -> dict[str, Any]:
    path = Path(metadata_path) if metadata_path is not None else DEFAULT_METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata.get("feature_names"), list):
        raise ValueError("Model metadata is missing the feature_names schema")
    return metadata


def _prepare_features(
    records: pd.DataFrame | Mapping[str, float] | Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, Mapping):
        frame = pd.DataFrame([records])
    else:
        frame = pd.DataFrame(list(records))
    expected = list(feature_names)
    missing = [name for name in expected if name not in frame.columns]
    unexpected = [name for name in frame.columns if name not in expected]
    if missing or unexpected:
        problems: list[str] = []
        if missing:
            problems.append(f"missing features: {missing}")
        if unexpected:
            problems.append(f"unexpected features: {unexpected}")
        raise ValueError("Input does not match the processed feature schema (" + "; ".join(problems) + ")")
    frame = frame.loc[:, expected].apply(pd.to_numeric, errors="raise")
    if frame.isna().any().any():
        raise ValueError("Input contains missing feature values")
    return frame


def _local_feature_impacts(model: object, frame: pd.DataFrame) -> pd.Series:
    """Use exact TreeSHAP contributions from XGBoost, with a safe generic fallback."""
    if hasattr(model, "get_booster"):
        try:
            import xgboost as xgb

            matrix = xgb.DMatrix(frame, feature_names=frame.columns.tolist())
            contributions = model.get_booster().predict(matrix, pred_contribs=True)[0, :-1]
            return pd.Series(contributions, index=frame.columns, dtype=float)
        except (ImportError, ValueError, TypeError):
            # The fallback preserves a useful response if this module is reused
            # with a non-XGBoost tree model.
            pass
    importances = np.asarray(getattr(model, "feature_importances_", np.ones(frame.shape[1])))
    return pd.Series(importances * frame.iloc[0].to_numpy(dtype=float), index=frame.columns)


def _top_impacts(impacts: pd.Series, positive: bool, top_n: int) -> list[dict[str, float | str]]:
    values = impacts[impacts > 0].sort_values(ascending=False) if positive else impacts[impacts < 0].sort_values()
    return [
        {"feature": str(feature), "impact": round(float(impact), 4)}
        for feature, impact in values.head(top_n).items()
    ]


def _confidence(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> float:
    """Adjust validation confidence when inputs exceed training quantile bounds."""
    calibration = metadata.get("confidence_calibration", {})
    baseline = float(calibration.get("baseline_confidence", 0.50))
    lower_bound = float(calibration.get("lower_bound", 0.50))
    upper_bound = float(calibration.get("upper_bound", 0.99))
    quantiles = metadata.get("feature_quantiles", {})
    out_of_distribution = 0
    for feature, value in frame.iloc[0].items():
        bounds = quantiles.get(feature)
        if bounds and (float(value) < float(bounds["p01"]) or float(value) > float(bounds["p99"])):
            out_of_distribution += 1
    ood_fraction = out_of_distribution / max(len(frame.columns), 1)
    adjusted = baseline * (1 - 0.40 * ood_fraction)
    return float(np.clip(adjusted, lower_bound, upper_bound))
