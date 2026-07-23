"""Production-facing inference functions for the behavioral risk classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .config import QUESTIONNAIRE_CATEGORICAL_COLUMNS, QUESTIONNAIRE_NUMERIC_COLUMNS, RiskConfig
from .feature_engineering import model_input_frame
from .personas import recommendation_summary, select_persona


DEFAULT_CONFIG = RiskConfig()


def load_model(model_path: str | Path | None = None) -> object:
    """Load the selected risk classifier without retraining it."""
    path = Path(model_path) if model_path is not None else DEFAULT_CONFIG.model_path
    if not path.exists():
        raise FileNotFoundError(f"Risk model not found: {path}. Run ml.risk.train_risk_model first.")
    return joblib.load(path)


def predict_risk(
    answers: pd.DataFrame | Mapping[str, object] | Sequence[Mapping[str, object]],
    *,
    model: object | None = None,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    encoder_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict risk class, probabilities, confidence, and behavioral persona.

    A mapping or one-row frame returns one dictionary; a sequence or multi-row
    frame returns a dictionary for each response. Confidence is the model's
    highest class probability, not a guarantee of investment suitability.
    """
    raw = model_input_frame(answers)
    active_model = model if model is not None else load_model(model_path)
    scaler = _load_artifact(scaler_path, DEFAULT_CONFIG.scaler_path, "risk scaler")
    encoder = _load_artifact(encoder_path, DEFAULT_CONFIG.encoder_path, "risk encoder")
    metadata = _load_metadata(metadata_path)
    matrix, feature_names = _transform_features(raw, scaler, encoder)
    _validate_model_schema(feature_names, metadata)
    probabilities = np.asarray(active_model.predict_proba(matrix), dtype=float)
    predictions = np.asarray(active_model.predict(matrix), dtype=int)
    outputs = [
        _base_response(int(prediction), probability, raw.iloc[index])
        for index, (prediction, probability) in enumerate(zip(predictions, probabilities, strict=True))
    ]
    return outputs[0] if len(outputs) == 1 else outputs


def predict_with_explanation(
    answer: pd.DataFrame | Mapping[str, object],
    *,
    model: object | None = None,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    encoder_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    top_n: int = 4,
) -> dict[str, Any]:
    """Return a single prediction plus signed, model-derived local factors."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")
    raw = model_input_frame(answer)
    if len(raw) != 1:
        raise ValueError("predict_with_explanation accepts exactly one questionnaire response")
    active_model = model if model is not None else load_model(model_path)
    scaler = _load_artifact(scaler_path, DEFAULT_CONFIG.scaler_path, "risk scaler")
    encoder = _load_artifact(encoder_path, DEFAULT_CONFIG.encoder_path, "risk encoder")
    metadata = _load_metadata(metadata_path)
    matrix, feature_names = _transform_features(raw, scaler, encoder)
    _validate_model_schema(feature_names, metadata)
    probabilities = np.asarray(active_model.predict_proba(matrix), dtype=float)[0]
    prediction = int(np.asarray(active_model.predict(matrix), dtype=int)[0])
    response = _base_response(prediction, probabilities, raw.iloc[0])
    impacts = _local_impacts(active_model, matrix, prediction)
    response["top_positive_factors"] = _explain_factors(impacts, feature_names, positive=True, top_n=top_n)
    response["top_negative_factors"] = _explain_factors(impacts, feature_names, positive=False, top_n=top_n)
    response["explanation_method"] = "SHAP" if _can_use_shap(active_model) else "model contribution fallback"
    return response


def _base_response(prediction: int, probabilities: np.ndarray, features: pd.Series) -> dict[str, Any]:
    labels = ("Low", "Medium", "High")
    if prediction < 0 or prediction >= len(labels):
        raise ValueError(f"Model returned unsupported class index: {prediction}")
    risk_profile = labels[prediction]
    persona = select_persona(risk_profile, features)
    probability_payload = {label.lower(): round(float(probabilities[index]), 4) for index, label in enumerate(labels)}
    return {
        "risk_profile": risk_profile,
        "persona": persona.name,
        "persona_details": persona.to_dict(),
        "confidence": round(float(np.max(probabilities)), 4),
        "probabilities": probability_payload,
        "recommendation_summary": recommendation_summary(risk_profile, persona),
    }


def _load_artifact(path_override: str | Path | None, default_path: Path, artifact_name: str) -> object:
    path = Path(path_override) if path_override is not None else default_path
    if not path.exists():
        raise FileNotFoundError(f"{artifact_name.capitalize()} not found: {path}. Run ml.risk.train_risk_model first.")
    return joblib.load(path)


def _load_metadata(metadata_path: str | Path | None) -> dict[str, Any]:
    path = Path(metadata_path) if metadata_path is not None else DEFAULT_CONFIG.metadata_path
    if not path.exists():
        raise FileNotFoundError(f"Risk model metadata not found: {path}. Run ml.risk.train_risk_model first.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("class_labels") != ["Low", "Medium", "High"]:
        raise ValueError("Risk metadata has an unsupported class-label schema")
    return metadata


def _transform_features(raw: pd.DataFrame, scaler: object, encoder: object) -> tuple[np.ndarray, list[str]]:
    numeric_columns = (*QUESTIONNAIRE_NUMERIC_COLUMNS, *DEFAULT_CONFIG.model_numeric_columns[len(QUESTIONNAIRE_NUMERIC_COLUMNS):])
    # ``model_numeric_columns`` provides the exact feature-engineering order.
    numeric = np.asarray(scaler.transform(raw.loc[:, numeric_columns]), dtype=float)
    categorical = np.asarray(encoder.transform(raw.loc[:, QUESTIONNAIRE_CATEGORICAL_COLUMNS]), dtype=float)
    names = [*numeric_columns, *encoder.get_feature_names_out(QUESTIONNAIRE_CATEGORICAL_COLUMNS).tolist()]
    return np.hstack((numeric, categorical)), names


def _validate_model_schema(feature_names: list[str], metadata: Mapping[str, Any]) -> None:
    expected = metadata.get("model_feature_names")
    if expected != feature_names:
        raise ValueError("Risk preprocessing artifacts do not match the trained model feature schema")


def _local_impacts(model: object, matrix: np.ndarray, predicted_class: int) -> np.ndarray:
    """Calculate signed contributions for the predicted class, preferring SHAP."""
    if _can_use_shap(model):
        try:
            import shap

            if hasattr(model, "coef_"):
                values = shap.LinearExplainer(model, np.zeros_like(matrix)).shap_values(matrix)
            else:
                values = shap.TreeExplainer(model).shap_values(matrix)
            tensor = _normalise_shap_values(values, matrix.shape[1])
            return tensor[min(predicted_class, tensor.shape[0] - 1), 0]
        except (ImportError, ValueError, TypeError, AttributeError, IndexError):
            pass
    if hasattr(model, "coef_"):
        coefficients = np.asarray(getattr(model, "coef_"), dtype=float)
        return coefficients[min(predicted_class, coefficients.shape[0] - 1)] * matrix[0]
    importances = np.asarray(getattr(model, "feature_importances_", np.ones(matrix.shape[1])), dtype=float)
    centered = matrix[0] - np.median(matrix[0])
    return importances * centered


def _can_use_shap(model: object) -> bool:
    return (
        hasattr(model, "coef_")
        or hasattr(model, "feature_importances_")
        or model.__class__.__module__.startswith("xgboost")
    )


def _normalise_shap_values(values: Any, feature_count: int) -> np.ndarray:
    if isinstance(values, list):
        tensor = np.stack([np.asarray(value, dtype=float) for value in values], axis=0)
    else:
        tensor = np.asarray(values, dtype=float)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, :, :]
        elif tensor.ndim == 3 and tensor.shape[1] == feature_count:
            tensor = np.moveaxis(tensor, -1, 0)
    if tensor.ndim != 3 or tensor.shape[2] != feature_count:
        raise ValueError("Unsupported SHAP value shape")
    return tensor


def _explain_factors(impacts: np.ndarray, feature_names: list[str], *, positive: bool, top_n: int) -> list[str]:
    pairs = [(name, float(impact)) for name, impact in zip(feature_names, impacts, strict=True) if (impact > 0 if positive else impact < 0)]
    pairs.sort(key=lambda item: item[1], reverse=positive)
    return [_friendly_factor(name) for name, _ in pairs[:top_n]]


def _friendly_factor(feature_name: str) -> str:
    labels = {
        "age": "Life-stage context",
        "monthly_income": "Monthly income level",
        "monthly_savings": "Monthly savings habit",
        "monthly_investment_budget": "Regular investment budget",
        "emergency_fund_months": "Emergency-fund coverage",
        "investment_horizon_years": "Long investment horizon",
        "expected_annual_return_percent": "Return expectations",
        "dependents": "Dependent obligations",
        "risk_tolerance_score": "Comfort with market volatility",
        "financial_preparedness": "Financial preparedness",
        "income_stability_index": "Income stability",
        "investment_experience_index": "Investment experience and knowledge",
        "liquidity_preference_index": "Willingness to commit funds",
        "behavioral_confidence_score": "Financial confidence",
        "loss_recovery_score": "Response to a 20% market loss",
        "investment_readiness_score": "Investment readiness",
        "long_term_orientation_score": "Long-term orientation",
    }
    if feature_name in labels:
        return labels[feature_name]
    if feature_name.startswith("reaction_to_20_percent_loss_"):
        return f"20% loss response: {feature_name.removeprefix('reaction_to_20_percent_loss_')}"
    if feature_name.startswith("previous_investment_experience_"):
        return f"Investment experience: {feature_name.removeprefix('previous_investment_experience_')}"
    if feature_name.startswith("preferred_investment_type_"):
        return f"Investment preference: {feature_name.removeprefix('preferred_investment_type_')}"
    if feature_name.startswith("preferred_liquidity_"):
        return f"Liquidity preference: {feature_name.removeprefix('preferred_liquidity_')}"
    return feature_name.replace("_", " ").capitalize()
