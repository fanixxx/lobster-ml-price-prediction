"""Evaluation helpers for model predictions, including label normalization and metric computation."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from fi2010_ml.train import decode_predictions_for_model


def normalize_predictions_for_metrics(model_name, predictions, labels, mapping):
    """
    Decode predictions for a given model.
    This is necessary for models like XGBoost that require label encoding, to ensure that metrics are computed on the original label space."""
    decoded = decode_predictions_for_model(model_name, predictions, mapping)
    if isinstance(decoded, pd.Series):
        return decoded
    return pd.Series(decoded, index=labels.index)


def evaluate_classifier(model, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    model_name = getattr(model, "_model_name", "logistic_regression")
    mapping = getattr(model, "_label_mapping", None)
    normalized = normalize_predictions_for_metrics(model_name, predictions, labels, mapping)
    return {
        "accuracy": accuracy_score(labels, normalized),
        "macro_f1": f1_score(labels, normalized, average="macro"),
    }


def build_classification_report(model, features: pd.DataFrame, labels: pd.Series) -> dict:
    predictions = model.predict(features)
    model_name = getattr(model, "_model_name", "logistic_regression")
    mapping = getattr(model, "_label_mapping", None)
    normalized = normalize_predictions_for_metrics(model_name, predictions, labels, mapping)
    return classification_report(labels, normalized, output_dict=True, zero_division=0)
