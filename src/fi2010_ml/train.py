"""Model-building and training utilities for linear, bagging, and boosting baselines."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def encode_labels_for_model(model_name: str, labels: pd.Series) -> tuple[pd.Series, dict[str, int] | None]:
    """
    Encode labels for a given model.
    """
    if model_name != "xgboost":
        return labels, None

    classes = sorted(labels.dropna().unique().tolist())
    mapping = {label: index for index, label in enumerate(classes)}
    encoded = labels.map(mapping)
    return encoded, mapping


def decode_predictions_for_model(
    model_name: str,
    predictions,
    mapping: dict[str, int] | None,
) -> pd.Series:
    """
    Decode predictions for a given model.
    """
    if model_name != "xgboost" or mapping is None:
        return pd.Series(predictions)

    inverse_mapping = {value: key for key, value in mapping.items()}
    return pd.Series(predictions).map(inverse_mapping)


def build_model(model_name: str):
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=42,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def train_model(
    model_name: str,
    features: pd.DataFrame,
    labels: pd.Series,
):
    """
    Train a model.
    """
    model = build_model(model_name)
    encoded_labels, mapping = encode_labels_for_model(model_name, labels)
    model.fit(features, encoded_labels)
    setattr(model, "_label_mapping", mapping)
    setattr(model, "_model_name", model_name)
    return model


def train_logistic_regression(
    features: pd.DataFrame,
    labels: pd.Series,
) -> Pipeline:
    return train_model("logistic_regression", features, labels)
