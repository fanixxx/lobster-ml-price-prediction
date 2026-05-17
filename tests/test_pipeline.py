"""Tests for time-order splitting, baseline training, and evaluation helpers."""

import pandas as pd

from fi2010_ml.evaluate import evaluate_classifier
from fi2010_ml.splits import time_order_split
from fi2010_ml.train import train_logistic_regression


def test_time_order_split_preserves_order():
    frame = pd.DataFrame({"value": list(range(10))})

    train, valid, test = time_order_split(frame, train_ratio=0.6, valid_ratio=0.2)

    assert list(train["value"]) == [0, 1, 2, 3, 4, 5]
    assert list(valid["value"]) == [6, 7]
    assert list(test["value"]) == [8, 9]


def test_train_logistic_regression_returns_fitted_model():
    features = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0],
        }
    )
    labels = pd.Series(["flat", "flat", "up", "up", "down", "down"])

    model = train_logistic_regression(features, labels)

    assert hasattr(model, "predict")


def test_evaluate_classifier_returns_metrics():
    features = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0],
        }
    )
    labels = pd.Series(["flat", "flat", "up", "up", "down", "down"])
    model = train_logistic_regression(features, labels)

    metrics = evaluate_classifier(model, features, labels)

    assert set(metrics) == {"accuracy", "macro_f1"}
