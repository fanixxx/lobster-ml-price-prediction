"""Tests for supported estimator construction and model fitting entrypoints."""

import pandas as pd

from fi2010_ml.train import build_model, train_model


def test_build_model_supports_random_forest():
    model = build_model("random_forest")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_build_model_supports_xgboost():
    model = build_model("xgboost")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_train_model_fits_requested_estimator():
    features = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0],
        }
    )
    labels = pd.Series(["flat", "flat", "up", "up", "down", "down"])

    model = train_model("random_forest", features, labels)

    assert hasattr(model, "predict")
