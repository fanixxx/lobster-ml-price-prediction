"""Tests for model-specific label encoding needed by XGBoost training."""

import pandas as pd

from fi2010_ml.train import encode_labels_for_model


def test_encode_labels_for_xgboost_returns_numeric_series_and_mapping():
    labels = pd.Series(["down", "flat", "up", "flat"])

    encoded, mapping = encode_labels_for_model("xgboost", labels)

    assert list(encoded) == [0, 1, 2, 1]
    assert mapping == {"down": 0, "flat": 1, "up": 2}
