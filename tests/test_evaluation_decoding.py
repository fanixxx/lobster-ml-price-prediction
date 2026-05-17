"""Tests for decoding numeric model outputs back into string labels for metrics."""

import pandas as pd

from fi2010_ml.evaluate import normalize_predictions_for_metrics


def test_normalize_predictions_for_metrics_decodes_xgboost_outputs():
    labels = pd.Series(["down", "flat", "up"])
    predictions = [0, 1, 2]
    mapping = {"down": 0, "flat": 1, "up": 2}

    normalized = normalize_predictions_for_metrics("xgboost", predictions, labels, mapping)

    assert list(normalized) == ["down", "flat", "up"]
