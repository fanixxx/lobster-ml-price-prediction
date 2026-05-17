"""Tests for fixed, adaptive, and macro-F1-oriented label generation from mid-price series."""

import pandas as pd

from fi2010_ml.labels import (
    add_adaptive_balanced_horizon_labels,
    add_adaptive_horizon_labels,
    add_fixed_horizon_labels,
    add_quantile_horizon_labels,
)


def test_add_fixed_horizon_labels_creates_three_classes():
    frame = pd.DataFrame({"mid_price": [100.0, 100.3, 100.0, 99.6]})

    result = add_fixed_horizon_labels(frame, horizon=1, threshold=0.002)

    assert list(result["label"].dropna()) == ["up", "down", "down"]


def test_add_adaptive_horizon_labels_adds_threshold_column():
    frame = pd.DataFrame({"mid_price": [100.0, 100.1, 100.2, 100.0, 99.8]})

    result = add_adaptive_horizon_labels(
        frame,
        horizon=1,
        volatility_window=2,
        volatility_scale=1.0,
    )

    assert "adaptive_threshold" in result.columns
    assert "future_return" in result.columns


def test_add_quantile_horizon_labels_creates_balanced_extremes():
    frame = pd.DataFrame({"mid_price": [100.0, 100.2, 100.4, 100.6, 100.5, 100.3, 100.1]})

    result = add_quantile_horizon_labels(
        frame,
        horizon=1,
        lower_quantile=0.25,
        upper_quantile=0.75,
    )

    labels = result["label"].dropna().tolist()
    assert "down" in labels
    assert "up" in labels
    assert "future_return" in result.columns
    assert "lower_threshold" in result.columns
    assert "upper_threshold" in result.columns


def test_add_adaptive_balanced_horizon_labels_exposes_threshold_strength():
    frame = pd.DataFrame({"mid_price": [100.0, 100.1, 100.3, 100.0, 99.7, 99.9, 100.2]})

    result = add_adaptive_balanced_horizon_labels(
        frame,
        horizon=1,
        volatility_window=3,
        base_scale=1.0,
        balance_scale=0.5,
    )

    assert "adaptive_threshold" in result.columns
    assert "effective_scale" in result.columns
    assert set(result["label"].dropna().unique()) <= {"up", "down", "flat"}
