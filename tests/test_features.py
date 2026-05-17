"""Tests for order book feature construction, depth aggregation, and rolling-statistic generation."""

import pandas as pd

from fi2010_ml.features import (
    add_depth_features,
    add_rolling_features,
    add_window_aggregate_features,
    build_basic_features,
)


def test_build_basic_features_adds_mid_spread_and_imbalance():
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0],
            "ask_price_1": [101.0],
            "bid_size_1": [20.0],
            "ask_size_1": [10.0],
        }
    )

    result = build_basic_features(frame)

    assert result.loc[0, "mid_price"] == 100.5
    assert result.loc[0, "spread"] == 1.0
    assert result.loc[0, "order_imbalance_1"] == 1 / 3
    assert result.loc[0, "micro_price"] == (101.0 * 20.0 + 100.0 * 10.0) / 30.0


def test_add_rolling_features_creates_requested_windows():
    frame = pd.DataFrame({"mid_price": [100.0, 101.0, 102.0]})

    result = add_rolling_features(frame, windows=[2])

    assert "return_1" in result.columns
    assert "mid_price_rolling_mean_2" in result.columns
    assert "mid_price_rolling_vol_2" in result.columns


def test_add_depth_features_creates_multi_level_aggregates():
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0, 100.0],
            "ask_price_1": [100.2, 100.2],
            "bid_size_1": [10.0, 12.0],
            "ask_size_1": [8.0, 9.0],
            "bid_size_2": [9.0, 10.0],
            "ask_size_2": [7.0, 6.0],
            "bid_size_3": [8.0, 11.0],
            "ask_size_3": [6.0, 5.0],
        }
    )

    result = add_depth_features(frame, depth_levels=[2, 3])

    assert "bid_depth_2" in result.columns
    assert "ask_depth_2" in result.columns
    assert "depth_imbalance_3" in result.columns
    assert "depth_ratio_3" in result.columns


def test_add_depth_features_clips_requested_level_to_available_depth():
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0],
            "ask_price_1": [100.2],
            "bid_size_1": [10.0],
            "ask_size_1": [8.0],
            "bid_size_2": [9.0],
            "ask_size_2": [7.0],
        }
    )

    result = add_depth_features(frame, depth_levels=[5])

    assert "bid_depth_2" in result.columns
    assert "depth_imbalance_2" in result.columns


def test_add_window_aggregate_features_creates_window_statistics():
    frame = pd.DataFrame(
        {
            "mid_price": [100.0, 100.1, 100.2, 100.0],
            "spread": [0.2, 0.2, 0.3, 0.2],
            "order_imbalance_1": [0.1, 0.2, 0.0, -0.1],
            "bid_depth_5": [50.0, 51.0, 49.0, 48.0],
        }
    )

    result = add_window_aggregate_features(
        frame,
        source_columns=["mid_price", "spread", "order_imbalance_1", "bid_depth_5"],
        windows=[2],
    )

    assert "mid_price_window_mean_2" in result.columns
    assert "mid_price_window_std_2" in result.columns
    assert "mid_price_window_delta_2" in result.columns
    assert "bid_depth_5_window_mean_2" in result.columns
