"""Tests for experiment-grid execution across models, label modes, horizons, and feature variants."""

import pandas as pd

from fi2010_ml.pipeline import run_experiment_grid


def test_run_experiment_grid_returns_all_model_label_horizon_combinations():
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0, 100.1, 100.2, 100.3, 100.1, 100.0, 99.9, 100.0, 100.2, 100.4] * 6,
            "ask_price_1": [100.2, 100.3, 100.4, 100.5, 100.3, 100.2, 100.1, 100.2, 100.4, 100.6] * 6,
            "bid_size_1": [10, 11, 12, 13, 12, 11, 10, 9, 10, 11] * 6,
            "ask_size_1": [9, 10, 11, 12, 11, 10, 9, 8, 9, 10] * 6,
            "bid_size_2": [9, 10, 11, 12, 11, 10, 9, 8, 9, 10] * 6,
            "ask_size_2": [8, 9, 10, 11, 10, 9, 8, 7, 8, 9] * 6,
        }
    )

    results = run_experiment_grid(
        frame=frame,
        model_names=["logistic_regression", "random_forest"],
        label_modes=["fixed", "adaptive", "quantile"],
        horizons=[1, 2],
        fixed_threshold=0.0005,
        volatility_window=3,
        volatility_scale=1.0,
        feature_variants=["baseline", "depth_plus_window"],
        depth_levels=[2],
        aggregate_windows=[2],
        quantile_bounds=(0.25, 0.75),
    )

    assert len(results) == 24
    assert set(results["model"]) == {"logistic_regression", "random_forest"}
    assert set(results["label_mode"]) == {"fixed", "adaptive", "quantile"}
    assert set(results["horizon"]) == {1, 2}
    assert set(results["feature_variant"]) == {"baseline", "depth_plus_window"}
    assert {"accuracy", "macro_f1", "train_size", "test_size"} <= set(results.columns)


def test_run_experiment_grid_skips_empty_label_splits():
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0] * 20,
            "ask_price_1": [100.1] * 20,
            "bid_size_1": [10.0] * 20,
            "ask_size_1": [10.0] * 20,
        }
    )

    results = run_experiment_grid(
        frame=frame,
        model_names=["logistic_regression"],
        label_modes=["fixed"],
        horizons=[1],
        fixed_threshold=1.0,
        volatility_window=3,
        volatility_scale=1.0,
        feature_variants=["baseline"],
        depth_levels=[2],
        aggregate_windows=[2],
        quantile_bounds=(0.25, 0.75),
    )

    assert len(results) == 1
    assert results.loc[0, "status"] == "skipped"


def test_run_experiment_grid_records_tuning_parameter_columns():
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0, 100.1, 100.2, 100.3, 100.1, 100.0, 99.9, 100.0, 100.2, 100.4] * 6,
            "ask_price_1": [100.2, 100.3, 100.4, 100.5, 100.3, 100.2, 100.1, 100.2, 100.4, 100.6] * 6,
            "bid_size_1": [10, 11, 12, 13, 12, 11, 10, 9, 10, 11] * 6,
            "ask_size_1": [9, 10, 11, 12, 11, 10, 9, 8, 9, 10] * 6,
            "bid_size_2": [9, 10, 11, 12, 11, 10, 9, 8, 9, 10] * 6,
            "ask_size_2": [8, 9, 10, 11, 10, 9, 8, 7, 8, 9] * 6,
        }
    )

    results = run_experiment_grid(
        frame=frame,
        model_names=["random_forest"],
        label_modes=["quantile"],
        horizons=[1],
        fixed_threshold=0.0005,
        volatility_window=3,
        volatility_scale=1.0,
        feature_variants=["depth_plus_window"],
        depth_levels=[2, 3],
        aggregate_windows=[2, 3],
        quantile_bounds=(0.2, 0.8),
        rolling_windows=[2, 3],
    )

    assert "depth_levels_key" in results.columns
    assert "aggregate_windows_key" in results.columns
    assert "rolling_windows_key" in results.columns
    assert "quantile_bounds_key" in results.columns
    assert results.loc[0, "quantile_bounds_key"] == "0.2|0.8"
