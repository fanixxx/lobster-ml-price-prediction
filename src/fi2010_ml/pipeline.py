"""End-to-end pipeline helpers that connect loading, preprocessing, features, labeling, training, and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fi2010_ml.data_loader import load_lobster_day
from fi2010_ml.evaluate import evaluate_classifier
from fi2010_ml.features import build_feature_variant
from fi2010_ml.labels import (
    add_adaptive_balanced_horizon_labels,
    add_adaptive_horizon_labels,
    add_fixed_horizon_labels,
    add_quantile_horizon_labels,
)
from fi2010_ml.preprocess import filter_trading_hours, resample_snapshots
from fi2010_ml.splits import time_order_split
from fi2010_ml.train import train_logistic_regression, train_model


def run_baseline_pipeline(
    frame: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> dict[str, float]:
    """Run a fixed-threshold logistic-regression baseline pipeline."""
    featured = build_feature_variant(frame, feature_variant="baseline")
    labeled = add_fixed_horizon_labels(featured, horizon=horizon, threshold=threshold)
    labeled = labeled.dropna(subset=["label"]).reset_index(drop=True)

    model_columns = [
        "mid_price",
        "spread",
        "order_imbalance_1",
        "micro_price",
        "return_1",
        "mid_price_rolling_mean_10",
        "mid_price_rolling_vol_10",
        "mid_price_rolling_mean_30",
        "mid_price_rolling_vol_30",
    ]

    usable = labeled[model_columns + ["label"]].dropna().reset_index(drop=True)
    train, _, test = time_order_split(usable)
    model = train_logistic_regression(train[model_columns], train["label"])
    return evaluate_classifier(model, test[model_columns], test["label"])


def prepare_lobster_dataset(
    message_path: str | Path,
    orderbook_path: str | Path,
    levels: int,
    snapshot_frequency: str,
    trading_start_seconds: float,
    trading_end_seconds: float,
) -> pd.DataFrame:
    """Load and preprocess a single LOBSTER day into a snapshot-level feature-ready frame."""
    frame = load_lobster_day(
        message_path=message_path,
        orderbook_path=orderbook_path,
        levels=levels,
    )
    frame = filter_trading_hours(
        frame,
        time_column="time",
        start_seconds=trading_start_seconds,
        end_seconds=trading_end_seconds,
    )
    frame = resample_snapshots(frame, time_column="time", frequency=snapshot_frequency)
    return frame


def _select_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("bid_price_", "ask_price_", "bid_size_", "ask_size_")
    excluded_columns = {
        "time",
        "event_type",
        "order_id",
        "size",
        "price",
        "direction",
        "label",
        "future_return",
        "adaptive_threshold",
        "lower_threshold",
        "upper_threshold",
        "effective_scale",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded_columns and not column.startswith(excluded_prefixes)
    ]


def _apply_label_mode(
    featured: pd.DataFrame,
    label_mode: str,
    horizon: int,
    fixed_threshold: float,
    volatility_window: int,
    volatility_scale: float,
    quantile_bounds: tuple[float, float],
) -> pd.DataFrame:
    if label_mode == "fixed":
        return add_fixed_horizon_labels(
            featured,
            horizon=horizon,
            threshold=fixed_threshold,
        )
    if label_mode == "adaptive":
        return add_adaptive_horizon_labels(
            featured,
            horizon=horizon,
            volatility_window=volatility_window,
            volatility_scale=volatility_scale,
        )
    if label_mode == "adaptive_balanced":
        return add_adaptive_balanced_horizon_labels(
            featured,
            horizon=horizon,
            volatility_window=volatility_window,
            base_scale=volatility_scale,
            balance_scale=0.5,
        )
    if label_mode == "quantile":
        return add_quantile_horizon_labels(
            featured,
            horizon=horizon,
            lower_quantile=quantile_bounds[0],
            upper_quantile=quantile_bounds[1],
        )
    raise ValueError(f"Unsupported label mode: {label_mode}")


def run_experiment_grid(
    frame: pd.DataFrame,
    model_names: list[str],
    label_modes: list[str],
    horizons: list[int],
    fixed_threshold: float,
    volatility_window: int,
    volatility_scale: float,
    feature_variants: list[str] | None = None,
    depth_levels: list[int] | None = None,
    aggregate_windows: list[int] | None = None,
    quantile_bounds: tuple[float, float] = (0.25, 0.75),
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Run a grid of experiments across feature variants, label modes, horizons, and models."""
    results: list[dict] = []
    feature_variants = feature_variants or ["baseline"]
    depth_levels_key = "|".join(str(level) for level in (depth_levels or []))
    aggregate_windows_key = "|".join(str(window) for window in (aggregate_windows or []))
    rolling_windows_key = "|".join(str(window) for window in (rolling_windows or []))
    quantile_bounds_key = f"{quantile_bounds[0]}|{quantile_bounds[1]}"

    for feature_variant in feature_variants:
        featured = build_feature_variant(
            frame,
            feature_variant=feature_variant,
            rolling_windows=rolling_windows,
            depth_levels=depth_levels,
            aggregate_windows=aggregate_windows,
        )
        feature_columns = _select_feature_columns(featured)

        for label_mode in label_modes:
            for horizon in horizons:
                labeled = _apply_label_mode(
                    featured=featured,
                    label_mode=label_mode,
                    horizon=horizon,
                    fixed_threshold=fixed_threshold,
                    volatility_window=volatility_window,
                    volatility_scale=volatility_scale,
                    quantile_bounds=quantile_bounds,
                )

                usable = labeled[feature_columns + ["label"]].dropna().reset_index(drop=True)
                train, _, test = time_order_split(usable)
                train_labels = train["label"].dropna()
                test_labels = test["label"].dropna()

                if train.empty or test.empty or train_labels.nunique() < 2 or test_labels.nunique() < 1:
                    for model_name in model_names:
                        results.append(
                            {
                                "model": model_name,
                                "label_mode": label_mode,
                                "feature_variant": feature_variant,
                                "depth_levels_key": depth_levels_key,
                                "aggregate_windows_key": aggregate_windows_key,
                                "rolling_windows_key": rolling_windows_key,
                                "quantile_bounds_key": quantile_bounds_key,
                                "horizon": horizon,
                                "accuracy": None,
                                "macro_f1": None,
                                "train_size": len(train),
                                "test_size": len(test),
                                "status": "skipped",
                            }
                        )
                    continue

                for model_name in model_names:
                    model = train_model(model_name, train[feature_columns], train["label"])
                    metrics = evaluate_classifier(model, test[feature_columns], test["label"])
                    results.append(
                        {
                            "model": model_name,
                            "label_mode": label_mode,
                            "feature_variant": feature_variant,
                            "depth_levels_key": depth_levels_key,
                            "aggregate_windows_key": aggregate_windows_key,
                            "rolling_windows_key": rolling_windows_key,
                            "quantile_bounds_key": quantile_bounds_key,
                            "horizon": horizon,
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "train_size": len(train),
                            "test_size": len(test),
                            "status": "ok",
                        }
                    )

    return pd.DataFrame(results)
