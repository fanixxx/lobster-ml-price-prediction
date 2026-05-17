"""Train/test workflow helpers for benchmark-style FI-2010 evaluation and backtesting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fi2010_ml.backtest import run_backtest
from fi2010_ml.evaluate import evaluate_classifier, normalize_predictions_for_metrics
from fi2010_ml.features import build_feature_variant
from fi2010_ml.fi2010_adapter import convert_fi2010_file
from fi2010_ml.labels import (
    add_adaptive_balanced_horizon_labels,
    add_adaptive_horizon_labels,
    add_fixed_horizon_labels,
    add_quantile_horizon_labels,
)
from fi2010_ml.pipeline import _select_feature_columns, prepare_lobster_dataset
from fi2010_ml.train import train_model


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
        return add_fixed_horizon_labels(featured, horizon=horizon, threshold=fixed_threshold)
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


def _prepare_featured_dataset(
    input_path: str | Path,
    converted_dir: str | Path,
    levels: int,
    snapshot_frequency: str,
    trading_start_seconds: float,
    trading_end_seconds: float,
    feature_variant: str,
    rolling_windows: list[int] | None,
    depth_levels: list[int] | None,
    aggregate_windows: list[int] | None,
) -> pd.DataFrame:
    message_path, orderbook_path = convert_fi2010_file(input_path, converted_dir)
    dataset = prepare_lobster_dataset(
        message_path=message_path,
        orderbook_path=orderbook_path,
        levels=levels,
        snapshot_frequency=snapshot_frequency,
        trading_start_seconds=trading_start_seconds,
        trading_end_seconds=trading_end_seconds,
    )
    return build_feature_variant(
        dataset,
        feature_variant=feature_variant,
        rolling_windows=rolling_windows,
        depth_levels=depth_levels,
        aggregate_windows=aggregate_windows,
    )


def run_train_test_workflow(
    train_input_path: str | Path,
    test_input_path: str | Path,
    converted_dir: str | Path,
    model_name: str,
    label_mode: str,
    horizon: int,
    fixed_threshold: float,
    volatility_window: int,
    volatility_scale: float,
    feature_variant: str,
    levels: int,
    depth_levels: list[int] | None,
    aggregate_windows: list[int] | None,
    rolling_windows: list[int] | None,
    snapshot_frequency: str,
    trading_start_seconds: float,
    trading_end_seconds: float,
    quantile_bounds: tuple[float, float],
    cost_bps: float = 1.0,
) -> dict[str, float | str | int]:
    """Run explicit train/test evaluation and simplified backtest for one FI-2010 split."""
    converted_dir = Path(converted_dir)
    train_featured = _prepare_featured_dataset(
        input_path=train_input_path,
        converted_dir=converted_dir / "train",
        levels=levels,
        snapshot_frequency=snapshot_frequency,
        trading_start_seconds=trading_start_seconds,
        trading_end_seconds=trading_end_seconds,
        feature_variant=feature_variant,
        rolling_windows=rolling_windows,
        depth_levels=depth_levels,
        aggregate_windows=aggregate_windows,
    )
    test_featured = _prepare_featured_dataset(
        input_path=test_input_path,
        converted_dir=converted_dir / "test",
        levels=levels,
        snapshot_frequency=snapshot_frequency,
        trading_start_seconds=trading_start_seconds,
        trading_end_seconds=trading_end_seconds,
        feature_variant=feature_variant,
        rolling_windows=rolling_windows,
        depth_levels=depth_levels,
        aggregate_windows=aggregate_windows,
    )

    train_labeled = _apply_label_mode(
        featured=train_featured,
        label_mode=label_mode,
        horizon=horizon,
        fixed_threshold=fixed_threshold,
        volatility_window=volatility_window,
        volatility_scale=volatility_scale,
        quantile_bounds=quantile_bounds,
    )
    test_labeled = _apply_label_mode(
        featured=test_featured,
        label_mode=label_mode,
        horizon=horizon,
        fixed_threshold=fixed_threshold,
        volatility_window=volatility_window,
        volatility_scale=volatility_scale,
        quantile_bounds=quantile_bounds,
    )

    train_feature_columns = _select_feature_columns(train_labeled)
    test_feature_columns = _select_feature_columns(test_labeled)
    shared_columns = sorted(set(train_feature_columns).intersection(test_feature_columns))
    train_feature_frame = train_labeled[shared_columns]
    test_feature_frame = test_labeled[shared_columns]
    valid_feature_columns = [
        column
        for column in shared_columns
        if not train_feature_frame[column].isna().all() and not test_feature_frame[column].isna().all()
    ]

    train_usable = train_labeled[valid_feature_columns + ["label"]].dropna().reset_index(drop=True)
    extra_columns = [
        column
        for column in ["label", "future_return", "mid_price"]
        if column not in valid_feature_columns
    ]
    aligned_test = test_labeled.reindex(columns=valid_feature_columns + extra_columns)
    test_usable = aligned_test.dropna().reset_index(drop=True)

    train_labels = train_usable["label"].dropna()
    test_labels = test_usable["label"].dropna()
    if train_usable.empty or test_usable.empty or train_labels.nunique() < 2 or test_labels.nunique() < 1:
        raise ValueError("Train/test split does not contain enough labeled samples for evaluation.")

    train_features = train_usable[valid_feature_columns].to_numpy()
    test_features = test_usable[valid_feature_columns].to_numpy()
    model = train_model(model_name, train_features, train_usable["label"])
    metrics = evaluate_classifier(model, test_features, test_usable["label"])

    predictions = model.predict(test_features)
    normalized = normalize_predictions_for_metrics(
        getattr(model, "_model_name", model_name),
        predictions,
        test_usable["label"],
        getattr(model, "_label_mapping", None),
    )
    backtest_input = pd.DataFrame(
        {
            "mid_price": test_labeled.loc[test_usable.index, "mid_price"].reset_index(drop=True),
            "future_return": test_usable["future_return"],
            "prediction": normalized,
        }
    )
    backtest_metrics = run_backtest(backtest_input, cost_bps=cost_bps)

    result: dict[str, float | str | int] = {
        "model": model_name,
        "label_mode": label_mode,
        "feature_variant": feature_variant,
        "horizon": horizon,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "train_size": len(train_usable),
        "test_size": len(test_usable),
        "status": "ok",
    }
    result.update(backtest_metrics)
    return result
