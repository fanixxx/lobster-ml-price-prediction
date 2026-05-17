"""Feature engineering functions for basic signals, depth aggregation, and window-based statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _available_book_levels(frame: pd.DataFrame) -> int:
    levels = []
    for column in frame.columns:
        if column.startswith("bid_size_"):
            try:
                levels.append(int(column.split("_")[-1]))
            except ValueError:
                continue
    return max(levels) if levels else 1


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.Series(
        np.where(denominator == 0, default, numerator / denominator),
        index=denominator.index,
    )


def build_basic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build basic level-1 price and imbalance features."""
    result = frame.copy()
    result["mid_price"] = (result["bid_price_1"] + result["ask_price_1"]) / 2
    result["spread"] = result["ask_price_1"] - result["bid_price_1"]

    total_size = result["bid_size_1"] + result["ask_size_1"]
    result["order_imbalance_1"] = _safe_ratio(
        result["bid_size_1"] - result["ask_size_1"],
        total_size,
    )
    result["micro_price"] = pd.Series(
        np.where(
            total_size == 0,
            result["mid_price"],
            (
                result["ask_price_1"] * result["bid_size_1"]
                + result["bid_price_1"] * result["ask_size_1"]
            )
            / total_size,
        ),
        index=result.index,
    )
    return result


def add_rolling_features(
    frame: pd.DataFrame,
    price_column: str = "mid_price",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and volatility features for a price column."""
    result = frame.copy()
    windows = windows or [10, 30]
    returns = result[price_column].pct_change().fillna(0.0)
    result["return_1"] = returns

    for window in windows:
        result[f"{price_column}_rolling_mean_{window}"] = (
            result[price_column].rolling(window=window, min_periods=1).mean()
        )
        result[f"{price_column}_rolling_vol_{window}"] = (
            returns.rolling(window=window, min_periods=1).std().fillna(0.0)
        )

    return result


def add_depth_features(
    frame: pd.DataFrame,
    depth_levels: list[int] | None = None,
) -> pd.DataFrame:
    """Add cumulative multi-level depth summaries and imbalance signals."""
    result = frame.copy()
    depth_levels = depth_levels or [5, 10]
    available_levels = _available_book_levels(result)
    additions: dict[str, pd.Series] = {}

    for requested_level in depth_levels:
        level = min(requested_level, available_levels)
        bid_columns = [f"bid_size_{index}" for index in range(1, level + 1) if f"bid_size_{index}" in result]
        ask_columns = [f"ask_size_{index}" for index in range(1, level + 1) if f"ask_size_{index}" in result]
        if not bid_columns or not ask_columns:
            continue

        bid_depth = result[bid_columns].sum(axis=1)
        ask_depth = result[ask_columns].sum(axis=1)
        total_depth = bid_depth + ask_depth

        additions[f"bid_depth_{level}"] = bid_depth
        additions[f"ask_depth_{level}"] = ask_depth
        additions[f"depth_diff_{level}"] = bid_depth - ask_depth
        additions[f"depth_ratio_{level}"] = _safe_ratio(bid_depth, ask_depth, default=1.0)
        additions[f"depth_imbalance_{level}"] = _safe_ratio(
            bid_depth - ask_depth,
            total_depth,
        )

    near_level = min(5, available_levels)
    far_level = available_levels
    if near_level >= 1 and far_level > near_level:
        near_bid = result[[f"bid_size_{index}" for index in range(1, near_level + 1) if f"bid_size_{index}" in result]].sum(axis=1)
        near_ask = result[[f"ask_size_{index}" for index in range(1, near_level + 1) if f"ask_size_{index}" in result]].sum(axis=1)
        far_bid = result[[f"bid_size_{index}" for index in range(near_level + 1, far_level + 1) if f"bid_size_{index}" in result]].sum(axis=1)
        far_ask = result[[f"ask_size_{index}" for index in range(near_level + 1, far_level + 1) if f"ask_size_{index}" in result]].sum(axis=1)
        additions["bid_near_far_ratio"] = _safe_ratio(near_bid, far_bid, default=1.0)
        additions["ask_near_far_ratio"] = _safe_ratio(near_ask, far_ask, default=1.0)
        additions["queue_pressure"] = _safe_ratio(near_bid - near_ask, near_bid + near_ask)

    if additions:
        result = pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)

    return result


def add_window_aggregate_features(
    frame: pd.DataFrame,
    source_columns: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add short-window aggregate statistics for selected source columns."""
    result = frame.copy()
    windows = windows or [5, 10, 30]
    additions: dict[str, pd.Series] = {}

    for column in source_columns:
        if column not in result.columns:
            continue

        for window in windows:
            rolling = result[column].rolling(window=window, min_periods=1)
            additions[f"{column}_window_mean_{window}"] = rolling.mean()
            additions[f"{column}_window_std_{window}"] = rolling.std().fillna(0.0)
            additions[f"{column}_window_min_{window}"] = rolling.min()
            additions[f"{column}_window_max_{window}"] = rolling.max()
            additions[f"{column}_window_delta_{window}"] = result[column] - result[column].shift(window - 1)

    if "mid_price" in result.columns:
        returns = result["mid_price"].pct_change().fillna(0.0)
        for window in windows:
            additions[f"mid_price_window_return_{window}"] = (
                (1 + returns).rolling(window=window, min_periods=1).apply(np.prod, raw=True) - 1
            )

    if additions:
        result = pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)

    return result


def build_feature_variant(
    frame: pd.DataFrame,
    feature_variant: str,
    rolling_windows: list[int] | None = None,
    depth_levels: list[int] | None = None,
    aggregate_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Build either baseline or upgraded depth-plus-window feature sets."""
    result = build_basic_features(frame)
    result = add_rolling_features(result, windows=rolling_windows)

    if feature_variant == "baseline":
        return result

    if feature_variant == "depth_plus_window":
        result = add_depth_features(result, depth_levels=depth_levels)
        source_columns = [
            "mid_price",
            "spread",
            "order_imbalance_1",
            "micro_price",
        ]
        source_columns.extend(
            column
            for column in result.columns
            if column.startswith("bid_depth_")
            or column.startswith("ask_depth_")
            or column.startswith("depth_imbalance_")
            or column in {"queue_pressure", "bid_near_far_ratio", "ask_near_far_ratio"}
        )
        result = add_window_aggregate_features(
            result,
            source_columns=source_columns,
            windows=aggregate_windows,
        )
        return result

    raise ValueError(f"Unsupported feature variant: {feature_variant}")
