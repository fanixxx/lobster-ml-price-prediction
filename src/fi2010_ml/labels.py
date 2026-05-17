"""Label-generation functions for fixed, adaptive, and macro-F1-oriented price direction tasks."""

from __future__ import annotations

import pandas as pd


def _future_return(frame: pd.DataFrame, horizon: int, price_column: str) -> pd.Series:
    future_price = frame[price_column].shift(-horizon)
    return (future_price - frame[price_column]) / frame[price_column]


def add_fixed_horizon_labels(
    frame: pd.DataFrame,
    horizon: int,
    threshold: float,
    price_column: str = "mid_price",
) -> pd.DataFrame:
    """Add fixed-threshold price direction labels."""
    result = frame.copy()
    future_return = _future_return(result, horizon=horizon, price_column=price_column)

    result["label"] = "flat"
    result.loc[future_return > threshold, "label"] = "up"
    result.loc[future_return < -threshold, "label"] = "down"
    result.loc[future_return.isna(), "label"] = pd.NA
    result["future_return"] = future_return
    return result


def add_adaptive_horizon_labels(
    frame: pd.DataFrame,
    horizon: int,
    volatility_window: int,
    volatility_scale: float,
    price_column: str = "mid_price",
) -> pd.DataFrame:
    """Add volatility-adaptive price direction labels."""
    result = frame.copy()
    future_return = _future_return(result, horizon=horizon, price_column=price_column)
    rolling_vol = (
        result[price_column]
        .pct_change()
        .rolling(window=volatility_window, min_periods=1)
        .std()
        .fillna(0.0)
    )
    adaptive_threshold = rolling_vol * volatility_scale

    result["label"] = "flat"
    result.loc[future_return > adaptive_threshold, "label"] = "up"
    result.loc[future_return < -adaptive_threshold, "label"] = "down"
    result.loc[future_return.isna(), "label"] = pd.NA
    result["future_return"] = future_return
    result["adaptive_threshold"] = adaptive_threshold
    return result


def add_quantile_horizon_labels(
    frame: pd.DataFrame,
    horizon: int,
    lower_quantile: float,
    upper_quantile: float,
    price_column: str = "mid_price",
) -> pd.DataFrame:
    """Add quantile-based labels to improve class balance."""
    result = frame.copy()
    future_return = _future_return(result, horizon=horizon, price_column=price_column)
    valid_future_return = future_return.dropna()

    lower_threshold = valid_future_return.quantile(lower_quantile)
    upper_threshold = valid_future_return.quantile(upper_quantile)

    result["label"] = "flat"
    result.loc[future_return < lower_threshold, "label"] = "down"
    result.loc[future_return > upper_threshold, "label"] = "up"
    result.loc[future_return.isna(), "label"] = pd.NA
    result["future_return"] = future_return
    result["lower_threshold"] = lower_threshold
    result["upper_threshold"] = upper_threshold
    return result


def add_adaptive_balanced_horizon_labels(
    frame: pd.DataFrame,
    horizon: int,
    volatility_window: int,
    base_scale: float,
    balance_scale: float,
    price_column: str = "mid_price",
) -> pd.DataFrame:
    """Add volatility-adaptive labels with a configurable balance-strength adjustment."""
    result = frame.copy()
    future_return = _future_return(result, horizon=horizon, price_column=price_column)
    rolling_vol = (
        result[price_column]
        .pct_change()
        .rolling(window=volatility_window, min_periods=1)
        .std()
        .fillna(0.0)
    )
    effective_scale = base_scale * balance_scale
    adaptive_threshold = rolling_vol * effective_scale

    result["label"] = "flat"
    result.loc[future_return > adaptive_threshold, "label"] = "up"
    result.loc[future_return < -adaptive_threshold, "label"] = "down"
    result.loc[future_return.isna(), "label"] = pd.NA
    result["future_return"] = future_return
    result["adaptive_threshold"] = adaptive_threshold
    result["effective_scale"] = effective_scale
    return result
