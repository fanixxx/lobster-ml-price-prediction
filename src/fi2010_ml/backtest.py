"""Simplified backtest utilities for mapping classification predictions into strategy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(frame: pd.DataFrame, cost_bps: float = 1.0) -> dict[str, float]:
    """Run a lightweight directional backtest from predicted classes and future returns."""
    if frame.empty:
        return {
            "cumulative_return": 0.0,
            "average_return": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
        }

    positions = frame["prediction"].map({"up": 1, "down": -1, "flat": 0}).fillna(0)
    gross_returns = positions * frame["future_return"].fillna(0.0)
    trade_flags = positions.ne(0).astype(int)
    cost = trade_flags * (cost_bps / 10000.0)
    strategy_returns = gross_returns - cost

    equity_curve = (1.0 + strategy_returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0

    trade_count = int(trade_flags.sum())
    trade_returns = strategy_returns.loc[trade_flags.eq(1)]
    win_rate = float((trade_returns > 0).mean()) if trade_count else 0.0
    average_return = float(trade_returns.mean()) if trade_count else 0.0
    sharpe_like = 0.0
    if strategy_returns.std(ddof=0) > 0:
        sharpe_like = float(
            np.sqrt(len(strategy_returns)) * strategy_returns.mean() / strategy_returns.std(ddof=0)
        )

    return {
        "cumulative_return": float(equity_curve.iloc[-1] - 1.0),
        "average_return": average_return,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "sharpe_like": sharpe_like,
        "max_drawdown": float(drawdown.min()),
    }
