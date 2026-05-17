"""Tests for simplified strategy backtesting on FI-2010 predictions."""

from __future__ import annotations

import pandas as pd

from fi2010_ml.backtest import run_backtest


def test_run_backtest_returns_basic_strategy_metrics() -> None:
    frame = pd.DataFrame(
        {
            "mid_price": [100.0, 101.0, 100.0, 102.0],
            "future_return": [0.01, -0.00990099, 0.02, 0.0],
            "prediction": ["up", "down", "flat", "up"],
        }
    )

    metrics = run_backtest(frame, cost_bps=1.0)

    assert "cumulative_return" in metrics
    assert "average_return" in metrics
    assert "win_rate" in metrics
    assert "trade_count" in metrics
    assert metrics["trade_count"] == 3

