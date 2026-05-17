"""Tests for trading-hour filtering and snapshot resampling behavior."""

import pandas as pd

from fi2010_ml.preprocess import filter_trading_hours, resample_snapshots


def test_filter_trading_hours_keeps_in_range_rows():
    frame = pd.DataFrame({"seconds": [34100.0, 34200.0, 35000.0, 57700.0]})

    result = filter_trading_hours(
        frame,
        time_column="seconds",
        start_seconds=34200.0,
        end_seconds=57600.0,
    )

    assert list(result["seconds"]) == [34200.0, 35000.0]


def test_resample_snapshots_forward_fills_seconds():
    frame = pd.DataFrame(
        {
            "seconds": [34200.0, 34201.4, 34203.0],
            "mid_price": [100.0, 100.2, 100.1],
        }
    )

    result = resample_snapshots(frame, time_column="seconds", frequency="1s")

    assert list(result["mid_price"]) == [100.0, 100.2, 100.2, 100.1]
