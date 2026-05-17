"""Tests for explicit FI-2010 train/test workflow execution."""

from __future__ import annotations

import pandas as pd

from fi2010_ml.workflow import run_train_test_workflow


def test_run_train_test_workflow_returns_metrics_and_backtest(tmp_path) -> None:
    train_path = tmp_path / "Train_Dst_NoAuction_DecPre_CF_1.txt"
    test_path = tmp_path / "Test_Dst_NoAuction_DecPre_CF_1.txt"

    train_frame = pd.DataFrame(
        [
            [
                (row + 1) * 0.01
                + col * 0.001
                + ((col % 7) - 3) * 0.003
                + ((-1) ** col) * 0.001
                for col in range(120)
            ]
            for row in range(45)
        ]
    )
    test_frame = pd.DataFrame(
        [
            [
                (row + 1) * 0.02
                + col * 0.001
                + ((col % 5) - 2) * 0.004
                + ((-1) ** (col + 1)) * 0.001
                for col in range(80)
            ]
            for row in range(45)
        ]
    )
    train_frame.to_csv(train_path, sep=" ", header=False, index=False)
    test_frame.to_csv(test_path, sep=" ", header=False, index=False)

    result = run_train_test_workflow(
        train_input_path=train_path,
        test_input_path=test_path,
        converted_dir=tmp_path / "converted",
        model_name="random_forest",
        label_mode="adaptive",
        horizon=5,
        fixed_threshold=0.0005,
        volatility_window=10,
        volatility_scale=0.5,
        feature_variant="depth_plus_window",
        levels=10,
        depth_levels=[5, 10],
        aggregate_windows=[5],
        rolling_windows=[5, 10],
        snapshot_frequency="1s",
        trading_start_seconds=0.0,
        trading_end_seconds=1000000000000.0,
        quantile_bounds=(0.2, 0.8),
        cost_bps=1.0,
    )

    assert result["model"] == "random_forest"
    assert result["label_mode"] == "adaptive"
    assert result["feature_variant"] == "depth_plus_window"
    assert "accuracy" in result
    assert "macro_f1" in result
    assert "cumulative_return" in result
