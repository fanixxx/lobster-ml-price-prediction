"""Tests for benchmark-style FI-2010 train/test CLI execution."""

from __future__ import annotations

import pandas as pd
import yaml

from fi2010_ml.benchmark_cli import main


def test_benchmark_cli_runs_one_split_end_to_end(tmp_path, monkeypatch) -> None:
    root = tmp_path / "BenchmarkDatasets" / "NoAuction" / "3.NoAuction_DecPre"
    train_dir = root / "NoAuction_DecPre_Training"
    test_dir = root / "NoAuction_DecPre_Testing"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    train_path = train_dir / "Train_Dst_NoAuction_DecPre_CF_1.txt"
    test_path = test_dir / "Test_Dst_NoAuction_DecPre_CF_1.txt"
    train_frame = pd.DataFrame(
        [[(row + 1) * 0.01 + col * 0.001 + ((col % 7) - 3) * 0.003 for col in range(120)] for row in range(45)]
    )
    test_frame = pd.DataFrame(
        [[(row + 1) * 0.02 + col * 0.001 + ((col % 5) - 2) * 0.004 for col in range(80)] for row in range(45)]
    )
    train_frame.to_csv(train_path, sep=" ", header=False, index=False)
    test_frame.to_csv(test_path, sep=" ", header=False, index=False)

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "benchmark_root": str(root),
                    "snapshot_frequency": "1s",
                    "trading_start_seconds": 0.0,
                    "trading_end_seconds": 1000000000000.0,
                    "cost_bps": 1.0,
                },
                "features": {
                    "levels": 10,
                    "rolling_windows": [5, 10],
                    "depth_levels": [5, 10],
                    "aggregate_windows": [5],
                    "feature_variants": ["depth_plus_window"],
                },
                "labels": {
                    "modes": ["adaptive"],
                    "horizons": [5],
                    "fixed_threshold": 0.0005,
                    "volatility_window": 10,
                    "volatility_scale": 0.5,
                    "quantile_lower": 0.2,
                    "quantile_upper": 0.8,
                },
                "models": {
                    "candidates": ["random_forest"],
                },
                "benchmark": {
                    "cf_splits": [1],
                },
            }
        ),
        encoding="utf-8",
    )

    results_path = tmp_path / "benchmark_results.csv"
    summary_path = tmp_path / "benchmark_summary.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_cli",
            "--config",
            str(config_path),
            "--results",
            str(results_path),
            "--summary",
            str(summary_path),
        ],
    )

    main()

    assert results_path.exists()
    assert summary_path.exists()
