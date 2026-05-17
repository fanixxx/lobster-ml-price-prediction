"""CLI tests for FI-2010 pseudo-LOBSTER conversion and experiment execution."""

from __future__ import annotations

import pandas as pd
import yaml

from fi2010_ml.fi2010_cli import main


def test_fi2010_cli_runs_end_to_end(tmp_path, monkeypatch) -> None:
    raw_path = tmp_path / "sample.txt"
    frame = pd.DataFrame(
        [[(row + 1) * 0.01 + col * 0.001 for col in range(40)] for row in range(45)]
    )
    frame.to_csv(raw_path, sep=" ", header=False, index=False)

    output_dir = tmp_path / "converted"
    feature_path = tmp_path / "features.csv"
    result_path = tmp_path / "results.csv"
    summary_path = tmp_path / "summary.csv"
    config_path = tmp_path / "fi2010_test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "snapshot_frequency": "1s",
                    "trading_start_seconds": 0.0,
                    "trading_end_seconds": 1000000000000.0,
                },
                "features": {
                    "levels": 10,
                    "rolling_windows": [5, 10],
                    "depth_levels": [5, 10],
                    "aggregate_windows": [5],
                    "feature_variants": ["baseline"],
                },
                "labels": {
                    "modes": ["quantile"],
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
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "fi2010_cli",
            "--config",
            str(config_path),
            "--input",
            str(raw_path),
            "--converted-dir",
            str(output_dir),
            "--output",
            str(feature_path),
            "--results",
            str(result_path),
            "--summary",
            str(summary_path),
        ],
    )

    main()

    assert feature_path.exists()
    assert result_path.exists()
    assert summary_path.exists()
