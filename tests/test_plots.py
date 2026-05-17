"""Tests for plot generation from aggregated experiment results."""

from pathlib import Path

import pandas as pd

from fi2010_ml.plots import generate_result_plots


def test_generate_result_plots_creates_png_files(tmp_path):
    results = pd.DataFrame(
        [
            {"symbol": "AAPL", "model": "a", "label_mode": "fixed", "horizon": 10, "macro_f1": 0.3, "accuracy": 0.9, "status": "ok"},
            {"symbol": "AAPL", "model": "b", "label_mode": "adaptive", "horizon": 60, "macro_f1": 0.4, "accuracy": 0.5, "status": "ok"},
            {"symbol": "MSFT", "model": "a", "label_mode": "fixed", "horizon": 10, "macro_f1": 0.2, "accuracy": 0.8, "status": "ok"},
        ]
    )

    outputs = generate_result_plots(results, tmp_path)

    assert len(outputs) == 2
    for output in outputs:
        assert Path(output).exists()
