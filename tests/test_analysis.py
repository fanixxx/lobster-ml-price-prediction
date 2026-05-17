"""Tests for experiment-summary table generation and executive summary text output."""

from pathlib import Path

import pandas as pd

from fi2010_ml.analysis import (
    build_executive_summary,
    build_label_distribution_summary,
    save_best_results_by_symbol,
)


def test_save_best_results_by_symbol_keeps_one_row_per_symbol(tmp_path):
    results = pd.DataFrame(
        [
            {"symbol": "AAPL", "model": "a", "macro_f1": 0.2, "accuracy": 0.9, "status": "ok"},
            {"symbol": "AAPL", "model": "b", "macro_f1": 0.3, "accuracy": 0.8, "status": "ok"},
            {"symbol": "MSFT", "model": "c", "macro_f1": 0.4, "accuracy": 0.7, "status": "ok"},
        ]
    )

    output_path = tmp_path / "best.csv"
    best = save_best_results_by_symbol(results, output_path)

    assert len(best) == 2
    assert set(best["symbol"]) == {"AAPL", "MSFT"}
    assert output_path.exists()


def test_build_executive_summary_mentions_best_model():
    results = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "model": "logistic_regression",
                "label_mode": "adaptive",
                "feature_variant": "depth_plus_window",
                "horizon": 60,
                "accuracy": 0.4,
                "macro_f1": 0.35,
                "status": "ok",
            }
        ]
    )

    summary = build_executive_summary(results)

    assert "AAPL" in summary
    assert "logistic_regression" in summary
    assert "depth_plus_window" in summary


def test_build_label_distribution_summary_counts_successful_runs():
    results = pd.DataFrame(
        [
            {"label_mode": "adaptive", "feature_variant": "baseline", "status": "ok"},
            {"label_mode": "adaptive", "feature_variant": "baseline", "status": "ok"},
            {"label_mode": "quantile", "feature_variant": "depth_plus_window", "status": "ok"},
            {"label_mode": "quantile", "feature_variant": "depth_plus_window", "status": "skipped"},
        ]
    )

    summary = build_label_distribution_summary(results)

    assert len(summary) == 2
    assert set(summary["label_mode"]) == {"adaptive", "quantile"}
