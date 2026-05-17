"""Tests for ranked summary-table export from experiment outputs."""

import pandas as pd

from fi2010_ml.reporting import save_summary_table, select_best_result


def test_save_summary_table_sorts_best_results_first(tmp_path):
    results = pd.DataFrame(
        [
            {"model": "a", "status": "ok", "macro_f1": 0.4, "accuracy": 0.5},
            {"model": "b", "status": "ok", "macro_f1": 0.6, "accuracy": 0.4},
            {"model": "c", "status": "skipped", "macro_f1": None, "accuracy": None},
        ]
    )

    output_path = tmp_path / "summary.csv"
    summary = save_summary_table(results, output_path)

    assert summary.loc[0, "model"] == "b"
    assert output_path.exists()


def test_select_best_result_returns_top_ranked_row():
    results = pd.DataFrame(
        [
            {"model": "a", "status": "ok", "macro_f1": 0.4, "accuracy": 0.5, "horizon": 30},
            {"model": "b", "status": "ok", "macro_f1": 0.6, "accuracy": 0.4, "horizon": 10},
            {"model": "c", "status": "skipped", "macro_f1": None, "accuracy": None, "horizon": 60},
        ]
    )

    best = select_best_result(results)

    assert best["model"] == "b"
    assert best["horizon"] == 10
