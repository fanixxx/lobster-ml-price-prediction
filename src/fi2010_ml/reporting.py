"""Reporting helpers for exporting ranked experiment tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _rank_successful_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.loc[results["status"] == "ok"]
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )


def save_summary_table(results: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    summary = _rank_successful_results(results)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return summary


def select_best_result(results: pd.DataFrame) -> pd.Series:
    """Return the single best successful result row ranked by macro-F1 then accuracy."""
    summary = _rank_successful_results(results)
    if summary.empty:
        return pd.Series(dtype="object")
    return summary.iloc[0]
