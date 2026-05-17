"""Analysis helpers for summarizing experiment outputs into report-ready text and tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_best_results_by_symbol(results: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    ok_results = results.loc[results["status"] == "ok"].copy()
    best = (
        ok_results.sort_values(["symbol", "macro_f1", "accuracy"], ascending=[True, False, False])
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(output, index=False)
    return best


def build_executive_summary(results: pd.DataFrame) -> str:
    ok_results = results.loc[results["status"] == "ok"].copy()
    if ok_results.empty:
        return "No successful experiment results were generated."

    best_rows = (
        ok_results.sort_values(["symbol", "macro_f1", "accuracy"], ascending=[True, False, False])
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    lines = ["# Experiment Executive Summary", ""]
    for row in best_rows.itertuples(index=False):
        lines.append(
            f"- {row.symbol}: best run is {row.model} with {row.label_mode} labels, "
            f"feature variant {getattr(row, 'feature_variant', 'baseline')}, at horizon {row.horizon}, "
            f"accuracy={row.accuracy:.4f}, macro_f1={row.macro_f1:.4f}."
        )
    return "\n".join(lines)


def build_label_distribution_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize how many successful runs exist by label mode and feature variant."""
    ok_results = results.loc[results["status"] == "ok"].copy()
    if ok_results.empty:
        return pd.DataFrame(columns=["label_mode", "feature_variant", "run_count"])

    group_columns = ["label_mode"]
    if "feature_variant" in ok_results.columns:
        group_columns.append("feature_variant")

    summary = (
        ok_results.groupby(group_columns, as_index=False)
        .size()
        .rename(columns={"size": "run_count"})
    )
    return summary
