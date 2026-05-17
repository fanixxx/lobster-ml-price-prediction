"""Plotting utilities for turning experiment results into course-report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_result_plots(results: pd.DataFrame, output_dir: str | Path) -> list[str]:
    ok_results = results.loc[results["status"] == "ok"].copy()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for model_name, group in ok_results.groupby("model"):
        ax1.scatter(group["accuracy"], group["macro_f1"], label=model_name, alpha=0.8)
    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("Macro F1")
    ax1.set_title("Accuracy vs Macro F1 by Model")
    ax1.legend()
    plot1 = output_path / "accuracy_vs_macro_f1.png"
    fig1.savefig(plot1, bbox_inches="tight")
    plt.close(fig1)
    outputs.append(str(plot1))

    pivot = (
        ok_results.groupby(["label_mode", "horizon"], as_index=False)["macro_f1"]
        .mean()
        .sort_values(["label_mode", "horizon"])
    )
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for label_mode, group in pivot.groupby("label_mode"):
        ax2.plot(group["horizon"], group["macro_f1"], marker="o", label=label_mode)
    ax2.set_xlabel("Prediction Horizon")
    ax2.set_ylabel("Average Macro F1")
    ax2.set_title("Average Macro F1 by Label Mode and Horizon")
    ax2.legend()
    plot2 = output_path / "macro_f1_by_label_and_horizon.png"
    fig2.savefig(plot2, bbox_inches="tight")
    plt.close(fig2)
    outputs.append(str(plot2))

    return outputs
