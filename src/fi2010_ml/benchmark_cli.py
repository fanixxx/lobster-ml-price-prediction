"""CLI for benchmark-style FI-2010 train/test evaluation on official CF splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fi2010_ml.config import load_config
from fi2010_ml.reporting import save_summary_table
from fi2010_ml.workflow import run_train_test_workflow


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for official train/test FI-2010 benchmark runs."""
    parser = argparse.ArgumentParser(
        description="Run FI-2010 benchmark train/test evaluation on NoAuction DecPre splits."
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--results",
        default="results/benchmark_experiment_results.csv",
        help="Path to save full benchmark result table.",
    )
    parser.add_argument(
        "--summary",
        default="results/benchmark_experiment_summary.csv",
        help="Path to save ranked benchmark summary table.",
    )
    return parser


def _resolve_split_paths(benchmark_root: Path, split_id: int) -> tuple[Path, Path]:
    train_path = benchmark_root / "NoAuction_DecPre_Training" / f"Train_Dst_NoAuction_DecPre_CF_{split_id}.txt"
    test_path = benchmark_root / "NoAuction_DecPre_Testing" / f"Test_Dst_NoAuction_DecPre_CF_{split_id}.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing benchmark pair for CF_{split_id}: {train_path} / {test_path}")
    return train_path, test_path


def main() -> None:
    """Run benchmark-style FI-2010 training, testing, and backtesting."""
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    benchmark_root = Path(config["data"]["benchmark_root"])
    split_ids = config.get("benchmark", {}).get("cf_splits", [7])
    results: list[dict] = []

    for split_id in split_ids:
        train_path, test_path = _resolve_split_paths(benchmark_root, split_id)
        for model_name in config["models"]["candidates"]:
            for label_mode in config["labels"]["modes"]:
                for feature_variant in config["features"].get("feature_variants", ["depth_plus_window"]):
                    for horizon in config["labels"]["horizons"]:
                        result = run_train_test_workflow(
                            train_input_path=train_path,
                            test_input_path=test_path,
                            converted_dir=Path("data/processed") / f"benchmark_cf_{split_id}",
                            model_name=model_name,
                            label_mode=label_mode,
                            horizon=horizon,
                            fixed_threshold=config["labels"]["fixed_threshold"],
                            volatility_window=config["labels"]["volatility_window"],
                            volatility_scale=config["labels"]["volatility_scale"],
                            feature_variant=feature_variant,
                            levels=config["features"]["levels"],
                            depth_levels=config["features"].get("depth_levels"),
                            aggregate_windows=config["features"].get("aggregate_windows"),
                            rolling_windows=config["features"].get("rolling_windows"),
                            snapshot_frequency=config["data"]["snapshot_frequency"],
                            trading_start_seconds=config["data"]["trading_start_seconds"],
                            trading_end_seconds=config["data"]["trading_end_seconds"],
                            quantile_bounds=(
                                config["labels"].get("quantile_lower", 0.25),
                                config["labels"].get("quantile_upper", 0.75),
                            ),
                            cost_bps=config["data"].get("cost_bps", 1.0),
                        )
                        result["split_id"] = split_id
                        results.append(result)

    results_frame = pd.DataFrame(results)
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(results_path, index=False)
    summary = save_summary_table(results_frame, args.summary)

    print(f"Saved benchmark results to {results_path}")
    print(f"Saved benchmark summary to {args.summary}")
    if not summary.empty:
        print("Top benchmark experiment:")
        print(summary.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
