"""Command-line entrypoint for converting FI-2010 files and running the existing experiment pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from fi2010_ml.config import load_config
from fi2010_ml.fi2010_adapter import convert_fi2010_file
from fi2010_ml.pipeline import prepare_lobster_dataset, run_experiment_grid
from fi2010_ml.reporting import save_summary_table


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for FI-2010 conversion and training."""
    parser = argparse.ArgumentParser(
        description="Run FI-2010 through the pseudo-LOBSTER experiment pipeline."
    )
    parser.add_argument("--config", default="configs/fi2010.yaml", help="Path to YAML config.")
    parser.add_argument("--input", required=True, help="Path to FI-2010 raw text file.")
    parser.add_argument(
        "--converted-dir",
        default="data/processed/fi2010_converted",
        help="Directory to save generated pseudo-LOBSTER CSV files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/fi2010_features.csv",
        help="Path to save processed feature dataset.",
    )
    parser.add_argument(
        "--results",
        default="results/fi2010_experiment_results.csv",
        help="Path to save experiment result table.",
    )
    parser.add_argument(
        "--summary",
        default="results/fi2010_experiment_summary.csv",
        help="Path to save sorted summary table.",
    )
    return parser


def main() -> None:
    """Convert one FI-2010 file and run the project experiment grid on it."""
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    message_path, orderbook_path = convert_fi2010_file(args.input, args.converted_dir)
    dataset = prepare_lobster_dataset(
        message_path=message_path,
        orderbook_path=orderbook_path,
        levels=config["features"]["levels"],
        snapshot_frequency=config["data"]["snapshot_frequency"],
        trading_start_seconds=config["data"]["trading_start_seconds"],
        trading_end_seconds=config["data"]["trading_end_seconds"],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    results = run_experiment_grid(
        frame=dataset,
        model_names=config["models"]["candidates"],
        label_modes=config["labels"]["modes"],
        horizons=config["labels"]["horizons"],
        fixed_threshold=config["labels"]["fixed_threshold"],
        volatility_window=config["labels"]["volatility_window"],
        volatility_scale=config["labels"]["volatility_scale"],
        feature_variants=config["features"].get("feature_variants", ["baseline"]),
        depth_levels=config["features"].get("depth_levels"),
        aggregate_windows=config["features"].get("aggregate_windows"),
        quantile_bounds=(
            config["labels"].get("quantile_lower", 0.25),
            config["labels"].get("quantile_upper", 0.75),
        ),
        rolling_windows=config["features"].get("rolling_windows"),
    )

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    summary = save_summary_table(results, args.summary)

    print(f"Saved pseudo-LOBSTER message to {message_path}")
    print(f"Saved pseudo-LOBSTER orderbook to {orderbook_path}")
    print(f"Saved processed dataset to {output_path}")
    print(f"Saved experiment results to {results_path}")
    print(f"Saved experiment summary to {args.summary}")
    if not summary.empty:
        print("Top experiment:")
        print(summary.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
