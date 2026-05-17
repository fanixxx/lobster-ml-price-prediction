# FI-2010 Price Direction Prediction

Machine learning pipeline for short-term price direction prediction on the `FI-2010` limit order book benchmark.

## Overview

This project focuses on benchmark-style prediction experiments on `FI-2010`, using the `NoAuction/DecPre` variant as the main dataset. The repository converts FI-2010 matrix files into a reusable pseudo order-book format, builds microstructure features, generates three-class labels, trains classification models, and exports evaluation results across official train/test splits.

The current default setup is:

- dataset: `NoAuction/DecPre`
- splits: official `CF_1 ~ CF_9`
- label mode: `adaptive`
- feature set: `depth_plus_window`
- primary model: `xgboost`

## What The Repository Includes

- FI-2010 matrix to pseudo order-book conversion
- feature engineering for price, imbalance, depth, and window statistics
- label generation for short-term direction classification
- benchmark train/test workflow on official split pairs
- experiment result export and ranked summaries
- automated tests for the full pipeline

## Repository Structure

```text
configs/        experiment configurations
src/fi2010_ml/  source code
tests/          automated tests
results/        example local outputs
```

## Installation

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout

Place the FI-2010 benchmark files under:

```text
data/
  BenchmarkDatasets/
    NoAuction/
      3.NoAuction_DecPre/
        NoAuction_DecPre_Training/
        NoAuction_DecPre_Testing/
```

The repository assumes benchmark files such as:

- `Train_Dst_NoAuction_DecPre_CF_1.txt`
- `Test_Dst_NoAuction_DecPre_CF_1.txt`

## Quick Start

Run the default benchmark configuration:

```bash
PYTHONPATH=src python -m fi2010_ml.benchmark_cli \
  --config configs/base.yaml \
  --results results/benchmark_experiment_results.csv \
  --summary results/benchmark_experiment_summary.csv
```

This command:

- trains on each official `Train_Dst_NoAuction_DecPre_CF_k.txt`
- evaluates on the matching `Test_Dst_NoAuction_DecPre_CF_k.txt`
- exports classification metrics for all configured splits

## Configurations

- `configs/base.yaml`
  - default global benchmark configuration
- `configs/benchmark_xgboost_h10.yaml`
  - fixed `xgboost + adaptive + depth_plus_window + horizon=10`
- `configs/fi2010.yaml`
  - broader FI-2010 experiment configuration
- `configs/fi2010_lite.yaml`
  - lightweight local test configuration

## Example Result Files

- `results/benchmark_xgboost_h10_results.csv`
- `results/benchmark_xgboost_h10_summary.csv`

These files contain split-level classification results such as:

- `accuracy`
- `macro_f1`
- `train_size`
- `test_size`

## Testing

```bash
PYTHONPATH=src python -m pytest -q
```

## Notes

- The repository is organized for reproducible prediction experiments, not for storing large raw datasets.
- The current focus is the prediction task itself rather than financial strategy evaluation.
