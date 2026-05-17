"""Time-aware dataset splitting utilities for leakage-safe train/validation/test partitions."""

from __future__ import annotations

import pandas as pd


def time_order_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered dataset into train/validation/test partitions."""    
    total = len(frame)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train = frame.iloc[:train_end].reset_index(drop=True) # drop=True表示不保留索引
    valid = frame.iloc[train_end:valid_end].reset_index(drop=True)
    test = frame.iloc[valid_end:].reset_index(drop=True)
    return train, valid, test
