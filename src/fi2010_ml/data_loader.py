"""Load raw LOBSTER message and orderbook files into aligned pandas DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_orderbook_columns(levels: int) -> list[str]:
    """Build column names for LOBSTER orderbook files."""
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


def build_message_columns() -> list[str]:
    """Build column names for LOBSTER message files."""
    return [
        "time",
        "event_type",
        "order_id",
        "size",
        "price",
        "direction",
    ]


def infer_levels_from_orderbook_file(path: str | Path) -> int:
    """Infer the number of LOBSTER book levels from the orderbook column count."""
    sample = pd.read_csv(path, header=None, nrows=1)
    column_count = sample.shape[1]
    if column_count % 4 != 0:
        raise ValueError(
            f"Orderbook file {path} has {column_count} columns, which is not divisible by 4."
        )
    return column_count // 4


def load_lobster_csv(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Load a generic LOBSTER CSV and assign validated columns."""
    frame = pd.read_csv(path, header=None)
    if len(columns) != frame.shape[1]:
        raise ValueError(
            f"Expected {len(columns)} columns, but file has {frame.shape[1]} columns."
        )
    frame.columns = columns
    return frame


def load_orderbook_csv(path: str | Path, levels: int | None = None) -> pd.DataFrame:
    """Load a LOBSTER orderbook file using explicit or inferred depth levels."""
    available_levels = infer_levels_from_orderbook_file(path)
    resolved_levels = available_levels if levels is None else min(levels, available_levels)
    columns = build_orderbook_columns(resolved_levels)
    return load_lobster_csv(path, columns=columns)


def load_message_csv(path: str | Path) -> pd.DataFrame:
    """Load a LOBSTER message file."""
    columns = build_message_columns()
    return load_lobster_csv(path, columns=columns)


def load_lobster_day(
    message_path: str | Path,
    orderbook_path: str | Path,
    levels: int | None = None,
) -> pd.DataFrame:
    """Load aligned message and orderbook files for a single LOBSTER day."""
    message = load_message_csv(message_path)
    orderbook = load_orderbook_csv(orderbook_path, levels=levels)

    if len(message) != len(orderbook):
        raise ValueError(
            "Message and orderbook files must have the same number of rows."
        )

    return pd.concat([message, orderbook], axis=1)
