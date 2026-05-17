"""Convert FI-2010 matrix files into pseudo-LOBSTER message and orderbook CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fi2010_ml.data_loader import build_message_columns, build_orderbook_columns


def load_fi2010_matrix(path: str | Path) -> pd.DataFrame:
    """Load a FI-2010 matrix and keep only the 40-row order-book block."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if frame.shape[0] < 40:
        raise ValueError(f"FI-2010 file {path} must contain at least 40 rows.")
    return frame.iloc[:40].copy()


def build_fi2010_orderbook_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    """Transpose the 40-row FI-2010 order-book block into 10 LOBSTER-style levels."""
    if matrix.shape[0] != 40:
        raise ValueError("FI-2010 order-book slice must contain exactly 40 rows.")
    orderbook = matrix.transpose().reset_index(drop=True)
    orderbook.columns = build_orderbook_columns(levels=10)
    return orderbook


def build_fi2010_message_frame(
    orderbook: pd.DataFrame,
    time_step: float = 1.0,
) -> pd.DataFrame:
    """Build a minimal synthetic message frame aligned to the pseudo orderbook rows."""
    times = [index * time_step for index in range(len(orderbook))]
    mid_price = (orderbook["bid_price_1"] + orderbook["ask_price_1"]) / 2
    next_move = mid_price.shift(-1) - mid_price
    direction = next_move.fillna(0.0).apply(
        lambda value: 1 if value > 0 else (-1 if value < 0 else 0)
    )

    message = pd.DataFrame(
        {
            "time": times,
            "event_type": 1,
            "order_id": range(1, len(orderbook) + 1),
            "size": orderbook["bid_size_1"] + orderbook["ask_size_1"],
            "price": mid_price,
            "direction": direction,
        }
    )
    message.columns = build_message_columns()
    return message


def convert_fi2010_file(
    input_path: str | Path,
    output_dir: str | Path,
    time_step: float = 1.0,
) -> tuple[Path, Path]:
    """Convert a single FI-2010 file into pseudo-LOBSTER CSV files."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_fi2010_matrix(input_path)
    orderbook = build_fi2010_orderbook_frame(matrix)
    message = build_fi2010_message_frame(orderbook, time_step=time_step)

    stem = input_path.stem
    message_path = output_dir / f"{stem}_message_10.csv"
    orderbook_path = output_dir / f"{stem}_orderbook_10.csv"
    message.to_csv(message_path, header=False, index=False)
    orderbook.to_csv(orderbook_path, header=False, index=False)
    return message_path, orderbook_path
