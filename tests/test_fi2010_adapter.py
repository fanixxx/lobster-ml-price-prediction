"""Tests for converting FI-2010 matrix files into pseudo-LOBSTER CSV frames."""

from __future__ import annotations

import pandas as pd
import pytest

from fi2010_ml.data_loader import load_lobster_day
from fi2010_ml.fi2010_adapter import (
    build_fi2010_message_frame,
    build_fi2010_orderbook_frame,
    convert_fi2010_file,
    load_fi2010_matrix,
)


def test_load_fi2010_matrix_keeps_only_orderbook_rows(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    frame = pd.DataFrame(
        [[row * 10 + col for col in range(3)] for row in range(45)]
    )
    frame.to_csv(path, sep=" ", header=False, index=False)

    matrix = load_fi2010_matrix(path)

    assert matrix.shape == (40, 3)
    assert matrix.iloc[-1, -1] == 392


def test_build_fi2010_orderbook_frame_uses_lobster_columns() -> None:
    matrix = pd.DataFrame(
        [[row * 100 + col for col in range(2)] for row in range(40)]
    )

    orderbook = build_fi2010_orderbook_frame(matrix)

    assert orderbook.shape == (2, 40)
    assert orderbook.columns[:4].tolist() == [
        "ask_price_1",
        "ask_size_1",
        "bid_price_1",
        "bid_size_1",
    ]
    assert orderbook.iloc[0]["ask_price_1"] == 0
    assert orderbook.iloc[0]["ask_size_1"] == 100
    assert orderbook.iloc[0]["bid_price_1"] == 200
    assert orderbook.iloc[0]["bid_size_1"] == 300


def test_build_fi2010_message_frame_matches_orderbook_rows() -> None:
    orderbook = pd.DataFrame(
        {
            "ask_price_1": [101.0, 102.0, 103.0],
            "ask_size_1": [11.0, 12.0, 13.0],
            "bid_price_1": [99.0, 100.0, 101.0],
            "bid_size_1": [9.0, 10.0, 11.0],
        }
    )

    message = build_fi2010_message_frame(orderbook, time_step=1.0)

    assert message.shape == (3, 6)
    assert message.columns.tolist() == [
        "time",
        "event_type",
        "order_id",
        "size",
        "price",
        "direction",
    ]
    assert message["time"].tolist() == [0.0, 1.0, 2.0]
    assert message["price"].tolist() == [100.0, 101.0, 102.0]
    assert message["direction"].tolist() == [1, 1, 0]


def test_load_fi2010_matrix_rejects_too_few_rows(tmp_path) -> None:
    path = tmp_path / "bad.txt"
    frame = pd.DataFrame([[row * 10 + col for col in range(3)] for row in range(39)])
    frame.to_csv(path, sep=" ", header=False, index=False)

    with pytest.raises(ValueError, match="at least 40 rows"):
        load_fi2010_matrix(path)


def test_convert_fi2010_file_writes_lobster_csvs(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    frame = pd.DataFrame(
        [[row * 10 + col for col in range(4)] for row in range(45)]
    )
    frame.to_csv(path, sep=" ", header=False, index=False)

    message_path, orderbook_path = convert_fi2010_file(path, tmp_path / "out")
    loaded = load_lobster_day(message_path, orderbook_path, levels=10)

    assert message_path.exists()
    assert orderbook_path.exists()
    assert loaded.shape[0] == 4
