"""Tests for raw LOBSTER file column templates, validation, and merged day loading."""

from pathlib import Path

import pytest

from fi2010_ml.data_loader import (
    build_message_columns,
    build_orderbook_columns,
    infer_levels_from_orderbook_file,
    load_lobster_csv,
    load_lobster_day,
)


def test_project_skeleton_exists():
    assert Path("src/fi2010_ml/__init__.py").exists()
    assert Path("configs/base.yaml").exists()


def test_load_lobster_csv_assigns_columns(tmp_path):
    csv_path = tmp_path / "orderbook.csv"
    csv_path.write_text("100,10,101,12\n99,8,102,15\n", encoding="utf-8")

    frame = load_lobster_csv(
        csv_path,
        columns=["bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1"],
    )

    assert list(frame.columns) == [
        "bid_price_1",
        "bid_size_1",
        "ask_price_1",
        "ask_size_1",
    ]
    assert frame.shape == (2, 4)


def test_load_lobster_csv_validates_column_count(tmp_path):
    csv_path = tmp_path / "message.csv"
    csv_path.write_text("1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_lobster_csv(csv_path, columns=["a", "b"])


def test_build_orderbook_columns_matches_level_count():
    columns = build_orderbook_columns(levels=2)

    assert columns == [
        "ask_price_1",
        "ask_size_1",
        "bid_price_1",
        "bid_size_1",
        "ask_price_2",
        "ask_size_2",
        "bid_price_2",
        "bid_size_2",
    ]


def test_build_message_columns_matches_lobster_format():
    assert build_message_columns() == [
        "time",
        "event_type",
        "order_id",
        "size",
        "price",
        "direction",
    ]


def test_load_lobster_day_merges_message_and_orderbook(tmp_path):
    message_path = tmp_path / "message.csv"
    orderbook_path = tmp_path / "orderbook.csv"
    message_path.write_text("34200,1,10,100,10000,1\n", encoding="utf-8")
    orderbook_path.write_text("10100,12,10000,10\n", encoding="utf-8")

    frame = load_lobster_day(message_path, orderbook_path, levels=1)

    assert list(frame.columns) == [
        "time",
        "event_type",
        "order_id",
        "size",
        "price",
        "direction",
        "ask_price_1",
        "ask_size_1",
        "bid_price_1",
        "bid_size_1",
    ]
    assert frame.shape == (1, 10)


def test_infer_levels_from_orderbook_file_uses_column_count(tmp_path):
    orderbook_path = tmp_path / "mystery_orderbook.csv"
    orderbook_path.write_text("10100,12,10000,10,10110,11,9990,9\n", encoding="utf-8")

    inferred = infer_levels_from_orderbook_file(orderbook_path)

    assert inferred == 2


def test_load_lobster_day_can_infer_levels_when_not_provided(tmp_path):
    message_path = tmp_path / "AAPL_message_50.csv"
    orderbook_path = tmp_path / "AAPL_orderbook_2.csv"
    message_path.write_text("34200,1,10,100,10000,1\n", encoding="utf-8")
    orderbook_path.write_text("10100,12,10000,10,10110,11,9990,9\n", encoding="utf-8")

    frame = load_lobster_day(message_path, orderbook_path, levels=None)

    assert "ask_price_2" in frame.columns
    assert "bid_size_2" in frame.columns


def test_load_lobster_day_caps_requested_levels_to_available_depth(tmp_path):
    message_path = tmp_path / "AAPL_message_50.csv"
    orderbook_path = tmp_path / "AAPL_orderbook_2.csv"
    message_path.write_text("34200,1,10,100,10000,1\n", encoding="utf-8")
    orderbook_path.write_text("10100,12,10000,10,10110,11,9990,9\n", encoding="utf-8")

    frame = load_lobster_day(message_path, orderbook_path, levels=50)

    assert "ask_price_2" in frame.columns
    assert "bid_size_2" in frame.columns
    assert "ask_price_3" not in frame.columns
