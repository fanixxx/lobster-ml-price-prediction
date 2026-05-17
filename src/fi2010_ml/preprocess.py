"""Preprocessing functions for trading-hour filtering and event-to-snapshot resampling."""

from __future__ import annotations

import pandas as pd


def filter_trading_hours(
    frame: pd.DataFrame,
    time_column: str,
    start_seconds: float,
    end_seconds: float,
) -> pd.DataFrame:
    """Filter a DataFrame to only include trading hours."""
    mask = frame[time_column].between(start_seconds, end_seconds) # between()用来筛选出在[start_seconds, end_seconds]之间的数据
    return frame.loc[mask].reset_index(drop=True) # loc用来筛选出在[start_seconds, end_seconds]之间的数据；reset_index()用来重置索引


def resample_snapshots(frame: pd.DataFrame, time_column: str, frequency: str) -> pd.DataFrame:
    """Resample a DataFrame to a snapshot frequency."""
    indexed = frame.copy() # copy()用来复制
    indexed["timestamp"] = pd.to_datetime(indexed[time_column], unit="s") #pd.to_datetime()用来将时间戳转换为日期时间,unit="s"表示时间戳单位为秒
    indexed = indexed.set_index("timestamp") #set_index()用来设置索引
    resampled = indexed.resample(frequency).last().ffill() # resample()用来重采样(对象是时间序列),ffill()用来填充缺失值,last()用来选择最后一个值，
    return resampled.reset_index(drop=True)
