"""Configuration loading utilities for YAML-based experiment settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load experiment configuration from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

"""
yaml.safe_load 函数会解析 YAML 格式的数据，并将其转换为 Python 的字典对象。
如果 YAML 数据是列表或映射结构，它将被转换为 Python 的列表或字典对象。
如果 YAML 数据包含嵌套结构，它将被转换为相应的 Python 对象。

例如，假设有一个名为 data.yaml 的文件，其中包含以下 YAML 数据：

name: John Doe
age: 30
hobbies:
  - reading
  - running
  - coding

使用 yaml.safe_load 函数将其转换为 Python 对象：

import yaml

with open("data.yaml", "r") as file:
    data = yaml.safe_load(file)

print(data)
输出结果为：

python
{'name': 'John Doe', 'age': 30, 'hobbies': ['reading', 'running', 'coding']}
"""