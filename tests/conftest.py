"""Pytest setup for making the local src package importable during test runs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] #resolve()获取绝对路径，parents[1]获取上两级目录(parents[0]是tests)
SRC = ROOT / "src"

if str(SRC) not in sys.path: #sys.path是一个列表，包含了Python解释器查找模块时会搜索的目录。我们将src目录添加到sys.path中，这样在测试文件中就可以直接import src.fi2010_ml.pipeline等模块了。
    sys.path.insert(0, str(SRC)) #insert(0, ...)将src目录添加到sys.path的开头，确保它优先于其他目录被搜索到。这对于避免与系统安装的同名模块发生冲突非常重要。