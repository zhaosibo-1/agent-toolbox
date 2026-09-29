"""测试夹具。

三条原则：

1. **不依赖任何 API Key**——离线规划器让整条 Agent 链路可在 CI 里完整跑通；
2. **不污染全局状态**——内置工具的安全边界是模块级变量，每个用例前后都重置；
3. **不碰真实网络**——HTTP 工具的用例只验证白名单拦截（在拦截点就报错，
   根本发不出请求）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from app.tools import builtin as builtin_tools
from app.tools.registry import ToolRegistry
from helpers import REPO_ROOT, make_settings

#: 内置工具的默认边界，用于用例前后复位
_DEFAULT_BOUNDARY: Dict[str, Any] = {
    "allowed_dirs": [],
    "http_hosts": [],
    "calc_max_length": 200,
    "project_root": str(REPO_ROOT),
}


@pytest.fixture
def settings() -> Any:
    return make_settings()


@pytest.fixture(autouse=True)
def _reset_tool_boundaries() -> Any:
    """内置工具的允许目录 / 白名单是模块级状态，用例之间必须隔离。"""
    builtin_tools.configure(**_DEFAULT_BOUNDARY)
    yield
    builtin_tools.configure(**_DEFAULT_BOUNDARY)


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """一个只含少量文件、已启用文件工具访问的沙箱目录。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    # 明确按 UTF-8 字节写入，避免 Windows 上 write_text 把 \n 转成 \r\n 影响字符计数。
    # 末尾刻意保留一个换行：用来守住「步骤之间流转的数据不被静默 strip」这件事。
    (docs / "note.txt").write_bytes(
        "你好，这是沙箱里的说明文件。\n第二行内容。\n".encode("utf-8")
    )
    (docs / "secret").mkdir()
    (docs / "secret" / "inner.txt").write_bytes("内层文件".encode("utf-8"))
    builtin_tools.configure(
        allowed_dirs=[str(docs)],
        http_hosts=["example.com"],
        calc_max_length=200,
        project_root=str(tmp_path),
    )
    return docs


@pytest.fixture
def registry() -> ToolRegistry:
    """只挂内置工具的注册表。"""
    reg = ToolRegistry()
    reg.register_module(builtin_tools)
    return reg
