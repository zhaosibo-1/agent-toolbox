"""工具子系统。

对外只需要认识三样东西：

- :func:`tool` —— 装饰器，把函数变成工具
- :class:`ToolSpec` —— 工具的元信息与执行体
- :class:`ToolRegistry` —— 工具注册与执行中心
"""

from .base import ToolError, ToolSpec, build_tool_spec, tool
from .registry import ToolRegistry

__all__ = [
    "ToolError",
    "ToolSpec",
    "ToolRegistry",
    "build_tool_spec",
    "tool",
]
