"""工具注册表：负责收集工具、生成模型可用的 schema、以及安全地执行调用。"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Dict, Iterable, List

from .base import ToolError, ToolSpec, build_tool_spec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表。

    职责：
    - **收集**：从模块或函数对象中提取 :class:`ToolSpec`
    - **暴露**：生成 OpenAI function calling 需要的 tools 数组
    - **执行**：按名字调用工具，把参数解析、超时、异常统一处理好
    - **可观测**：记录调用次数，供前端展示与调试
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._order: List[str] = []

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(self, spec: ToolSpec) -> ToolSpec:
        """注册单个工具。同名会被覆盖（便于测试时替换）。"""
        if spec.name not in self._tools:
            self._order.append(spec.name)
        else:
            logger.debug("工具 %s 被重新注册，已覆盖", spec.name)
        self._tools[spec.name] = spec
        return spec

    def register_callable(self, func: Any, **kwargs: Any) -> ToolSpec:
        """把普通函数注册为工具（若已带 ``__tool_spec__`` 则复用其 schema）。"""
        existing = getattr(func, "__tool_spec__", None)
        if isinstance(existing, ToolSpec) and not kwargs:
            return self.register(existing.clone())
        return self.register(build_tool_spec(func, **kwargs))

    def register_module(self, module: Any) -> int:
        """扫描一个模块，把它里面所有 ``@tool`` 装饰过的函数全部注册进来。

        用「扫模块」而不是「手工维护工具列表」的好处是：
        新增工具只需要写函数，不需要记得去别处登记。
        """
        count = 0
        for _, obj in inspect.getmembers(module, inspect.isfunction):
            spec = getattr(obj, "__tool_spec__", None)
            if isinstance(spec, ToolSpec):
                # 存副本，避免把调用统计写回函数上的全局 spec
                self.register(spec.clone())
                count += 1
        return count

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            self._order.remove(name)
            return True
        return False

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(
                f"工具 {name} 不存在。可用工具：{', '.join(self.names())}"
            )
        return spec

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> List[ToolSpec]:
        """按注册顺序返回全部工具。"""
        return [self._tools[n] for n in self._order]

    def names(self) -> List[str]:
        return list(self._order)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # 输出给模型 / 前端
    # ------------------------------------------------------------------
    def openai_schemas(self, only: Iterable[str] | None = None) -> List[Dict[str, Any]]:
        """生成 OpenAI ``tools`` 参数。

        Args:
            only: 只导出指定的工具（用于按场景裁剪工具集，减少模型选错）。
        """
        specs = self.all()
        if only is not None:
            allow = set(only)
            specs = [s for s in specs if s.name in allow]
        return [s.to_openai_schema() for s in specs]

    def describe(self) -> List[Dict[str, Any]]:
        """给前端展示的工具清单。"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "params": s.describe_params(),
                "required": s.required_params(),
                "dangerous": s.dangerous,
                "timeout": s.timeout,
                "call_count": s.call_count,
            }
            for s in self.all()
        ]

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    async def aexecute(self, name: str, arguments: Dict[str, Any] | str | None) -> str:
        """执行工具并返回**字符串**结果（供回灌给模型）。

        任何失败都会抛出 :class:`ToolError`，由 Agent 循环捕获后转成
        工具消息回灌——让模型看到错误信息，它往往能自己纠正参数重试。
        """
        spec = self.get(name)

        # 模型有时会把参数写成 JSON 字符串而不是对象，这里做一次兼容
        if arguments is None:
            args: Dict[str, Any] = {}
        elif isinstance(arguments, str):
            raw = arguments.strip()
            if not raw:
                args = {}
            else:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ToolError(
                        f"工具参数不是合法 JSON：{exc.msg}。"
                        f"请按此格式传参：{spec.describe_params()}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ToolError("工具参数必须是 JSON 对象")
                args = parsed
        elif isinstance(arguments, dict):
            args = dict(arguments)
        else:
            raise ToolError("工具参数类型不支持")

        # 提前拦掉模型编造的参数名，给出可用参数清单
        allowed = set(spec.parameters.get("properties", {}))
        unknown = set(args) - allowed
        if unknown:
            raise ToolError(
                f"{name} 收到了未知参数：{', '.join(sorted(unknown))}。"
                f"可用参数：{spec.describe_params()}"
            )

        # 空字符串 / 只有空白的字符串都算没填——否则会把 "   " 直接塞进工具，
        # 报出「表达式不能为空」这种让模型摸不着头脑的错误。
        missing = [
            p
            for p in spec.required_params()
            if p not in args or args[p] is None or (isinstance(args[p], str) and not args[p].strip())
        ]
        if missing:
            raise ToolError(
                f"{name} 缺少必填参数：{', '.join(missing)}。"
                f"期望参数：{spec.describe_params()}"
            )

        logger.info("调用工具 %s，参数 %s", name, args)
        result = await spec.ainvoke(args)

        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(result)
