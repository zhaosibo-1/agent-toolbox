"""工具抽象层：把一个普通 Python 函数变成大模型能调用的「工具」。

这是整个项目最核心的设计——**函数签名即接口定义**。

写一个工具不需要手写 JSON Schema，只要：

1. 给函数加上 `@tool` 装饰器；
2. 参数用标准的 Python 类型标注；
3. docstring 用 Google 风格的 `Args:` 段描述每个参数。

本模块会自动完成类型映射（`str` → `string`、`int` → `integer`、
`list[str]` → `array`…）、必填项判定（有默认值即选填）、以及参数说明提取。

这样做的好处是**接口永远不会和实现脱节**：改了函数签名，schema 自动跟着变，
不会出现「schema 里说必填、代码里其实可选」这类经典事故。
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import re
import types
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)


class ToolError(RuntimeError):
    """工具执行失败。异常信息会回灌给模型，让它有机会换个思路重试。"""


# ===========================================================================
# 类型 -> JSON Schema 映射
# ===========================================================================
#: Python 基础类型到 JSON Schema 类型的映射表
_PRIMITIVE_MAP: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    """把 Python 类型标注转换成 JSON Schema 片段。

    支持的写法（覆盖工具函数的绝大多数场景）::

        str                          -> {"type": "string"}
        int / float / bool           -> 对应标量类型
        list[str] / List[int]        -> {"type": "array", "items": {...}}
        dict[str, Any]               -> {"type": "object"}
        Optional[str]                -> {"type": "string"}      （可空在 required 里体现）
        str | int                    -> {"anyOf": [...]}        （联合类型）
        Literal["a", "b"]            -> {"type": "string", "enum": [...]}

    无法识别的类型退化为 ``{"type": "string"}``——宁可宽松，也不要因为
    一个没见过的类型让整个工具注册失败。
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    # 未被 get_type_hints 解析掉的字符串注解（前向引用失败）也退化为 string，
    # 但要留下痕迹，避免「全是 string」这种问题悄无声息地发生。
    if isinstance(annotation, str):
        logger.warning("参数标注 %r 未能解析为类型对象，退化为 string", annotation)
        return {"type": "string"}

    # 处理 Optional[X] / Union[X, Y]（Python 3.10+ 的 X | Y 也会归一化成 Union）
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _annotation_to_schema(args[0])
        if args:
            return {"anyOf": [_annotation_to_schema(a) for a in args]}
        return {"type": "string"}

    # 处理 list[X] / List[X] / Sequence[X]
    if origin in (list, List, Sequence):
        args = get_args(annotation)
        item_schema = _annotation_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    # 处理 dict[str, X]
    if origin in (dict, Dict):
        return {"type": "object"}

    # 基础类型
    if annotation in _PRIMITIVE_MAP:
        return {"type": _PRIMITIVE_MAP[annotation]}

    # 枚举（Literal）取第一个值作为语义提示
    if origin is not None and str(origin).endswith("Literal"):
        values = [v for v in get_args(annotation) if isinstance(v, (str, int, float))]
        if values and all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": list(values)}

    # 未知类型：退化为字符串，保证注册不失败
    logger.debug("工具参数类型无法识别，退化为 string：%r", annotation)
    return {"type": "string"}


def _is_optional(annotation: Any) -> bool:
    """判断类型标注是否允许 None。"""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return type(None) in get_args(annotation)
    return annotation is type(None)


# ===========================================================================
# docstring 解析
# ===========================================================================
# 匹配 Args / Arguments / 参数 段落
_ARGS_SECTION_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:Args|Arguments|参数)[ \t]*:[ \t]*\n(.*?)"
    r"(?=\n[ \t]*(?:Returns?|Raises?|Examples?|Yields?|Notes?|返回|异常|示例|注意)[ \t]*:|\Z)",
    re.S | re.I,
)
# 匹配 Args 段落里的一行参数说明：`name (type): 说明`
_ARG_LINE_RE = re.compile(r"^[ \t]*(\*{0,2}\w+)[ \t]*(?:\([^)]*\))?[ \t]*:[ \t]*(.+)$", re.M)


def _parse_docstring(doc: Optional[str]) -> tuple[str, Dict[str, str]]:
    """从 docstring 中提取「简介」和「参数说明表」。

    Returns:
        ``(简介, {参数名: 说明})``
    """
    if not doc:
        return "", {}

    doc = inspect.cleandoc(doc)

    # 简介 = 第一个空行之前的内容（合并成一行）
    head = re.split(r"\n\s*\n", doc, maxsplit=1)[0]
    summary = " ".join(line.strip() for line in head.splitlines() if line.strip())

    # 参数说明
    params: Dict[str, str] = {}
    section = _ARGS_SECTION_RE.search(doc)
    if section:
        for match in _ARG_LINE_RE.finditer(section.group(1)):
            name = match.group(1).lstrip("*")
            desc = " ".join(match.group(2).split())
            if desc:
                params[name] = desc
    return summary, params


# ===========================================================================
# 工具定义
# ===========================================================================
@dataclass
class ToolSpec:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]
    #: 是否为危险操作（会在前端高亮，也可用于接入人工确认流程）
    dangerous: bool = False
    #: 单次执行超时秒数
    timeout: float = 15.0
    #: 参数名 -> 说明，用于生成更易读的展示信息
    param_docs: Dict[str, str] = field(default_factory=dict)
    #: 调用统计
    call_count: int = 0

    # ------------------------------------------------------------------
    def to_openai_schema(self) -> Dict[str, Any]:
        """转成 OpenAI function calling 所需的 tools 数组元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def required_params(self) -> List[str]:
        return list(self.parameters.get("required", []))

    def clone(self) -> "ToolSpec":
        """复制一份给注册表使用。

        ``@tool`` 装饰器会把 spec 挂在函数对象上，那是一个**全局单例**。
        如果注册表直接持有它，调用统计、超时覆盖这类运行期状态就会跨注册表
        （以及跨测试）互相污染——生产环境只有一个注册表时看不出问题，
        但它是个真实的状态泄漏。
        """
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=copy.deepcopy(self.parameters),
            func=self.func,
            dangerous=self.dangerous,
            timeout=self.timeout,
            param_docs=dict(self.param_docs),
            call_count=0,
        )

    def describe_params(self) -> str:
        """生成一行人类可读的参数说明，用于日志与前端展示。"""
        props: Dict[str, Any] = self.parameters.get("properties", {})
        if not props:
            return "(无参数)"
        required = set(self.required_params())
        parts = []
        for pname, pschema in props.items():
            ptype = pschema.get("type", "any")
            mark = "" if pname in required else "?"
            parts.append(f"{pname}{mark}: {ptype}")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    def invoke(self, arguments: Dict[str, Any]) -> Any:
        """同步执行工具，把异常统一包装成 :class:`ToolError`。"""
        self.call_count += 1
        try:
            return self.func(**arguments)
        except TypeError as exc:
            # 参数不匹配是最常见的失败原因（模型编了个不存在的参数名）
            raise ToolError(
                f"调用 {self.name} 的参数不正确：{exc}。"
                f"期望参数：{self.describe_params()}"
            ) from exc
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转成可回灌给模型的错误
            raise ToolError(f"{self.name} 执行失败：{type(exc).__name__}: {exc}") from exc

    async def ainvoke(self, arguments: Dict[str, Any]) -> Any:
        """异步执行，带超时保护，避免某个工具卡死整个 Agent 循环。"""
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.invoke, arguments), timeout=self.timeout
            )
            return result
        except asyncio.TimeoutError as exc:
            raise ToolError(
                f"{self.name} 执行超时（超过 {self.timeout:.0f} 秒），已中止"
            ) from exc


# ===========================================================================
# 装饰器
# ===========================================================================
def build_tool_spec(
    func: Callable[..., Any],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    dangerous: bool = False,
    timeout: float = 15.0,
) -> ToolSpec:
    """由函数对象构造 :class:`ToolSpec`（schema 自动生成）。"""
    signature = inspect.signature(func)

    # 解析类型标注：get_type_hints 能处理 `from __future__ import annotations`
    # 带来的字符串化注解（Python 3.14 起默认行为）。
    try:
        hints = get_type_hints(func)
    except Exception as exc:  # noqa: BLE001
        # 这里必须留日志：一旦静默失败，所有参数都会退化成 string，
        # 而工具依然「能注册、能调用」，是最难排查的一类问题。
        logger.warning(
            "工具 %s 的类型标注解析失败，参数将退化为 string：%s: %s",
            func.__name__,
            type(exc).__name__,
            exc,
        )
        hints = {}

    doc_summary, param_docs = _parse_docstring(func.__doc__)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for pname, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # 不收 *args / **kwargs：模型无法理解它们的 schema
            logger.warning("工具 %s 的参数 %s 是可变参数，已忽略", func.__name__, pname)
            continue

        annotation = hints.get(pname, param.annotation)
        schema = _annotation_to_schema(annotation)

        # 参数说明优先取 docstring 的 Args 段
        if pname in param_docs:
            schema["description"] = param_docs[pname]

        # 有默认值 或 可空 或 是可选参数 -> 非必填
        has_default = param.default is not inspect.Parameter.empty
        if has_default:
            if param.default is not None:
                schema["default"] = param.default
        elif not _is_optional(annotation):
            required.append(pname)

        properties[pname] = schema

    parameters: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    # 明确禁止模型编造不存在的参数，能在入口挡掉一类幻觉
    parameters["additionalProperties"] = False

    return ToolSpec(
        name=name or func.__name__,
        description=description or doc_summary or f"调用 {func.__name__}",
        parameters=parameters,
        func=func,
        dangerous=dangerous,
        timeout=timeout,
        param_docs=param_docs,
    )


def tool(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    dangerous: bool = False,
    timeout: float = 15.0,
) -> Any:
    """把一个函数标记为工具。既可裸用 ``@tool``，也可带参 ``@tool(name="x")``。

    被装饰的函数会被附加 ``__tool_spec__`` 属性，注册表据此收集工具，
    因此不影响函数被当作普通函数直接调用（便于单元测试）。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        spec = build_tool_spec(
            func,
            name=name,
            description=description,
            dangerous=dangerous,
            timeout=timeout,
        )
        func.__tool_spec__ = spec  # type: ignore[attr-defined]
        return func

    if _func is not None:
        return decorator(_func)
    return decorator
