"""工具 Schema 自动生成：这是项目的核心卖点，必须有足够的用例守住。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import pytest

from app.tools.base import ToolError, ToolSpec, build_tool_spec, tool


# ===========================================================================
# 基础类型映射
# ===========================================================================
def test_primitive_type_mapping() -> None:
    def fn(text: str, count: int, ratio: float, enabled: bool) -> str:
        """演示函数。

        Args:
            text: 文本参数。
            count: 整数参数。
            ratio: 浮点参数。
            enabled: 布尔参数。
        """
        return text

    spec = build_tool_spec(fn)
    props = spec.parameters["properties"]
    assert props["text"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["enabled"]["type"] == "boolean"


def test_all_parameters_required_without_default() -> None:
    def fn(a: str, b: int) -> str:
        """无默认值。

        Args:
            a: 参数 a。
            b: 参数 b。
        """
        return a

    spec = build_tool_spec(fn)
    assert spec.required_params() == ["a", "b"]


def test_default_value_makes_parameter_optional() -> None:
    def fn(a: str, b: int = 3) -> str:
        """带默认值。

        Args:
            a: 必填。
            b: 选填。
        """
        return a

    spec = build_tool_spec(fn)
    assert spec.required_params() == ["a"]
    assert spec.parameters["properties"]["b"]["default"] == 3


def test_optional_annotation_is_not_required() -> None:
    def fn(a: str, b: Optional[str] = None, c: Optional[int] = None) -> str:
        """可空参数。

        Args:
            a: 必填。
            b: 可空字符串。
            c: 可空整数。
        """
        return a

    spec = build_tool_spec(fn)
    # 有默认值时本来就不进 required；这里主要确认可空标注不会报错
    assert spec.required_params() == ["a"]
    assert spec.parameters["properties"]["b"]["type"] == "string"
    assert spec.parameters["properties"]["c"]["type"] == "integer"


def test_optional_without_default_is_not_required() -> None:
    """``Optional[X]`` 没有默认值时，语义上允许 None，不该要求模型必填。"""

    def fn(a: str, b: Optional[str]) -> str:
        """可空无默认。

        Args:
            a: 必填。
            b: 可空。
        """
        return a

    spec = build_tool_spec(fn)
    assert spec.required_params() == ["a"]


def test_list_type_mapping() -> None:
    def fn(tags: List[str], nums: list[int]) -> str:
        """数组参数。

        Args:
            tags: 字符串数组。
            nums: 整数数组。
        """
        return ",".join(tags)

    spec = build_tool_spec(fn)
    assert spec.parameters["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "字符串数组。",
    }
    assert spec.parameters["properties"]["nums"]["items"]["type"] == "integer"


def test_dict_type_mapping() -> None:
    def fn(payload: Dict[str, Any]) -> str:
        """对象参数。

        Args:
            payload: 任意对象。
        """
        return str(payload)

    spec = build_tool_spec(fn)
    assert spec.parameters["properties"]["payload"]["type"] == "object"


def test_literal_becomes_enum() -> None:
    def fn(mode: Literal["fast", "slow"]) -> str:
        """枚举参数。

        Args:
            mode: 模式。
        """
        return mode

    spec = build_tool_spec(fn)
    prop = spec.parameters["properties"]["mode"]
    assert prop["type"] == "string"
    assert prop["enum"] == ["fast", "slow"]


def test_unknown_type_falls_back_to_string() -> None:
    """没见过的类型不能让整个工具注册失败，退化为 string 更稳妥。"""

    def fn(value: complex) -> str:
        """未知类型。

        Args:
            value: 复数。
        """
        return str(value)

    spec = build_tool_spec(fn)
    assert spec.parameters["properties"]["value"]["type"] == "string"


def test_varargs_are_ignored() -> None:
    def fn(a: str, *args: int, **kwargs: str) -> str:
        """可变参数。

        Args:
            a: 必填。
        """
        return a

    spec = build_tool_spec(fn)
    assert set(spec.parameters["properties"]) == {"a"}


# ===========================================================================
# 文档与描述
# ===========================================================================
def test_docstring_summary_becomes_description() -> None:
    def fn(a: str) -> str:
        """这是简介第一行。
        这是简介第二行。

        Args:
            a: 参数说明。
        """
        return a

    spec = build_tool_spec(fn)
    assert "这是简介第一行" in spec.description
    assert "这是简介第二行" in spec.description


def test_docstring_args_are_extracted() -> None:
    def fn(expression: str, precision: int = 4) -> str:
        """计算表达式。

        Args:
            expression: 待计算的表达式。
            precision: 保留的小数位数。
        """
        return expression

    spec = build_tool_spec(fn)
    assert spec.param_docs["expression"] == "待计算的表达式。"
    assert spec.parameters["properties"]["expression"]["description"] == "待计算的表达式。"
    assert spec.parameters["properties"]["precision"]["description"] == "保留的小数位数。"


def test_explicit_description_overrides_docstring() -> None:
    def fn(a: str) -> str:
        """原始简介。

        Args:
            a: 参数。
        """
        return a

    spec = build_tool_spec(fn, description="自定义描述")
    assert spec.description == "自定义描述"


def test_missing_args_section_is_tolerated() -> None:
    def fn(a: str) -> str:
        """没有 Args 段的函数。"""
        return a

    spec = build_tool_spec(fn)
    assert "description" not in spec.parameters["properties"]["a"]
    assert spec.description == "没有 Args 段的函数。"


# ===========================================================================
# OpenAI schema 形态
# ===========================================================================
def test_openai_schema_shape() -> None:
    def fn(a: str) -> str:
        """示例。

        Args:
            a: 参数。
        """
        return a

    spec = build_tool_spec(fn, name="demo_tool")
    schema = spec.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "demo_tool"
    assert set(schema["function"]) == {"name", "description", "parameters"}


def test_additional_properties_is_disabled() -> None:
    """禁止模型编造不存在的参数，能在入口挡掉一类幻觉。"""

    def fn(a: str) -> str:
        """示例。

        Args:
            a: 参数。
        """
        return a

    spec = build_tool_spec(fn)
    assert spec.parameters["additionalProperties"] is False


def test_describe_params_readable() -> None:
    def fn(a: str, b: int = 1) -> str:
        """示例。

        Args:
            a: 必填。
            b: 选填。
        """
        return a

    spec = build_tool_spec(fn)
    assert spec.describe_params() == "a: string, b?: integer"


def test_describe_params_without_arguments() -> None:
    def fn() -> str:
        """无参函数。"""
        return "ok"

    spec = build_tool_spec(fn)
    assert spec.describe_params() == "(无参数)"


# ===========================================================================
# 装饰器
# ===========================================================================
def test_tool_decorator_attaches_spec() -> None:
    @tool
    def demo(a: str) -> str:
        """装饰器示例。

        Args:
            a: 参数。
        """
        return a

    assert isinstance(demo.__tool_spec__, ToolSpec)
    assert demo.__tool_spec__.name == "demo"


def test_tool_decorator_with_options() -> None:
    @tool(name="renamed", description="描述", dangerous=True, timeout=3.5)
    def demo(a: str) -> str:
        """原简介。

        Args:
            a: 参数。
        """
        return a

    spec = demo.__tool_spec__
    assert spec.name == "renamed"
    assert spec.description == "描述"
    assert spec.dangerous is True
    assert spec.timeout == 3.5


def test_decorated_function_still_callable() -> None:
    """装饰器不能破坏原函数，否则单元测试就没法直接调了。"""

    @tool
    def add(a: int, b: int) -> int:
        """求和。

        Args:
            a: 加数。
            b: 加数。
        """
        return a + b

    assert add(2, 3) == 5


# ===========================================================================
# 调用与超时
# ===========================================================================
def test_invoke_wraps_type_error() -> None:
    def fn(a: str) -> str:
        """示例。

        Args:
            a: 参数。
        """
        return a

    spec = build_tool_spec(fn)
    with pytest.raises(ToolError) as excinfo:
        spec.invoke({"b": "x"})
    assert "参数不正确" in str(excinfo.value)


def test_invoke_wraps_arbitrary_exception() -> None:
    def fn(a: str) -> str:
        """示例。

        Args:
            a: 参数。
        """
        raise ValueError("内部炸了")

    spec = build_tool_spec(fn)
    with pytest.raises(ToolError) as excinfo:
        spec.invoke({"a": "x"})
    assert "ValueError" in str(excinfo.value)
    assert "内部炸了" in str(excinfo.value)


def test_invoke_passes_tool_error_through() -> None:
    """工具自己抛的 ToolError 要原样上抛，不能被再包一层。"""

    def fn(a: str) -> str:
        """示例。

        Args:
            a: 参数。
        """
        raise ToolError("我自己的错误")

    spec = build_tool_spec(fn)
    with pytest.raises(ToolError) as excinfo:
        spec.invoke({"a": "x"})
    assert str(excinfo.value) == "我自己的错误"


def test_invoke_counts_calls() -> None:
    def fn(a: str) -> str:
        """示例。

        Args:
            a: 参数。
        """
        return a

    spec = build_tool_spec(fn)
    spec.invoke({"a": "1"})
    spec.invoke({"a": "2"})
    assert spec.call_count == 2


async def test_ainvoke_timeout() -> None:
    """慢工具必须被超时掐断，否则会拖垮整个会话。"""

    import time

    def slow(seconds: float = 1.0) -> str:
        """慢工具。

        Args:
            seconds: 睡眠秒数。
        """
        time.sleep(seconds)
        return "done"

    spec = build_tool_spec(slow, timeout=0.05)
    with pytest.raises(ToolError) as excinfo:
        await spec.ainvoke({"seconds": 1.0})
    assert "执行超时" in str(excinfo.value)


async def test_ainvoke_success() -> None:
    def fast(a: str) -> str:
        """快工具。

        Args:
            a: 参数。
        """
        return f"收到 {a}"

    spec = build_tool_spec(fast)
    assert await spec.ainvoke({"a": "x"}) == "收到 x"
