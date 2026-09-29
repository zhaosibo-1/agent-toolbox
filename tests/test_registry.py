"""工具注册表：收集、暴露、安全执行。"""

from __future__ import annotations

import json

import pytest

from app.tools import builtin as bt
from app.tools.base import ToolError, ToolSpec, build_tool_spec
from app.tools.registry import ToolRegistry

EXPECTED_TOOLS = {
    "calculator",
    "convert_unit",
    "current_time",
    "http_get",
    "json_query",
    "list_directory",
    "read_text_file",
    "text_statistics",
}


# ===========================================================================
# 注册
# ===========================================================================
def test_register_module_collects_all_tools(registry: ToolRegistry) -> None:
    assert set(registry.names()) == EXPECTED_TOOLS
    assert len(registry) == len(EXPECTED_TOOLS)


def test_register_module_ignores_plain_functions() -> None:
    """模块里没被 @tool 装饰的函数不应被误收。"""
    import types

    module = types.ModuleType("fake")

    def plain(a: str) -> str:
        """普通函数。"""
        return a

    plain.__module__ = "fake"
    module.plain = plain

    reg = ToolRegistry()
    assert reg.register_module(module) == 0
    assert len(reg) == 0


def test_register_preserves_order(registry: ToolRegistry) -> None:
    names = registry.names()
    assert names.index("calculator") < names.index("current_time")


def test_duplicate_registration_overwrites(registry: ToolRegistry) -> None:
    def fake_calculator(expression: str) -> str:
        """假的。

        Args:
            expression: 表达式。
        """
        return "假结果"

    spec = build_tool_spec(fake_calculator, name="calculator")
    registry.register(spec)
    assert len(registry) == len(EXPECTED_TOOLS)
    assert registry.get("calculator").func(expression="1+1") == "假结果"


def test_register_callable_plain_function(registry: ToolRegistry) -> None:
    def helper(a: str) -> str:
        """辅助。

        Args:
            a: 参数。
        """
        return a

    spec = registry.register_callable(helper)
    assert isinstance(spec, ToolSpec)
    assert "helper" in registry


def test_register_callable_reuses_existing_schema(registry: ToolRegistry) -> None:
    """已被 @tool 装饰过的函数应复用同一份 schema，避免两处定义分叉。"""
    spec = registry.register_callable(bt.calculator)
    assert spec.name == bt.calculator.__tool_spec__.name
    assert spec.parameters == bt.calculator.__tool_spec__.parameters


def test_registered_spec_is_a_copy_of_the_global_one(registry: ToolRegistry) -> None:
    """注册表持有的是副本：调用统计不能写回函数上的全局 spec。

    否则多个注册表（以及测试用例）之间会互相污染计数。
    """
    spec = registry.get("calculator")
    spec.invoke({"expression": "1+1"})
    assert spec.call_count == 1
    assert bt.calculator.__tool_spec__.call_count == 0


def test_unregister(registry: ToolRegistry) -> None:
    assert registry.unregister("calculator") is True
    assert "calculator" not in registry
    assert registry.unregister("calculator") is False


# ===========================================================================
# 查询与导出
# ===========================================================================
def test_get_unknown_tool_lists_available(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError) as excinfo:
        registry.get("nope")
    assert "可用工具" in str(excinfo.value)


def test_has_and_contains(registry: ToolRegistry) -> None:
    assert registry.has("calculator")
    assert "calculator" in registry
    assert not registry.has("nope")


def test_openai_schemas_shape(registry: ToolRegistry) -> None:
    schemas = registry.openai_schemas()
    assert len(schemas) == len(EXPECTED_TOOLS)
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["additionalProperties"] is False


def test_openai_schemas_filtered(registry: ToolRegistry) -> None:
    """按场景裁剪工具集：工具越少，模型选错的概率越低。"""
    schemas = registry.openai_schemas(only=["calculator", "current_time"])
    assert {s["function"]["name"] for s in schemas} == {"calculator", "current_time"}


def test_describe_shape(registry: ToolRegistry) -> None:
    described = registry.describe()
    item = next(d for d in described if d["name"] == "calculator")
    assert item["params"] == "expression: string, precision?: integer"
    assert item["required"] == ["expression"]
    assert item["dangerous"] is False


def test_describe_marks_dangerous_tools(registry: ToolRegistry) -> None:
    described = {d["name"]: d for d in registry.describe()}
    assert described["http_get"]["dangerous"] is True
    assert described["read_text_file"]["dangerous"] is True
    assert described["list_directory"]["dangerous"] is True


def test_calculator_schema_types_are_not_degraded(registry: ToolRegistry) -> None:
    """守住「所有参数都退化成 string」这类静默故障。"""
    props = registry.get("calculator").parameters["properties"]
    assert props["expression"]["type"] == "string"
    assert props["precision"]["type"] == "integer"

    props = registry.get("convert_unit").parameters["properties"]
    assert props["value"]["type"] == "number"


# ===========================================================================
# 执行
# ===========================================================================
async def test_aexecute_ok(registry: ToolRegistry) -> None:
    assert await registry.aexecute("calculator", {"expression": "2+3"}) == "5"


async def test_aexecute_accepts_json_string_arguments(registry: ToolRegistry) -> None:
    """有些模型会把参数塞成 JSON 字符串，必须兼容。"""
    assert await registry.aexecute("calculator", '{"expression": "2+3"}') == "5"


async def test_aexecute_accepts_none_arguments(registry: ToolRegistry) -> None:
    result = await registry.aexecute("current_time", None)
    assert "UTC" in result


async def test_aexecute_rejects_invalid_json(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError) as excinfo:
        await registry.aexecute("calculator", "{不是 json}")
    assert "不是合法 JSON" in str(excinfo.value)


async def test_aexecute_rejects_non_object_json(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError) as excinfo:
        await registry.aexecute("calculator", "[1, 2]")
    assert "必须是 JSON 对象" in str(excinfo.value)


async def test_aexecute_rejects_unsupported_argument_type(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError) as excinfo:
        await registry.aexecute("calculator", 42)
    assert "类型不支持" in str(excinfo.value)


async def test_aexecute_rejects_unknown_parameter(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError) as excinfo:
        await registry.aexecute("calculator", {"expression": "1+1", "mode": "fast"})
    assert "未知参数" in str(excinfo.value)
    assert "mode" in str(excinfo.value)


async def test_aexecute_rejects_missing_required(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError) as excinfo:
        await registry.aexecute("calculator", {"precision": 2})
    assert "缺少必填参数" in str(excinfo.value)


async def test_aexecute_rejects_blank_required(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError) as excinfo:
        await registry.aexecute("calculator", {"expression": "   "})
    assert "缺少必填参数" in str(excinfo.value)

async def test_aexecute_serializes_non_string_results() -> None:
    """工具返回 dict/list 时要转成 JSON 字符串再回灌给模型。"""

    def dict_tool(a: str) -> dict:
        """返回字典。

        Args:
            a: 参数。
        """
        return {"received": a, "n": 2}

    reg = ToolRegistry()
    reg.register_callable(dict_tool)
    result = await reg.aexecute("dict_tool", {"a": "x"})
    assert json.loads(result) == {"received": "x", "n": 2}


async def test_aexecute_increments_call_count(registry: ToolRegistry) -> None:
    await registry.aexecute("calculator", {"expression": "1+1"})
    await registry.aexecute("calculator", {"expression": "2+2"})
    assert registry.get("calculator").call_count == 2
