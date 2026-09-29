"""Agent 主循环：项目里最容易出 bug、也最值得被测透的地方。

用脚本化模型把每条分支都钉死：

- 正常收敛（工具 → 最终答复）
- 工具报错后把错误回灌给模型（Agent 鲁棒性的关键）
- 模型编了不存在的工具名 / 参数不是合法 JSON
- 步数用尽时强制收尾，不能无限循环
- ReAct 文本协议通道
- 服务端不支持 function calling 时自动降级
- 纯离线（无 API Key）也能端到端跑完
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.agent import AgentRunner
from app.llm import LLMClient, LLMError, LLMResponse
from app.tools.registry import ToolRegistry
from helpers import (
    ScriptedLLM,
    event_types,
    make_settings,
    scripted_answer,
    scripted_call,
    scripted_multi_calls,
    scripted_raw_arguments,
    scripted_react,
    scripted_react_final,
)


def runner(registry: ToolRegistry, llm: Any, **settings_overrides: Any) -> AgentRunner:
    return AgentRunner(registry, make_settings(**settings_overrides), llm)


async def collect(agent: AgentRunner, message: str, **kwargs: Any) -> List[Dict[str, Any]]:
    return [event async for event in agent.astream(message, **kwargs)]


# ===========================================================================
# Function Calling 通道
# ===========================================================================
async def test_happy_path_single_tool(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_call("calculator", {"expression": "2+3"}), scripted_answer("答案是 5")])
    agent = runner(registry, llm)
    result = await agent.arun("算一下 2+3")

    assert result.answer == "答案是 5"
    assert result.tools_called == ["calculator"]
    assert result.steps_used == 2
    assert result.mode == "function_call"
    assert result.error is None
    assert [s["type"] for s in result.steps] == ["tool_call", "tool_result"]
    assert result.steps[1]["result"] == "5"


async def test_multi_step_chain(registry: ToolRegistry) -> None:
    """模型可以先调一个工具，拿到结果后再调另一个。"""
    llm = ScriptedLLM(
        [
            scripted_call("calculator", {"expression": "6*7"}, "c1"),
            scripted_call("current_time", {}, "c2"),
            scripted_answer("乘积 42，时间已获取"),
        ]
    )
    agent = runner(registry, llm)
    result = await agent.arun("先算 6*7 再告诉我时间")

    assert result.tools_called == ["calculator", "current_time"]
    assert result.steps_used == 3
    assert result.answer == "乘积 42，时间已获取"


async def test_parallel_tool_calls_in_one_round(registry: ToolRegistry) -> None:
    """同一轮里的多个工具调用会被并行执行，事件里都能看到。"""
    llm = ScriptedLLM(
        [
            scripted_multi_calls(
                [("calculator", {"expression": "1+1"}), ("calculator", {"expression": "2+2"})]
            ),
            scripted_answer("都算完了"),
        ]
    )
    agent = runner(registry, llm)
    result = await agent.arun("算两个式子")

    assert result.answer == "都算完了"
    assert len([s for s in result.steps if s["type"] == "tool_call"]) == 2
    assert len([s for s in result.steps if s["type"] == "tool_result"]) == 2


async def test_intermediate_thought_is_recorded(registry: ToolRegistry) -> None:
    """模型一边调工具一边解释思路时，这段想法要出现在轨迹里。"""
    llm = ScriptedLLM(
        [
            scripted_multi_calls([("calculator", {"expression": "2+2"})], content="我需要先算一下"),
            scripted_answer("4"),
        ]
    )
    agent = runner(registry, llm)
    result = await agent.arun("2+2")

    assert result.steps[0]["type"] == "thought"
    assert result.steps[0]["content"] == "我需要先算一下"


async def test_tool_error_is_fed_back_to_model(registry: ToolRegistry) -> None:
    """工具报错不该中断任务，而要作为 Observation 回灌，让模型自己纠正。

    这是 Agent 鲁棒性最关键的一环，必须测到。
    """
    llm = ScriptedLLM(
        [
            scripted_call("calculator", {"expression": "1/0"}),
            scripted_answer("除零了，换个算法"),
        ]
    )
    agent = runner(registry, llm)
    result = await agent.arun("算 1/0")

    # 第二轮请求里应该能看到错误信息
    second_call_messages = llm.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert tool_messages, "模型没有收到工具结果回灌"
    assert "除数不能为零" in tool_messages[0]["content"]

    assert result.answer == "除零了，换个算法"
    assert "calculator" not in result.tools_called  # 失败的调用不计入成功列表
    assert result.steps[1]["error"] is not None


async def test_unknown_tool_is_reported_but_not_fatal(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_call("magic_tool", {}), scripted_answer("好的，我自己算")])
    agent = runner(registry, llm)
    result = await agent.arun("用魔法工具")

    assert result.error is None
    error = result.steps[1]["error"]
    assert error and "不存在" in error
    assert "可用工具" in error  # 错误里要带上可用清单，模型才知道改用哪个
    assert result.answer == "好的，我自己算"


async def test_malformed_json_arguments_are_tolerated(registry: ToolRegistry) -> None:
    """模型把参数写成非法 JSON 时，要变成一条可诊断的错误，而不是崩溃。"""
    llm = ScriptedLLM(
        [
            scripted_raw_arguments("calculator", "{这根本不是 JSON"),
            scripted_answer("我重新组织一下参数"),
        ]
    )
    agent = runner(registry, llm)
    result = await agent.arun("算一下")

    assert result.error is None
    assert result.steps[1]["error"] is not None
    # 参数解析失败 → 空字典 → 注册表报「缺少必填参数」
    assert "缺少必填参数" in result.steps[1]["error"]


async def test_max_steps_forces_final_answer(registry: ToolRegistry) -> None:
    """步数用尽必须强制收尾，绝不能无限循环。"""
    llm = ScriptedLLM(
        [scripted_call("calculator", {"expression": "1+1"})] * 5,
        stream_text="强制收尾：我已经算完了",
    )
    agent = runner(registry, llm, max_steps=2)
    result = await agent.arun("一直算下去")

    assert result.steps_used == 2
    assert result.answer == "强制收尾：我已经算完了"


async def test_disabling_result_feedback_stops_after_one_round(registry: ToolRegistry) -> None:
    """关掉结果回灌 = 只跑一轮工具就收尾（省 token 的取舍）。

    工具结果仍然要留在上下文里供最后那次生成使用，只是不再多绕一轮推理。
    """
    llm = ScriptedLLM([scripted_call("calculator", {"expression": "3+4"})], stream_text="答案是 7")
    agent = runner(registry, llm, enable_tool_result_feedback=False)
    result = await agent.arun("3+4")

    assert result.steps_used == 1
    assert result.tools_called == ["calculator"]
    assert result.answer == "答案是 7"

    # 第 2 次调用是流式收尾，且上下文里能看到工具结果
    assert len(llm.calls) == 2
    assert llm.calls[1].get("stream") is True
    assert any(m.get("role") == "tool" and "7" in str(m.get("content")) for m in llm.calls[1]["messages"])


async def test_llm_error_produces_error_event(registry: ToolRegistry) -> None:
    class BrokenLLM(ScriptedLLM):
        async def chat(self, messages: List[Dict[str, Any]], tools: Any = None, **kwargs: Any) -> Any:
            raise LLMError("网络不可达")

    agent = runner(registry, BrokenLLM())
    events = await collect(agent, "随便问点什么")

    assert "error" in event_types(events)
    done = next(e for e in events if e["type"] == "done")
    assert done["answer"] == ""
    assert done["steps"][-1]["type"] == "error"


# ===========================================================================
# 事件流形态
# ===========================================================================
async def test_event_sequence(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_call("calculator", {"expression": "1+1"}), scripted_answer("2")])
    agent = runner(registry, llm)
    events = await collect(agent, "1+1")

    types = event_types(events)
    assert types[0] == "start"
    assert "thinking" in types
    assert types.index("tool_call") < types.index("tool_result")
    assert "delta" in types
    assert types[-1] == "done"


async def test_start_event_payload(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_answer("好")])
    agent = runner(registry, llm, max_steps=6)
    events = await collect(agent, "你好")
    start = events[0]

    assert start["mode"] == "function_call"
    assert start["max_steps"] == 6
    assert "calculator" in start["tools"]


async def test_done_event_payload(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_call("calculator", {"expression": "3*4"}), scripted_answer("12")])
    agent = runner(registry, llm)
    events = await collect(agent, "3*4")
    done = events[-1]

    assert done["type"] == "done"
    assert done["answer"] == "12"
    assert done["tools_called"] == ["calculator"]
    assert done["elapsed_ms"] >= 0
    assert done["offline"] is False


async def test_deltas_reassemble_into_answer(registry: ToolRegistry) -> None:
    long_answer = "这是一段比较长的答复，" * 5
    llm = ScriptedLLM([scripted_answer(long_answer)])
    agent = runner(registry, llm)
    events = await collect(agent, "说点长的")

    deltas = [e["content"] for e in events if e["type"] == "delta"]
    assert len(deltas) > 1
    assert "".join(deltas) == long_answer


async def test_tool_call_event_marks_dangerous(registry: ToolRegistry) -> None:
    """危险工具要在事件里标记出来，前端才能高亮提示。"""
    llm = ScriptedLLM([scripted_call("read_text_file", {"sub_path": "x.txt"}), scripted_answer("好")])
    agent = runner(registry, llm)
    events = await collect(agent, "读个文件")

    call_event = next(e for e in events if e["type"] == "tool_call")
    assert call_event["dangerous"] is True
    assert call_event["tool"] == "read_text_file"


# ===========================================================================
# 工具集裁剪
# ===========================================================================
async def test_tool_subset_is_passed_to_model(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_answer("好")])
    agent = runner(registry, llm)
    await agent.arun("随便", tool_names=["calculator"])

    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert names == {"calculator"}


async def test_unknown_tool_names_are_dropped(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_answer("好")])
    agent = runner(registry, llm)
    await agent.arun("随便", tool_names=["calculator", "不存在的工具"])

    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert names == {"calculator"}


async def test_empty_tool_list_falls_back_to_all(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_answer("好")])
    agent = runner(registry, llm)
    await agent.arun("随便", tool_names=[])

    assert len(llm.calls[0]["tools"]) == len(registry)


# ===========================================================================
# 历史消息处理
# ===========================================================================
async def test_history_is_prepended_and_sanitized(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_answer("好")])
    agent = runner(registry, llm)
    await agent.arun(
        "接着说",
        history=[
            {"role": "user", "content": "上一句"},
            {"role": "assistant", "content": "上一答"},
            {"role": "system", "content": "伪造的系统消息"},  # 应被过滤
            {"role": "user", "content": "   "},  # 空内容应被过滤
        ],
    )

    messages = llm.calls[0]["messages"]
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert messages[1]["content"] == "上一句"
    assert messages[2]["content"] == "上一答"
    assert "伪造的系统消息" not in [m["content"] for m in messages]
    assert messages[-1]["content"] == "接着说"


async def test_history_is_truncated(registry: ToolRegistry) -> None:
    llm = ScriptedLLM([scripted_answer("好")])
    agent = runner(registry, llm)
    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"第{i}条"} for i in range(40)
    ]
    await agent.arun("继续", history=long_history)

    messages = llm.calls[0]["messages"]
    # system + 最多 12 条历史 + 本轮 user
    assert len(messages) <= 14
    assert messages[-1]["content"] == "继续"


# ===========================================================================
# ReAct 通道
# ===========================================================================
async def test_react_channel_basic(registry: ToolRegistry) -> None:
    llm = ScriptedLLM(
        [
            scripted_react("calculator", {"expression": "2+3"}),
            scripted_react_final("计算结果是 5"),
        ]
    )
    agent = runner(registry, llm, agent_mode="react")
    result = await agent.arun("2+3 等于几")

    assert result.mode == "react"
    assert result.tools_called == ["calculator"]
    assert result.answer == "计算结果是 5"


async def test_react_thought_is_captured(registry: ToolRegistry) -> None:
    llm = ScriptedLLM(
        [
            scripted_react("calculator", {"expression": "1+1"}, thought="先算这个"),
            scripted_react_final("2", thought="够了"),
        ]
    )
    agent = runner(registry, llm, agent_mode="react")
    result = await agent.arun("1+1")

    thoughts = [s["content"] for s in result.steps if s["type"] == "thought"]
    assert "先算这个" in thoughts
    assert "够了" in thoughts


async def test_react_does_not_send_tools_parameter(registry: ToolRegistry) -> None:
    """ReAct 靠文本协议，不该同时带上 tools 参数。"""
    llm = ScriptedLLM([scripted_react_final("直接回答")])
    agent = runner(registry, llm, agent_mode="react")
    await agent.arun("你好")

    assert llm.calls[0]["tools"] is None
    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "calculator" in system_prompt


async def test_react_observation_is_sent_back(registry: ToolRegistry) -> None:
    llm = ScriptedLLM(
        [
            scripted_react("calculator", {"expression": "6*7"}),
            scripted_react_final("42"),
        ]
    )
    agent = runner(registry, llm, agent_mode="react")
    await agent.arun("6*7")

    second_messages = llm.calls[1]["messages"]
    # 注意：系统提示词里也含 "Observation:" 这个词，所以要按角色过滤
    observations = [
        m["content"]
        for m in second_messages
        if m.get("role") == "user" and str(m.get("content", "")).startswith("Observation:")
    ]
    assert observations and "42" in observations[0]


async def test_react_handles_json_code_fence(registry: ToolRegistry) -> None:
    """模型常把 Action Input 包在 ```json 里，解析要能容忍。"""
    llm = ScriptedLLM(
        [
            LLMResponse(
                content='Thought: 算一下\nAction: calculator\nAction Input: ```json\n{"expression": "8*8"}\n```',
                finish_reason="stop",
            ),
            scripted_react_final("64"),
        ]
    )
    agent = runner(registry, llm, agent_mode="react")
    result = await agent.arun("8*8")

    assert result.tools_called == ["calculator"]
    assert result.answer == "64"


async def test_react_without_action_treats_output_as_answer(registry: ToolRegistry) -> None:
    """模型没按格式来的时候，可用的文本也不该浪费。"""
    llm = ScriptedLLM([LLMResponse(content="我直接回答：这是答案", finish_reason="stop")])
    agent = runner(registry, llm, agent_mode="react")
    result = await agent.arun("随便")

    assert result.answer == "我直接回答：这是答案"
    assert result.tools_called == []


async def test_react_max_steps_forces_final(registry: ToolRegistry) -> None:
    """ReAct 通道步数用尽时，也要收敛到一个答复。"""
    llm = ScriptedLLM(
        [
            scripted_react("calculator", {"expression": "1+1"}),
            scripted_react("calculator", {"expression": "2+2"}),
            scripted_answer("兜底答复"),
        ]
    )
    agent = runner(registry, llm, agent_mode="react", max_steps=2)
    result = await agent.arun("一直算")

    assert result.steps_used == 2
    assert result.answer == "兜底答复"


async def test_react_forced_final_strips_stubborn_action(registry: ToolRegistry) -> None:
    """模型不听话、收尾时还在输出 Action，也不能把协议文本原样丢给用户。"""
    llm = ScriptedLLM(
        [
            scripted_react("calculator", {"expression": "1+1"}),
            scripted_react("calculator", {"expression": "2+2"}),
            scripted_react("calculator", {"expression": "3+3"}, thought="我再试一次"),
        ]
    )
    agent = runner(registry, llm, agent_mode="react", max_steps=2)
    result = await agent.arun("一直算")

    assert "Action" not in result.answer
    assert "我再试一次" in result.answer


# ===========================================================================
# 自动降级
# ===========================================================================
async def test_auto_falls_back_to_react(registry: ToolRegistry) -> None:
    """服务端不认 tools 参数时，auto 模式要自动切到 ReAct 重跑。"""
    llm = ScriptedLLM([scripted_react_final("降级后的答复")], fail_with_tools=True)
    agent = runner(registry, llm, agent_mode="auto")
    events = await collect(agent, "你好")

    assert "fallback" in event_types(events)
    done = next(e for e in events if e["type"] == "done")
    assert done["mode"] == "react"
    assert done["answer"] == "降级后的答复"
    # 第一轮带 tools（→失败），第二轮不带 tools（→成功）
    assert llm.calls[0]["tools"] is not None
    assert llm.calls[1]["tools"] is None


async def test_function_call_mode_does_not_fall_back(registry: ToolRegistry) -> None:
    """显式指定 function_call 时不该偷偷降级，而要把错误暴露出来。"""
    llm = ScriptedLLM([], fail_with_tools=True)
    agent = runner(registry, llm, agent_mode="function_call")
    events = await collect(agent, "你好")

    assert "fallback" not in event_types(events)
    assert "error" in event_types(events)
    done = next(e for e in events if e["type"] == "done")
    assert done["mode"] == "function_call"


# ===========================================================================
# 离线模式端到端
# ===========================================================================
async def test_offline_end_to_end_math(registry: ToolRegistry) -> None:
    """没有 API Key 也要能把整条链路跑通——这是可复现性的基础。"""
    settings = make_settings()
    llm = LLMClient(settings)
    assert llm.is_offline is True

    agent = AgentRunner(registry, settings, llm)
    result = await agent.arun("帮我算一下 (3+5)*2/4")

    assert result.tools_called == ["calculator"]
    assert "4" in result.answer
    assert "离线演示模式" in result.answer


async def test_offline_end_to_end_multi_tool(registry: ToolRegistry) -> None:
    settings = make_settings()
    agent = AgentRunner(registry, settings, LLMClient(settings))
    result = await agent.arun("现在几点了，顺便帮我算 12*8")

    assert set(result.tools_called) == {"calculator", "current_time"}
    assert "96" in result.answer


async def test_offline_file_tool_error_is_reported(registry: ToolRegistry, sandbox: Any) -> None:
    """离线模式下读一个不存在的文件，错误也要如实呈现。"""
    settings = make_settings(allowed_file_dirs=str(sandbox))
    agent = AgentRunner(registry, settings, LLMClient(settings))
    result = await agent.arun("读取文件 nope.txt")

    errors = [s for s in result.steps if s["type"] == "tool_result" and s["error"]]
    assert errors, "应当记录到一次失败的文件读取"
    assert "文件不存在" in errors[0]["error"]


async def test_offline_unknown_intent_returns_guide(registry: ToolRegistry) -> None:
    settings = make_settings()
    agent = AgentRunner(registry, settings, LLMClient(settings))
    result = await agent.arun("今天天气怎么样")

    assert result.tools_called == []
    assert "离线演示模式" in result.answer
    assert "LLM_API_KEY" in result.answer


async def test_offline_uses_real_tools_not_fake_output(registry: ToolRegistry) -> None:
    """离线模式只是「规划」是规则的，工具执行必须是真实的。"""
    settings = make_settings()
    agent = AgentRunner(registry, settings, LLMClient(settings))
    result = await agent.arun("把 12 km 换算成 mile")

    assert result.tools_called == ["convert_unit"]
    assert "7.456" in result.answer  # 12 / 1.609344


async def test_offline_chains_file_read_into_statistics(
    registry: ToolRegistry, sandbox: Any
) -> None:
    """多步感知：读文件之后统计，统计的应该是**文件内容**而不是用户问句。

    否则「先读文件再统计」就成了一句废话——这条用例守住这个语义。
    """
    settings = make_settings(allowed_file_dirs=str(sandbox))
    agent = AgentRunner(registry, settings, LLMClient(settings))
    result = await agent.arun("读取文件 note.txt，然后统计它的字数")

    assert result.tools_called == ["read_text_file", "text_statistics"]

    stats = next(s for s in result.steps if s["tool"] == "text_statistics" and s["type"] == "tool_result")
    # 要求字符数**严格**相等：文件末尾的换行也必须原样传下去，
    # 步骤之间流转的数据不能被静默 strip。
    expected = len((sandbox / "note.txt").read_bytes().decode("utf-8"))
    assert f"总字符数：{expected}" in stats["result"]

    # 统计对象是文件正文，不应该出现只在用户问句里才有的词
    assert "沙箱" in stats["result"]
    assert "note" not in stats["result"]


# ===========================================================================
# 非流式结果对象
# ===========================================================================
async def test_arun_collects_steps_and_usage(registry: ToolRegistry) -> None:
    llm = ScriptedLLM(
        [
            scripted_multi_calls([("calculator", {"expression": "1+1"})]),
            LLMResponse(content="2", usage={"total_tokens": 10}, finish_reason="stop"),
        ]
    )
    agent = runner(registry, llm)
    result = await agent.arun("1+1")

    assert result.usage.get("total_tokens") == 10  # 最后一次调用的用量
    assert len(result.steps) == 2


async def test_arun_records_error(registry: ToolRegistry) -> None:
    class BrokenLLM(ScriptedLLM):
        async def chat(self, messages: List[Dict[str, Any]], tools: Any = None, **kwargs: Any) -> Any:
            raise LLMError("炸了")

    agent = runner(registry, BrokenLLM())
    result = await agent.arun("你好")

    assert result.error == "炸了"
    assert result.answer == ""


@pytest.mark.parametrize(
    "message,expected_tool",
    [
        ("帮我算 100*log10(1000)", "calculator"),
        ("现在几点了", "current_time"),
        ("目录下有哪些文件", "list_directory"),
        ("把 1 kg 换算成 pound", "convert_unit"),
        ("把 12 km 换算成 mile", "convert_unit"),
        ("把 100 公顷 = 多少 亩", "convert_unit"),
    ],
)
async def test_offline_intent_routing(
    registry: ToolRegistry, sandbox: Any, message: str, expected_tool: str
) -> None:
    """离线规划器的「意图 → 工具」路由。"""
    settings = make_settings(allowed_file_dirs=str(sandbox))
    agent = AgentRunner(registry, settings, LLMClient(settings))
    result = await agent.arun(message)

    assert result.tools_called == [expected_tool], result.steps
