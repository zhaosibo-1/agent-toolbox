"""测试辅助工具：脚本化模型与消息构造器。

放在独立模块里，是为了让 conftest 只管夹具，测试文件按需 import，
避免「从 conftest 里 import 东西」这种容易踩坑的写法。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from app.config import Settings
from app.llm import LLMError, LLMResponse, ToolCall

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_settings(**overrides: Any) -> Settings:
    """构造一份测试配置：默认不带 API Key（即离线模式）。"""
    base: Dict[str, Any] = {
        "_env_file": None,
        "llm_api_key": "",
        "llm_preset": "deepseek",
        "allowed_file_dirs": str(REPO_ROOT / "examples"),
        "http_allowed_hosts": "example.com",
        "calc_max_length": 200,
        "max_steps": 4,
        "tool_timeout": 5.0,
    }
    base.update(overrides)
    return Settings(**base)


class ScriptedLLM:
    """按剧本返回结果的假模型。

    用来精确命中 Agent 主循环的每一条分支：正常收敛、工具报错后纠正、
    步数用尽强制收尾、服务端不支持 function calling 触发降级……
    这些路径用真实模型很难稳定复现，但恰恰是最容易出 bug 的地方。
    """

    def __init__(
        self,
        responses: Optional[List[LLMResponse]] = None,
        *,
        fail_with_tools: bool = False,
        stream_text: str = "这是流式最终答复。",
    ) -> None:
        self.responses: List[LLMResponse] = list(responses or [])
        self.fail_with_tools = fail_with_tools
        self.stream_text = stream_text
        #: 记录每次被调用的入参，便于断言「模型到底看到了什么」
        self.calls: List[Dict[str, Any]] = []
        self._tools_unsupported = False

    # --- 与 LLMClient 对齐的接口 ---
    @property
    def is_offline(self) -> bool:
        return False

    @property
    def tools_unsupported(self) -> bool:
        return self._tools_unsupported

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})

        if self.fail_with_tools and tools:
            self._tools_unsupported = True
            raise LLMError("该模型不支持 tools 参数", tools_unsupported=True)

        if not self.responses:
            return LLMResponse(content="（剧本已用尽）", finish_reason="stop")
        return self.responses.pop(0)

    async def stream(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[str]:
        self.calls.append({"messages": messages, "tools": tools, "stream": True})
        for index in range(0, len(self.stream_text), 8):
            yield self.stream_text[index : index + 8]
        await asyncio.sleep(0)

    async def ping(self) -> Dict[str, Any]:
        return {"ok": True, "offline": False, "message": "脚本化模型"}

    # --- 断言辅助 ---
    def last_messages(self) -> List[Dict[str, Any]]:
        return self.calls[-1]["messages"] if self.calls else []

    def saw_tool_message(self) -> bool:
        """模型是否收到过 ``role=tool`` 的结果回灌。"""
        return any(
            message.get("role") == "tool"
            for call in self.calls
            for message in call["messages"]
        )


def scripted_call(name: str, arguments: Dict[str, Any], call_id: str = "call_1") -> LLMResponse:
    """构造「模型要求调用某工具」的返回。"""
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id=call_id, name=name, arguments=json.dumps(arguments, ensure_ascii=False))
        ],
        finish_reason="tool_calls",
    )


def scripted_answer(text: str) -> LLMResponse:
    """构造「模型给出最终答复」的返回。"""
    return LLMResponse(content=text, finish_reason="stop")


def scripted_multi_calls(
    pairs: List[Any], content: str = ""
) -> LLMResponse:
    """构造「同一轮里并行调用多个工具」的返回。

    Args:
        pairs: ``[(工具名, 参数字典), ...]``
        content: 模型同时给出的思考文本。
    """
    calls = [
        ToolCall(
            id=f"call_{index}",
            name=name,
            arguments=json.dumps(args, ensure_ascii=False),
        )
        for index, (name, args) in enumerate(pairs)
    ]
    return LLMResponse(content=content, tool_calls=calls, finish_reason="tool_calls")


def scripted_raw_arguments(name: str, arguments: str, call_id: str = "call_raw") -> LLMResponse:
    """构造「参数不是合法 JSON」的返回，用于测试容错路径。"""
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        finish_reason="tool_calls",
    )


def scripted_react(action: str, arguments: Dict[str, Any], thought: str = "需要借助工具") -> LLMResponse:
    """构造一段 ReAct 格式的文本返回（含 Action / Action Input）。"""
    payload = json.dumps(arguments, ensure_ascii=False)
    text = f"Thought: {thought}\nAction: {action}\nAction Input: {payload}"
    return LLMResponse(content=text, finish_reason="stop")


def scripted_react_final(answer: str, thought: str = "信息已足够") -> LLMResponse:
    """构造 ReAct 格式的最终答复。"""
    return LLMResponse(content=f"Thought: {thought}\nFinal Answer: {answer}", finish_reason="stop")


def event_types(events: List[Dict[str, Any]]) -> List[str]:
    return [event["type"] for event in events]
