"""Agent 主循环：规划 → 调用 → 观察 → 再规划，直到收敛。

这里是整个项目的「骨架」。刻意**不套 LangChain**，因为这条循环只有两百行，
自己写出来能把每一处细节都讲清楚：

- **步数上限**：``MAX_STEPS`` 硬性截断，避免模型陷入「调工具 → 报错 → 再调同一个」的死循环；
- **错误回灌**：工具抛出的异常不会中断任务，而是作为 ``role=tool`` 消息喂回模型，
  它往往能自己修正参数重试——这是 Agent 鲁棒性最关键的一环；
- **并行执行**：同一轮里互不依赖的工具调用并发跑，不比逐个串行慢；
- **轨迹可观测**：每一步都以事件形式 yield 出去，前端能实时看到「思考 / 调用 / 观察」，
  而不是盯着一个转圈等最终答案；
- **超时保护**：每个工具单独立超时，一个工具卡死不会拖垮整个会话。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from .config import Settings
from .llm import LLMClient, LLMError, ToolCall
from .tools.base import ToolError
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

#: 系统提示词（Function Calling 通道）
SYSTEM_PROMPT = """你是一个严谨的任务型智能体，可以通过调用工具来获取信息或完成计算。

工作原则：
1. 需要外部事实、精确计算或文件内容时，必须调用工具，不要凭记忆猜测。
2. 一次可以调用多个互不依赖的工具，拿到结果后再决定下一步。
3. 工具返回错误时，仔细阅读错误信息，修正参数后重试；不要用同样的参数重复调用。
4. 最终答复必须基于工具返回的真实结果。信息不足就直说缺什么，绝对不要编造。
5. 用简体中文回答，简洁、直接，先给结论再给必要的细节。"""

#: ReAct 模板（给不支持 function calling 的模型用）
REACT_TEMPLATE = """你是一个严谨的任务型智能体。你可以调用下列工具：

{tool_list}

请严格按下面的格式一步步推理：

Thought: 你的思考过程
Action: 工具名
Action Input: {{"参数名": "参数值"}}

然后系统会返回：
Observation: 工具执行结果

你可以继续 Thought / Action / Action Input，直到信息足够回答，此时输出：

Thought: 我已经掌握足够信息
Final Answer: 最终答案

硬性要求：
1. Action Input 必须是合法 JSON 对象，字符串值用双引号。
2. 一次只调用一个工具。
3. Final Answer 里不要出现 Observation 中没有的事实。
4. 用简体中文回答。"""

#: 超过步数上限后强制收尾的提示词
FORCE_FINAL_PROMPT = (
    "已达到本次任务的最大工具调用轮数。请立即基于以上所有工具返回的结果给出最终答复；"
    "如果信息仍然不足，请明确说明还缺什么，不要再请求调用工具。"
)

#: 历史消息最多保留的条数（避免上下文无限膨胀）
MAX_HISTORY_MESSAGES = 12
#: 回灌给模型的工具结果截断长度
MAX_TOOL_RESULT_CHARS = 6000
#: 推给前端的工具结果截断长度
MAX_EVENT_RESULT_CHARS = 3000

#: ReAct 输出的解析正则
# 关键点：`Final Answer` 中间是**有空格**的。用共享片段拼装可以避免
# 「终结符写错一个空格、lookahead 一路滑到文本末尾」这种隐蔽故障。
_REACT_ACTION_KW = r"Action(?:\s*Input)?"
_REACT_FINAL_KW = r"Final\s*Answer"
_REACT_STOP = rf"(?=\n\s*(?:{_REACT_ACTION_KW}|{_REACT_FINAL_KW})\s*[:：]|\Z)"

_REACT_FINAL_RE = re.compile(rf"{_REACT_FINAL_KW}\s*[:：]\s*([\s\S]+)", re.I)
_REACT_ACTION_RE = re.compile(r"^[ \t>*-]*Action\s*[:：]\s*([A-Za-z_][\w\-]*)", re.I | re.M)
_REACT_ACTION_LINE_RE = re.compile(r"^[ \t>*-]*Action(?:\s*Input)?\s*[:：].*$", re.I | re.M)
_REACT_INPUT_RE = re.compile(rf"Action\s*Input\s*[:：]\s*([\s\S]*?){_REACT_STOP}", re.I)
_REACT_THOUGHT_RE = re.compile(rf"Thought\s*[:：]\s*([\s\S]*?){_REACT_STOP}", re.I)


@dataclass
class AgentResult:
    """一次完整运行的产物。"""

    answer: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "function_call"
    elapsed_ms: float = 0.0
    steps_used: int = 0
    tools_called: List[str] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class _RunContext:
    """一次运行的可变状态。集中管理，免得在方法之间传六七个参数。"""

    __slots__ = ("steps", "tools_called", "usage", "steps_used")

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.tools_called: List[str] = []
        self.usage: Dict[str, Any] = {}
        self.steps_used: int = 0

    def reset(self) -> None:
        self.steps.clear()
        self.tools_called.clear()
        self.usage.clear()
        self.steps_used = 0

    def add_step(self, **kwargs: Any) -> Dict[str, Any]:
        step = {"index": len(self.steps) + 1, **kwargs}
        self.steps.append(step)
        return step


class AgentRunner:
    """把「模型 + 工具注册表」组装成一个能跑的 Agent。"""

    def __init__(self, registry: ToolRegistry, settings: Settings, llm: LLMClient) -> None:
        self.registry = registry
        self.settings = settings
        self.llm = llm
        #: 本次会话实际使用的通道，可能因自动降级而与配置不同
        self.active_mode = self._resolve_mode()

    # ------------------------------------------------------------------
    def _resolve_mode(self) -> str:
        """把配置里的 ``auto`` 解析成实际通道。"""
        configured = self.settings.agent_mode
        if configured == "auto":
            return "react" if self.llm.tools_unsupported else "function_call"
        return configured

    # ==================================================================
    # 对外入口
    # ==================================================================
    async def astream(
        self,
        message: str,
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        tool_names: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """执行一次任务，逐步产出事件。

        ==============  ==================================================
        ``start``       开始，带上实际通道、启用工具、步数上限
        ``thinking``    即将向模型发起一次推理（第 N 轮）
        ``thought``     模型在两轮工具调用之间输出的中间想法
        ``tool_call``   请求调用某个工具（含参数、是否危险、超时）
        ``tool_result`` 工具返回（含耗时；失败时带 ``error``）
        ``fallback``    服务端不支持 tools，自动降级到 ReAct 通道
        ``delta``       最终答复的增量文本
        ``done``        结束，带上完整轨迹与统计
        ``error``       致命错误
        ==============  ==================================================
        """
        started = time.perf_counter()
        ctx = _RunContext()
        answer_parts: List[str] = []
        enabled = self._enabled_tool_names(tool_names)
        step_limit = max_steps or self.settings.max_steps

        yield {
            "type": "start",
            "mode": self.active_mode,
            "offline": self.llm.is_offline,
            "tools": enabled,
            "max_steps": step_limit,
        }

        try:
            if self.active_mode == "react":
                async for event in self._pump(
                    self._run_react(message, history, enabled, step_limit, ctx), answer_parts
                ):
                    yield event
            else:
                try:
                    async for event in self._pump(
                        self._run_function_call(message, history, enabled, step_limit, ctx),
                        answer_parts,
                    ):
                        yield event
                except LLMError as exc:
                    # auto 模式下服务端不认 tools 参数 → 换文本协议再来一遍
                    if self.settings.agent_mode == "auto" and exc.tools_unsupported:
                        logger.warning("服务端不支持 function calling，降级到 ReAct：%s", exc)
                        self.active_mode = "react"
                        ctx.reset()
                        answer_parts.clear()
                        yield {"type": "fallback", "mode": "react", "reason": str(exc)}
                        async for event in self._pump(
                            self._run_react(message, history, enabled, step_limit, ctx),
                            answer_parts,
                        ):
                            yield event
                    else:
                        raise

        except LLMError as exc:
            logger.warning("模型调用失败：%s", exc)
            ctx.add_step(type="error", content=str(exc))
            yield {"type": "error", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - 兜底，保证 SSE 一定收尾
            logger.exception("Agent 运行异常")
            ctx.add_step(type="error", content=f"{type(exc).__name__}: {exc}")
            yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}

        elapsed = (time.perf_counter() - started) * 1000
        yield {
            "type": "done",
            "answer": "".join(answer_parts),
            "steps": ctx.steps,
            "steps_used": ctx.steps_used,
            "mode": self.active_mode,
            "offline": self.llm.is_offline,
            "elapsed_ms": round(elapsed, 1),
            "tools_called": list(ctx.tools_called),
            "usage": dict(ctx.usage),
        }

    async def arun(
        self,
        message: str,
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        tool_names: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ) -> AgentResult:
        """非流式执行，把事件流收敛成一个结果对象。"""
        result = AgentResult(mode=self.active_mode)
        async for event in self.astream(
            message, history=history, tool_names=tool_names, max_steps=max_steps
        ):
            kind = event["type"]
            if kind == "done":
                result.answer = event["answer"]
                result.steps = event["steps"]
                result.mode = event["mode"]
                result.elapsed_ms = event["elapsed_ms"]
                result.steps_used = event["steps_used"]
                result.tools_called = event["tools_called"]
                result.usage = event["usage"]
            elif kind == "error":
                result.error = event["error"]
        return result

    # ==================================================================
    # 事件转发
    # ==================================================================
    async def _pump(
        self, source: AsyncIterator[Dict[str, Any]], answer_parts: List[str]
    ) -> AsyncIterator[Dict[str, Any]]:
        """转发子通道的事件，同时把 ``delta`` 拼成完整答复。"""
        async for event in source:
            if event["type"] == "delta":
                answer_parts.append(event["content"])
            yield event

    # ==================================================================
    # Function Calling 通道
    # ==================================================================
    async def _run_function_call(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]],
        enabled: List[str],
        step_limit: int,
        ctx: _RunContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        messages = self._build_messages(message, history)
        schemas = self.registry.openai_schemas(only=enabled)
        final_content = ""
        exhausted = True

        for round_index in range(step_limit):
            ctx.steps_used = round_index + 1
            yield {"type": "thinking", "index": ctx.steps_used}

            response = await self.llm.chat(messages, tools=schemas)
            if response.usage:
                ctx.usage.update(response.usage)

            # 模型一边调工具一边解释思路时，把这段想法也记进轨迹
            if response.content and response.tool_calls:
                thought = response.content.strip()
                ctx.add_step(type="thought", content=thought)
                yield {"type": "thought", "content": thought}

            if not response.tool_calls:
                final_content = response.content
                exhausted = False
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [call.to_message_part() for call in response.tool_calls],
                }
            )

            # 先把「要调什么」全部抛出去，界面立刻有反馈；同轮工具并行执行
            for call in response.tool_calls:
                yield self._build_call_step(call, ctx)

            outcomes = await asyncio.gather(
                *(self._invoke_tool(call, ctx) for call in response.tool_calls)
            )
            for call, (payload, event) in zip(response.tool_calls, outcomes):
                yield event
                messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})

            # 关闭结果回灌 = 只跑一轮工具就直接收尾（省 token 的取舍）。
            # 注意工具结果仍留在 messages 里供最后那次生成使用，只是不再多绕一轮推理。
            if not self.settings.enable_tool_result_feedback:
                exhausted = True
                break

        # --- 收尾 ---
        if exhausted:
            messages.append({"role": "user", "content": FORCE_FINAL_PROMPT})
            async for chunk in self.llm.stream(messages):
                yield {"type": "delta", "content": chunk}
        elif final_content:
            # 完整答复已经在手，按小块回放给前端，保证打字机效果一致
            for chunk in _split(final_content, 24):
                yield {"type": "delta", "content": chunk}
        else:
            # 模型只给了空答复，再催一次（这一次走真流式）
            async for chunk in self.llm.stream(messages):
                yield {"type": "delta", "content": chunk}

    # ==================================================================
    # ReAct 通道
    # ==================================================================
    async def _run_react(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]],
        enabled: List[str],
        step_limit: int,
        ctx: _RunContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": REACT_TEMPLATE.format(tool_list=self._tool_listing(enabled))}
        ]
        messages.extend(self._sanitize_history(history))
        messages.append({"role": "user", "content": message})

        for round_index in range(step_limit):
            ctx.steps_used = round_index + 1
            yield {"type": "thinking", "index": ctx.steps_used}

            # ReAct 要拿到完整文本才能解析格式，所以这一步不走流式
            response = await self.llm.chat(messages, tools=None)
            transcript = response.content

            thought = self._extract_thought(transcript)

            final_match = _REACT_FINAL_RE.search(transcript)
            if final_match:
                if thought:
                    ctx.add_step(type="thought", content=thought)
                    yield {"type": "thought", "content": thought}
                for chunk in _split(final_match.group(1).strip(), 24):
                    yield {"type": "delta", "content": chunk}
                return

            action_match = _REACT_ACTION_RE.search(transcript)
            if not action_match:
                # 模型没按格式来：把这轮文本当成答案，别让整轮白跑
                ctx.add_step(type="final", content=transcript.strip()[:2000])
                for chunk in _split(transcript, 24):
                    yield {"type": "delta", "content": chunk}
                return

            if thought:
                ctx.add_step(type="thought", content=thought)
                yield {"type": "thought", "content": thought}

            call = ToolCall(
                id=f"react_{round_index}",
                name=action_match.group(1),
                arguments=self._extract_action_input(transcript),
            )
            yield self._build_call_step(call, ctx)
            payload, event = await self._invoke_tool(call, ctx)
            yield event

            messages.append({"role": "assistant", "content": transcript})
            messages.append({"role": "user", "content": f"Observation: {payload}"})

            if not self.settings.enable_tool_result_feedback:
                break

        # 步数用尽 → 强制收尾
        messages.append({"role": "user", "content": FORCE_FINAL_PROMPT})
        response = await self.llm.chat(messages, tools=None)
        for chunk in _split(self._extract_final_text(response.content), 24):
            yield {"type": "delta", "content": chunk}

    # ==================================================================
    # 工具执行（两个通道共用）
    # ==================================================================
    def _build_call_step(self, call: ToolCall, ctx: _RunContext) -> Dict[str, Any]:
        """登记并返回 ``tool_call`` 事件。"""
        spec = self.registry.get(call.name) if self.registry.has(call.name) else None
        arguments = call.parsed_arguments()
        dangerous = bool(spec.dangerous) if spec else False

        ctx.add_step(
            type="tool_call", tool=call.name, arguments=arguments, dangerous=dangerous
        )
        return {
            "type": "tool_call",
            "index": len(ctx.steps),
            "tool": call.name,
            "arguments": arguments,
            "dangerous": dangerous,
            "timeout": spec.timeout if spec else None,
            "known": spec is not None,
        }

    async def _invoke_tool(
        self, call: ToolCall, ctx: _RunContext
    ) -> "tuple[str, Dict[str, Any]]":
        """执行工具，返回 ``(回灌给模型的内容, tool_result 事件)``。

        这里**绝不向外抛异常**：工具失败也是一种「观察结果」，要喂回模型让它自己纠正。
        """
        error: Optional[str] = None
        arguments = call.parsed_arguments()
        started = time.perf_counter()

        if not self.registry.has(call.name):
            error = f"工具 {call.name} 不存在。可用工具：{', '.join(self.registry.names())}"
            payload = f"错误：{error}"
        else:
            try:
                payload = await self.registry.aexecute(call.name, arguments)
            except ToolError as exc:
                error = str(exc)
                payload = f"错误：{error}"
            except Exception as exc:  # noqa: BLE001 - 注册表理论上已兜住，这里再保一层
                error = f"{type(exc).__name__}: {exc}"
                payload = f"错误：{error}"

        duration = round((time.perf_counter() - started) * 1000, 1)
        if error is None:
            ctx.tools_called.append(call.name)

        ctx.add_step(
            type="tool_result",
            tool=call.name,
            result=payload[:MAX_EVENT_RESULT_CHARS],
            error=error,
            duration_ms=duration,
        )
        return (
            payload[:MAX_TOOL_RESULT_CHARS],
            {
                "type": "tool_result",
                "index": len(ctx.steps),
                "tool": call.name,
                "result": payload[:MAX_EVENT_RESULT_CHARS],
                "error": error,
                "duration_ms": duration,
            },
        )

    # ==================================================================
    # 消息组装
    # ==================================================================
    def _build_messages(
        self, message: str, history: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._sanitize_history(history))
        messages.append({"role": "user", "content": message})
        return messages

    @staticmethod
    def _sanitize_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """只保留最近若干条 user/assistant 消息，并过滤掉空内容。"""
        if not history:
            return []
        cleaned: List[Dict[str, Any]] = []
        for item in history:
            role = str(item.get("role", "")).lower()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                cleaned.append({"role": role, "content": content})
        return cleaned[-MAX_HISTORY_MESSAGES:]

    def _enabled_tool_names(self, requested: Optional[List[str]]) -> List[str]:
        known = self.registry.names()
        if not requested:
            return known
        # 只保留真实存在的工具名，顺序以注册顺序为准
        wanted = {name for name in requested if name in known}
        return [name for name in known if name in wanted]

    def _tool_listing(self, enabled: List[str]) -> str:
        lines = []
        for name in enabled:
            spec = self.registry.get(name)
            lines.append(f"- {spec.name}({spec.describe_params()})：{spec.description}")
        return "\n".join(lines) or "(当前没有可用工具，请直接回答用户问题)"

    @staticmethod
    def _extract_thought(transcript: str) -> str:
        match = _REACT_THOUGHT_RE.search(transcript)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_final_text(transcript: str) -> str:
        """从模型的最后一轮输出里抠出正文。

        优先取 ``Final Answer``；如果模型还在固执地输出 ``Action``，说明它没听懂
        「步数已用完」——此时把 Action 行剔掉，只留它能给出的正文，
        总比把一段无法执行的协议文本原样丢给用户要好。
        """
        final_match = _REACT_FINAL_RE.search(transcript)
        if final_match:
            return final_match.group(1).strip()
        cleaned = _REACT_ACTION_LINE_RE.sub("", transcript).strip()
        return cleaned or transcript.strip()

    @staticmethod
    def _extract_action_input(transcript: str) -> str:
        """抽出 ``Action Input`` 后面的 JSON，容忍 ```json 代码块包裹。"""
        match = _REACT_INPUT_RE.search(transcript)
        if not match:
            return "{}"
        raw = match.group(1).strip()
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        return raw or "{}"


# ===========================================================================
# 辅助
# ===========================================================================
def _split(text: str, size: int) -> List[str]:
    """把文本切成小块，用于前端的打字机效果。"""
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]
