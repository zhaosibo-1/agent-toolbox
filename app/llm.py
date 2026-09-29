"""大模型客户端：一套代码对接所有 OpenAI 兼容服务。

设计要点：

- **只走 OpenAI 兼容协议**：DeepSeek / 通义 / Kimi / 智谱 / SiliconFlow / Ollama
  全部通用，换厂商只改 ``.env``，不改一行代码。
- **双通道**：`function_call` 走原生 tools 协议；`react` 走文本协议
  （``Thought / Action / Action Input / Final Answer``），用于那些不支持
  function calling 的开源模型。
- **自动降级**：``AGENT_MODE=auto`` 时优先试 function calling，一旦服务端
  明确回「不认识 tools 参数」，就自动切到 ReAct，并把这件事作为事件推给前端。
- **离线可跑**：没配 Key 时交给 :class:`~app.offline.OfflinePlanner`，
  整条 Agent 链路依然可复现。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .config import Settings
from .offline import OfflinePlanner

logger = logging.getLogger(__name__)

#: 单次模型调用的超时（秒）
_REQUEST_TIMEOUT = 120.0


class LLMError(RuntimeError):
    """模型调用失败。

    Args:
        tools_unsupported: 服务端是否明确表示不支持 ``tools`` 参数。
            为 True 时上层会考虑从 function calling 降级到 ReAct。
    """

    def __init__(self, message: str, *, tools_unsupported: bool = False) -> None:
        super().__init__(message)
        self.tools_unsupported = tools_unsupported


@dataclass
class ToolCall:
    """模型请求的一次工具调用。"""

    id: str
    name: str
    #: 原始 JSON 字符串（保留原样，便于把解析错误也当作可诊断信息回灌给模型）
    arguments: str

    def parsed_arguments(self) -> Dict[str, Any]:
        """尽力把参数解析成字典；解析不出来就返回空字典（由注册表统一报错）。"""
        if not self.arguments.strip():
            return {}
        try:
            parsed = json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def to_message_part(self) -> Dict[str, Any]:
        """还原成 OpenAI assistant 消息里的 tool_calls 元素。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class LLMResponse:
    """一次模型返回。"""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    model: str = ""


class LLMClient:
    """OpenAI 兼容协议的对话客户端。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._offline = OfflinePlanner()
        #: function calling 被服务端拒绝过之后置为 True（auto 模式下会切 ReAct）
        self._tools_unsupported = False

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def is_offline(self) -> bool:
        """未配置 API Key → 离线演示模式。"""
        return not self.settings.llm_enabled

    @property
    def tools_unsupported(self) -> bool:
        return self._tools_unsupported

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """发一次对话请求（非流式）。

        Args:
            messages: OpenAI 格式的消息列表。
            tools: ``tools`` 参数；为空或 None 表示纯文本对话。
            temperature: 覆盖默认温度。
            max_tokens: 覆盖默认最大 token 数。

        Raises:
            LLMError: 网络错误、鉴权失败、返回体不合法等。
        """
        if self.is_offline:
            return self._offline.chat(messages, tools)

        payload: Dict[str, Any] = {
            "model": self.settings.resolved_llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": self.settings.llm_max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = await self._post(payload)
        return self._parse_response(data)

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """流式产出正文增量文本。

        工具调用不走流式（各家对 ``tool_calls`` 分片字段的实现差异太大，
        实用主义取舍：只在**最终答复**这一步用流式，换取稳定的解析）。
        """
        if self.is_offline:
            text = self._offline.final_text(messages)
            for chunk in _chunk(text, 24):
                yield chunk
            return

        payload: Dict[str, Any] = {
            "model": self.settings.resolved_llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.settings.resolved_llm_base_url}/chat/completions"
        headers = self._headers()

        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", errors="ignore")
                        raise self._http_error(response.status_code, body)

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if not chunk or chunk == "[DONE]":
                            continue
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        for choice in data.get("choices") or []:
                            delta = choice.get("delta") or {}
                            piece = delta.get("content")
                            if piece:
                                yield piece
        except httpx.HTTPError as exc:
            raise LLMError(f"流式请求失败：{type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------
    async def ping(self) -> Dict[str, Any]:
        """连通性自检，供前端「测试连接」按钮使用。"""
        if self.is_offline:
            return {
                "ok": True,
                "offline": True,
                "message": "未配置 LLM_API_KEY，当前为离线演示模式（规则化规划器 + 真实工具执行）",
            }
        try:
            response = await self.chat(
                [{"role": "user", "content": "ping"}], temperature=0.0, max_tokens=16
            )
        except LLMError as exc:
            return {"ok": False, "offline": False, "message": str(exc)}

        return {
            "ok": True,
            "offline": False,
            "message": f"连接正常：{self.settings.resolved_llm_model}",
            "sample": response.content[:80],
        }

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.llm_api_key.strip()}",
            "Content-Type": "application/json",
        }

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.settings.resolved_llm_base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise LLMError(f"请求大模型失败：{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            raise self._http_error(response.status_code, response.text)

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise LLMError(f"返回内容不是合法 JSON：{response.text[:200]}") from exc

    def _http_error(self, status: int, body: str) -> LLMError:
        """把 HTTP 错误翻译成人话，并判断是否属于「不支持 tools」。"""
        snippet = body.strip().replace("\n", " ")[:300]
        lowered = snippet.lower()
        tools_unsupported = status in (400, 404, 422) and any(
            keyword in lowered
            for keyword in ("tool", "function", "not supported", "unsupported", "unknown parameter")
        )
        if tools_unsupported:
            self._tools_unsupported = True

        hints = {
            401: "鉴权失败：请检查 LLM_API_KEY 是否正确、是否与 LLM_PRESET 匹配",
            403: "无权访问：该 Key 可能没有调用此模型的权限",
            404: "接口不存在：请检查 LLM_BASE_URL（很多厂商需要带 /v1）",
            429: "请求过于频繁或额度用尽",
        }
        hint = hints.get(status, "")
        message = f"大模型返回 HTTP {status}{('：' + hint) if hint else ''}。原始信息：{snippet}"
        return LLMError(message, tools_unsupported=tools_unsupported)

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"返回体中没有 choices 字段：{json.dumps(data, ensure_ascii=False)[:300]}")

        choice = choices[0]
        message = choice.get("message") or {}

        raw_calls = message.get("tool_calls") or []
        tool_calls: List[ToolCall] = []
        for index, raw in enumerate(raw_calls):
            func = raw.get("function") or {}
            name = func.get("name") or ""
            if not name:
                continue
            tool_calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{index}",
                    name=name,
                    arguments=func.get("arguments") or "{}",
                )
            )

        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            usage=data.get("usage") or {},
            finish_reason=choice.get("finish_reason") or "",
            model=data.get("model") or "",
        )


def _chunk(text: str, size: int) -> List[str]:
    """把文本切成小块，用于模拟打字机效果（离线模式复用同一套前端渲染）。"""
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]
