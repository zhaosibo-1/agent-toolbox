"""接口数据模型（Pydantic）。

前后端之间只通过这里定义的结构通信。把契约集中在一个文件里，
好处是**前端读什么字段，后端就必须返回什么字段**，不会出现
「界面上显示空白、查半天发现后端少返回一个 key」这类问题。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# 工具
# ===========================================================================
class ToolInfo(BaseModel):
    """单个工具的描述信息。"""

    name: str
    description: str
    #: 一行人类可读的参数签名，例如 `expression: string, precision?: integer`
    params: str
    required: List[str] = Field(default_factory=list)
    dangerous: bool = False
    timeout: float = 15.0
    call_count: int = 0
    #: 自动生成的原始 JSON Schema，前端可展开查看
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolListResponse(BaseModel):
    total: int
    tools: List[ToolInfo] = Field(default_factory=list)


# ===========================================================================
# 对话
# ===========================================================================
class ChatMessage(BaseModel):
    """历史消息（仅保留 user / assistant 两种角色）。"""

    role: str
    content: str = ""

    @field_validator("role")
    @classmethod
    def _check_role(cls, v: str) -> str:
        role = (v or "").strip().lower()
        if role not in {"user", "assistant"}:
            raise ValueError("role 只能是 user 或 assistant")
        return role


class ChatRequest(BaseModel):
    """一次 Agent 运行请求。"""

    message: str = Field(min_length=1, max_length=8000)
    history: List[ChatMessage] = Field(default_factory=list)
    #: 只启用这些工具；为空表示全部启用（用于演示「按场景裁剪工具集」）
    tools: Optional[List[str]] = None
    #: 覆盖默认的最大步数
    max_steps: Optional[int] = Field(default=None, ge=1, le=30)
    #: 是否流式返回（SSE）
    stream: bool = True


class AgentStep(BaseModel):
    """Agent 运行过程中的一条轨迹记录。"""

    index: int = 0
    #: thought / tool_call / tool_result / final / error
    type: str
    content: str = ""
    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    dangerous: bool = False


class ChatResponse(BaseModel):
    """非流式返回。"""

    answer: str
    steps: List[AgentStep] = Field(default_factory=list)
    steps_used: int = 0
    mode: str = "function_call"
    offline: bool = False
    elapsed_ms: float = 0.0
    tools_called: List[str] = Field(default_factory=list)
    usage: Dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# 配置 / 状态
# ===========================================================================
class ConfigResponse(BaseModel):
    llm: Dict[str, Any]
    agent: Dict[str, Any]
    safety: Dict[str, Any]
    tools_unsupported: bool = False


class ToolCallStat(BaseModel):
    name: str
    call_count: int
    dangerous: bool = False


class StatsResponse(BaseModel):
    tool_total: int
    tool_calls_total: int
    per_tool: List[ToolCallStat] = Field(default_factory=list)
    llm_enabled: bool = False
    mode: str = "function_call"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    llm_enabled: bool
    tool_total: int
    offline: bool
