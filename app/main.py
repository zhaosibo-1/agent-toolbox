"""FastAPI 服务入口。

对外提供四类能力：

1. **Agent 运行**：``/api/chat``（一次性 JSON）与 ``/api/chat/stream``（SSE，
   实时推送「思考 / 调用 / 观察 / 答复」全轨迹）；
2. **工具自省**：``/api/tools`` 把自动生成的 JSON Schema 暴露给前端，让界面上的
   工具清单永远和代码保持一致；
3. **单工具调试**：``/api/tools/{name}/run`` 跳过模型直接调用工具，排查问题时非常省事；
4. **运行状态**：``/api/health``、``/api/config``、``/api/stats``。
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .agent import AgentRunner
from .config import PROJECT_ROOT, Settings, get_settings
from .llm import LLMClient
from .schemas import (
    AgentStep,
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    HealthResponse,
    StatsResponse,
    ToolCallStat,
    ToolInfo,
    ToolListResponse,
)
from .tools import ToolError
from .tools import builtin as builtin_tools
from .tools.registry import ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("agent-toolbox")

WEB_DIR = PROJECT_ROOT / "web"
ALLOWED_UPLOAD_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log", ".yaml", ".yml"}


# ===========================================================================
# 应用装配
# ===========================================================================
def build_registry(settings: Settings) -> ToolRegistry:
    """装配工具注册表，并把安全边界注入内置工具。"""
    # 相对路径按项目根目录解析，避免受启动时工作目录影响
    allowed_dirs = [
        str((PROJECT_ROOT / d).resolve()) if not Path(d).is_absolute() else d
        for d in settings.allowed_dir_list
    ]
    builtin_tools.configure(
        allowed_dirs=allowed_dirs,
        http_hosts=settings.http_host_list,
        calc_max_length=settings.calc_max_length,
        project_root=PROJECT_ROOT,
    )
    registry = ToolRegistry()
    count = registry.register_module(builtin_tools)
    # 统一应用 TOOL_TIMEOUT，让「单工具超时」真的可配置，而不是写在装饰器里改不动
    for spec in registry.all():
        spec.timeout = settings.tool_timeout
    logger.info("已注册 %d 个工具：%s", count, ", ".join(registry.names()))
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    registry = build_registry(settings)
    llm = LLMClient(settings)

    app.state.settings = settings
    app.state.registry = registry
    app.state.llm = llm
    app.state.started_at = time.time()

    mode = "离线演示模式（未配置 LLM_API_KEY）" if llm.is_offline else settings.resolved_llm_model
    logger.info("Agent Toolbox 就绪 | 模型：%s | 通道：%s", mode, settings.agent_mode)
    try:
        yield
    finally:
        logger.info("Agent Toolbox 已关闭")


app = FastAPI(
    title="Agent Toolbox",
    description="手写 Function Calling 循环的 Agent 工具箱：函数签名即工具 Schema",
    version=__version__,
    lifespan=lifespan,
)

_settings_for_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings_for_cors.cors_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _registry() -> ToolRegistry:
    return app.state.registry


def _settings() -> Settings:
    return app.state.settings


# ===========================================================================
# 状态类接口
# ===========================================================================
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    llm: LLMClient = app.state.llm
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_enabled=not llm.is_offline,
        tool_total=len(_registry()),
        offline=llm.is_offline,
    )


@app.get("/api/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    llm: LLMClient = app.state.llm
    described = _settings().describe()
    return ConfigResponse(
        llm=described["llm"],
        agent=described["agent"],
        safety=described["safety"],
        tools_unsupported=llm.tools_unsupported,
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    registry = _registry()
    llm: LLMClient = app.state.llm
    specs = registry.all()
    return StatsResponse(
        tool_total=len(specs),
        tool_calls_total=sum(s.call_count for s in specs),
        per_tool=[
            ToolCallStat(name=s.name, call_count=s.call_count, dangerous=s.dangerous)
            for s in specs
        ],
        llm_enabled=not llm.is_offline,
        mode=_settings().agent_mode,
    )


# ===========================================================================
# 工具
# ===========================================================================
@app.get("/api/tools", response_model=ToolListResponse)
async def list_tools() -> ToolListResponse:
    registry = _registry()
    infos = [
        ToolInfo(
            name=item["name"],
            description=item["description"],
            params=item["params"],
            required=item["required"],
            dangerous=item["dangerous"],
            timeout=item["timeout"],
            call_count=item["call_count"],
            parameters=registry.get(item["name"]).parameters,
        )
        for item in registry.describe()
    ]
    return ToolListResponse(total=len(infos), tools=infos)


@app.post("/api/tools/{name}/run")
async def run_tool(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """直接调用某个工具（跳过模型），用于调试与演示。

    请求体就是工具的参数字典，例如 ``{"expression": "sqrt(2)*100"}``。
    """
    registry = _registry()
    if not registry.has(name):
        raise HTTPException(status_code=404, detail=f"工具 {name} 不存在")

    started = time.perf_counter()
    try:
        result = await registry.aexecute(name, payload or {})
    except ToolError as exc:
        # 工具层面的失败不算服务错误，返回 200 + error 字段，便于前端原样展示
        return {
            "ok": False,
            "tool": name,
            "error": str(exc),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    return {
        "ok": True,
        "tool": name,
        "result": result,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }


# ===========================================================================
# Agent
# ===========================================================================
def _make_runner() -> AgentRunner:
    return AgentRunner(_registry(), _settings(), app.state.llm)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """非流式执行一次任务。"""
    runner = _make_runner()
    result = await runner.arun(
        request.message,
        history=[m.model_dump() for m in request.history],
        tool_names=request.tools,
        max_steps=request.max_steps,
    )
    if result.error and not result.answer:
        raise HTTPException(status_code=502, detail=result.error)

    return ChatResponse(
        answer=result.answer,
        steps=[AgentStep(**step) for step in result.steps],
        steps_used=result.steps_used,
        mode=result.mode,
        offline=app.state.llm.is_offline,
        elapsed_ms=result.elapsed_ms,
        tools_called=result.tools_called,
        usage=result.usage,
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE 流式执行，实时推送完整轨迹。"""
    runner = _make_runner()
    history = [m.model_dump() for m in request.history]

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in runner.astream(
                request.message,
                history=history,
                tool_names=request.tools,
                max_steps=request.max_steps,
            ):
                yield _sse(event["type"], event)
        except Exception as exc:  # noqa: BLE001 - 任何异常都要转成 error 事件，别让连接悬着
            logger.exception("SSE 处理异常")
            yield _sse("error", {"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        yield "event: [DONE]\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉 Nginx 之类的缓冲，否则流式会被攒成一坨
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event_type: str, payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type}\ndata: {data}\n\n"


# ===========================================================================
# 示例文件（给文件类工具提供可读的内容）
# ===========================================================================
@app.get("/api/examples")
async def list_examples() -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for directory in _settings().allowed_dir_list:
        target = (PROJECT_ROOT / directory).resolve() if not Path(directory).is_absolute() else Path(directory)
        if not target.is_dir():
            continue
        for path in sorted(target.iterdir()):
            if path.is_file():
                files.append(
                    {
                        "name": path.name,
                        "rel": f"{directory.rstrip('/')}/{path.name}",
                        "size": path.stat().st_size,
                    }
                )
    return {"total": len(files), "files": files}


@app.post("/api/examples/upload")
async def upload_example(file: UploadFile = File(...)) -> Dict[str, Any]:
    """上传一个小文本文件到允许目录，方便现场演示「Agent 读取本地文件」。

    文件名会做白名单校验，且始终落在第一个允许目录内——上传接口本身就是一个
    攻击面，不能因为「演示用」就省掉校验。
    """
    settings = _settings()
    if not settings.allowed_dir_list:
        raise HTTPException(status_code=400, detail="未配置 ALLOWED_FILE_DIRS，文件功能已禁用")

    raw_name = Path(file.filename or "").name
    safe_name = re.sub(r"[^\w\u4e00-\u9fff.\-]+", "_", raw_name).lstrip(".")
    if not safe_name:
        raise HTTPException(status_code=400, detail="文件名不合法")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"只允许上传 {'/'.join(sorted(ALLOWED_UPLOAD_SUFFIXES))} 类型的文件",
        )

    payload = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(payload) > limit:
        raise HTTPException(status_code=413, detail=f"文件过大（上限 {settings.max_upload_mb}MB）")

    first_dir = settings.allowed_dir_list[0]
    target_dir = (PROJECT_ROOT / first_dir).resolve() if not Path(first_dir).is_absolute() else Path(first_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_name
    target.write_bytes(payload)

    logger.info("已保存示例文件 %s（%d 字节）", target, len(payload))
    return {
        "ok": True,
        "name": safe_name,
        "rel": f"{first_dir.rstrip('/')}/{safe_name}",
        "size": len(payload),
    }


@app.get("/api/llm/ping")
async def llm_ping() -> Dict[str, Any]:
    return await app.state.llm.ping()


# ===========================================================================
# 静态页面
# ===========================================================================
if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="web/index.html 不存在")
    return FileResponse(page, media_type="text/html; charset=utf-8")
