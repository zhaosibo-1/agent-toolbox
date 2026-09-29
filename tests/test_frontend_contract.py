"""前后端字段契约。

这类问题在项目 1 里真实发生过一次：前端读 ``cfg.embedding.dim``，
后端却没返回这个字段，界面上只显示了一个「维」字，肉眼很难发现。

与其靠人工对齐，不如让测试去读前端源码，把它引用的每个字段
和真实接口返回的字段做个差集——差集非空就说明契约破了。

为了让静态扫描可靠，前端刻意给每个接口载荷用了**不会重名**的变量：
``cfg`` / ``tool`` / ``stats`` / ``stat`` / ``ev`` / ``res`` / ``ping``。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from helpers import make_settings

HTML_PATH = Path(__file__).resolve().parent.parent / "web" / "index.html"


def html_source() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def js_source() -> str:
    """只取 <script> 里的代码。

    必须排除 CSS：样式表里的 ``.tool.off`` / ``.tool.expanded`` 这类类名选择器
    会被字段扫描误当成 ``tool.off`` 字段访问。
    """
    blocks = re.findall(r"<script>([\s\S]*?)</script>", html_source())
    assert blocks, "web/index.html 里找不到内联脚本"
    return "\n".join(blocks)


def fields_used(pattern: str) -> Set[str]:
    """从内联脚本里扫出符合某个访问模式的所有字段名。"""
    return set(re.findall(pattern, js_source()))


def parse_sse(raw: str) -> List[Tuple[str, Dict[str, Any]]]:
    events: List[Tuple[str, Dict[str, Any]]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name, data = "", ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        if name and data and data != "[DONE]":
            events.append((name, json.loads(data)))
    return events


@pytest.fixture
def client(monkeypatch: Any, tmp_path: Path) -> Any:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.txt").write_bytes("沙箱说明文件\n".encode("utf-8"))

    settings = make_settings(allowed_file_dirs=str(docs), max_steps=4)
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    with TestClient(main_module.app) as test_client:
        yield test_client


def assert_covered(prefix: str, payload_groups: List[Dict[str, Any]]) -> None:
    """断言前端读的 ``prefix.*`` 字段都能在至少一组真实载荷里找到。"""
    used = fields_used(rf"\b{prefix}\.(\w+)")
    assert used, f"前端没有再读 {prefix}.* —— 契约测试需要同步更新"
    available: Set[str] = set()
    for group in payload_groups:
        available |= set(group)
    missing = used - available
    assert not missing, f"{prefix} 载荷缺少这些字段：{sorted(missing)}"


# ===========================================================================
def test_frontend_html_exists() -> None:
    assert HTML_PATH.is_file(), "web/index.html 必须存在，否则首页就是 404"


def test_config_fields_used_by_frontend_exist(client: TestClient) -> None:
    body = client.get("/api/config").json()

    for group in ("llm", "agent", "safety"):
        used = fields_used(rf"cfg\.{group}\.(\w+)")
        assert used, f"前端没有再读 cfg.{group}.* —— 契约测试需要同步更新"
        missing = used - set(body[group])
        assert not missing, f"/api/config 的 {group} 缺少字段：{sorted(missing)}"

    top_level = fields_used(r"cfg\.(\w+)(?!\w*\.)")
    assert top_level <= set(body), f"/api/config 缺少顶层字段：{sorted(top_level - set(body))}"


def test_tool_fields_used_by_frontend_exist(client: TestClient) -> None:
    tools = client.get("/api/tools").json()["tools"]
    assert tools
    assert_covered("tool", tools)


def test_stats_fields_used_by_frontend_exist(client: TestClient) -> None:
    stats = client.get("/api/stats").json()
    assert_covered("stats", [stats])

    per_tool = stats["per_tool"]
    assert per_tool
    assert_covered("stat", per_tool)


def test_sse_event_fields_used_by_frontend_exist(client: TestClient) -> None:
    """把 SSE 事件的字段并集和前端读的字段比一比。"""
    queries = [
        "读取文件 examples/note.txt，然后统计它的字数",
        "帮我算 (3+5)*2/4",
        "你好呀",
    ]
    groups: List[Dict[str, Any]] = []
    for query in queries:
        raw = client.post("/api/chat/stream", json={"message": query}).text
        groups.extend(payload for _, payload in parse_sse(raw))

    assert groups, "SSE 没有产出任何事件"
    assert_covered("ev", groups)


def test_tool_run_response_fields(client: TestClient) -> None:
    ok_group = client.post("/api/tools/calculator/run", json={"expression": "1+1"}).json()
    err_group = client.post("/api/tools/calculator/run", json={"expression": "1/0"}).json()
    assert_covered("res", [ok_group, err_group])


def test_ping_response_fields(client: TestClient) -> None:
    assert_covered("ping", [client.get("/api/llm/ping").json()])


def test_frontend_references_all_key_endpoints() -> None:
    html = html_source()
    for endpoint in (
        "/api/config",
        "/api/tools",
        "/api/stats",
        "/api/chat/stream",
        "/api/llm/ping",
    ):
        assert endpoint in html, f"前端没有调用 {endpoint}"


def test_frontend_declares_dark_theme_variables() -> None:
    """页面必须是深色主题：背景深、正文浅。"""
    html = html_source()
    for variable in ("--bg:", "--panel:", "--text:", "--border:", "--accent:"):
        assert variable in html, f"缺少主题变量 {variable}"
