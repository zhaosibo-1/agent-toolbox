"""HTTP 接口层：契约、SSE 事件流、错误码。

这一层最容易出「前后端字段对不上」的问题，所以用例集中在**契约**上：
前端读哪个字段，后端就必须返回哪个字段；SSE 的事件名与顺序也不能变。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app import __version__
from helpers import make_settings


@pytest.fixture
def client(monkeypatch: Any, tmp_path: Path) -> Any:
    """带独立沙箱目录的测试客户端（避免往仓库的 examples/ 里写文件）。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.txt").write_text("沙箱里的说明文件。", encoding="utf-8")

    settings = make_settings(
        allowed_file_dirs=str(docs),
        http_allowed_hosts="example.com",
        max_steps=4,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    with TestClient(main_module.app) as test_client:
        yield test_client


def parse_sse(body: str) -> List[Tuple[str, Dict[str, Any]]]:
    """把 SSE 原始文本解析成 ``[(事件名, 载荷), ...]``。"""
    events: List[Tuple[str, Dict[str, Any]]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if not event_name:
            continue
        payload = json.loads(data) if data and data != "[DONE]" else {"raw": data}
        events.append((event_name, payload))
    return events


# ===========================================================================
# 状态类
# ===========================================================================
def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["tool_total"] == 8
    assert body["offline"] is True
    assert body["llm_enabled"] is False


def test_config_exposes_fields_used_by_frontend(client: TestClient) -> None:
    """契约测试：前端页面读的每个字段都必须存在。"""
    body = client.get("/api/config").json()

    assert set(body["llm"]) >= {"preset", "model", "base_url", "api_key", "enabled"}
    assert set(body["agent"]) >= {"mode", "max_steps", "tool_timeout", "result_feedback"}
    assert set(body["safety"]) >= {"allowed_file_dirs", "http_allowed_hosts"}
    assert body["tools_unsupported"] is False


def test_config_masks_api_key(client: TestClient) -> None:
    body = client.get("/api/config").json()
    assert body["llm"]["api_key"] == "(未配置)"


def test_stats(client: TestClient) -> None:
    body = client.get("/api/stats").json()
    assert body["tool_total"] == 8
    assert body["tool_calls_total"] == 0
    assert {item["name"] for item in body["per_tool"]} >= {"calculator", "http_get"}
    assert body["llm_enabled"] is False


def test_llm_ping_offline(client: TestClient) -> None:
    body = client.get("/api/llm/ping").json()
    assert body["ok"] is True
    assert body["offline"] is True
    assert "LLM_API_KEY" in body["message"]


# ===========================================================================
# 工具
# ===========================================================================
def test_list_tools(client: TestClient) -> None:
    body = client.get("/api/tools").json()
    assert body["total"] == 8

    names = {tool["name"] for tool in body["tools"]}
    assert names == {
        "calculator",
        "convert_unit",
        "current_time",
        "http_get",
        "json_query",
        "list_directory",
        "read_text_file",
        "text_statistics",
    }


def test_tool_info_shape(client: TestClient) -> None:
    tools = {tool["name"]: tool for tool in client.get("/api/tools").json()["tools"]}

    calc = tools["calculator"]
    assert calc["params"] == "expression: string, precision?: integer"
    assert calc["required"] == ["expression"]
    assert calc["dangerous"] is False
    assert calc["parameters"]["additionalProperties"] is False
    assert calc["parameters"]["properties"]["precision"]["type"] == "integer"

    assert tools["http_get"]["dangerous"] is True
    # 只有一个必填参数的当前时间工具
    assert tools["current_time"]["required"] == []


def test_run_tool_directly(client: TestClient) -> None:
    response = client.post("/api/tools/calculator/run", json={"expression": "sqrt(2)*100"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["result"] == "141.4214"
    assert body["duration_ms"] >= 0


def test_run_tool_returns_error_payload(client: TestClient) -> None:
    """工具层面的失败不是服务错误，返回 200 + error，前端原样展示。"""
    response = client.post("/api/tools/calculator/run", json={"expression": "1/0"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "除数不能为零" in body["error"]


def test_run_tool_unknown_404(client: TestClient) -> None:
    response = client.post("/api/tools/nope/run", json={})
    assert response.status_code == 404


def test_run_tool_uses_sandbox(client: TestClient) -> None:
    """文件工具的边界要在真实服务里也生效。"""
    ok = client.post("/api/tools/read_text_file/run", json={"sub_path": "note.txt"}).json()
    assert ok["ok"] is True
    assert "沙箱里的说明文件" in ok["result"]

    blocked = client.post(
        "/api/tools/read_text_file/run", json={"sub_path": "../../../etc/passwd"}
    ).json()
    assert blocked["ok"] is False


# ===========================================================================
# 对话（非流式）
# ===========================================================================
def test_chat_offline(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "帮我算一下 (3+5)*2/4", "stream": False},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["offline"] is True
    assert body["mode"] == "function_call"
    assert body["tools_called"] == ["calculator"]
    assert "4" in body["answer"]
    assert body["steps_used"] >= 2
    assert body["elapsed_ms"] >= 0


def test_chat_steps_shape(client: TestClient) -> None:
    body = client.post("/api/chat", json={"message": "帮我算 7*8"}).json()
    steps = body["steps"]
    assert [s["type"] for s in steps] == ["tool_call", "tool_result"]
    assert steps[0]["tool"] == "calculator"
    assert steps[1]["result"] == "56"
    assert steps[1]["error"] is None


def test_chat_respects_tool_subset(client: TestClient) -> None:
    """只给 calculator，问时间就调不到 current_time。"""
    body = client.post(
        "/api/chat",
        json={"message": "现在几点了", "tools": ["calculator"]},
    ).json()
    assert body["tools_called"] == []


def test_chat_multiple_tools(client: TestClient) -> None:
    body = client.post("/api/chat", json={"message": "现在几点了，顺便帮我算 12*8"}).json()
    assert set(body["tools_called"]) == {"calculator", "current_time"}
    assert "96" in body["answer"]


def test_chat_with_history(client: TestClient) -> None:
    body = client.post(
        "/api/chat",
        json={
            "message": "再算一次",
            "history": [
                {"role": "user", "content": "帮我算 1+1"},
                {"role": "assistant", "content": "结果是 2"},
            ],
        },
    ).json()
    assert body["answer"]


def test_chat_rejects_empty_message(client: TestClient) -> None:
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_bad_history_role(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "你好", "history": [{"role": "system", "content": "x"}]},
    )
    assert response.status_code == 422


def test_chat_rejects_too_many_steps(client: TestClient) -> None:
    assert client.post("/api/chat", json={"message": "你好", "max_steps": 99}).status_code == 422


# ===========================================================================
# 对话（SSE）
# ===========================================================================
def test_chat_stream_event_sequence(client: TestClient) -> None:
    response = client.post("/api/chat/stream", json={"message": "帮我算 6*7"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    names = [name for name, _ in events]

    assert names[0] == "start"
    assert names[-1] == "[DONE]"
    assert "thinking" in names
    assert names.index("tool_call") < names.index("tool_result")
    assert names[-2] == "done"


def test_chat_stream_start_payload(client: TestClient) -> None:
    events = dict(parse_sse(client.post("/api/chat/stream", json={"message": "你好"}).text))
    start = events["start"]
    assert start["mode"] == "function_call"
    assert start["offline"] is True
    assert "calculator" in start["tools"]


def test_chat_stream_tool_events(client: TestClient) -> None:
    events = parse_sse(client.post("/api/chat/stream", json={"message": "帮我算 9*9"}).text)
    payloads = {name: body for name, body in events}

    assert payloads["tool_call"]["tool"] == "calculator"
    assert payloads["tool_call"]["arguments"] == {"expression": "9*9"}
    assert payloads["tool_result"]["result"] == "81"


def test_chat_stream_deltas_reassemble(client: TestClient) -> None:
    events = parse_sse(client.post("/api/chat/stream", json={"message": "帮我算 3+4"}).text)
    deltas = "".join(body["content"] for name, body in events if name == "delta")
    done = next(body for name, body in events if name == "done")

    assert deltas == done["answer"]
    assert "7" in deltas


def test_chat_stream_done_payload(client: TestClient) -> None:
    events = parse_sse(client.post("/api/chat/stream", json={"message": "帮我算 5*5"}).text)
    done = next(body for name, body in events if name == "done")

    assert done["tools_called"] == ["calculator"]
    assert done["steps_used"] >= 2
    assert done["offline"] is True
    assert len(done["steps"]) == 2


def test_chat_stream_multi_tool(client: TestClient) -> None:
    events = parse_sse(
        client.post("/api/chat/stream", json={"message": "现在几点了，顺便帮我算 12*8"}).text
    )
    called = [body["tool"] for name, body in events if name == "tool_call"]
    assert set(called) == {"calculator", "current_time"}


# ===========================================================================
# 示例文件
# ===========================================================================
def test_list_examples(client: TestClient) -> None:
    body = client.get("/api/examples").json()
    assert body["total"] >= 1
    assert any(item["name"] == "note.txt" for item in body["files"])


def test_upload_example(client: TestClient) -> None:
    response = client.post(
        "/api/examples/upload",
        files={"file": ("新增说明.txt", "这是新上传的文件内容。".encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["name"] == "新增说明.txt"

    # 上传后应该立刻能被文件工具读到
    read = client.post("/api/tools/read_text_file/run", json={"sub_path": "新增说明.txt"}).json()
    assert read["ok"] is True
    assert "新上传的文件内容" in read["result"]


def test_upload_rejects_bad_extension(client: TestClient) -> None:
    response = client.post(
        "/api/examples/upload",
        files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_sanitizes_filename(client: TestClient) -> None:
    """上传接口本身就是攻击面，文件名必须清洗且不能跑出允许目录。"""
    response = client.post(
        "/api/examples/upload",
        files={"file": ("../../evil name.txt", b"x", "text/plain")},
    )
    assert response.status_code == 200
    name = response.json()["name"]
    assert "/" not in name and "\\" not in name and ".." not in name


def test_upload_rejects_oversize(client: TestClient) -> None:
    big = b"a" * (6 * 1024 * 1024)
    response = client.post(
        "/api/examples/upload", files={"file": ("big.txt", big, "text/plain")}
    )
    assert response.status_code == 413


# ===========================================================================
# 页面与 OpenAPI
# ===========================================================================
def test_index_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Agent" in response.text


def test_openapi_available(client: TestClient) -> None:
    body = client.get("/openapi.json").json()
    assert body["info"]["title"] == "Agent Toolbox"
    assert "/api/chat/stream" in body["paths"]


def test_static_mount(client: TestClient) -> None:
    assert client.get("/static/index.html").status_code == 200
