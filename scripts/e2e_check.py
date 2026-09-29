"""端到端验收脚本：对真实运行的服务做一轮黑盒验证。

只依赖标准库（urllib），所以任何环境都能跑：
    python _tools/e2e_check.py [base_url]
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111"

# 绕过系统代理：本地回环不该走代理
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        FAILED.append(name)
        print(f"  ✗ {name}  {detail}")


def get(path: str) -> tuple[int, str]:
    try:
        with OPENER.open(BASE + path, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="ignore")


def post(path: str, payload: dict | None = None) -> tuple[int, str]:
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with OPENER.open(request, timeout=120) as resp:
            return resp.status, resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="ignore")


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
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
        if name:
            events.append((name, json.loads(data) if data and data != "[DONE]" else {}))
    return events


print("=" * 74)
print("Agent Toolbox 端到端验收")
print("=" * 74)

# ---------------------------------------------------------------- 1. 基础状态
print("\n[1] 基础状态")
status, body = get("/api/health")
health = json.loads(body)
check("GET /api/health 返回 200", status == 200)
check("健康检查报告 8 个工具", health.get("tool_total") == 8, body)
check("未配置 Key 时进入离线模式", health.get("offline") is True)

status, body = get("/api/config")
config = json.loads(body)
check("GET /api/config 返回 200", status == 200)
check("配置里含前端读取的全部字段",
      set(config["llm"]) >= {"preset", "model", "base_url", "api_key", "enabled"}
      and set(config["agent"]) >= {"mode", "max_steps", "tool_timeout", "result_feedback"}
      and set(config["safety"]) >= {"allowed_file_dirs", "http_allowed_hosts"})
check("API Key 已脱敏", "sk-xxxx" not in json.dumps(config), config["llm"]["api_key"])

# ---------------------------------------------------------------- 2. 工具 Schema
print("\n[2] 工具 Schema（由函数签名自动生成）")
status, body = get("/api/tools")
tools = json.loads(body)
by_name = {t["name"]: t for t in tools["tools"]}
check("工具总数 8", tools["total"] == 8)
check("calculator 参数类型未被退化成 string",
      by_name["calculator"]["parameters"]["properties"]["precision"]["type"] == "integer",
      by_name["calculator"]["params"])
check("convert_unit 的 value 是 number",
      by_name["convert_unit"]["parameters"]["properties"]["value"]["type"] == "number")
check("additionalProperties 被显式关闭",
      all(t["parameters"]["additionalProperties"] is False for t in tools["tools"]))
check("危险工具被标记",
      by_name["http_get"]["dangerous"] and by_name["read_text_file"]["dangerous"]
      and by_name["list_directory"]["dangerous"])
check("非危险工具未被误标", by_name["calculator"]["dangerous"] is False)

# ---------------------------------------------------------------- 3. 单工具调试
print("\n[3] 单工具直接调用 /api/tools/{name}/run")
status, body = post("/api/tools/calculator/run", {"expression": "sqrt(2)*100"})
ok = json.loads(body)
check("计算器返回正确结果", status == 200 and ok["ok"] and ok["result"] == "141.4214", body)

status, body = post("/api/tools/calculator/run", {"expression": "1/0"})
bad = json.loads(body)
check("除零返回 ok=false 且状态仍是 200", status == 200 and bad["ok"] is False)
check("除零错误信息可读", "除数不能为零" in bad["error"], body)

status, body = post("/api/tools/calculator/run", {"expression": "__import__('os')"})
check("代码注入被拒绝", status == 200 and json.loads(body)["ok"] is False)

status, _ = post("/api/tools/nope/run", {})
check("未知工具返回 404", status == 404)

# ---------------------------------------------------------------- 4. 安全边界
print("\n[4] 安全边界")
for evil in ("../../../.env", "../../.env", "C:/Windows/win.ini"):
    status, body = post("/api/tools/read_text_file/run", {"sub_path": evil})
    payload = json.loads(body)
    leaked = payload.get("ok") and "LLM_API_KEY" in str(payload.get("result", ""))
    check(f"路径穿越被拒绝: {evil}", payload["ok"] is False and not leaked, body[:160])

status, body = post("/api/tools/read_text_file/run", {"sub_path": "服务器巡检记录.txt"})
check("允许目录内可正常读取",
      json.loads(body)["ok"] and "数据中心机房巡检记录" in json.loads(body)["result"])

status, body = post("/api/tools/list_directory/run", {"sub_path": "."})
check("目录列举可用", json.loads(body)["ok"])

status, body = post("/api/tools/http_get/run", {"url": "file:///etc/passwd"})
check("非 http(s) 协议被拒绝", json.loads(body)["ok"] is False)

status, body = post("/api/tools/http_get/run", {"url": "https://evil.internal/x"})
check("域名白名单生效", json.loads(body)["ok"] is False, body[:160])

# ---------------------------------------------------------------- 5. SSE 多步链
print("\n[5] Agent 多步工具链（SSE 全轨迹）")
status, raw = post(
    "/api/chat/stream",
    {"message": "读取文件 examples/服务器巡检记录.txt，然后统计它的字数"},
)
events = parse_sse(raw)
names = [n for n, _ in events]
check("SSE 返回 200", status == 200)
check("事件以 start 开头", names[0] == "start", names[:3])
check("事件以 done + [DONE] 结尾", names[-1] == "[DONE]" and names[-2] == "done", names[-3:])
check("工具调用与结果成对出现",
      names.count("tool_call") == 2 and names.count("tool_result") == 2, str(names))

calls = [p["tool"] for n, p in events if n == "tool_call"]
check("主循环按顺序编排了两步工具", calls == ["read_text_file", "text_statistics"], str(calls))
check("有 3 轮推理（两次工具 + 一次收尾）",
      sum(1 for n, _ in events if n == "thinking") == 3)

results = {p["tool"]: p for n, p in events if n == "tool_result"}
check("文件读取成功", results["read_text_file"]["error"] is None)
stats_text = results["text_statistics"]["result"]

# 断言统计对象确实是文件正文：字符数与文件严格一致，且不含只在问句里出现的词
from pathlib import Path

sample = Path("examples/服务器巡检记录.txt")
expected_chars = len(sample.read_bytes().decode("utf-8"))
check("统计对象是文件正文而非用户问句",
      f"总字符数：{expected_chars}" in stats_text
      and "读取" not in stats_text
      and "txt" not in stats_text,
      f"期望 总字符数：{expected_chars}；实际 {stats_text.splitlines()[0]}")
done = next(p for n, p in events if n == "done")
check("done 事件带 steps_used", done.get("steps_used") == 3, str(done.get("steps_used")))
check("done 事件的 tools_called 正确",
      done["tools_called"] == ["read_text_file", "text_statistics"], str(done["tools_called"]))
deltas = "".join(p["content"] for n, p in events if n == "delta")
check("delta 拼起来等于最终答复", deltas == done["answer"] and len(deltas) > 0)

# ---------------------------------------------------------------- 6. 错误回灌
print("\n[6] 工具失败后 Agent 仍能收敛")
status, raw = post("/api/chat/stream", {"message": "读取文件 examples/不存在的文件.txt"})
events = parse_sse(raw)
names = [n for n, _ in events]
failed = [p for n, p in events if n == "tool_result" and p["error"]]
check("失败被记录成 tool_result.error", len(failed) == 1, str(names))
check("失败没有中断任务（仍有 done）", "done" in names)
check("错误信息可读", "不存在" in failed[0]["error"], failed[0]["error"])
check("失败的工具不计入 tools_called",
      next(p for n, p in events if n == "done")["tools_called"] == [])

# ---------------------------------------------------------------- 7. 工具集裁剪
print("\n[7] 按场景裁剪工具集")
status, raw = post("/api/chat/stream", {"message": "现在几点了", "tools": ["calculator"]})
events = parse_sse(raw)
start = next(p for n, p in events if n == "start")
check("只启用了 calculator", start["tools"] == ["calculator"], str(start["tools"]))
check("裁剪后问时间调不到工具",
      next(p for n, p in events if n == "done")["tools_called"] == [])

# ---------------------------------------------------------------- 8. 参数校验
print("\n[8] 请求校验")
check("空消息被拒绝", post("/api/chat", {"message": ""})[0] == 422)
check("非法 role 被拒绝",
      post("/api/chat", {"message": "hi", "history": [{"role": "system", "content": "x"}]})[0] == 422)
check("超范围 max_steps 被拒绝", post("/api/chat", {"message": "hi", "max_steps": 99})[0] == 422)

status, body = post("/api/chat", {"message": "帮我算 (3+5)*2/4"})
payload = json.loads(body)
check("非流式接口可用", status == 200 and "4" in payload["answer"])
check("非流式接口返回 steps_used", payload["steps_used"] >= 2)

# ---------------------------------------------------------------- 9. 示例文件
print("\n[9] 示例文件与上传")
status, body = get("/api/examples")
check("示例目录可见", json.loads(body)["total"] >= 2, body)

# ---------------------------------------------------------------- 10. 页面
print("\n[10] 前端页面")
status, html = get("/")
check("首页返回 HTML", status == 200 and "<!DOCTYPE html>" in html)
check("页面引用了必备的接口", "/api/chat/stream" in html and "/api/tools" in html)
check("页面声明了深色主题变量", "--panel:" in html and "--accent:" in html)
check("静态资源可访问", get("/static/index.html")[0] == 200)
check("OpenAPI 文档可用", get("/openapi.json")[0] == 200)

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 74)
print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
if FAILED:
    print("失败项：")
    for name in FAILED:
        print("  -", name)
print("=" * 74)
sys.exit(1 if FAILED else 0)
