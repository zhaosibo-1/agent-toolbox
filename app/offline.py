"""离线规划器：未配置 API Key 时的降级「大脑」。

先说清楚它**不是**什么：它不是一个小型语言模型，也不会假装自己懂语义。
它是一个**规则化的演示规划器**——用关键词判断用户想干什么，挑选对应的工具，
在拿到工具执行结果后把结果汇总成答复。

为什么值得写这一层：

1. **可复现**：clone 仓库的人不配 Key 也能把完整的 Agent 循环（规划 → 调用 →
   观察 → 再规划 → 收敛）跑起来，看到真实的工具调用轨迹，而不是一句
   "请先配置 API Key"。
2. **可测试**：CI 里不需要任何密钥就能对 Agent 主循环做端到端断言。
3. **诚实**：所有输出都会带「离线演示模式」前缀，不会让人误以为是模型生成的。

一旦配置了 ``LLM_API_KEY``，这一层立即被真实模型取代。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# ===========================================================================
# 表达式抽取
# ===========================================================================
#: 表达式中允许出现的字符（刻意不含字母以外的其他符号）
_EXPR_CHARS = set("0123456789.+-*/%() \t")
#: 支持的数学函数名
_FUNC_NAMES = (
    "sqrt", "log2", "log10", "log", "exp", "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2", "hypot", "abs", "floor", "ceil",
    "factorial", "degrees", "radians",
)
_FUNC_CALL_RE = re.compile(
    r"(?:" + "|".join(_FUNC_NAMES) + r")\s*\([^()]{0,60}\)"
)
#: 纯算术片段的「字符游程」——先尽量贪心，再回头修括号
_RUN_RE = re.compile(r"[0-9\s\.\+\-\*/%\(\)]{2,160}")
#: 必须至少含一个运算符，否则 "2027" 这种数字会被误当成算式
_OPERATOR_RE = re.compile(r"[+\-*/%]")
#: 长得像日期 / 版本的片段（避免 "2026-03-31" 被送去计算器）
_DATE_LIKE_RE = re.compile(r"^\d{1,4}\s*[-/.]\s*\d{1,2}(\s*[-/.]\s*\d{1,4})?$")


def _balanced(expr: str) -> bool:
    """括号是否配对。"""
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _repair(expr: str) -> str:
    """把贪心截出来的片段修成括号配对的合法式子。

    例：``"3+5)*2/4"`` 会被一路砍成 ``"3+5"``。
    """
    expr = expr.strip()
    while expr.startswith(")"):
        expr = expr[1:]
    while expr and not _balanced(expr):
        expr = expr[:-1]
    return expr.strip(" .+-*/%")


def _extract_expression(text: str) -> Optional[str]:
    """从自然语言里抠出一个可计算的数学表达式。

    分两条路径，因为带函数名（``sqrt(2)*100``）时字符游程会被字母打断：

    1. 命中数学函数 → 以函数调用为锚点，向左右扩展算术字符；
    2. 否则 → 取最长的算术字符游程，再修括号。
    """
    # --- 路径 1：带数学函数 ---
    func_match = _FUNC_CALL_RE.search(text)
    if func_match:
        start, end = func_match.start(), func_match.end()
        while start > 0 and text[start - 1] in _EXPR_CHARS:
            start -= 1
        while end < len(text) and text[end] in _EXPR_CHARS:
            end += 1
        candidate = text[start:end].strip()
        if candidate and _OPERATOR_RE.search(candidate):
            return candidate or None

    # --- 路径 2：纯算术 ---
    best = ""
    for run in _RUN_RE.finditer(text):
        candidate = _repair(run.group(0))
        if _DATE_LIKE_RE.match(candidate):
            continue
        if not re.search(r"\d", candidate):
            continue
        if not _OPERATOR_RE.search(candidate):
            continue
        if len(candidate) > len(best):
            best = candidate
    return best or None


# ===========================================================================
# 单位换算抽取
# ===========================================================================
#: 中文单位别名 → 工具内部使用的符号
_UNIT_ALIAS: Dict[str, str] = {
    "米": "m", "千米": "km", "公里": "km", "厘米": "cm", "毫米": "mm",
    "英寸": "inch", "英尺": "foot", "英里": "mile",
    "千克": "kg", "公斤": "kg", "克": "g", "毫克": "mg", "吨": "ton",
    "磅": "pound", "盎司": "ounce",
    "平方米": "m2", "平方公里": "km2", "平方厘米": "cm2",
    "公顷": "hectare", "亩": "mu", "英亩": "acre",
}
#: 单位符号：字母或中文开头，最长 8 个字符。
#: 上限不能太小——`pound` / `ounce` / `inch` 这些恰好是 5 个字母，
#: 卡到 4 位会被截成 `poun` 这种查不到的单位。
_UNIT_TOKEN = r"[a-zA-Z\u4e00-\u9fff][a-zA-Z0-9\u4e00-\u9fff]{0,7}"
_UNIT_SEP = (
    r"(?:换算成|换算为|转换成|转为|转成|换成|换算|折合|等于多少|等于|是多少|是|=|->|→|\s+to\s+)"
)
#: 分隔符后面常跟着「多少 / 几」，它不是单位，必须先吃掉
_UNIT_FILLER = r"(?:多少|几)?\s*"
_UNIT_RE = re.compile(
    rf"(-?\d+(?:\.\d+)?)\s*({_UNIT_TOKEN})\s*{_UNIT_SEP}\s*{_UNIT_FILLER}({_UNIT_TOKEN})"
)


def _normalize_unit(token: str) -> str:
    key = token.strip()
    return _UNIT_ALIAS.get(key, key.lower())


# ===========================================================================
# 其它意图
# ===========================================================================
_URL_RE = re.compile(r"https?://[^\s，。；、\"'<>）)]+")
_JSON_RE = re.compile(r"\{[\s\S]{4,4000}\}")
_JSON_PATH_RE = re.compile(r"(?:路径|path)\s*[:：]?\s*([A-Za-z_][\w.\-]*)")
#: 「统计 … 的字数：正文」——冒号前的限定语最多 12 个字符，避免把正文也吞进去
_STATS_RE = re.compile(r"(?:统计|分析|计算)[^：:\n]{0,12}[:：]\s*([\s\S]{2,})")
_FILE_PATH_RE = re.compile(r"([A-Za-z0-9_\-./\\\u4e00-\u9fff]+\.[A-Za-z0-9]{1,6})")
_DIR_PATH_RE = re.compile(r"([A-Za-z0-9_\-]+/)\s")

#: 各意图的触发关键词
_TIME_WORDS = ("现在几点", "几点", "当前时间", "现在时间", "今天的日期", "当前日期", "时间", "日期")
_JSON_WORDS = ("json", "JSON", "Json", "取字段", "提取字段")
_STATS_WORDS = ("统计", "字数", "词频", "多少个字", "多少字")
_FILE_WORDS = ("读取文件", "读一下", "读文件", "打开文件", "查看文件", "列出", "目录", "文件清单", "有哪些文件")
_READ_WORDS = ("读", "打开", "查看", "内容")

#: 离线模式下「没听懂」时的引导语
_GUIDE = """（离线演示模式）我没能从这句话里识别出可以调用的工具。

可以直接试试这些指令：
- 帮我算一下 (3+5)*2/4
- 帮我算 sqrt(2)*100
- 现在几点了
- 统计一下这段话的字数：人工智能正在改变软件开发的方式
- 把 12 km 换算成 mile
- 抓取 https://example.com
- 读取文件 examples/服务器巡检记录.txt
- 目录下有哪些文件
- 从这段 JSON 里取值 路径 data.items.0.name：{"data":{"items":[{"name":"张三"}]}}

在 `.env` 里配置 `LLM_API_KEY` 之后，规划这一步交给真实大模型完成，
它就能理解任意自然语言、自主编排多步工具调用了。"""


class OfflinePlanner:
    """规则化的规划器，对外接口与 LLM 客户端保持一致。"""

    # ------------------------------------------------------------------
    def plan(self, user_text: str, available: List[str]) -> List[Dict[str, Any]]:
        """根据用户输入，列出「按什么顺序调用哪些工具」。

        Args:
            user_text: 用户本轮输入。
            available: 当前启用的工具名列表。

        Returns:
            ``[{"name": 工具名, "arguments": {...}}, ...]``，已按优先级排序。
        """
        plans: List[Dict[str, Any]] = []

        def add(name: str, args: Dict[str, Any]) -> None:
            # 同名工具只排一次，避免「换算+计算」把同一个工具重复塞进去
            if name in available and all(p["name"] != name for p in plans):
                plans.append({"name": name, "arguments": args})

        # 1) 单位换算（比计算器更具体，优先）
        unit_match = _UNIT_RE.search(user_text)
        if unit_match:
            add(
                "convert_unit",
                {
                    "value": float(unit_match.group(1)),
                    "from_unit": _normalize_unit(unit_match.group(2)),
                    "to_unit": _normalize_unit(unit_match.group(3)),
                },
            )

        # 2) 数学计算
        expression = _extract_expression(user_text)
        if expression:
            add("calculator", {"expression": expression})

        # 3) 抓取网页
        url_match = _URL_RE.search(user_text)
        if url_match:
            add("http_get", {"url": url_match.group(0)})

        # 4) JSON 取值
        json_match = _JSON_RE.search(user_text)
        looks_like_json = bool(json_match) and any(word in user_text for word in _JSON_WORDS)
        if looks_like_json:
            path_match = _JSON_PATH_RE.search(user_text)
            add(
                "json_query",
                {
                    "json_text": json_match.group(0) if json_match else "",
                    "path": path_match.group(1) if path_match else "",
                },
            )

        # 5) 文件操作（已经判定是 JSON 场景时跳过，否则会把 JSON 里的路径误当文件名）
        if not looks_like_json:
            file_path = self._extract_file_path(user_text)
            if file_path and any(word in user_text for word in _READ_WORDS):
                add("read_text_file", {"sub_path": file_path})
            elif any(word in user_text for word in _FILE_WORDS):
                add("list_directory", {"sub_path": file_path or "."})

        # 6) 文本统计
        if any(word in user_text for word in _STATS_WORDS):
            stats_match = _STATS_RE.search(user_text)
            body = stats_match.group(1).strip() if stats_match else user_text
            add("text_statistics", {"text": body})

        # 7) 时间放最后：「时间」是个很宽泛的词，别把更具体的意图挤掉
        if any(word in user_text for word in _TIME_WORDS):
            add("current_time", {})

        return plans

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_file_path(text: str) -> str:
        """从文本里找出一个像文件 / 目录路径的片段。"""
        file_match = _FILE_PATH_RE.search(text)
        if file_match:
            return file_match.group(1).replace("\\", "/")
        dir_match = _DIR_PATH_RE.search(text)
        if dir_match:
            return dir_match.group(1)
        return ""

    # ------------------------------------------------------------------
    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> Any:
        """模拟一次模型调用，返回 :class:`~app.llm.LLMResponse`。"""
        from .llm import LLMResponse, ToolCall  # 延迟导入，打破循环依赖

        available = [t["function"]["name"] for t in (tools or [])]
        user_text = self.last_user_text(messages)
        already: Set[str] = self._already_called(messages)
        plans: List[Dict[str, Any]] = self.plan(user_text, available)

        for index, plan in enumerate(plans):
            if plan["name"] in already:
                continue

            # 多步感知：上一步刚读过文件，这一步要统计的显然是**文件内容**
            # 而不是用户的原始问句——这是让「读文件 + 统计」这类链式任务
            # 真正有意义的关键一步。
            if plan["name"] == "text_statistics":
                content = self._last_tool_result(messages, "read_text_file")
                if content:
                    plan["arguments"]["text"] = content

            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"offline_{len(already)}_{index}",
                        name=plan["name"],
                        arguments=json.dumps(plan["arguments"], ensure_ascii=False),
                    )
                ],
                finish_reason="tool_calls",
                model="offline-planner",
            )

        return LLMResponse(
            content=self.final_text(messages),
            tool_calls=[],
            finish_reason="stop",
            model="offline-planner",
        )

    # ------------------------------------------------------------------
    def final_text(self, messages: List[Dict[str, Any]]) -> str:
        """把会话里出现过的工具结果汇总成一份可读答复。"""
        results: List[Tuple[str, str]] = []
        names_by_id: Dict[str, str] = {}

        for message in messages:
            for call in message.get("tool_calls") or []:
                names_by_id[call.get("id", "")] = (call.get("function") or {}).get("name", "?")
            if message.get("role") == "tool":
                content = str(message.get("content", "")).strip()
                if content:
                    results.append((names_by_id.get(message.get("tool_call_id", ""), "工具"), content))

        if not results:
            return _GUIDE

        blocks = [f"**{name}**\n```\n{content}\n```" for name, content in results]
        return (
            "（离线演示模式：以下结论由工具执行结果直接汇总，未经大模型润色）\n\n"
            + "\n\n".join(blocks)
            + "\n\n在 `.env` 里配置 `LLM_API_KEY` 后，同样的工具轨迹会交给大模型组织成更自然的回答。"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def last_user_text(messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    @staticmethod
    def _last_tool_result(messages: List[Dict[str, Any]], tool_name: str) -> str:
        """取出指定工具最近一次成功返回的内容。

        注意这里**不做 strip**：这一步的数据会原样喂给下一个工具，
        悄悄改掉首尾空白（比如文件末尾的换行）会让下游的统计结果对不上。
        """
        names_by_id: Dict[str, str] = {}
        for message in messages:
            for call in message.get("tool_calls") or []:
                names_by_id[call.get("id", "")] = (call.get("function") or {}).get("name", "")

        for message in reversed(messages):
            if message.get("role") != "tool":
                continue
            if names_by_id.get(message.get("tool_call_id", "")) != tool_name:
                continue
            content = str(message.get("content", ""))
            if content.strip() and not content.lstrip().startswith("错误"):
                return content
        return ""

    @staticmethod
    def _already_called(messages: List[Dict[str, Any]]) -> Set[str]:
        """扫一遍历史，看这次任务里已经调用过哪些工具。"""
        called: Set[str] = set()
        for message in messages:
            for call in message.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name")
                if name:
                    called.add(name)
        return called
