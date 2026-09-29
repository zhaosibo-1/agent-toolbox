"""内置工具箱。

每个工具都用 ``@tool`` 装饰，schema 由函数签名与 docstring 自动生成。
这里刻意包含了几类**安全敏感**的工具（执行表达式、访问网络、读写文件），
用来演示「Agent 手上的工具必须有边界」这件事：

- 计算器**不用** ``eval``，而是走 AST 白名单，从根本上杜绝代码注入；
- 网络工具带域名白名单、响应体大小上限与超时；
- 文件工具被限制在配置允许的目录内，且做路径穿越校验。
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .base import ToolError, tool

# ===========================================================================
# 运行时安全配置（由 registry 在初始化时注入）
# ===========================================================================
_ALLOWED_DIRS: List[Path] = []
_HTTP_HOSTS: List[str] = []
_CALC_MAX_LENGTH: int = 200
_PROJECT_ROOT: Path = Path.cwd().resolve()


def configure(
    allowed_dirs: Sequence[str] | None = None,
    http_hosts: Sequence[str] | None = None,
    calc_max_length: int = 200,
    project_root: str | Path | None = None,
) -> None:
    """注入安全边界配置（由应用启动时调用）。

    Args:
        allowed_dirs: 允许文件工具访问的目录白名单。
        http_hosts: HTTP 工具的域名白名单；为空表示不限制（仅本地开发）。
        calc_max_length: 计算器允许的表达式最大长度。
        project_root: 项目根目录，用于解析那种「多带了一层目录名」的路径
            （模型经常把 ``examples/x.txt`` 直接丢进来，而允许目录本身就是
            ``examples``，此时需要能自动剥掉这一层）。
    """
    global _ALLOWED_DIRS, _HTTP_HOSTS, _CALC_MAX_LENGTH, _PROJECT_ROOT

    _ALLOWED_DIRS = []
    for raw in allowed_dirs or []:
        try:
            _ALLOWED_DIRS.append(Path(raw).expanduser().resolve())
        except Exception:  # noqa: BLE001
            continue

    _HTTP_HOSTS = [h.strip().lower() for h in (http_hosts or []) if h.strip()]
    _CALC_MAX_LENGTH = max(16, int(calc_max_length))
    _PROJECT_ROOT = Path(project_root).resolve() if project_root else Path.cwd().resolve()


# ===========================================================================
# 工具 1：安全计算器
# ===========================================================================
_ALLOWED_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_NAMES: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}
_ALLOWED_FUNCS: Dict[str, Any] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "floor": math.floor,
    "ceil": math.ceil,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": math.factorial,
    "gcd": math.gcd,
}
# 幂运算的指数上限：防止 `9**9**9` 这类表达式把进程算死
_MAX_POW_EXPONENT = 1000


def _eval_node(node: ast.AST) -> Any:
    """递归求值 AST 节点，只放行白名单内的语法结构。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # 字面量：数字、字符串（字符串在数学表达式里没有意义，但也不算危险）
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError(f"计算器不支持的字面量：{node.value!r}")

    # 二元运算 a + b
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINARY_OPS:
            raise ToolError(f"不支持的运算符：{op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and isinstance(right, (int, float)) and right > _MAX_POW_EXPONENT:
            raise ToolError(f"幂运算指数过大（上限 {_MAX_POW_EXPONENT}）")
        try:
            return _ALLOWED_BINARY_OPS[op_type](left, right)
        except ZeroDivisionError as exc:
            raise ToolError("除数不能为零") from exc

    # 一元运算 -a / +a
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ToolError(f"不支持的一元运算符：{op_type.__name__}")
        return _ALLOWED_UNARY_OPS[op_type](_eval_node(node.operand))

    # 函数调用 sqrt(x)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolError("只支持调用内置数学函数，不支持属性调用")
        fname = node.func.id
        if fname not in _ALLOWED_FUNCS:
            raise ToolError(
                f"不支持的函数 {fname}，可用函数：{', '.join(sorted(_ALLOWED_FUNCS))}"
            )
        if node.keywords:
            raise ToolError("不支持关键字参数")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[fname](*args)

    # 常量名 pi / e
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ToolError(f"未知的标识符 {node.id}，只支持 pi / e / tau")

    # 列表字面量（min([1,2,3]) 这种用法）
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e) for e in node.elts]

    raise ToolError(f"表达式中包含不支持的语法：{type(node).__name__}")


@tool(description="计算数学表达式，支持 + - * / // % ** 以及 sqrt/log/sin 等常用数学函数")
def calculator(expression: str, precision: int = 4) -> str:
    """计算一个数学表达式的值。

    Args:
        expression: 待计算的数学表达式，例如 "(3+5)*2/4" 或 "sqrt(2)*100"。
        precision: 结果保留的小数位数。
    """
    expr = expression.strip()
    if not expr:
        raise ToolError("表达式不能为空")
    if len(expr) > _CALC_MAX_LENGTH:
        raise ToolError(f"表达式过长（上限 {_CALC_MAX_LENGTH} 字符）")

    # 先在语法层面拒绝一切非算术内容，再做求值
    if re.search(r"(?:__|\bimport\b|\blambda\b|\bexec\b|\beval\b|\bopen\b)", expr):
        raise ToolError("表达式包含不允许的关键字")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"表达式语法错误：{exc.msg}") from exc

    value = _eval_node(tree)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return f"{value}"
        return f"{value:.{max(0, min(precision, 12))}f}".rstrip("0").rstrip(".") or "0"
    return str(value)


# ===========================================================================
# 工具 2：时间
# ===========================================================================
_WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@tool(description="获取当前的日期与时间，可指定时区偏移")
def current_time(utc_offset_hours: float = 8.0) -> str:
    """获取当前时间。

    Args:
        utc_offset_hours: 相对 UTC 的时区偏移小时数，默认 8（北京时间）。
    """
    offset = int(utc_offset_hours) if float(utc_offset_hours).is_integer() else utc_offset_hours
    # 负偏移自带减号，别再加一个（否则会渲染成 UTC--5）
    label = f"{offset:+}"
    try:
        tz = timezone(timedelta(hours=utc_offset_hours))
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"时区偏移不合法：{utc_offset_hours}") from exc

    now = datetime.now(tz)
    return (
        f"{now.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(UTC{label}, {_WEEKDAY_CN[now.weekday()]})"
    )


# ===========================================================================
# 工具 3：文本统计
# ===========================================================================
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-zA-Z]+|\d+")


@tool(description="统计一段文本的字符数、词数、行数以及高频词")
def text_statistics(text: str, top_n: int = 8) -> str:
    """对文本做基础统计分析。

    Args:
        text: 待统计的文本内容。
        top_n: 返回出现频率最高的前 N 个词。
    """
    if not text.strip():
        raise ToolError("文本内容为空")

    cjk_count = len(_CJK_RE.findall(text))
    words = _WORD_RE.findall(text.lower())
    lines = [line for line in text.splitlines() if line.strip()]

    # 中文按二字组合统计（无分词库时的常用近似做法）
    grams = [text[i : i + 2] for i in range(len(text) - 1) if _CJK_RE.match(text[i]) and _CJK_RE.match(text[i + 1])]
    counter: Counter[str] = Counter(words)
    counter.update(grams)

    top_items = [f"{w}×{c}" for w, c in counter.most_common(max(1, min(top_n, 20)))]
    return (
        f"总字符数：{len(text)}\n"
        f"中文字数：{cjk_count}\n"
        f"英文单词/数字：{len(words)}\n"
        f"非空行数：{len(lines)}\n"
        f"高频词：{', '.join(top_items) if top_items else '(无)'}"
    )


# ===========================================================================
# 工具 4：HTTP 请求
# ===========================================================================
_MAX_RESPONSE_BYTES = 200_000


@tool(description="抓取一个网页或接口的文本内容（GET 请求）", dangerous=True)
def http_get(url: str, max_chars: int = 3000) -> str:
    """用 GET 方式请求一个 URL 并返回响应文本。

    Args:
        url: 完整的 URL，必须以 http:// 或 https:// 开头。
        max_chars: 最多返回的字符数。
    """
    import httpx
    from urllib.parse import urlparse

    if not url.startswith(("http://", "https://")):
        raise ToolError("URL 必须以 http:// 或 https:// 开头")

    host = (urlparse(url).hostname or "").lower()
    if not host:
        raise ToolError("无法解析 URL 中的主机名")

    # 域名白名单：配置为空表示不限制（本地开发）；生产环境应显式收紧
    if _HTTP_HOSTS and not any(host == h or host.endswith("." + h) for h in _HTTP_HOSTS):
        raise ToolError(
            f"域名 {host} 不在白名单内。当前允许：{', '.join(_HTTP_HOSTS)}"
        )

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "agent-toolbox/1.0"})
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"请求失败：{type(exc).__name__}: {exc}") from exc

    if resp.status_code >= 400:
        raise ToolError(f"目标返回 HTTP {resp.status_code}")

    raw = resp.content[:_MAX_RESPONSE_BYTES]
    charset = resp.encoding or "utf-8"
    try:
        text = raw.decode(charset, errors="ignore")
    except LookupError:
        text = raw.decode("utf-8", errors="ignore")

    # 粗粒度去标签，让模型看到的是正文而不是一堆 HTML
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    limit = max(200, min(max_chars, 20000))
    if len(text) > limit:
        text = text[:limit] + f"\n…（已截断，原文共 {len(text)} 字符）"
    return text or "(响应内容为空)"


# ===========================================================================
# 工具 5：读取本地文件
# ===========================================================================
def _containing_allowed(path: Path) -> Path | None:
    """返回包含该路径的允许目录；若不在任何允许目录内则返回 None。"""
    for allowed in _ALLOWED_DIRS:
        try:
            path.relative_to(allowed)
            return allowed
        except ValueError:
            continue
    return None


def _iter_candidates(raw: Path) -> Iterator[Path]:
    """按「最可能命中」的顺序生成候选绝对路径。

    这么做是为了兼容模型常见的几种写法：

    - ``服务器巡检记录.txt``            → 相对允许目录
    - ``examples/服务器巡检记录.txt``   → 多带了一层允许目录名（自动剥掉）
    - ``/abs/path/服务器巡检记录.txt``  → 绝对路径（仍要校验在允许目录内）
    """
    if raw.is_absolute():
        yield raw
        return

    # 1) 相对项目根目录
    yield _PROJECT_ROOT / raw

    # 2) 去掉与允许目录同名的前缀
    names = {d.name for d in _ALLOWED_DIRS}
    parts = raw.parts
    trimmed: Path | None = None
    if len(parts) > 1 and parts[0] in names:
        trimmed = Path(*parts[1:])

    # 3) 相对每个允许目录
    for allowed in _ALLOWED_DIRS:
        yield allowed / raw
        if trimmed is not None:
            yield allowed / trimmed


def _resolve_within_allowed(path_str: str) -> Path:
    """把用户给的路径解析成绝对路径，并校验它确实位于允许的目录内。

    这一步是必须的：不做校验的话，模型可以让 Agent 去读 ``../../.env``。
    注意校验是在 ``resolve()`` **之后**做的——只有解析完符号链接与 ``..``
    才能真正判断最终落点是否越界。
    """
    if not _ALLOWED_DIRS:
        raise ToolError(
            "文件访问未启用。如需使用，请在 .env 中配置 ALLOWED_FILE_DIRS 指定允许的目录"
        )

    raw = Path(path_str).expanduser()
    fallback: Path | None = None

    for candidate in _iter_candidates(raw):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if _containing_allowed(resolved) is None:
            continue
        if fallback is None:
            fallback = resolved
        if resolved.exists():
            return resolved

    # 有落在允许目录内、但文件不存在的候选 → 交给调用方报「不存在」，信息更有用
    if fallback is not None:
        return fallback

    raise ToolError(
        f"拒绝访问：{path_str} 不在允许的目录内"
        f"（允许：{', '.join(str(d) for d in _ALLOWED_DIRS)}）"
    )


@tool(description="列出允许目录下的文件清单", dangerous=True)
def list_directory(sub_path: str = ".") -> str:
    """列出目录内容。

    Args:
        sub_path: 相对于允许目录的路径，默认列出根目录。
    """
    target = _resolve_within_allowed(sub_path)
    if not target.exists():
        raise ToolError(f"路径不存在：{sub_path}")
    if not target.is_dir():
        raise ToolError(f"不是目录：{sub_path}")

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))[:60]
    if not entries:
        return "(目录为空)"

    lines = []
    for p in entries:
        if p.is_dir():
            lines.append(f"[目录] {p.name}/")
        else:
            lines.append(f"[文件] {p.name}  ({p.stat().st_size} 字节)")
    return "\n".join(lines)


@tool(description="读取允许目录下某个文本文件的内容", dangerous=True)
def read_text_file(sub_path: str, max_chars: int = 4000) -> str:
    """读取文本文件内容。

    Args:
        sub_path: 相对于允许目录的文件路径。
        max_chars: 最多返回的字符数。
    """
    target = _resolve_within_allowed(sub_path)
    if not target.exists():
        raise ToolError(f"文件不存在：{sub_path}")
    if not target.is_file():
        raise ToolError(f"不是文件：{sub_path}")

    raw = target.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")

    limit = max(200, min(max_chars, 40000))
    if len(text) > limit:
        text = text[:limit] + f"\n…（已截断，原文共 {len(text)} 字符）"
    return text


# ===========================================================================
# 工具 6：JSON 查询
# ===========================================================================
def _walk_json(data: Any, path: List[str]) -> Any:
    """按 ``a.b.0.c`` 形式的路径逐层取值。"""
    current = data
    for token in path:
        if isinstance(current, dict):
            if token not in current:
                raise ToolError(f"路径中不存在键 {token}，可用键：{', '.join(list(current)[:10])}")
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise ToolError(f"{token} 不是合法的数组下标") from exc
            if not -len(current) <= index < len(current):
                raise ToolError(f"数组下标越界：{index}（长度 {len(current)}）")
            current = current[index]
        else:
            raise ToolError(f"无法在 {type(current).__name__} 上继续取路径 {token}")
    return current


@tool(description="从 JSON 文本中按路径提取字段，路径写法如 data.items.0.name")
def json_query(json_text: str, path: str = "") -> str:
    """查询 JSON 中的字段。

    Args:
        json_text: 合法的 JSON 字符串。
        path: 点分隔的访问路径；留空则返回整体摘要。
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolError(f"JSON 解析失败：{exc.msg}（第 {exc.lineno} 行）") from exc

    if not path.strip():
        if isinstance(data, dict):
            return f"这是一个对象，顶层键：{', '.join(list(data)[:20])}"
        if isinstance(data, list):
            return f"这是一个数组，长度 {len(data)}"
        return f"这是一个 {type(data).__name__}：{data!r}"

    tokens = [t for t in path.strip().split(".") if t]
    value = _walk_json(data, tokens)

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)[:4000]
    return str(value)


# ===========================================================================
# 工具 7：单位换算（演示多类型参数与枚举）
# ===========================================================================
_UNIT_TABLE: Dict[str, Dict[str, float]] = {
    "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "inch": 0.0254, "foot": 0.3048, "mile": 1609.344},
    "weight": {"kg": 1.0, "g": 0.001, "mg": 1e-6, "ton": 1000.0, "pound": 0.45359237, "ounce": 0.028349523},
    "area": {"m2": 1.0, "km2": 1e6, "cm2": 1e-4, "hectare": 10000.0, "mu": 666.6667, "acre": 4046.8564},
}


@tool(description="在常用单位之间换算（长度、重量、面积）")
def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算。

    Args:
        value: 待换算的数值。
        from_unit: 源单位，如 km、kg、m2、hectare。
        to_unit: 目标单位。
    """
    src, dst = from_unit.strip().lower(), to_unit.strip().lower()

    category = None
    for name, table in _UNIT_TABLE.items():
        if src in table and dst in table:
            category = name
            break

    if category is None:
        supported = "; ".join(
            f"{name}: {', '.join(sorted(table))}" for name, table in _UNIT_TABLE.items()
        )
        raise ToolError(
            f"不支持从 {from_unit} 换算到 {to_unit}（需同属一个量纲）。可用：{supported}"
        )

    table = _UNIT_TABLE[category]
    result = float(value) * table[src] / table[dst]
    return f"{value} {from_unit} = {result:.6g} {to_unit}"
