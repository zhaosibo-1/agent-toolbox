"""内置工具的行为与**安全边界**。

安全相关的用例是这个文件里最重要的部分——「Agent 手上的工具必须有边界」
是项目的核心主张，必须有测试兜住：

- 计算器不许变成任意代码执行入口；
- 文件工具不许被路径穿越绕出去；
- HTTP 工具不许被当成 SSRF 跳板。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools import builtin as bt
from app.tools.base import ToolError


# ===========================================================================
# 计算器
# ===========================================================================
def test_calculator_basic_arithmetic() -> None:
    assert bt.calculator("(3+5)*2/4") == "4"


def test_calculator_respects_precision() -> None:
    assert bt.calculator("1/3", precision=2) == "0.33"
    assert bt.calculator("1/3", precision=6) == "0.333333"


def test_calculator_trailing_zeros_are_trimmed() -> None:
    assert bt.calculator("10/4", precision=4) == "2.5"


def test_calculator_math_functions() -> None:
    assert bt.calculator("sqrt(16)") == "4"
    assert bt.calculator("log10(1000)") == "3"
    assert bt.calculator("max(3, 9)") == "9"
    assert bt.calculator("floor(2.9)") == "2"


def test_calculator_constants() -> None:
    assert bt.calculator("round(pi, 2)") == "3.14"


def test_calculator_rejects_div_by_zero() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.calculator("1/0")
    assert "除数不能为零" in str(excinfo.value)


def test_calculator_rejects_import_injection() -> None:
    """经典攻击：不要指望 eval 的沙箱，直接从语法层拒绝。"""
    for evil in (
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "exec('x=1')",
        "lambda: 1",
    ):
        with pytest.raises(ToolError):
            bt.calculator(evil)


def test_calculator_rejects_attribute_access() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.calculator("(1).__class__")
    assert "不允许的关键字" in str(excinfo.value)


def test_calculator_rejects_unknown_function() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.calculator("danger(1)")
    assert "不支持的函数" in str(excinfo.value)


def test_calculator_rejects_huge_power() -> None:
    """`9**9**9` 会把 CPU 算死，必须拦掉。"""
    with pytest.raises(ToolError) as excinfo:
        bt.calculator("9**99999")
    assert "指数过大" in str(excinfo.value)


def test_calculator_rejects_overlong_expression() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.calculator("1+" * 200 + "1")
    assert "表达式过长" in str(excinfo.value)


def test_calculator_rejects_empty() -> None:
    with pytest.raises(ToolError):
        bt.calculator("   ")


def test_calculator_rejects_syntax_error() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.calculator("1+*2")
    assert "语法错误" in str(excinfo.value)


def test_calculator_respects_configured_max_length() -> None:
    bt.configure(calc_max_length=20, project_root=str(Path.cwd()))
    with pytest.raises(ToolError) as excinfo:
        bt.calculator("1" * 30)
    assert "表达式过长" in str(excinfo.value)


# ===========================================================================
# 时间
# ===========================================================================
def test_current_time_default_beijing() -> None:
    result = bt.current_time()
    assert "UTC+8" in result
    assert "星期" in result


def test_current_time_negative_offset() -> None:
    result = bt.current_time(-5)
    assert "UTC-5" in result


def test_current_time_custom_offset() -> None:
    assert "UTC+0" in bt.current_time(0)


# ===========================================================================
# 文本统计
# ===========================================================================
def test_text_statistics_counts() -> None:
    result = bt.text_statistics("你好世界 hello world")
    assert "总字符数" in result
    assert "中文字数：4" in result
    assert "英文单词/数字：2" in result


def test_text_statistics_lines() -> None:
    result = bt.text_statistics("第一行\n\n第二行\n")
    assert "非空行数：2" in result


def test_text_statistics_top_n_clamped() -> None:
    # top_n 超过上限不会崩，会被夹到 20
    result = bt.text_statistics("你好世界你好世界", top_n=999)
    assert "高频词" in result


def test_text_statistics_rejects_empty() -> None:
    with pytest.raises(ToolError):
        bt.text_statistics("   ")


# ===========================================================================
# 单位换算
# ===========================================================================
def test_convert_unit_length() -> None:
    assert bt.convert_unit(1, "km", "m") == "1 km = 1000 m"


def test_convert_unit_area() -> None:
    result = bt.convert_unit(100, "hectare", "mu")
    assert "1500" in result


def test_convert_unit_weight() -> None:
    assert "1000" in bt.convert_unit(1, "kg", "g")


def test_convert_unit_case_insensitive() -> None:
    assert bt.convert_unit(1, "KM", "M") == "1 KM = 1000 M"


def test_convert_unit_cross_dimension_rejected() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.convert_unit(1, "km", "kg")
    assert "不支持从" in str(excinfo.value)


def test_convert_unit_unknown_unit() -> None:
    with pytest.raises(ToolError):
        bt.convert_unit(1, "banana", "kg")


# ===========================================================================
# JSON 查询
# ===========================================================================
def test_json_query_dict_key() -> None:
    assert bt.json_query('{"name": "张三", "age": 20}', "name") == "张三"


def test_json_query_nested_path() -> None:
    payload = '{"data": {"items": [{"id": 7}]}}'
    assert bt.json_query(payload, "data.items.0.id") == "7"


def test_json_query_negative_index() -> None:
    assert bt.json_query('{"a": [1, 2, 3]}', "a.-1") == "3"


def test_json_query_object_returns_pretty_json() -> None:
    result = bt.json_query('{"a": {"b": 1}}', "a")
    assert '"b": 1' in result


def test_json_query_summary_when_no_path() -> None:
    assert "顶层键" in bt.json_query('{"a": 1, "b": 2}')
    assert "长度 3" in bt.json_query("[1, 2, 3]")
    assert "int" in bt.json_query("42")


def test_json_query_missing_key_lists_available() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.json_query('{"a": 1, "b": 2}', "c")
    assert "可用键" in str(excinfo.value)


def test_json_query_index_out_of_range() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.json_query("[1, 2]", "5")
    assert "越界" in str(excinfo.value)


def test_json_query_bad_json() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.json_query("{不是 json}", "a")
    assert "JSON 解析失败" in str(excinfo.value)


# ===========================================================================
# 文件工具：路径穿越防护
# ===========================================================================
def test_file_tools_disabled_by_default() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.read_text_file("note.txt")
    assert "未启用" in str(excinfo.value)


def test_read_text_file_ok(sandbox: Path) -> None:
    assert "沙箱里的说明文件" in bt.read_text_file("note.txt")


def test_read_text_file_strips_redundant_dir_prefix(sandbox: Path) -> None:
    """模型经常把 `docs/note.txt` 整条丢进来，而允许目录本身就是 docs。"""
    assert "沙箱里的说明文件" in bt.read_text_file("docs/note.txt")


def test_read_text_file_subdirectory(sandbox: Path) -> None:
    assert bt.read_text_file("secret/inner.txt") == "内层文件"


def test_read_text_file_truncates(sandbox: Path) -> None:
    (sandbox / "big.txt").write_text("啊" * 3000, encoding="utf-8")
    result = bt.read_text_file("big.txt", max_chars=500)
    assert "已截断" in result


def test_read_text_file_missing(sandbox: Path) -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.read_text_file("nope.txt")
    assert "文件不存在" in str(excinfo.value)


@pytest.mark.parametrize(
    "evil",
    [
        "../outside.txt",
        "../../../etc/passwd",
        "secret/../../outside.txt",
        "/etc/passwd",
        "C:/Windows/win.ini",
    ],
)
def test_read_text_file_blocks_traversal(sandbox: Path, evil: str) -> None:
    """这是本文件最关键的一组用例：路径穿越必须被拦住。"""
    (sandbox.parent / "outside.txt").write_text("越界内容", encoding="utf-8")
    with pytest.raises(ToolError) as excinfo:
        bt.read_text_file(evil)
    assert "拒绝访问" in str(excinfo.value) or "不存在" in str(excinfo.value)


def test_list_directory_ok(sandbox: Path) -> None:
    result = bt.list_directory(".")
    assert "[文件] note.txt" in result
    assert "[目录] secret/" in result


def test_list_directory_nested(sandbox: Path) -> None:
    assert "[文件] inner.txt" in bt.list_directory("secret")


def test_list_directory_blocks_traversal(sandbox: Path) -> None:
    with pytest.raises(ToolError):
        bt.list_directory("../..")


def test_list_directory_rejects_file(sandbox: Path) -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.list_directory("note.txt")
    assert "不是目录" in str(excinfo.value)


def test_list_directory_missing(sandbox: Path) -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.list_directory("nosuchdir")
    assert "路径不存在" in str(excinfo.value)


def test_list_directory_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    bt.configure(allowed_dirs=[str(empty)], project_root=str(tmp_path))
    assert bt.list_directory(".") == "(目录为空)"


def test_read_text_file_gbk_fallback(sandbox: Path) -> None:
    (sandbox / "gbk.txt").write_bytes("中文编码测试".encode("gb18030"))
    assert "中文编码测试" in bt.read_text_file("gbk.txt")


# ===========================================================================
# HTTP 工具
# ===========================================================================
def test_http_get_rejects_bad_scheme() -> None:
    with pytest.raises(ToolError) as excinfo:
        bt.http_get("file:///etc/passwd")
    assert "http://" in str(excinfo.value)


def test_http_get_blocks_host_outside_whitelist() -> None:
    """白名单在发请求之前就拦截，因此这条用例不会碰网络。"""
    bt.configure(http_hosts=["example.com"], project_root=str(Path.cwd()))
    with pytest.raises(ToolError) as excinfo:
        bt.http_get("https://evil.internal/steal")
    assert "不在白名单内" in str(excinfo.value)


def test_http_get_allows_subdomain_of_whitelist() -> None:
    """白名单里的域名，其子域也应放行——但这一步会真发请求，所以用必然失败的域名验证「已通过白名单」。"""
    bt.configure(http_hosts=["invalid"], project_root=str(Path.cwd()))
    with pytest.raises(ToolError) as excinfo:
        bt.http_get("https://nope.invalid/")
    # 过了白名单，失败在请求阶段
    assert "请求失败" in str(excinfo.value)


def test_http_get_empty_host() -> None:
    with pytest.raises(ToolError):
        bt.http_get("http:///nohost")
