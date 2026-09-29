"""配置模块：统一从 `.env` / 环境变量读取，内置常见厂商预设。"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 大模型厂商预设（均使用 OpenAI 兼容协议）
LLM_PRESETS: Dict[str, Dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
    },
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "custom": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
}


class Settings(BaseSettings):
    """运行时配置。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- 大模型 ----------------
    llm_preset: str = "deepseek"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=64, le=32768)

    # ---------------- Agent 行为 ----------------
    max_steps: int = Field(default=8, ge=1, le=30, description="最多工具调用轮数")
    enable_tool_result_feedback: bool = True
    agent_mode: str = Field(default="auto", description="auto / function_call / react")
    tool_timeout: float = Field(default=15.0, gt=0, le=120)

    # ---------------- 工具安全 ----------------
    allowed_file_dirs: str = "./examples"
    http_allowed_hosts: str = ""
    calc_max_length: int = Field(default=200, ge=16, le=2000)

    # ---------------- 服务 ----------------
    cors_origins: str = "*"
    max_upload_mb: int = Field(default=5, ge=1)

    # ------------------------------------------------------------------
    @property
    def resolved_llm_base_url(self) -> str:
        if self.llm_base_url.strip():
            return self.llm_base_url.strip().rstrip("/")
        preset = LLM_PRESETS.get(self.llm_preset.lower(), LLM_PRESETS["deepseek"])
        return preset["base_url"].rstrip("/")

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_model.strip():
            return self.llm_model.strip()
        preset = LLM_PRESETS.get(self.llm_preset.lower(), LLM_PRESETS["deepseek"])
        return preset["model"]

    @property
    def llm_enabled(self) -> bool:
        """是否配置了可用的大模型 Key。未配置时进入「脚本化演示」模式。"""
        key = self.llm_api_key.strip()
        return bool(key) and not key.startswith("sk-xxxx")

    @property
    def allowed_dir_list(self) -> List[str]:
        return [d.strip() for d in self.allowed_file_dirs.split(",") if d.strip()]

    @property
    def http_host_list(self) -> List[str]:
        return [h.strip() for h in self.http_allowed_hosts.split(",") if h.strip()]

    @property
    def cors_list(self) -> List[str]:
        raw = self.cors_origins.strip()
        if raw == "*" or not raw:
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    @field_validator("agent_mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        allowed = {"auto", "function_call", "react"}
        value = v.strip().lower()
        if value not in allowed:
            raise ValueError(f"agent_mode 只能是 {allowed} 之一，收到 {v!r}")
        return value

    def describe(self) -> Dict[str, object]:
        """脱敏后的配置摘要。"""

        def mask(key: str) -> str:
            if not key:
                return "(未配置)"
            return f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"

        return {
            "llm": {
                "preset": self.llm_preset,
                "model": self.resolved_llm_model,
                "base_url": self.resolved_llm_base_url,
                "api_key": mask(self.llm_api_key),
                "enabled": self.llm_enabled,
            },
            "agent": {
                "mode": self.agent_mode,
                "max_steps": self.max_steps,
                "tool_timeout": self.tool_timeout,
                "result_feedback": self.enable_tool_result_feedback,
            },
            "safety": {
                "allowed_file_dirs": self.allowed_dir_list,
                "http_allowed_hosts": self.http_host_list or "(不限制)",
            },
        }


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
