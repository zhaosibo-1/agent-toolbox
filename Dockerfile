# ---------- 构建阶段：安装依赖 ----------
FROM python:3.12-slim AS builder

WORKDIR /build

# 先只拷贝依赖清单，充分利用 Docker 层缓存：
# 只要 requirements.txt 没变，改代码就不会重新装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- 运行阶段：只带运行时需要的东西 ----------
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Agent Toolbox" \
      org.opencontainers.image.description="手写 Function Calling 循环的 Agent 工具箱：函数签名即工具 Schema" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --from=builder /install /usr/local

# 只拷贝运行所需的文件（测试与脚本镜像内不需要）
COPY app/ ./app/
COPY web/ ./web/
COPY examples/ ./examples/
COPY .env.example README.md LICENSE ./

# 以非 root 用户运行；examples/ 需要可写（上传接口会把文件存进去）
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查指向内置的 /api/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
