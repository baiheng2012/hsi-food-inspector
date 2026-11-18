# uv 官方镜像（自带 uv 和 Python 3.12)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 构建时编译字节码；拷贝而不是硬链接，避免跨文件系统警告
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# 第 1 层：只装第三方依赖。pyproject.toml 和 uv.lock 不变时，这一层会被缓存，改代码不重装依赖
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# 第 2 层：拷贝代码，安装本项目
COPY src ./src
COPY scripts ./scripts
COPY params.yaml ./
RUN uv sync --frozen --no-dev

# 之后直接用虚拟环境里的 python / uvicorn
ENV PATH="/app/.venv/bin:$PATH"

# 第 3 层：构建时训练并导出模型（数据由模拟器按固定种子确定性生成），镜像自包含
RUN python scripts/export_model.py

# 小矩阵上多线程只增加抖动
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HSI_MODEL_PATH=/app/models/engine.npz

# 非 root 运行（不 chown，避免把整个 .venv 在镜像里复制一份）
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "hsi.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
