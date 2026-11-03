# syntax=docker/dockerfile:1

# ---------- build: wheels for the app and its dependencies --------------------------------------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[mcp]"

# ---------- runtime: small, non-root, healthchecked ---------------------------------------------
FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="shiny-pancake" \
      org.opencontainers.image.description="Multi-agent LLM hub: agent team, multi-worker summarization, job queue, evals, MCP" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SP_DATA_DIR=/data \
    SP_WORKSPACE=/app/workspace

RUN useradd --create-home --uid 10001 pancake
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels shiny-pancake[mcp] && rm -rf /wheels

WORKDIR /app
COPY workspace ./workspace
COPY evals ./evals
COPY examples ./examples
RUN mkdir -p /data && chown -R pancake:pancake /data /app
USER pancake
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"

ENTRYPOINT ["shiny-pancake"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
