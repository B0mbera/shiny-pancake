from __future__ import annotations

import os
from pathlib import Path

import pytest

# Tests never talk to a real provider and never wait on simulated latency.
for var in ("ANTHROPIC_API_KEY", "SP_PROVIDER", "SP_OPENAI_BASE_URL", "OPENAI_API_KEY", "SP_API_KEY"):
    os.environ.pop(var, None)
os.environ["SP_OFFLINE_LATENCY"] = "0"
os.environ["SP_NO_ART"] = "1"

from shiny_pancake.config import Settings  # noqa: E402
from shiny_pancake.hub.service import Hub  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "notes").mkdir(parents=True)
    (ws / "notes" / "kickoff.md").write_text(
        "# Kickoff\n\nBence will ship the MCP bridge by Friday, November 7.\n"
        "Dora must review the write_file tool before launch.\n"
        "Prompt injection in incident reports is a real risk.\n",
        encoding="utf-8",
    )
    (ws / "readme.txt").write_text("Pancakes are best served warm.", encoding="utf-8")
    return ws


@pytest.fixture
def settings(tmp_path: Path, workspace: Path) -> Settings:
    return Settings(
        provider="offline",
        model="offline-demo",
        worker_model="offline-demo",
        workspace=workspace,
        data_dir=tmp_path / "data",
        offline_latency=0.0,
        summary_workers=3,
        chunk_chars=800,
        inline_workers=0,
    )


@pytest.fixture
async def hub(settings: Settings):
    async with Hub(settings) as h:
        yield h


LONG_DOC = "\n\n".join(
    f"Section {i}. The team must review component {i} before the release. "
    f"Risk {i}: latency could grow when traffic spikes. Component {i} handles {i * 10} requests per second. "
    "Every tool call is traced, validated and bounded by a timeout so failures stay visible."
    for i in range(1, 13)
)
