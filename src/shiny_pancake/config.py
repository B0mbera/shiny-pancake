"""Configuration. Every setting comes from an environment variable (see `.env.example`)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

PROVIDERS = ("anthropic", "openai", "offline")

# Defaults per provider: (main model, worker model, base URL)
DEFAULT_MODELS = {
    "anthropic": ("claude-sonnet-5-5", "claude-haiku-5-5"),
    "openai": ("llama3.2", None),  # OpenAI-compatible default targets a local Ollama
    "offline": ("offline-demo", "offline-demo"),
}
DEFAULT_OPENAI_BASE_URL = "http://localhost:11434/v1"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    provider: str = "offline"
    model: str = "offline-demo"
    worker_model: str = "offline-demo"
    fallback_provider: str | None = None

    max_steps: int = 8
    max_tokens: int = 16000
    worker_max_tokens: int = 4096
    effort: str | None = None
    worker_effort: str | None = "low"
    request_timeout: float = 120.0
    max_retries: int = 2
    max_concurrency: int = 8
    prompt_cache: bool = True
    refusal_fallbacks: bool = True

    anthropic_api_key: str | None = None
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_api_key: str | None = None

    workspace: Path = Path("workspace")
    data_dir: Path = Path("data")

    summary_workers: int = 4
    chunk_chars: int = 6000

    inline_workers: int = 2
    approval: str = "auto"
    api_key: str | None = None
    offline_latency: float = 0.05

    # ------------------------------------------------------------------ derived paths
    @property
    def trace_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def memory_path(self) -> Path:
        return self.data_dir / "memory.json"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hub.db"

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Build settings from the environment; keyword overrides win (``None`` means "not given")."""
        overrides = {k: v for k, v in overrides.items() if v is not None}
        anthropic_key = _env("ANTHROPIC_API_KEY")
        openai_key = _env("SP_OPENAI_API_KEY", _env("OPENAI_API_KEY"))

        provider = overrides.pop("provider", None) or _env("SP_PROVIDER")
        if provider is None:
            if anthropic_key:
                provider = "anthropic"
            elif _env("SP_OPENAI_BASE_URL") or openai_key:
                provider = "openai"
            else:
                provider = "offline"
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}; expected one of {PROVIDERS}")

        default_main, default_worker = DEFAULT_MODELS[provider]
        model = (
            overrides.pop("model", None)
            or (_env("SP_MODEL") if provider != "offline" else None)
            or default_main
        )
        worker_model = (
            overrides.pop("worker_model", None)
            or (_env("SP_WORKER_MODEL") if provider != "offline" else None)
            or default_worker
            or model
        )

        values: dict[str, Any] = dict(
            provider=provider,
            model=model,
            worker_model=worker_model,
            fallback_provider=_env("SP_FALLBACK_PROVIDER"),
            max_steps=_env_int("SP_MAX_STEPS", cls.max_steps),
            max_tokens=_env_int("SP_MAX_TOKENS", cls.max_tokens),
            worker_max_tokens=_env_int("SP_WORKER_MAX_TOKENS", cls.worker_max_tokens),
            effort=_env("SP_EFFORT"),
            worker_effort=_env("SP_WORKER_EFFORT", cls.worker_effort),
            request_timeout=_env_float("SP_REQUEST_TIMEOUT", cls.request_timeout),
            max_retries=_env_int("SP_MAX_RETRIES", cls.max_retries),
            max_concurrency=_env_int("SP_MAX_CONCURRENCY", cls.max_concurrency),
            prompt_cache=_env_bool("SP_PROMPT_CACHE", cls.prompt_cache),
            refusal_fallbacks=_env_bool("SP_REFUSAL_FALLBACKS", cls.refusal_fallbacks),
            anthropic_api_key=anthropic_key,
            openai_base_url=_env("SP_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            openai_api_key=openai_key,
            workspace=Path(_env("SP_WORKSPACE", "workspace")),
            data_dir=Path(_env("SP_DATA_DIR", "data")),
            summary_workers=_env_int("SP_SUMMARY_WORKERS", cls.summary_workers),
            chunk_chars=_env_int("SP_CHUNK_CHARS", cls.chunk_chars),
            inline_workers=_env_int("SP_INLINE_WORKERS", cls.inline_workers),
            approval=_env("SP_APPROVAL", cls.approval),
            api_key=_env("SP_API_KEY"),
            offline_latency=_env_float("SP_OFFLINE_LATENCY", cls.offline_latency),
        )
        unknown = set(overrides) - {f.name for f in fields(cls)}
        if unknown:
            raise TypeError(f"Unknown settings: {sorted(unknown)}")
        values.update(overrides)
        settings = cls(**values)
        settings.validate()
        return settings

    def with_overrides(self, **overrides: Any) -> Settings:
        changed = replace(self, **{k: v for k, v in overrides.items() if v is not None})
        changed.validate()
        return changed

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {self.provider!r}")
        for name in ("effort", "worker_effort"):
            value = getattr(self, name)
            if value is not None and value not in EFFORT_LEVELS:
                raise ValueError(f"{name} must be one of {EFFORT_LEVELS}, got {value!r}")
        if self.approval not in ("auto", "deny"):
            raise ValueError("approval must be 'auto' or 'deny'")
        if self.max_steps < 1 or self.summary_workers < 1 or self.chunk_chars < 500:
            raise ValueError("max_steps and summary_workers must be >= 1, chunk_chars >= 500")

    def public_dict(self) -> dict[str, Any]:
        """Settings without secrets, for /api/info and logs."""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name in {"anthropic_api_key", "openai_api_key", "api_key"}:
                out[f.name] = "set" if getattr(self, f.name) else None
                continue
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else value
        return out
