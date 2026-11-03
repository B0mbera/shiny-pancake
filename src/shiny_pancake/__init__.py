"""shiny-pancake: a production-minded multi-agent LLM hub.

Public entry points (imported lazily so `import shiny_pancake` stays cheap)::

    from shiny_pancake import Hub, Settings

    async with Hub(Settings.from_env(provider="offline")) as hub:
        answer = await hub.ask("What is 12 * (3 + 4)?")
        summary = await hub.summarize(text, strategy="ensemble")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

__all__ = ["Hub", "HubResult", "Settings", "__version__"]

if TYPE_CHECKING:  # pragma: no cover
    from .config import Settings
    from .hub.service import Hub, HubResult


def __getattr__(name: str) -> Any:
    if name == "Settings":
        from .config import Settings

        return Settings
    if name in {"Hub", "HubResult"}:
        from .hub import service

        return getattr(service, name)
    raise AttributeError(f"module 'shiny_pancake' has no attribute {name!r}")
