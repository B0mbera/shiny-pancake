"""Pluggable long-term memory shared across runs.

The default store is a small JSON file with keyword-overlap retrieval: dependency-free and good
enough for a demo. Swap in a vector store by implementing the same three methods.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

_WORD = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 2}


@dataclass
class MemoryItem:
    text: str
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: float = field(default_factory=time.time)


class MemoryStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._items: list[MemoryItem] = []
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
            self._items = [MemoryItem(**item) for item in raw]

    def add(self, text: str, tags: list[str] | tuple[str, ...] = ()) -> MemoryItem:
        item = MemoryItem(text=text.strip(), tags=[t.lower() for t in tags])
        with self._lock:
            self._items.append(item)
            self._flush()
        return item

    def search(self, query: str, k: int = 5) -> list[MemoryItem]:
        q = _tokens(query)
        scored = []
        with self._lock:
            items = list(self._items)
        for item in items:
            overlap = len(q & _tokens(item.text)) + 2 * len(q & set(item.tags))
            if overlap:
                scored.append((overlap, item.created_at, item))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [s[2] for s in scored[:k]]

    def all(self) -> list[MemoryItem]:
        with self._lock:
            return list(self._items)

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([asdict(i) for i in self._items], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.replace(tmp, self.path)  # atomic on POSIX and Windows
