"""Multi-worker summarization: chunking, worker pool, map-reduce / refine / ensemble strategies."""

from .chunking import Chunk, chunk_text
from .pipeline import STRATEGIES, SummaryOrchestrator, SummaryResult
from .profiles import PROFILES, SummaryProfile
from .workers import PartialSummary, SummaryError, SummaryWorker, SummaryWorkerPool

__all__ = [
    "Chunk",
    "chunk_text",
    "STRATEGIES",
    "SummaryOrchestrator",
    "SummaryResult",
    "PROFILES",
    "SummaryProfile",
    "PartialSummary",
    "SummaryError",
    "SummaryWorker",
    "SummaryWorkerPool",
]
