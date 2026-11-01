"""Semantic cache of tool choices (circuitAnt agent).

A request similar to an earlier one starts with a short first LLM turn offering only the tools the
earlier request used; see manager.py.  The experiments on the cache are in experiments/semantic_cache/.
"""

from .entry import CacheDecision, CacheEntry, CandidateMatch, ToolCall
from .store import SemanticCacheStore

__all__ = [
    "SemanticCacheStore",
    "CacheDecision",
    "CacheEntry",
    "CandidateMatch",
    "ToolCall",
]
