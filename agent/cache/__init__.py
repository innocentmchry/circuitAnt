"""CircuitAnt Semantic Tool-Call Caching Package.

Provides transparent LLM-planning avoidance through hybrid semantic retrieval,
cross-encoder reranking, and GenCache structural anti-unification.
"""

from .entry import CacheDecision, CacheEntry, CandidateMatch, ToolCall
from .manager import AgentCacheManager
from .store import SemanticCacheStore

__all__ = [
    "AgentCacheManager",
    "SemanticCacheStore",
    "CacheDecision",
    "CacheEntry",
    "CandidateMatch",
    "ToolCall",
]
