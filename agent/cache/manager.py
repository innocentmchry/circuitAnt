"""The cache the agent uses: the requests answered so far, with the engine tools each one ran.

A new request is embedded and compared with them.  When the most similar one is close enough
(cosine_threshold) and the two do not disagree on an opposite-word pair (fanin / fanout, count /
list, ... see validation/constants.py), the request is a hit: the agent starts it with a short first
LLM turn that offers only the tools that request ran (see agent.Agent._answer).  The cache chooses
tools only; the model writes their arguments from the new request.

The cache lives in memory and fills during a run.  Config section `cache`:
  enabled           false: the agent runs without it (no import, no model)
  embedding_model   sentence-transformers model (default all-MiniLM-L6-v2)
  cosine_threshold  the similarity a hit needs (default 0.80)
"""
from dataclasses import dataclass

from .entry import CacheEntry, ToolCall
from .models.embeddings import EmbeddingModel
from .store import SemanticCacheStore
from .validation.polarity import check_negative_constraints

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.80


@dataclass
class Lookup:
    """What the cache found for one request."""
    entry: CacheEntry | None = None  # the most similar earlier request (None: nothing cached yet)
    similarity: float = 0.0
    rejected: str = ""               # why the entry is not used ("" = a hit)
    refused: bool = False            # set by the agent: the model answered the short turn without its tools

    @property
    def hit(self) -> bool:
        return self.entry is not None and not self.rejected

    @property
    def tools(self) -> list[str]:
        """The distinct engine tools the earlier request ran, in order."""
        return list(dict.fromkeys(call.tool for call in self.entry.tool_calls)) if self.entry else []

    def status(self) -> str:
        """hit / refused / miss, for the run analysis."""
        return "refused" if self.refused else "hit" if self.hit else "miss"


class AgentCacheManager:
    def __init__(self, config: dict) -> None:
        self.store = SemanticCacheStore()
        self.embedding_model = EmbeddingModel(model_name=config.get("embedding_model", DEFAULT_MODEL))
        self.threshold = float(config.get("cosine_threshold", DEFAULT_THRESHOLD))
        self.embedding_model.encode("warm-up")  # load the model now: a missing package fails at start

    def lookup(self, request: str) -> Lookup:
        """The most similar earlier request, and whether it is a hit."""
        entries = [(e.entry_id, e.embedding) for e in self.store.get_all_entries() if e.embedding]
        if not entries:
            return Lookup(rejected="cache empty")
        embedding, _ = self.embedding_model.encode(request)
        entry_id, similarity, _ = self.embedding_model.retrieve_top_k(embedding, entries, top_k=1, threshold=-1.0)[0]
        found = Lookup(self.store.entries[entry_id], similarity)
        if similarity < self.threshold:
            found.rejected = f"similarity {similarity:.3f} below {self.threshold}"
        else:
            agree, reason = check_negative_constraints(request, found.entry.original_request)
            if not agree:
                found.rejected = reason
        return found

    def record(self, request: str, case: str, calls: list[tuple[str, dict]]) -> None:
        """Remember a request and the engine calls it ran successfully ([] : nothing to remember)."""
        if not calls:
            return
        tool_calls = [ToolCall(tool=name, args=args) for name, args in calls]
        known = self.store.get_exact_match(request)
        if known and [c.tool for c in known.tool_calls] == [c.tool for c in tool_calls]:
            return  # the same request with the same tools is cached already
        embedding, _ = self.embedding_model.encode(request)
        self.store.insert(request_id=f"{case}_{len(self.store.entries) + 1}", testcase=case,
                          original_request=request, tool_calls=tool_calls, netlist_id=case,
                          embedding=embedding)
