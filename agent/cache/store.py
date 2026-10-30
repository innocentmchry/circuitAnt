import copy
import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .entry import CacheDecision, CacheEntry, ToolCall


class SemanticCacheStore:
    """Manages the in-memory cache state, lookup, updates, and event logging."""

    def __init__(
        self,
        capacity: Optional[int] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.capacity = capacity
        self.event_callback = event_callback
        self.entries: Dict[str, CacheEntry] = {}  # entry_id -> CacheEntry
        self.exact_index: Dict[str, str] = {}  # normalized_request -> entry_id
        self.version = 0

    def _log_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        if self.event_callback:
            event_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "cache_version": self.version,
                "cache_size": len(self.entries),
                **metadata,
            }
            self.event_callback(event_type, event_data)

    def normalize_request(self, text: str) -> str:
        """Conservative normalization: strip extra spaces, preserve case for tokens."""
        return " ".join(text.strip().split())

    def insert(
        self,
        request_id: str,
        testcase: str,
        original_request: str,
        tool_calls: List[ToolCall],
        netlist_id: str = "",
        context_summary: str = "",
        embedding: Optional[List[float]] = None,
        template_str: Optional[str] = None,
        slots: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry:
        """Insert a newly verified plan into the cache."""
        norm_req = self.normalize_request(original_request)
        entry_id = f"entry_{len(self.entries) + 1:04d}_{hashlib.md5(norm_req.encode('utf-8')).hexdigest()[:8]}"

        # Eviction if capacity exceeded (LRU based on access_count / created_at)
        if self.capacity and len(self.entries) >= self.capacity:
            oldest_id = min(self.entries.keys(), key=lambda k: (self.entries[k].access_count, self.entries[k].created_at))
            evicted = self.entries.pop(oldest_id)
            if evicted.normalized_request in self.exact_index:
                del self.exact_index[evicted.normalized_request]
            self._log_event("INVALIDATION", {
                "entry_id": oldest_id,
                "reason": "capacity_eviction",
            })

        entry = CacheEntry(
            entry_id=entry_id,
            request_id=request_id,
            testcase=testcase,
            original_request=original_request,
            normalized_request=norm_req,
            context_summary=context_summary,
            tool_calls=tool_calls,
            netlist_id=netlist_id,
            embedding=embedding,
            template_str=template_str,
            slots=slots or {},
            metadata=metadata or {},
        )

        self.entries[entry_id] = entry
        self.exact_index[norm_req] = entry_id
        self.version += 1

        self._log_event("INSERT", {
            "entry_id": entry_id,
            "request_id": request_id,
            "testcase": testcase,
            "request": original_request,
            "num_tool_calls": len(tool_calls),
            "tool_signatures": [tc.signature_str() for tc in tool_calls],
        })
        return entry

    def get_exact_match(self, request: str) -> Optional[CacheEntry]:
        norm = self.normalize_request(request)
        entry_id = self.exact_index.get(norm)
        if entry_id and entry_id in self.entries:
            return self.entries[entry_id]
        return None

    def record_hit(self, entry_id: str, request_id: str, hit_type: str = "EXACT") -> None:
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            entry.access_count += 1
            entry.last_hit_at = datetime.now(timezone.utc).isoformat()
            self._log_event("HIT", {
                "entry_id": entry_id,
                "request_id": request_id,
                "hit_type": hit_type,
            })

    def record_miss(self, request_id: str, reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        data = {
            "request_id": request_id,
            "reason": reason,
        }
        if metadata:
            data.update(metadata)
        self._log_event("MISS", data)

    def get_all_entries(self) -> List[CacheEntry]:
        return list(self.entries.values())

    def clear(self) -> None:
        self.entries.clear()
        self.exact_index.clear()
        self.version = 0

    def snapshot(self) -> Dict[str, Any]:
        """Deep copy of cache state for reproducible evaluations."""
        return {
            "version": self.version,
            "entries": copy.deepcopy(self.entries),
            "exact_index": copy.deepcopy(self.exact_index),
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        self.version = snap["version"]
        self.entries = copy.deepcopy(snap["entries"])
        self.exact_index = copy.deepcopy(snap["exact_index"])
