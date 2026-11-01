from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "args": self.args}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(tool=data["tool"], args=dict(data.get("args", {})))

    def signature_str(self) -> str:
        arg_items = [f"{k}={v!r}" for k, v in sorted(self.args.items())]
        return f"{self.tool}({', '.join(arg_items)})"


@dataclass
class CacheEntry:
    entry_id: str
    request_id: str
    testcase: str
    original_request: str
    normalized_request: str
    context_summary: str
    tool_calls: List[ToolCall]
    netlist_id: str
    embedding: Optional[List[float]] = None
    template_str: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0
    last_hit_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "request_id": self.request_id,
            "testcase": self.testcase,
            "original_request": self.original_request,
            "normalized_request": self.normalized_request,
            "context_summary": self.context_summary,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "netlist_id": self.netlist_id,
            "template_str": self.template_str,
            "slots": self.slots,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_hit_at": self.last_hit_at,
        }


@dataclass
class CandidateMatch:
    entry: CacheEntry
    cosine_similarity: float = 0.0
    cross_encoder_score: Optional[float] = None
    reliability_score: Optional[float] = None
    rank: int = 0
    validation_passed: bool = False
    rejection_reason: Optional[str] = None
    bound_tool_calls: Optional[List[ToolCall]] = None
    extracted_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry.entry_id,
            "candidate_request": self.entry.original_request,
            "candidate_testcase": self.entry.testcase,
            "cosine_similarity": round(self.cosine_similarity, 4),
            "cross_encoder_score": round(self.cross_encoder_score, 4) if self.cross_encoder_score is not None else None,
            "reliability_score": round(self.reliability_score, 4) if self.reliability_score is not None else None,
            "rank": self.rank,
            "validation_passed": self.validation_passed,
            "rejection_reason": self.rejection_reason,
            "extracted_params": self.extracted_params,
            "bound_tool_calls": [tc.to_dict() for tc in self.bound_tool_calls] if self.bound_tool_calls else None,
        }


@dataclass
class CacheDecision:
    accepted: bool
    decision_type: str  # "HIT", "MISS", "FALLBACK"
    reason: str
    selected_candidate: Optional[CandidateMatch] = None
    tool_calls: Optional[List[ToolCall]] = None
    candidate_scores: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "decision_type": self.decision_type,
            "reason": self.reason,
            "selected_candidate": self.selected_candidate.to_dict() if self.selected_candidate else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls] if self.tool_calls else None,
            "candidate_scores": self.candidate_scores,
        }
