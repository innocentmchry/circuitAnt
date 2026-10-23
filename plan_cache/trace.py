"""Trace recording data structures for agent tool calls and LLM metrics."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """Record of a single tool execution."""
    name: str
    arguments: dict[str, Any]
    result: Any
    elapsed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallRecord:
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            elapsed=float(data.get("elapsed", 0.0)),
        )


@dataclass
class ExecutionTrace:
    """Complete trace of a request run: tool calls, timing, and token metrics."""
    request: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_clock_time: float = 0.0
    llm_time: float = 0.0
    tool_time: float = 0.0
    success: bool = True
    answer: str = ""
    context_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "wall_clock_time": self.wall_clock_time,
            "llm_time": self.llm_time,
            "tool_time": self.tool_time,
            "success": self.success,
            "answer": self.answer,
            "context_summary": self.context_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionTrace:
        return cls(
            request=data["request"],
            tool_calls=[ToolCallRecord.from_dict(tc) for tc in data.get("tool_calls", [])],
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            wall_clock_time=float(data.get("wall_clock_time", 0.0)),
            llm_time=float(data.get("llm_time", 0.0)),
            tool_time=float(data.get("tool_time", 0.0)),
            success=bool(data.get("success", True)),
            answer=data.get("answer", ""),
            context_summary=data.get("context_summary", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> ExecutionTrace:
        return cls.from_dict(json.loads(json_str))
