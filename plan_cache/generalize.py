"""Template generalization: transforms linear execution traces into reusable parameterized plans."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import MUTATING_TOOLS
from .masking import MaskResult, mask_request
from .trace import ExecutionTrace, ToolCallRecord


@dataclass
class PlanTemplate:
    masked_intent: str
    steps: list[dict[str, Any]]
    slot_schema: dict[str, str]  # placeholder -> type (e.g. {"<SIGNAL:0>": "signal"})
    mutates_design: bool
    tool_sequence: str
    use_count: int = 0
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanTemplate:
        return cls(
            masked_intent=data["masked_intent"],
            steps=data["steps"],
            slot_schema=data["slot_schema"],
            mutates_design=bool(data["mutates_design"]),
            tool_sequence=data["tool_sequence"],
            use_count=int(data.get("use_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
        )


def find_reference_path(target: Any, result: Any, current_path: str = "") -> str | None:
    """Search for target value in nested result dict/list and return dotted path."""
    if target == result:
        return current_path
    if isinstance(result, dict):
        for k, v in result.items():
            sub_path = f"{current_path}.{k}" if current_path else k
            found = find_reference_path(target, v, sub_path)
            if found is not None:
                return found
    elif isinstance(result, list):
        for idx, item in enumerate(result):
            sub_path = f"{current_path}[{idx}]"
            found = find_reference_path(target, item, sub_path)
            if found is not None:
                return found
    return None


def is_linear_plan(tool_calls: list[ToolCallRecord]) -> tuple[bool, str | None]:
    """Check whether a trace is a linear plan without loops or per-item repetition."""
    if not tool_calls:
        return False, "Trace has no tool calls"
    names = [tc.name for tc in tool_calls]
    # Check for consecutive identical tool calls (per-item repetition)
    for i in range(len(names) - 1):
        if names[i] == names[i + 1]:
            return False, f"Per-item repetition detected: tool '{names[i]}' called consecutively"
    # Check for repeating loop patterns (e.g. [A, B, A, B])
    if len(names) >= 4:
        for period in range(1, len(names) // 2 + 1):
            pattern = names[:period]
            matches = True
            for i in range(period, len(names)):
                if names[i] != pattern[i % period]:
                    matches = False
                    break
            if matches:
                return False, f"Loop detected with repeating tool pattern: {pattern}"
    return True, None


def trace_to_template(trace: ExecutionTrace) -> tuple[PlanTemplate | None, str | None]:
    """Generalize an execution trace into a PlanTemplate.

    Returns (template, None) on success, or (None, rejection_reason) if non-cacheable.
    """
    if not trace.success or not trace.tool_calls:
        return None, "Trace was not successful or has no tool calls"

    linear, reason = is_linear_plan(trace.tool_calls)
    if not linear:
        return None, f"Non-linear plan: {reason}"

    mask_res = mask_request(trace.request)
    slot_lookup: dict[Any, str] = {}
    for placeholder, val in mask_res.slots.items():
        slot_lookup[val] = placeholder
        if isinstance(val, str):
            slot_lookup[val.lower()] = placeholder
            slot_lookup[val.upper()] = placeholder

    steps: list[dict[str, Any]] = []
    earlier_results: list[Any] = []

    for step_idx, tc in enumerate(trace.tool_calls):
        gen_args: dict[str, Any] = {}
        for arg_name, arg_val in tc.arguments.items():
            # 1. Slot check: does arg_val match any slot extracted from request?
            if arg_val in slot_lookup:
                gen_args[arg_name] = slot_lookup[arg_val]
                continue
            if isinstance(arg_val, str) and arg_val.upper() in slot_lookup:
                gen_args[arg_name] = slot_lookup[arg_val.upper()]
                continue

            # 2. Reference check: does arg_val equal a value in an earlier step's result?
            ref_found = False
            for prev_idx, prev_res in enumerate(earlier_results):
                ref_path = find_reference_path(arg_val, prev_res)
                if ref_path is not None:
                    prefix = f"$step_{prev_idx}.result"
                    ref_expr = f"{prefix}.{ref_path}" if not ref_path.startswith("[") else f"{prefix}{ref_path}"
                    gen_args[arg_name] = ref_expr
                    ref_found = True
                    break

            if ref_found:
                continue

            # 3. Constant: keep literal value
            gen_args[arg_name] = arg_val

        steps.append({
            "step": step_idx,
            "tool": tc.name,
            "args": gen_args,
        })
        earlier_results.append(tc.result)

    tool_seq = ",".join(tc.name for tc in trace.tool_calls)
    mutates = any(tc.name in MUTATING_TOOLS for tc in trace.tool_calls)

    template = PlanTemplate(
        masked_intent=mask_res.masked_text,
        steps=steps,
        slot_schema=mask_res.slot_types,
        mutates_design=mutates,
        tool_sequence=tool_seq,
        use_count=0,
        failure_count=0,
    )
    return template, None
