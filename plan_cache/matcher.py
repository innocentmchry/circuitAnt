"""Plan matching and deterministic slot filling against OpenDB design."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .constants import CLASS_KEYWORDS, GATE_TYPES, OPPOSITE_PAIRS
from .generalize import PlanTemplate
from .masking import MaskResult, mask_request
from .store import PlanStore


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity in pure Python (standard library math)."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def check_opposite_words(req_text: str, template_intent: str) -> tuple[bool, str | None]:
    """Check if request and template disagree on opposite word pairs.

    Returns (True, None) if check passes, or (False, reason) on disagreement.
    """
    req_lower = req_text.lower()
    tpl_lower = template_intent.lower()

    for w1, w2 in OPPOSITE_PAIRS:
        r_w1 = bool(re.search(rf"\b{re.escape(w1)}\b", req_lower))
        r_w2 = bool(re.search(rf"\b{re.escape(w2)}\b", req_lower))
        t_w1 = bool(re.search(rf"\b{re.escape(w1)}\b", tpl_lower))
        t_w2 = bool(re.search(rf"\b{re.escape(w2)}\b", tpl_lower))

        # Direct opposite mismatch: one has w1, the other has w2
        if (r_w1 and t_w2) or (r_w2 and t_w1):
            return False, f"Opposite word mismatch: '{w1}' vs '{w2}'"

        # Disagreement when one text specifies a polarity/action and the other specifies its alternative
        if (r_w1 != t_w1) and (r_w2 or t_w2):
            return False, f"Opposite word disagreement on pair ('{w1}', '{w2}')"

    return True, None


def validate_slot_in_design(slot_type: str, slot_val: Any, tools: Any) -> tuple[bool, str | None]:
    """Validate that a slot value exists deterministically in the loaded design."""
    if tools.design is None:
        return False, "No design currently loaded in tools"

    d = tools.design

    if slot_type == "signal":
        val_str = str(slot_val)
        if val_str in CLASS_KEYWORDS:
            return True, None
        # Check net in OpenDB
        net = d.block.findNet(val_str) or d.block.findNet(d.escaped(val_str))
        if net is not None:
            return True, None
        # Check bterm (port)
        bterm = d.block.findBTerm(val_str) or d.block.findBTerm(d.escaped(val_str))
        if bterm is not None:
            return True, None
        return False, f"Signal {val_str!r} not found in design netlist"

    elif slot_type == "gate":
        val_str = str(slot_val)
        inst = d.block.findInst(val_str) or d.block.findInst(d.escaped(val_str))
        if inst is not None:
            return True, None
        return False, f"Gate instance {val_str!r} not found in design netlist"

    elif slot_type == "gate_type":
        val_str = str(slot_val).upper()
        if val_str not in GATE_TYPES:
            return False, f"Unknown gate type {val_str!r}"
        return True, None

    elif slot_type == "number":
        try:
            int(slot_val)
            return True, None
        except (ValueError, TypeError):
            return False, f"Invalid number {slot_val!r}"

    elif slot_type == "path":
        # Paths are strings
        return True, None

    return True, None


@dataclass
class MatchResult:
    matched: bool
    template: PlanTemplate | None = None
    template_id: int | None = None
    filled_slots: dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0
    reason: str | None = None


class PlanMatcher:
    """Matches a new request against cached plan templates."""

    def __init__(self, store: PlanStore, embedding_fn: Any | None = None) -> None:
        self.store = store
        self.embedding_fn = embedding_fn

    def get_embedding(self, text: str) -> list[float]:
        if self.embedding_fn is None:
            raise ValueError("No embedding function provided to PlanMatcher")
        return self.embedding_fn(text)

    def find_match(
        self,
        request: str,
        tools: Any,
        similarity_threshold: float = 0.85,
    ) -> MatchResult:
        """Find the best matching template if all validation rules pass."""
        templates = self.store.get_all_templates()
        if not templates:
            return MatchResult(matched=False, reason="Cache is empty")

        mask_res = mask_request(request)
        req_embedding = self.get_embedding(mask_res.masked_text)

        available_tools = {
            name for name, fn in tools.__class__.__dict__.items()
            if not name.startswith("_") and callable(fn)
        }

        best_sim = -1.0
        best_candidate: tuple[int, PlanTemplate, dict[str, Any]] | None = None
        rejection_reasons: list[str] = []

        for tid, tpl, t_emb in templates:
            sim = cosine_similarity(req_embedding, t_emb)
            if sim < similarity_threshold:
                continue

            # (d) Opposite word check
            opp_ok, opp_reason = check_opposite_words(request, tpl.masked_intent)
            if not opp_ok:
                rejection_reasons.append(f"Template {tid}: {opp_reason}")
                continue

            # (c) Every tool in template still exists
            tools_exist = all(step["tool"] in available_tools for step in tpl.steps)
            if not tools_exist:
                rejection_reasons.append(f"Template {tid}: Missing tool in runtime")
                continue

            # (b) Every slot can be filled deterministically
            slots_ok = True
            filled_slots: dict[str, Any] = {}

            # Collect required slots across all steps
            for step in tpl.steps:
                for arg_val in step["args"].values():
                    if isinstance(arg_val, str) and arg_val.startswith("<") and arg_val.endswith(">"):
                        if arg_val not in mask_res.slots:
                            slots_ok = False
                            rejection_reasons.append(f"Template {tid}: Cannot fill slot {arg_val}")
                            break
                        slot_val = mask_res.slots[arg_val]
                        slot_type = tpl.slot_schema.get(arg_val, "unknown")
                        # Validate slot value exists in netlist
                        valid, v_reason = validate_slot_in_design(slot_type, slot_val, tools)
                        if not valid:
                            slots_ok = False
                            rejection_reasons.append(f"Template {tid}: {v_reason}")
                            break
                        filled_slots[arg_val] = slot_val
                if not slots_ok:
                    break

            if not slots_ok:
                continue

            if sim > best_sim:
                best_sim = sim
                best_candidate = (tid, tpl, filled_slots)

        if best_candidate is not None:
            tid, tpl, filled_slots = best_candidate
            return MatchResult(
                matched=True,
                template=tpl,
                template_id=tid,
                filled_slots=filled_slots,
                similarity=best_sim,
            )

        reason_str = "; ".join(rejection_reasons[:3]) if rejection_reasons else "No candidate met similarity threshold"
        return MatchResult(matched=False, reason=reason_str)
