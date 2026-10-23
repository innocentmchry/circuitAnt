"""Masking engine: replaces netlist entities, numbers, paths, and gate types with typed slots."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .constants import CLASS_KEYWORDS, GATE_TYPES


@dataclass
class MaskResult:
    masked_text: str
    slots: dict[str, Any] = field(default_factory=dict)
    slot_types: dict[str, str] = field(default_factory=dict)


def mask_request(request: str) -> MaskResult:
    """Mask entities in the request into typed placeholders.

    Entity types recognized:
    - PATH: file paths ending in .v, .blif, .lef, .lib, .txt, .log or containing / or \\
    - GATE_TYPE: known gate logic types (NAND, NOR, XOR, DFF, etc.)
    - SIGNAL: signals/nets/ports (e.g. n12, n25[0], r3[0])
    - GATE: gate instance names (e.g. g825, g27319)
    - NUMBER: standalone numeric literals
    """
    spans: list[tuple[int, int, str, Any, str]] = []  # (start, end, entity_type, raw_val, norm_val)

    # 1. File paths
    path_pattern = re.compile(
        r"""(?P<path>['"]?(?:[\w\-./\\]+?\.(?:v|blif|lef|lib|txt|log)|[\w\-.]+[/\\][\w\-./\\]+)['"]?)""",
        re.IGNORECASE,
    )
    for m in path_pattern.finditer(request):
        val = m.group("path").strip("'\"")
        spans.append((m.start(), m.end(), "PATH", val, val))

    # 2. Signals with bus indexing: e.g. n25[0], r3_reg[1], data[31]
    indexed_sig = re.compile(r"""\b(?P<sig>[a-zA-Z_][a-zA-Z0-9_]*\[\d+\])(?!\w)""")
    for m in indexed_sig.finditer(request):
        val = m.group("sig")
        spans.append((m.start(), m.end(), "SIGNAL", val, val))

    # 3. Known Gate Types (word boundary, case-insensitive)
    sorted_types = sorted(GATE_TYPES, key=len, reverse=True)
    gate_type_pattern = re.compile(rf"""\b(?P<type>{'|'.join(sorted_types)})\b""", re.IGNORECASE)
    for m in gate_type_pattern.finditer(request):
        raw = m.group("type")
        val = raw.upper()
        spans.append((m.start(), m.end(), "GATE_TYPE", raw, val))

    # 4. Gate instance names: e.g. g825, g27319, g0
    gate_pattern = re.compile(r"""\b(?P<gate>g\d+)\b""")
    for m in gate_pattern.finditer(request):
        val = m.group("gate")
        spans.append((m.start(), m.end(), "GATE", val, val))

    # 5. Simple wire/signal names: e.g. n12, n4042 (distinct from g-gates)
    net_pattern = re.compile(r"""\b(?P<net>n\d+)\b""")
    for m in net_pattern.finditer(request):
        val = m.group("net")
        spans.append((m.start(), m.end(), "SIGNAL", val, val))

    # 6. Standalone numbers
    num_pattern = re.compile(r"""\b(?P<num>\d+)\b""")
    for m in num_pattern.finditer(request):
        val = int(m.group("num"))
        spans.append((m.start(), m.end(), "NUMBER", m.group("num"), val))

    # Filter out CLASS_KEYWORDS if matched
    spans = [s for s in spans if str(s[3]).upper() not in CLASS_KEYWORDS]

    # Resolve overlapping spans: sort by start asc, longer length desc
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    non_overlapping: list[tuple[int, int, str, Any, str]] = []
    last_end = 0
    for start, end, etype, raw, norm in spans:
        if start >= last_end:
            non_overlapping.append((start, end, etype, raw, norm))
            last_end = end

    # Replace from left to right with indexed placeholders
    counts: dict[str, int] = {}
    slots: dict[str, Any] = {}
    slot_types: dict[str, str] = {}

    out_parts: list[str] = []
    cur_idx = 0
    for start, end, etype, raw, norm in non_overlapping:
        out_parts.append(request[cur_idx:start])
        slot_idx = counts.get(etype, 0)
        counts[etype] = slot_idx + 1
        placeholder = f"<{etype}:{slot_idx}>"
        out_parts.append(placeholder)
        slots[placeholder] = norm
        slot_types[placeholder] = etype.lower()
        cur_idx = end
    out_parts.append(request[cur_idx:])

    masked_text = "".join(out_parts)
    return MaskResult(masked_text=masked_text, slots=slots, slot_types=slot_types)
