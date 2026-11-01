import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .constants import CLASS_KEYWORDS, GATE_TYPES


@dataclass
class ExtractedEntities:
    signals: List[str] = field(default_factory=list)
    gates: List[str] = field(default_factory=list)
    gate_types: List[str] = field(default_factory=list)
    numbers: List[int] = field(default_factory=list)
    paths: List[str] = field(default_factory=list)
    directions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": self.signals,
            "gates": self.gates,
            "gate_types": self.gate_types,
            "numbers": self.numbers,
            "paths": self.paths,
            "directions": self.directions,
        }


def extract_entities(text: str) -> ExtractedEntities:
    """Deterministically extracts EDA entities from a natural-language prompt."""
    entities = ExtractedEntities()

    # 1. File paths (ending in .v, .blif, .lef, .lib, .txt, .log, or directory + file descriptions)
    # Check for pattern like: "file foo.v located in the directory dir/bar/"
    dir_file_pattern = re.compile(
        r"""(?:file\s+['"]?(?P<f1>[\w\-.]+\.v)['"]?\s+located\s+in\s+(?:the\s+)?directory\s+['"]?(?P<d1>[\w\-./\\]+)['"]?|directory\s+['"]?(?P<d2>[\w\-./\\]+)['"]?.*?file\s+['"]?(?P<f2>[\w\-.]+\.v)['"]?)""",
        re.IGNORECASE,
    )
    df_match = dir_file_pattern.search(text)
    if df_match:
        d = (df_match.group("d1") or df_match.group("d2") or "").rstrip("/\\")
        # Strip trailing dot representing current dir or punctuation: e.g. testcase/test002/. -> testcase/test002
        if d.endswith("/.") or d.endswith("\\."):
            d = d[:-2]
        elif d.endswith("."):
            d = d.rstrip(".")
        f = df_match.group("f1") or df_match.group("f2") or ""
        if d and f:
            norm_p = os.path.normpath(f"{d}/{f}").replace("\\", "/")
            if norm_p not in entities.paths:
                entities.paths.append(norm_p)

    path_pattern = re.compile(
        r"""(?P<path>['"]?(?:[\w\-./\\]+?\.(?:v|blif|lef|lib|txt|log)|[\w\-.]+[/\\][\w\-./\\]+)['"]?)""",
        re.IGNORECASE,
    )
    for m in path_pattern.finditer(text):
        val = m.group("path").strip("'\"").rstrip(".,;")
        val = os.path.normpath(val).replace("\\", "/")
        if val not in entities.paths and not val.endswith((".", ",")):
            entities.paths.append(val)

    # 2. Signals with bus indexing: e.g. n7[59], n11[25], r3[0]
    indexed_sig = re.compile(r"""\b(?P<sig>[a-zA-Z_][a-zA-Z0-9_]*\[\d+\])(?!\w)""")
    for m in indexed_sig.finditer(text):
        val = m.group("sig")
        if val not in entities.signals and val not in CLASS_KEYWORDS:
            entities.signals.append(val)

    # 3. Known Gate Types
    sorted_types = sorted(GATE_TYPES, key=len, reverse=True)
    gate_type_pattern = re.compile(rf"""\b(?P<type>{'|'.join(sorted_types)})\b""", re.IGNORECASE)
    for m in gate_type_pattern.finditer(text):
        val = m.group("type").upper()
        if val not in entities.gate_types:
            entities.gate_types.append(val)

    # 4. Gate instance names: e.g. g3055, g27319, g0
    gate_pattern = re.compile(r"""\b(?P<gate>g\d+)\b""")
    for m in gate_pattern.finditer(text):
        val = m.group("gate")
        if val not in entities.gates:
            entities.gates.append(val)

    # 5. Simple wire/signal names: e.g. n12, n20, n4042 (distinct from gates)
    net_pattern = re.compile(r"""\b(?P<net>n\d+)\b""")
    for m in net_pattern.finditer(text):
        val = m.group("net")
        if val not in entities.signals and val not in CLASS_KEYWORDS:
            entities.signals.append(val)

    # 6. Direction keywords: fanin / fanout
    if re.search(r"\bfanin\b", text, re.IGNORECASE):
        entities.directions.append("fanin")
    if re.search(r"\bfanout\b", text, re.IGNORECASE):
        entities.directions.append("fanout")

    # 7. Standalone numbers
    num_pattern = re.compile(r"""\b(?P<num>\d+)\b""")
    for m in num_pattern.finditer(text):
        try:
            val = int(m.group("num"))
            # Avoid duplicating numbers that are part of signals already matched
            entities.numbers.append(val)
        except ValueError:
            pass

    return entities
