"""Structural patterns: inverter pairs, duplicate gates, flip-flop enable/hold logic.

Purely structural (connectivity and gate types).  Each query lists the first
`inline_limit` matches and writes all of them to a file when there are more
(inline_limit=None: all of them, no file).
"""
from collections import deque
from pathlib import Path

from .. import utils
from ..design import Design


def find_inverter_pairs(d: Design, inline_limit: int | None = 10, out_dir: str | Path = ".") -> dict:
    """NOT gates fed directly by another NOT gate (back-to-back inverters).

    {"count", "matches": [{"first", "second", "input", "output"}] (sorted by first gate),
     "file_path" (all, if more)}.
    """
    drivers, _ = utils.netlist_view(d)
    matches = []
    for out, (name, kind, ins, _) in drivers.items():
        if d.cells[kind].kind != "inverter" or ins[0] not in drivers:
            continue
        first, first_kind, first_ins, _ = drivers[ins[0]]
        if d.cells[first_kind].kind == "inverter":
            matches.append({"first": first, "second": name, "input": first_ins[0], "output": out})
    matches.sort(key=lambda m: m["first"])
    result = {"count": len(matches), "matches": matches[:inline_limit]}
    if inline_limit is not None and len(matches) > inline_limit:
        result["file_path"] = utils.write_list(out_dir, "inverter_pairs", [str(m) for m in matches])
    return result


def find_duplicates(d: Design, inline_limit: int | None = 10, out_dir: str | Path = ".") -> dict:
    """Groups of gates of the same type on the same input signals (structural duplicates).

    {"count" (groups), "removable_gates" (all but one per group),
     "matches": [{"type", "inputs", "gates"}], "file_path" (all, if more)}.
    """
    drivers, _ = utils.netlist_view(d)
    groups = {}
    for name, kind, _, pins in {v[0]: v for v in drivers.values()}.values():  # once per gate
        groups.setdefault((kind, d.cells[kind].input_key(pins)), []).append(name)  # swappable inputs sorted
    matches = sorted(({"type": kind.upper(), "inputs": [s for group in key for s in group], "gates": sorted(names)}
                      for (kind, key), names in groups.items() if len(names) > 1),
                     key=lambda m: m["gates"])
    result = {"count": len(matches), "removable_gates": sum(len(m["gates"]) - 1 for m in matches),
              "matches": matches[:inline_limit]}
    if inline_limit is not None and len(matches) > inline_limit:
        result["file_path"] = utils.write_list(out_dir, "duplicates", [str(m) for m in matches])
    return result


def find_enable_hold(d: Design, near: int = 6, inline_limit: int | None = 10, out_dir: str | Path = ".") -> dict:
    """Flip-flops with enable or hold structures in their D input logic.

    A structural heuristic: a flop matches when its D depends on its
    own output (Q, or QN = inverted Q) through combinational logic and either
      * D is Q through only BUF/NOT (even inversions; odd from QN) -> "direct_hold", or
      * a gate, searched breadth-first back from D, joins a branch with Q within
        `near` gates upstream and a branch without it (the "mux") -> "and_gated_hold_candidate"
        if that gate is an AND, else "mux_or_gated_hold_candidate".
    {"count", "dffs_examined", "self_feedback_candidates" (D depends on own Q at all),
     "classification_breakdown", "matches": [{"dff", "q_signal", "d_signal",
     "classification", "merge_gate"}], "file_path" (all, if more)}.
    """
    drivers, dffs = utils.netlist_view(d)
    cells = {flop: d.cell(d.inst(flop)) for flop in dffs}
    qs = {pins[p] for flop, pins in dffs.items() for p in cells[flop].outputs if pins.get(p)}

    def merge_gate(q: str, dsig: str) -> tuple | None:  # first gate joining a near-Q and a far branch
        memo = {}

        def near_q(s: str, budget: int) -> bool:  # Q within `budget` gates upstream of s
            if s == q:
                return True
            if budget <= 0 or s in qs or s not in drivers:
                return False
            if (s, budget) not in memo:
                memo[(s, budget)] = any(near_q(i, budget - 1) for i in drivers[s][2])
            return memo[(s, budget)]

        queue, visited = deque([dsig]), set()
        while queue:
            s = queue.popleft()
            if s in visited or s == q or s not in drivers:
                continue
            visited.add(s)
            flags = [near_q(i, near) for i in drivers[s][2]]
            if any(flags) and not all(flags):
                return drivers[s]
            queue.extend(i for i, f in zip(drivers[s][2], flags) if f)  # follow only near-Q branches
        return None

    matches, self_feedback, breakdown = [], 0, {}
    for flop, pins in sorted(dffs.items()):
        cell = cells[flop]
        dsig = pins.get(cell.data_inputs[0]) if cell.data_inputs else None
        outs = [(pins[p], p in cell.inverted_outputs) for p in cell.outputs if pins.get(p)]
        fed = [(q, inverted) for q, inverted in outs  # outputs D depends on, with their polarity
               if dsig and utils.depends_on(drivers, dsig, q, stop=qs)]
        if not fed:
            continue
        self_feedback += 1
        q = fed[0][0]
        gate = merge_gate(q, dsig)
        if utils.trace_buf_not(d, drivers, dsig) in fed:  # D = Q, or D = NOT(QN)
            kind = "direct_hold"
        elif gate is not None:
            kind = "and_gated_hold_candidate" if d.cells[gate[1]].logic == "AND" else "mux_or_gated_hold_candidate"
        else:
            continue
        breakdown[kind] = breakdown.get(kind, 0) + 1
        matches.append({"dff": flop, "q_signal": q, "d_signal": dsig, "classification": kind,
                        "merge_gate": gate[0] if gate else None})
    result = {"count": len(matches), "dffs_examined": len(dffs), "self_feedback_candidates": self_feedback,
              "classification_breakdown": dict(sorted(breakdown.items())), "matches": matches[:inline_limit]}
    if inline_limit is not None and len(matches) > inline_limit:
        result["file_path"] = utils.write_list(out_dir, "enable_hold", [str(m) for m in matches])
    return result
