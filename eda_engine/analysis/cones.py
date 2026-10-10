"""Transitive fanin / fanout cones."""
from pathlib import Path

from .. import utils
from ..design import Design


def get_cone(d: Design, signal: str, direction: str = "fanin", shared_with: str | None = None,
             inline_limit: int | None = 10, out_dir: str | Path = ".") -> dict:
    """Gates in the transitive fanin or fanout cone of `signal` (see utils.cone_gates).

    shared_with: another signal; only gates in both fanin/fanout cones are reported.
    {"count", "by_type", "gates" (first `inline_limit`, sorted), "file_path" (all, if more)},
    plus "through_dff" when a flip-flop output was resolved to its D input.
    """
    gates, resolved = utils.cone_gates(d, signal, direction)
    result = {"signal": signal, "direction": direction}
    if resolved.get("through_dff"):
        result.update(resolved)
    if shared_with is not None:
        other, other_resolved = utils.cone_gates(d, shared_with, direction)
        result.update(shared_with=shared_with, signal_cone_count=len(gates),
                      other_cone_count=len(other))
        if other_resolved.get("through_dff"):
            result["shared_with_through_dff"] = other_resolved["through_dff"]
        gates &= other
    names = sorted(gates)
    by_type = {}
    for g in names:
        kind = d.inst(g).getMaster().getName().upper()
        by_type[kind] = by_type.get(kind, 0) + 1
    result.update(count=len(names), by_type=dict(sorted(by_type.items())),
                  gates=names[:inline_limit])
    if inline_limit is not None and len(names) > inline_limit:
        tag = f"_shared_{shared_with}" if shared_with else ""
        result["file_path"] = utils.write_list(
            out_dir, f"cone_{direction}_{signal}{tag}",
            [f"{g} ({d.inst(g).getMaster().getName()})" for g in names])
    return result
