"""Functional questions about individual signals: equivalence and symmetry.

Combinational, with flip-flop outputs as free inputs: two signal cones compared
by ABC `cec` (blif.compare_cones).
"""
from .. import blif, utils
from ..design import Design


def check_signal_equivalence(d: Design, a: str, b: str, timeout: int = 120) -> dict:
    """Do signals `a` and `b` have the same value for every input combination?

    {"equivalent": bool, "status": PASS | FAIL | TIMEOUT | ERROR, "detail"}.
    """
    d.net(a), d.net(b)  # raise a clear error for unknown names
    if a == b:
        return {"a": a, "b": b, "equivalent": True, "status": "PASS", "detail": "same signal"}
    proof = blif.compare_cones(d, (a, {}), (b, {}), timeout)
    return {"a": a, "b": b, "equivalent": proof["status"] == "PASS", **proof}


def check_symmetry(d: Design, signal: str, input_a: str, input_b: str, timeout: int = 120) -> dict:
    """Is the function at `signal` unchanged when inputs `input_a` and `input_b` are swapped?

    {"symmetric": bool, "status": PASS | FAIL | TIMEOUT | ERROR, "detail"}.
    """
    for s in (signal, input_a, input_b):
        d.net(s)  # raise a clear error for unknown names
    head = {"signal": signal, "input_a": input_a, "input_b": input_b}
    if input_a == input_b:
        return {**head, "symmetric": True, "status": "PASS", "detail": "identical inputs"}
    drivers, _ = utils.netlist_view(d)
    driven = [s for s in (input_a, input_b) if s in drivers]
    if driven:  # swapping only acts on cone inputs; a gate-driven signal would be silently ignored
        raise ValueError(f"{', '.join(driven)} is driven by a gate; symmetry inputs must be "
                         "primary inputs or flip-flop outputs")
    proof = blif.compare_cones(d, (signal, {}), (signal, {input_a: input_b, input_b: input_a}), timeout)
    return {**head, "symmetric": proof["status"] == "PASS", **proof}
