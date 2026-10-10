"""Functional equivalence of the current design with the netlist it was loaded from.

Checked at sequential-cell boundaries (the same convention as cada1067): both
designs must have the same ports and the same sequential cells (by instance name);
each cell's output pins are then free inputs and its input pins are outputs, so
the question is purely combinational.  Proof: both designs written as BLIF with
cells by name, compared by ABC `cec` using the designs' Liberty files (inputs and
outputs matched by name).
"""
import tempfile
from pathlib import Path

from .. import blif, utils
from ..design import Design
from ..design_io import read_design


def check_equivalence(d: Design, timeout: int = 260) -> dict:
    """Is the current design functionally equivalent to the file it was read from?

    {"equivalent", "status": PASS | FAIL | TIMEOUT | ERROR, "method", "original", "reason",
     "failing_outputs" (some outputs that differ, on FAIL)}.
    """
    original = read_design(d.path, **d.read_args)  # read the same way as the current design
    result = {"original": d.path}
    port_bits = lambda design: {bt.getName(): bt.getIoType() for bt in design.block.getBTerms()}
    if port_bits(original) != port_bits(d):
        return {**result, "equivalent": False, "status": "FAIL", "method": "ports",
                "reason": "the ports differ from the original"}
    drivers, flops = utils.netlist_view(d)
    gold_drivers, gold_flops = utils.netlist_view(original)
    if set(flops) != set(gold_flops):
        return {**result, "equivalent": False, "status": "FAIL", "method": "flip-flops",
                "reason": "the flip-flops differ from the original: "
                          f"only in original {sorted(set(gold_flops) - set(flops))[:10]}, "
                          f"only in current {sorted(set(flops) - set(gold_flops))[:10]}"}
    if (drivers, flops) == (gold_drivers, gold_flops):
        return {**result, "equivalent": True, "status": "PASS", "method": "structurally identical"}

    with tempfile.TemporaryDirectory(prefix="circuitant_equiv_") as tmp:
        gold, gate = Path(tmp) / "gold.blif", Path(tmp) / "gate.blif"
        blif.write_blif(original, gold)
        blif.write_blif(d, gate)
        proof = blif.abc_cec(d.liberty, gold, gate, timeout)
    status = {"PROVED": "PASS", "FAILED": "FAIL"}.get(proof["status"], proof["status"])
    result.update(equivalent=status == "PASS", status=status,
                  method="ABC cec at flip-flop boundaries", reason=proof["detail"])
    if proof.get("failing_outputs"):
        result["failing_outputs"] = proof["failing_outputs"]  # e.g. _cut_<flop>_D: where they differ
    return result
