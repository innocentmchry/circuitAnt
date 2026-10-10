"""Resynthesis with ABC: reducing logic depth and mapping onto a set of gate types.

A set of gates is handed to ABC and replaced by ABC's result, entirely through
OpenROAD: its BLIF writer (rmp::create_blif / blif_add_instance / blif_dump) writes
the gates, with the signals around them as inputs and outputs, and its BLIF reader
(rmp::blif_read) puts ABC's netlist in their place.  OpenROAD labels these commands
as exposed for testing.  ABC reads the full library, turns the gates into an AIG,
runs the script, then maps onto the allowed cells only (read_lib -X for the rest).
ABC cannot map without an inverter, so NAND-only / NOR-only results use the
standard circuits in DECOMPOSITIONS instead.  A buffer is always available to ABC:
without one it writes outputs equal to an input as `.barbuf` lines, which OpenROAD's
reader drops; buffers that are not allowed become plain wires afterwards.
"""
import re
import tempfile
from pathlib import Path

import odb

from .. import analysis, blif, edit, utils
from ..design import Design

# Standard NAND-only and NOR-only circuits for 1- and 2-input gates: each step is
# (output, input, input) of one target gate; A, B are the gate's inputs, Y its output.
DECOMPOSITIONS = {
    "NAND": {"NOT": [("Y", "A", "A")],
             "BUF": [("n1", "A", "A"), ("Y", "n1", "n1")],
             "AND": [("n1", "A", "B"), ("Y", "n1", "n1")],
             "OR": [("n1", "A", "A"), ("n2", "B", "B"), ("Y", "n1", "n2")],
             "NOR": [("n1", "A", "A"), ("n2", "B", "B"), ("n3", "n1", "n2"), ("Y", "n3", "n3")],
             "XOR": [("n1", "A", "B"), ("n2", "A", "n1"), ("n3", "B", "n1"), ("Y", "n2", "n3")],
             "XNOR": [("n1", "A", "B"), ("n2", "A", "n1"), ("n3", "B", "n1"), ("n4", "n2", "n3"),
                      ("Y", "n4", "n4")]},
    "NOR": {"NOT": [("Y", "A", "A")],
            "BUF": [("n1", "A", "A"), ("Y", "n1", "n1")],
            "OR": [("n1", "A", "B"), ("Y", "n1", "n1")],
            "AND": [("n1", "A", "A"), ("n2", "B", "B"), ("Y", "n1", "n2")],
            "NAND": [("n1", "A", "A"), ("n2", "B", "B"), ("n3", "n1", "n2"), ("Y", "n3", "n3")],
            "XNOR": [("n1", "A", "B"), ("n2", "A", "n1"), ("n3", "B", "n1"), ("Y", "n2", "n3")],
            "XOR": [("n1", "A", "B"), ("n2", "A", "n1"), ("n3", "B", "n1"), ("n4", "n2", "n3"),
                    ("Y", "n4", "n4")]},
}


def resynthesize(d: Design, gates: set[str], abc_script: str = "", allowed: list[str] | None = None) -> None:
    """Replace `gates` (instance names) by ABC's result for them.

    abc_script: ABC commands run on the AIG before mapping (e.g. "dch").  allowed:
    the logic types the result may use (Cell.logic, e.g. ["NAND", "NOT"]; it must
    include an inverter); None for every cell of the library.
    """
    if not gates:
        return
    logic_cells = {name: c for name, c in d.cells.items() if c.kind in ("combinational", "buffer", "inverter")}
    mapped = None if allowed is None else allowed + ["BUF"]
    exclude = [] if mapped is None else [n for n, c in logic_cells.items() if c.logic not in mapped]
    if allowed is not None and not any(c.kind == "inverter" for n, c in logic_cells.items() if n not in exclude):
        raise ValueError(f"ABC needs an inverter among the allowed gates {allowed}")
    tie = {kind: min((c.area, n) for n, c in d.cells.items() if c.kind == kind)[1] for kind in ("constant1", "constant0")}
    # The reader names new nets or_<id><name> and reuses an existing net of that name:
    # use an id no earlier resynthesis left in the design.
    used = {int(m[1]) for net in d.block.getNets() if (m := re.match(r"or_(\d+)", net.getName()))}
    call_id = max(used, default=0) + 1
    touched = {it.getNet().getName() for name in gates for it in d.inst(name).getITerms() if it.getNet()}
    existing = {inst.getName() for inst in d.block.getInsts()}
    with tempfile.TemporaryDirectory(prefix="circuitant_resyn_") as tmp:
        before, after = Path(tmp) / "gates.blif", Path(tmp) / "abc.blif"
        d.tcl(f"set blif [rmp::create_blif {tie['constant1']} {d.cells[tie['constant1']].outputs[0]} "
              f"{tie['constant0']} {d.cells[tie['constant0']].outputs[0]} {call_id}]")
        for name in gates:
            d.tcl(f"rmp::blif_add_instance $blif {{{name}}}")
        d.tcl(f"rmp::blif_dump $blif {{{before}}}")
        out = blif.run_abc(f"{blif.read_lib(d.liberty)}; read_blif {before}; strash; {abc_script}; "
                           f"{blif.read_lib(d.liberty, exclude)}; map; write_blif {after}")
        if out is None or not after.exists():
            raise ValueError(f"ABC failed: {(out or 'timeout').strip()[-500:]}")
        if d.tcl(f"rmp::blif_read $blif {{{after}}}") != "1":
            raise ValueError("OpenROAD could not read ABC's result back (see the OpenROAD log)")
    for name in touched:  # the replaced gates' internal nets, now unconnected
        net = d.block.findNet(name)
        if net is not None and not net.getITerms() and not net.getBTerms():
            odb.dbNet_destroy(net)
    d.scan_constants()  # the reader adds its own tie cells
    if allowed is not None and "BUF" not in allowed:  # ABC's buffers become wires
        for inst in [i for i in d.block.getInsts() if i.getName() not in existing and d.cell(i).kind == "buffer"]:
            a, y = (inst.findITerm(p).getNet() for p in (d.cell(inst).inputs[0], d.cell(inst).outputs[0]))
            edit.remove_gate(d, inst)
            if a is not None and y is not None:
                edit.merge_nets(d, a, y)


def reduce_depth(d: Design) -> dict:
    """Restructure all combinational logic for the smallest maximum logic depth
    (ABC: dch, then delay-oriented mapping); flip-flops stay.
    {"depth_before", "depth_after", "gates_before", "gates_after"}.
    """
    depth = lambda: analysis.get_max_depth(d, "ANY_START", "ANY_END")["depth"]
    result = {"depth_before": depth(), "gates_before": analysis.count_gates(d)["total_gates"]}
    resynthesize(d, {i.getName() for i in d.block.getInsts() if d.is_gate(i)}, "dch")
    return {**result, "depth_after": depth(), "gates_after": analysis.count_gates(d)["total_gates"]}


def remap_gates(d: Design, allowed: list[str], signal: str | None = None) -> dict:
    """Rebuild all combinational logic -- or only the fanin cone of `signal` -- with
    only the `allowed` logic types (e.g. ["NAND", "NOT"]).  A single NAND or NOR is
    allowed too (inverters then become NAND(x, x) / NOR(x, x)).
    {"allowed", "signal", "gates_replaced", "gates_before", "gates_after", "breakdown_after"}.
    """
    allowed = [a.upper() for a in allowed]
    universal = allowed in (["NAND"], ["NOR"])
    gates = scope(d, signal)
    before = analysis.count_gates(d)["total_gates"]
    resynthesize(d, gates, "", allowed + ["NOT"] if universal else allowed)
    if universal:  # the inverters ABC needed, as NAND(x, x) / NOR(x, x)
        replace_gate_type(d, "NOT", allowed, signal)
    counts = analysis.count_gates(d)
    return {"allowed": allowed, "signal": signal, "gates_replaced": len(gates), "gates_before": before,
            "gates_after": counts["total_gates"], "breakdown_after": {k: v for k, v in counts["breakdown"].items() if v}}


def replace_gate_type(d: Design, gate_type: str, allowed: list[str], signal: str | None = None) -> dict:
    """Replace every gate of logic type `gate_type` (in the fanin cone of `signal`, or
    everywhere) by equivalent logic using only the `allowed` types.  A single NAND or
    NOR uses the standard circuits (e.g. XOR -> 4 NAND); otherwise ABC maps each set.
    {"gate_type", "allowed", "signal", "replaced", "removed", "added" ({type: n}),
     "gates_before", "gates_after"}.
    """
    gate_type, allowed = gate_type.upper(), [a.upper() for a in allowed]
    targets = {g for g in scope(d, signal) if d.cell(d.inst(g)).logic == gate_type}
    before = analysis.count_gates(d)["breakdown"]
    if allowed in (["NAND"], ["NOR"]):
        steps = DECOMPOSITIONS[allowed[0]].get(gate_type)
        cells = sorted((c.area, n) for n, c in d.cells.items() if c.logic == allowed[0] and len(c.inputs) == 2)
        if steps is None or not cells:
            raise ValueError(f"no {allowed[0]}-only circuit for {gate_type}")
        cell = d.cells[cells[0][1]]
        for name in targets:
            inst = d.inst(name)
            source = d.cell(inst)
            if len(source.inputs) > 2:
                raise ValueError(f"{name}: only 1- and 2-input gates have a {allowed[0]}-only circuit")
            nets = {k: inst.findITerm(p).getNet() for k, p in zip(("A", "B"), source.inputs)}
            nets["Y"] = inst.findITerm(source.outputs[0]).getNet()
            names = {k: n.getName() for k, n in nets.items() if n}
            edit.remove_gate(d, inst)
            nets = {k: d.block.findNet(n) for k, n in names.items()}  # removal may delete unused nets
            for out, a, b in steps:
                if nets.get(out) is None:
                    nets[out] = edit.new_net(d, f"{name}_{out}")
                edit.add_gate(d, cells[0][1], {cell.inputs[0]: nets[a], cell.inputs[1]: nets[b],
                                               cell.outputs[0]: nets[out]}, name if out == "Y" else f"{name}_{out}")
    else:
        resynthesize(d, targets, "", allowed)
    after = analysis.count_gates(d)["breakdown"]
    return {"gate_type": gate_type, "allowed": allowed, "signal": signal, "replaced": len(targets),
            "removed": {k: before[k] - after.get(k, 0) for k in before if before[k] > after.get(k, 0)},
            "added": {k: after[k] - before.get(k, 0) for k in after if after[k] > before.get(k, 0)},
            "gates_before": sum(before.values()), "gates_after": sum(after.values())}


def scope(d: Design, signal: str | None) -> set[str]:
    """The combinational gates in the fanin cone of `signal`, or all of them."""
    if signal is not None:
        return utils.cone_gates(d, signal)[0]
    return {i.getName() for i in d.block.getInsts() if d.is_gate(i)}
