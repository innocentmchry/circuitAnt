"""Removing logic that is unused or repeated, and simplifying gates on constants.

Each function edits the design in place and returns counts for the agent.
Gate counts (gates_before / gates_after) are count_gates totals: tie cells
are constants, not gates.
"""
from .. import analysis, edit, utils
from ..design import TIE_VALUE, Design


def remove_dangling_gates(d: Design) -> dict:
    """Remove every gate that feeds no primary output and no flip-flop input
    (flip-flops themselves are kept).  {"gates_removed", "by_type", "gates_before", "gates_after"}.
    """
    before = analysis.count_gates(d)["total_gates"]
    sinks = {bt.getNet().getName() for bt in d.block.getBTerms()
             if bt.getIoType() == "OUTPUT" and bt.getNet()}
    sinks |= {it.getNet().getName() for inst in d.block.getInsts() if inst.getMaster().isSequential()
              for it in inst.getITerms() if it.isInputSignal() and it.getNet()}
    needed = utils.reach_backward(d, sinks)
    by_type = {}
    for inst in list(d.block.getInsts()):
        outs = {it.getNet().getName() for it in inst.getITerms() if it.isOutputSignal() and it.getNet()}
        if d.cell(inst).kind == "sequential" or outs & needed:
            continue
        if d.cell(inst).kind not in TIE_VALUE:  # tie cells are constants, not gates
            kind = inst.getMaster().getName()
            by_type[kind] = by_type.get(kind, 0) + 1
        edit.remove_gate(d, inst)
    return {"gates_removed": sum(by_type.values()), "by_type": dict(sorted(by_type.items())),
            "gates_before": before, "gates_after": analysis.count_gates(d)["total_gates"]}


def collapse_inverter_pairs(d: Design) -> dict:
    """Replace every NOT -> NOT chain by a wire: the second NOT's loads move to the
    first NOT's input; the first NOT is removed too if nothing else uses it.
    Repeats until no pair is left, so longer chains collapse fully.
    {"pairs_collapsed", "gates_removed", "gates_before", "gates_after"}.
    """
    before = analysis.count_gates(d)["total_gates"]
    pairs_collapsed = removed = 0
    while True:
        pairs = analysis.find_inverter_pairs(d, inline_limit=None)["matches"]
        gone = set()  # gates removed this round: their pairs are stale
        for p in pairs:
            if p["first"] in gone or p["second"] in gone:
                continue
            first, second = d.inst(p["first"]), d.inst(p["second"])
            source = next(it for it in first.getITerms() if it.isInputSignal()).getNet()
            output = next(it for it in second.getITerms() if it.isOutputSignal()).getNet()
            output_name = output.getName()
            edit.remove_gate(d, second)
            if d.block.findNet(output_name) is not None:  # still has loads or a port
                edit.merge_nets(d, source, output)
            gone.add(p["second"])
            pairs_collapsed += 1
            removed += 1
            first_out = next(it for it in first.getITerms() if it.isOutputSignal()).getNet()
            if first_out is None or not (first_out.getBTerms() or
                                         any(it.isInputSignal() for it in first_out.getITerms())):
                edit.remove_gate(d, first)
                gone.add(p["first"])
                removed += 1
        if not gone:
            break
    return {"pairs_collapsed": pairs_collapsed, "gates_removed": removed,
            "gates_before": before, "gates_after": analysis.count_gates(d)["total_gates"]}


def merge_duplicate_gates(d: Design) -> dict:
    """Merge structural duplicates (same cell type on the same inputs, swappable inputs
    in any order): one gate of each group is kept and drives all the others' loads.  Gates are visited inputs
    first, so gates that become identical through earlier merges (the next level of
    a repeated block) are merged in the same pass (structural hashing).  The pass
    repeats until it merges nothing: a merge onto a port net renames a net already
    hashed, which the next pass sees.
    {"gates_merged", "passes", "gates_before", "gates_after"}.
    """
    def merge_pass() -> int:
        drivers, _ = utils.netlist_view(d)
        gates = {}  # gate -> (input signals, output signals)
        for out, (cell, _, ins, _) in drivers.items():
            gates.setdefault(cell, (ins, []))[1].append(out)
        outputs = {out for _, outs in gates.values() for out in outs}
        signals = outputs | {s for ins, _ in gates.values() for s in ins}
        order = utils.topo_order(gates, signals - outputs, signals)
        table, merged = {}, 0  # (cell type, input key) -> the gate kept for it
        for name in order:
            inst = d.inst(name)
            pins = {it.getMTerm().getName(): d.signal(it.getNet())
                    for it in inst.getITerms() if it.isInputSignal() and it.getNet()}
            kept = table.setdefault((inst.getMaster().getName(), d.cell(inst).input_key(pins)), inst)
            if kept is inst:
                continue
            outs = {p: inst.findITerm(p).getNet() for p in d.cell(inst).outputs}
            names = {p: n.getName() for p, n in outs.items() if n}
            edit.remove_gate(d, inst)
            merged += 1
            for p, name in names.items():  # each output's loads move to the kept gate's output
                if d.block.findNet(name) is None:
                    continue
                keep = kept.findITerm(p).getNet()
                if keep is None:  # the kept gate drove nothing there: it takes over the net
                    kept.findITerm(p).connect(outs[p])
                else:
                    edit.merge_nets(d, keep, outs[p])
        return merged

    before = analysis.count_gates(d)["total_gates"]
    merged = passes = 0
    while True:
        passes += 1
        n = merge_pass()
        merged += n
        if n == 0:
            break
    return {"gates_merged": merged, "passes": passes, "gates_before": before,
            "gates_after": analysis.count_gates(d)["total_gates"]}


def simplify_gates_with_constant_inputs(d: Design, gate_type: str | None = None) -> dict:
    """Simplify 2-input AND/NAND/OR/NOR/XOR/XNOR-type gates (by function, Cell.logic) with
    one constant input (1'b0 / 1'b1) when the result is a single gate: e.g. NAND(x, 1) ->
    NOT(x), AND(x, 1) -> BUF(x), XOR(x, 1) -> NOT(x), using the library's smallest
    inverter / buffer.  The new gate keeps the old name.  Gates that reduce to a constant
    (e.g. AND(x, 0)) are left unchanged and counted under "skipped".  gate_type: only
    this cell type.  {"gate_type", "simplified_gates", "replacement_breakdown", "skipped"}.
    """
    # (gate, constant input) -> the gate it becomes on its other input, or the constant it is
    rules = {("AND", "1'b1"): "BUF", ("AND", "1'b0"): "1'b0", ("OR", "1'b0"): "BUF", ("OR", "1'b1"): "1'b1",
             ("NAND", "1'b1"): "NOT", ("NAND", "1'b0"): "1'b1", ("NOR", "1'b0"): "NOT", ("NOR", "1'b1"): "1'b0",
             ("XOR", "1'b0"): "BUF", ("XOR", "1'b1"): "NOT", ("XNOR", "1'b1"): "BUF", ("XNOR", "1'b0"): "NOT"}
    breakdown, skipped = {}, {}
    for constant in ("1'b0", "1'b1"):
        for g in analysis.find_gates(d, gate_type, connected_to=constant, inline_limit=None)["gates"]:
            cell = d.cell(d.inst(g["gate"]))
            rule = rules.get((cell.logic, constant))
            others = [p for p in cell.inputs if g["pins"].get(p) not in ("1'b0", "1'b1", None)]
            if rule is None or len(cell.inputs) != 2 or len(others) != 1:  # not a 2-input gate with one constant
                continue
            label = f"{cell.logic}(x, {constant}) -> {rule}"
            if rule not in ("BUF", "NOT"):
                skipped[label] = skipped.get(label, 0) + 1
                continue
            new, a, y = d.smallest_cell("buffer" if rule == "BUF" else "inverter")
            x_name, y_name = g["pins"][others[0]], g["pins"].get(cell.outputs[0])
            edit.remove_gate(d, d.inst(g["gate"]))  # may delete a net only this gate used
            pins = {a: d.block.findNet(x_name), y: d.block.findNet(y_name) if y_name else None}
            edit.add_gate(d, new, {p: n for p, n in pins.items() if n is not None}, g["gate"])
            breakdown[label] = breakdown.get(label, 0) + 1
    return {"gate_type": gate_type.upper() if gate_type else "ALL",
            "simplified_gates": sum(breakdown.values()), "replacement_breakdown": breakdown,
            "skipped": skipped}
