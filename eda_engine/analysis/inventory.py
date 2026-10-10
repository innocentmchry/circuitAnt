"""What is in the design: gate counts, finding gates, and the ports."""
from fnmatch import fnmatchcase
from pathlib import Path

from .. import utils
from ..design import CONSTANT, LOGIC_TYPES, TIE_VALUE, Design


def count_gates(d: Design) -> dict:
    """{"total_gates": n, "breakdown": {cell type: n}} with every gate and flip-flop type
    of the library listed, zeros included (e.g. AND ... DFF for the ICCAD library).
    Tie cells are constants, not gates; physical-only cells are left out."""
    counts = {name: 0 for name, c in d.cells.items() if c.kind not in ("other", *TIE_VALUE)}
    for inst in d.block.getInsts():
        kind = inst.getMaster().getName()
        if kind in counts:
            counts[kind] += 1
    return {"total_gates": sum(counts.values()), "breakdown": counts}


def find_gates(d: Design, gate_type: str | None = None, connected_to: str | None = None,
               pin: str | None = None, name: str | None = None, inline_limit: int | None = 10,
               out_dir: str | Path = ".") -> dict:
    """Gates matching every filter given, each as {"gate", "type", "pins": {pin: signal}}.

    gate_type: a logic type (LOGIC_TYPES: AND, NAND, ..., AOI21, MUX2, FA, HA), matched by
    what the cell computes (Cell.logic, any library and input count); anything else is
    a cell-name pattern, matched case-insensitively (DFF, NAND2_X1, NAND2_*).
    connected_to: a signal on any of the gate's pins (only on `pin` if given, e.g. CK
    for flip-flops on a clock), or 1'b0 / 1'b1 for gates tied to a constant.  name: one
    gate instance.  {"count", "by_cell" ({cell type: count}), "gates" (first
    `inline_limit`, sorted by name), "file_path" (all, if more)}.
    """
    def wanted(inst) -> bool:
        if gate_type is None:
            return True
        if gate_type.upper() in LOGIC_TYPES:
            return d.cell(inst).logic == gate_type.upper()
        return fnmatchcase(inst.getMaster().getName().upper(), gate_type.upper())

    if name is not None:
        candidates = [d.inst(name)]
    elif connected_to is not None:
        if connected_to in CONSTANT.values():  # 1'b0 / 1'b1: every net of that constant
            nets = [n for n in d.block.getNets() if d.signal(n) == connected_to]
        else:
            nets = [d.net(connected_to)]
        candidates = {it.getInst().getName(): it.getInst() for net in nets for it in net.getITerms()
                      if pin is None or it.getMTerm().getName() == pin.upper()}.values()
    else:
        candidates = d.block.getInsts()
    gates = []
    for inst in candidates:
        if d.cell(inst).kind in TIE_VALUE:  # tie cells are constants, not gates
            continue
        if wanted(inst):
            gates.append(utils.gate_info(d, inst))
    gates.sort(key=lambda g: g["gate"])
    by_cell = {}
    for g in gates:
        by_cell[g["type"]] = by_cell.get(g["type"], 0) + 1
    result = {k: v for k, v in (("gate_type", gate_type), ("connected_to", connected_to), ("pin", pin),
                                ("name", name)) if v is not None}
    result.update(count=len(gates), by_cell=dict(sorted(by_cell.items())), gates=gates[:inline_limit])
    if inline_limit is not None and len(gates) > inline_limit:
        stem = "gates_" + "_".join(str(v).replace("*", "x") for v in (gate_type, connected_to, pin) if v is not None)
        result["file_path"] = utils.write_list(
            out_dir, stem, [f"{g['gate']} {g['type']} " + " ".join(f"{p}={s}" for p, s in g["pins"].items())
                            for g in gates])
    return result


def get_ports(d: Design, direction: str | None = None) -> dict:
    """Primary inputs / outputs in source order, with widths.

    direction: "input", "output", or None for both.  For each:
    {"ports": declared ports, "bits": total bits, "list": [{"name", "width", "range"}]}.
    """
    if direction not in (None, "input", "output"):
        raise ValueError("direction must be 'input', 'output' or omitted")
    result = {}
    for io in ("input", "output"):
        if direction not in (None, io):
            continue
        ports = [{"name": n, "width": 1 if msb is None else abs(msb - lsb) + 1,
                  "range": None if msb is None else f"[{msb}:{lsb}]"}
                 for n, kind, msb, lsb in d.port_list() if kind == io]
        result[f"{io}s"] = {"ports": len(ports), "bits": sum(p["width"] for p in ports), "list": ports}
    return result
