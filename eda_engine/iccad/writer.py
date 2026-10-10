"""Write a Design back out as ICCAD gate-level Verilog.

The reverse of the reader's rewrite: AND/OR/... cells become primitives with
positional pins (output first), DFF becomes dff with named pins, and tie cells
disappear -- the nets they drive are written as 1'b1 / 1'b0.

We don't use OpenROAD's write_verilog: it writes primitives with named pins
(`and g1 (.Y(n1), ...)`, rejected by strict tools), sorts the ports, and
leaves 1'b0 / 1'b1 as undriven `zero_` / `one_` wires.
"""
import re
from pathlib import Path

from ..design import CONSTANT, TIE_VALUE, Design



def write_verilog(design: Design, path: str | Path) -> None:
    """Write the design as ICCAD gate-primitive Verilog."""
    lines = _header(design)
    lines += _wires(design)
    lines += _assigns(design)
    lines += _instances(design)
    lines.append("endmodule")
    Path(path).write_text("\n".join(lines) + "\n")


def _port_bits(design: Design) -> set[str]:
    """Names of all port bits, e.g. n12, n25[0] ... n25[16]."""
    bits = set()
    for name, _, msb, lsb in design.source_header:
        if msb is None:
            bits.add(name)
        else:
            lo, hi = sorted((msb, lsb))
            bits.update(f"{name}[{i}]" for i in range(lo, hi + 1))
    return bits


def _header(design: Design) -> list[str]:
    lines = [f"module {design.top}({', '.join(p[0] for p in design.source_header)});"]
    for name, direction, msb, lsb in design.source_header:
        rng = "" if msb is None else f"[{msb}:{lsb}] "
        lines.append(f"  {direction} {rng}{name};")
    return lines


def _wires(design: Design) -> list[str]:
    """Internal nets; name[i] bits are grouped back into bus declarations."""
    port_bits = _port_bits(design)
    scalars, buses = [], {}
    for net in design.block.getNets():
        name = net.getName()
        if name in port_bits or net.getSigType() in CONSTANT or name in design.constant_nets:
            continue
        m = re.fullmatch(r"(.+)\[(\d+)\]", name)
        if m:
            buses.setdefault(m.group(1), []).append(int(m.group(2)))
        else:
            scalars.append(name)
    lines = [f"  wire [{max(bits)}:{min(bits)}] {base};" for base, bits in buses.items()]
    lines += [f"  wire {', '.join(scalars[i:i + 8])};" for i in range(0, len(scalars), 8)]
    return lines


def _assigns(design: Design) -> list[str]:
    """Output ports whose net carries another name (assign y = a; / 1'b1)."""
    lines = []
    for bt in design.block.getBTerms():
        net = bt.getNet()
        if bt.getIoType() == "OUTPUT" and net and net.getName() != bt.getName():
            lines.append(f"  assign {bt.getName()} = {design.signal(net)};")
    return lines


def _instances(design: Design) -> list[str]:
    lines = []
    for inst in design.block.getInsts():
        kind = inst.getMaster().getName()
        if design.cell(inst).kind in TIE_VALUE:  # its net is written as 1'b1 / 1'b0 instead
            continue
        pins = [(it.getMTerm().getName(), design.signal(it.getNet()) if it.getNet() else "")
                for it in inst.getITerms() if it.getMTerm().getSigType() not in ("POWER", "GROUND")]
        if design.cell(inst).kind == "sequential":  # dff: named pins
            conns = ", ".join(f".{p}({s})" for p, s in pins)
        else:  # primitive gates are positional: output first, then inputs
            conns = ", ".join(s for _, s in pins)
        lines.append(f"  {kind.lower()} {inst.getName()} ({conns});")
    return lines
