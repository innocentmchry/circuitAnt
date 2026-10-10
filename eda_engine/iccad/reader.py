"""Read an ICCAD gate-primitive Verilog netlist into OpenDB.

The contest netlists use Verilog primitives (`and g1(y, a, b);`) and literal
constants (`.SN(1'b1)`).  They are first rewritten into an ordinary standard-cell
netlist -- `AND g1 (.Y(y), .A(a), .B(b));`, constants driven by TIEHI / TIELO
cells -- and read by the generic reader against cells/primitives.{lef,lib}.  From
then on the design is like any other.  The top module must be named `top`.  The
port order and bus ranges are taken from the source header, so the writer can
reproduce it exactly.
"""
import re
import tempfile
from pathlib import Path

from ..design import Design
from ..generic import reader as generic

CELL_DIR = Path(__file__).resolve().parent / "cells"  # the ICCAD primitives as standard cells
LEF = CELL_DIR / "primitives.lef"
LIBERTY = CELL_DIR / "primitives.lib"
TOP = "top"


# Primitive -> (cell, output pin + input pins in positional order).
CELLS = {"and": "AND", "or": "OR", "not": "NOT", "nand": "NAND", "nor": "NOR",
         "xor": "XOR", "xnor": "XNOR", "buf": "BUF"}
PINS = ["Y", "A", "B"]
TIES = {"1'b1": "TIEHI", "1'b0": "TIELO"}


def read_verilog(path: str | Path) -> Design:
    text = Path(path).read_text()
    with tempfile.NamedTemporaryFile("w", suffix=".v") as cells:
        cells.write(_to_standard_cells(text))
        cells.flush()
        design = generic.read_verilog(cells.name, [LEF], [LIBERTY], TOP)
    design.path = str(path)  # the original file, read again by check_equivalence
    design.read_args = {"format": "iccad"}
    design.source_header = _parse_ports(text)
    return design


def _to_standard_cells(text: str) -> str:
    """ICCAD primitives -> AND/OR/... cell instances with named pins; dff -> DFF;
    each literal constant -> a net driven by one TIEHI / TIELO cell."""
    def gate(m: re.Match) -> str:
        signals = [s.strip() for s in m[4].split(",")]
        if len(signals) > len(PINS):
            raise ValueError(f"{m[2]} {m[3]}: {len(signals) - 1}-input gates are not in the gate library")
        pins = ", ".join(f".{p}({s})" for p, s in zip(PINS, signals))
        return f"{m[1]}{CELLS[m[2]]} {m[3]} ({pins});"

    text = re.sub(rf"^(\s*)({'|'.join(CELLS)})\s+(\w+)\s*\(([^;]*)\)\s*;", gate, text, flags=re.M)
    text = re.sub(r"^(\s*)dff\s+(\w+)\s*\(", r"\1DFF \2 (", text, flags=re.M)
    ties = []
    for literal, cell in TIES.items():
        if literal in text:
            net = f"_{cell.lower()}_"
            while re.search(rf"\b{net}\b", text):  # keep the name unique
                net += "_"
            text = text.replace(literal, net)
            ties.append(f"  wire {net};\n  {cell} {net}cell (.Y({net}));\n")
    return text.replace("endmodule", "".join(ties) + "endmodule", 1)


def _parse_ports(text: str) -> list[tuple[str, str, int | None, int | None]]:
    """[(name, direction, msb, lsb)] in module-header order."""
    # Comments are stripped so a commented-out declaration isn't read as a port.
    text = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", text), flags=re.S)
    body = re.search(rf"\bmodule\s+{TOP}\s*\((.*?)\)\s*;(.*?)\bendmodule", text, re.S)
    if body is None:
        raise ValueError(f"module {TOP!r} not found")
    header, decls = body.group(1), body.group(2)
    info = {}
    for direction, msb, lsb, names in re.findall(
            r"\b(input|output|inout)\s*(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*([^;]+);", decls):
        for name in re.split(r"\s*,\s*", names.strip()):
            info[name] = (direction, int(msb) if msb else None, int(lsb) if lsb else None)
    return [(n, *info[n]) for n in re.split(r"\s*,\s*", header.strip()) if n]
