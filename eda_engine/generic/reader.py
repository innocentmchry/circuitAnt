"""Read a technology-mapped gate-level Verilog netlist into OpenDB, given its LEF and
Liberty files.

The netlist is expected to be flattened: a single module, which is the top.
OpenROAD reads the files (OpenSTA parses the Verilog and links it against the
Liberty cells).  The only thing read from the text is that module's name, which
OpenROAD needs in order to link; ports and everything else come from OpenROAD
(Design.port_list).
"""
import re
from pathlib import Path

import openroad

from ..design import Design


def read_verilog(path: str | Path, lef: list[str], liberty: list[str], top: str | None = None) -> Design:
    """Load `path` with the given LEF files (technology first) and Liberty files, and
    link module `top` -- by default the netlist's one module."""
    if top is None:
        top = single_module(path)
    tech = openroad.Tech()
    for f in lef:
        tech.readLef(str(f))
    for f in liberty:
        tech.readLiberty(str(f))
    or_design = openroad.Design(tech)
    try:
        or_design.readVerilog(str(path))
    except RuntimeError as e:  # OpenROAD's exception text is only the message code
        raise ValueError(f"{path}: Verilog syntax error ({e}); the OpenROAD log has the line") from None
    try:
        or_design.link(top)
    except RuntimeError as e:  # OpenROAD's exception text is only the message code
        raise ValueError(f"{path}: cannot link top module {top!r} ({e}); see the OpenROAD log") from None
    block = or_design.getBlock()
    if block is None or not block.getInsts():
        raise ValueError(f"{path}: no instances linked -- are all cell types in the LEF files?")
    liberty = [str(f) for f in liberty]
    return Design(path, top, tech, or_design, liberty,
                  read_args={"format": "generic", "lef": [str(f) for f in lef], "liberty": liberty, "top": top})


def single_module(path: str | Path) -> str:
    """The name of the one module a flattened netlist defines (comments ignored)."""
    text = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", Path(path).read_text()), flags=re.S)
    modules = re.findall(r"^\s*module\s+(\\\S+|\w+)", text, re.M)
    if len(modules) != 1:
        raise ValueError(f"{path}: expected one (flattened) module, found {len(modules)}"
                         + (f": {', '.join(modules[:5])}" if modules else ""))
    return modules[0]
