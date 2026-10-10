"""Read and write a design, in the format the configuration names.

    iccad   : ICCAD gate-primitive netlists (iccad/), on the built-in gate library
    generic : any gate-level Verilog OpenROAD can read, with the design's own
              LEF and Liberty files (generic/)
"""
from pathlib import Path

from . import generic, iccad
from .design import Design


def read_design(path: str | Path, format: str = "iccad", lef: list[str] = (),
                liberty: list[str] = (), top: str | None = None) -> Design:
    """Load a netlist.  lef / liberty (required) and top are used by the generic format only;
    without top, the netlist's one module is the top (generic netlists are flattened)."""
    if format == "iccad":
        return iccad.read_verilog(path)
    if format == "generic":
        if not lef or not liberty:
            raise ValueError("the generic format needs `lef` and `liberty` files in the design config")
        return generic.read_verilog(path, list(lef), list(liberty), top)
    raise ValueError(f"unknown netlist format {format!r} (iccad or generic)")


def write_design(d: Design, path: str | Path) -> None:
    """Write the design in the format it was read in."""
    if d.read_args["format"] == "iccad":
        iccad.write_verilog(d, path)
    else:
        generic.write_verilog(d, path)
