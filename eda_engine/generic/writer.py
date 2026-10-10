"""Write a generic (standard-cell) design back to Verilog with OpenROAD's write_verilog.

OpenROAD writes the ports in its own (alphabetical) order, not the source order.
"""
from pathlib import Path

from ..design import Design


def write_verilog(design: Design, path: str | Path) -> None:
    design.tcl(f"write_verilog {{{Path(path)}}}")
