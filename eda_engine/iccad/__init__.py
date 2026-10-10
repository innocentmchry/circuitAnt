"""The ICCAD contest netlist format: Verilog gate primitives (and/or/not/... and dff),
read through the cell library in iccad/cells (primitives.lef / .lib), and written back the same way."""
from .reader import read_verilog
from .writer import write_verilog

__all__ = ["read_verilog", "write_verilog"]
