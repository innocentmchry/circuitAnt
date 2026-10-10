"""Renaming gates and wires."""
import re

from .. import edit
from ..design import Design


def rename_gate(d: Design, old_name: str, new_name: str) -> dict:
    """Rename gate instance `old_name` to `new_name`.  {"old_name", "new_name"}."""
    inst = d.inst(old_name)
    if not re.fullmatch(r"[A-Za-z_]\w*", new_name):
        raise ValueError(f"{new_name!r} is not a plain Verilog identifier")
    if d.block.findInst(new_name) is not None:
        raise ValueError(f"a gate named {new_name!r} already exists")
    inst.rename(new_name)
    return {"old_name": old_name, "new_name": new_name}


def rename_wire(d: Design, old_name: str, new_name: str) -> dict:
    """Rename wire `old_name` to `new_name`; every connection follows.  Port signals
    cannot be renamed (that changes the design's interface).  {"old_name", "new_name"}."""
    net = d.net(old_name)
    if net.getBTerms():
        raise ValueError(f"{old_name!r} is a port; renaming it would change the interface")
    if not re.fullmatch(r"[A-Za-z_]\w*", new_name):
        raise ValueError(f"{new_name!r} is not a plain Verilog identifier")
    if d.block.findNet(new_name) is not None:
        raise ValueError(f"a wire named {new_name!r} already exists")
    edit.rename_net(d, net, new_name)
    return {"old_name": old_name, "new_name": new_name}
