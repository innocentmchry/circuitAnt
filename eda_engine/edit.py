"""Editing the netlist in OpenDB: the primitives transformations are built from.

New nets and instances get unique, Verilog-safe names here.  One rule lives in
merge_nets: a net on a port keeps its name, because port names are the
design's interface.  Design.constant_nets (nets driven by tie cells) is kept
up to date here, so netlist edits (including renaming a net) go through this file.
"""
import re

import odb

from .design import TIE_VALUE, Design


def unique_name(base: str, taken) -> str:
    """`base` as a plain identifier, with _1, _2, ... appended while `taken(name)` finds it."""
    base = re.sub(r"\W", "_", base)
    name, i = base, 0
    while taken(name) is not None:
        i += 1
        name = f"{base}_{i}"
    return name


def new_net(d: Design, base: str):  # -> odb.dbNet
    """A new, unconnected net named after `base`."""
    return odb.dbNet_create(d.block, unique_name(base, d.block.findNet))


def add_gate(d: Design, master: str, pins: dict, base: str):  # -> odb.dbInst
    """A new instance of library cell `master` named after `base`, with {pin: dbNet} connected."""
    inst = odb.dbInst_create(d.block, d.master(master), unique_name(base, d.block.findInst))
    for pin, net in pins.items():
        inst.findITerm(pin).connect(net)
    return inst


def remove_gate(d: Design, inst) -> None:
    """Delete an instance; nets it leaves with no connection at all are deleted too."""
    nets = {it.getNet().getName(): it.getNet() for it in inst.getITerms() if it.getNet()}
    if d.cell(inst).kind in TIE_VALUE:
        for name in nets:
            d.constant_nets.pop(name, None)
    odb.dbInst_destroy(inst)
    for name, net in nets.items():
        if not net.getITerms() and not net.getBTerms():
            d.constant_nets.pop(name, None)
            odb.dbNet_destroy(net)


def rename_net(d: Design, net, new_name: str) -> None:
    """Rename a net (its connections follow)."""
    old_name = net.getName()
    net.rename(new_name)
    if old_name in d.constant_nets:
        d.constant_nets[new_name] = d.constant_nets.pop(old_name)


def merge_nets(d: Design, keep, drop):  # -> odb.dbNet
    """Make `drop` part of `keep`: whatever `drop` fed is now fed by `keep`'s driver.

    `drop` must have no driver left (remove it first).  Returns the surviving net:
    `keep`, or `drop` when only `drop` is on a port (the port net keeps its name).
    When both are on ports, both stay and a buffer drives `drop` from `keep`.
    """
    if drop.getBTerms() and keep.getBTerms():
        buf, a, y = d.buffer_cell()
        add_gate(d, buf, {a: keep, y: drop}, f"{drop.getName()}_buf")
        return keep
    if drop.getBTerms():
        keep, drop = drop, keep
    for it in list(drop.getITerms()):
        it.disconnect()
        it.connect(keep)
    for bt in list(drop.getBTerms()):
        bt.disconnect()
        bt.connect(keep)
    if drop.getName() in d.constant_nets:
        d.constant_nets[keep.getName()] = d.constant_nets.pop(drop.getName())
    odb.dbNet_destroy(drop)
    return keep
