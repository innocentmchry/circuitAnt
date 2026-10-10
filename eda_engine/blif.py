"""BLIF networks for ABC (whole design or signal cones) and ABC's equivalence check.

Cells are written by name (`.gate <master> pin=net ...`) and ABC reads their
functions from the design's Liberty files; sequential cells are cut so every
check is combinational.  Used by the verification package.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

from . import utils
from .design import Design

BLIF_CONST = {"1'b0": "_const_0", "1'b1": "_const_1"}  # ABC's built-in constants _const0_ / _const1_
BLIF_CONST_GATES = [".gate _const0_ z=_const_0", ".gate _const1_ z=_const_1"]


def blif_cell(master: str, pins: dict[str, str], rename: dict[str, str] | None = None) -> str:
    """One `.gate <master> pin=net ...` line (constants and renamed signals substituted)."""
    rename = rename or {}
    return f".gate {master} " + " ".join(f"{p}={BLIF_CONST.get(n, rename.get(n, n))}" for p, n in pins.items())


def compare_cones(d: Design, first: tuple[str, dict], second: tuple[str, dict], timeout: int = 120) -> dict:
    """Do two signal cones compute the same function?

    first / second: (signal, rename) -- the cone of `signal`, its inputs renamed
    through `rename` (e.g. two inputs swapped for a symmetry check; {} for none).
    Combinational: flip-flop outputs and other cone leaves are free inputs.  Each
    cone becomes a BLIF network with output `out`, both declaring the same inputs
    so ABC `cec` can match them by name.
    {"status": PASS | FAIL | TIMEOUT | ERROR, "detail"}.
    """
    drivers, _ = utils.netlist_view(d)
    buf, a, y = d.buffer_cell()

    def cone(signal: str, rename: dict) -> tuple[list[str], set[str]]:
        """(BLIF lines computing `out`, input names) for the cone of `signal`."""
        seen, stack, lines, leaves = set(), [signal], [], set()
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            if s in drivers:
                cell, master, ins, pins = drivers[s]
                lines.append(blif_cell(master, pins, rename))
                stack.extend(ins)
            elif s not in BLIF_CONST:
                leaves.add(rename.get(s, s))
        source = signal if signal in drivers else BLIF_CONST.get(signal, rename.get(signal, signal))
        return lines + [f".gate {buf} {a}={source} {y}=out"], leaves

    (lines1, leaves1), (lines2, leaves2) = cone(*first), cone(*second)
    inputs = " ".join(sorted(leaves1 | leaves2))
    network = lambda lines: "\n".join([".model cone", f".inputs {inputs}", ".outputs out",
                                       *BLIF_CONST_GATES, *lines, ".end", ""])
    with tempfile.TemporaryDirectory(prefix="circuitant_cone_") as tmp:
        one, two = Path(tmp) / "first.blif", Path(tmp) / "second.blif"
        one.write_text(network(lines1))
        two.write_text(network(lines2))
        proof = abc_cec(d.liberty, one, two, timeout)
    return {"status": {"PROVED": "PASS", "FAILED": "FAIL"}.get(proof["status"], proof["status"]),
            "detail": proof["detail"]}


def run_abc(commands: str, timeout: int = 260) -> str | None:
    """ABC's output (stdout + stderr) for a command script, or None on timeout."""
    abc = os.environ.get("ABC_BIN", "abc")  # set by circuitant / openroad_python.sh (default ~/abc/abc)
    with tempfile.TemporaryDirectory(prefix="circuitant_abc_") as cwd:  # ABC leaves abc.history there
        try:
            run = subprocess.run([abc, "-c", commands], capture_output=True, text=True, timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired:
            return None
    return run.stdout + run.stderr


def abc_cec(liberty: list[str], first: str | Path, second: str | Path, timeout: int = 260) -> dict:
    """Are two BLIF networks (cells by name, inputs/outputs matched by name) equivalent?

    ABC loads the cell functions from the Liberty files, then runs `cec`.
    {"status": "PROVED" | "FAILED" | "TIMEOUT" | "ERROR", "detail", "failing_outputs"}.
    """
    out = run_abc(f"{read_lib(liberty)}; cec {first} {second}", timeout)
    if out is None:
        return {"status": "TIMEOUT", "detail": f"no answer within {timeout} s"}
    if "NOT EQUIVALENT" in out:
        failing = re.search(r"Verification failed for at least \d+ outputs?:(.*)", out)
        return {"status": "FAILED", "detail": "ABC cec: networks are NOT equivalent",
                "failing_outputs": failing.group(1).split()[:10] if failing else []}
    if "are equivalent" in out:
        return {"status": "PROVED", "detail": "ABC cec: networks are equivalent"}
    return {"status": "ERROR", "detail": out.strip()[-800:]}


def read_lib(liberty: list[str], exclude: list[str] = ()) -> str:
    """ABC commands loading the Liberty files (merged), leaving out the cells in `exclude`."""
    skip = "".join(f" -X {cell}" for cell in exclude)
    return "; ".join(f"read_lib -w{' -m' if i else ''}{skip} {lib}" for i, lib in enumerate(liberty))


def write_blif(design: Design, path: str | Path) -> None:
    """Write the combinational view used for equivalence checking, as BLIF for ABC.

    Library-independent: every non-sequential cell is written by name
    (`.gate <master> pin=net ...`; ABC reads the functions from the Liberty files).
    Sequential cells are cut: each output pin becomes an input `_cut_<cell>_<pin>`
    and each input pin an output `_cut_<cell>_<pin>` (named after the cell, so two
    designs line up by name even if a transform changed the nets around it),
    joined to the netlist through the library's buffer.
    """
    drivers, flops = utils.netlist_view(design)
    buf, a, y = design.buffer_cell()
    alias = lambda src, dst: f".gate {buf} {a}={BLIF_CONST.get(src, src)} {y}={dst}"
    inputs = [bt.getName() for bt in design.block.getBTerms() if bt.getIoType() == "INPUT"]
    outputs = [bt.getName() for bt in design.block.getBTerms() if bt.getIoType() == "OUTPUT"]
    cells = {cell: blif_cell(master, pins) for cell, master, _, pins in drivers.values()}
    lines = list(cells.values())
    for flop in flops:
        for it in design.inst(flop).getITerms():
            if not it.getNet():
                continue
            cut, net = f"_cut_{flop}_{it.getMTerm().getName()}", design.signal(it.getNet())
            if it.getIoType() == "OUTPUT":
                inputs.append(cut)
                lines.append(alias(cut, net))
            else:
                outputs.append(cut)
                lines.append(alias(net, cut))
    for bt in design.block.getBTerms():  # outputs whose net carries another name (assign y = a;)
        net = bt.getNet()
        if bt.getIoType() == "OUTPUT" and net and net.getName() != bt.getName():
            lines.append(alias(design.signal(net), bt.getName()))
    Path(path).write_text("\n".join([".model top", f".inputs {' '.join(inputs)}",
                                     f".outputs {' '.join(outputs)}", *BLIF_CONST_GATES,
                                     *lines, ".end", ""]))
