"""Boolean expressions of signals, read straight off the gates (structural, no solver).

Operators: * AND, + OR, ^ XOR, ~ NOT.
"""
import re
from pathlib import Path

from .. import utils
from ..design import Design

MAX_FLAT_CHARS = 5000  # beyond this, shared signals stay named instead of being copied in


def derive_boolean_equation(d: Design, signal: str, inline_limit: int | None = 20,
                            out_dir: str | Path = ".") -> dict:
    """Boolean equation of `signal` in terms of its cone's inputs.

    The cone stops at primary inputs, flip-flop outputs and constants.  The equation
    is fully substituted unless that would exceed MAX_FLAT_CHARS (signals used more
    than once get copied every time); then signals used more than once in the cone
    stay as named terms, each listed in "shared_terms" with its own equation.
    {"driver_type": primary_input | constant | dff | combinational | undriven,
     "equation", "primary_inputs_used", "state_inputs" (flip-flop outputs used),
     "primary_input_only", "cone_gate_count", "gate_equations" (first `inline_limit`,
     inputs to output), "file_path" (all gate equations and the equation, if more)}.
    """
    drivers, dffs = utils.netlist_view(d)
    name = d.net(signal).getName()  # raise a clear error for an unknown name
    pis = {bt.getNet().getName() for bt in d.block.getBTerms()
           if bt.getIoType() == "INPUT" and bt.getNet()}
    qs = {pins[p]: (flop, p) for flop, pins in dffs.items()  # flip-flop output net -> (flop, pin)
          for p in d.cell(d.inst(flop)).outputs if pins.get(p)}
    result = {"signal": signal}

    if name not in drivers:  # no gate drives it: input, constant, flip-flop or nothing
        if name in qs:
            flop, pin = qs[name]
            data = ", ".join(f"{p} = {dffs[flop].get(p)}" for p in d.cell(d.inst(flop)).data_inputs)
            return {**result, "driver_type": "dff", "equation": f"{signal} = {flop}.{pin}",
                    "primary_input_only": False, "primary_inputs_used": [], "state_inputs": [name],
                    "note": f"{signal} is stored state (flip-flop {flop}), not a function of "
                            f"the current inputs; its next value comes from {data}"}
        kind = ("primary_input" if name in pis else
                "constant" if d.signal(d.net(name)) != name else "undriven")
        value = d.signal(d.net(name))
        return {**result, "driver_type": kind,
                "equation": f"{signal} = {value}" if kind != "undriven" else f"{signal} is undriven",
                "primary_input_only": kind != "undriven",
                "primary_inputs_used": [name] if kind == "primary_input" else [], "state_inputs": []}

    # Cone gates from inputs to output (depth-first post-order, no recursion).
    order, done, stack = [], set(), [(name, False)]
    while stack:
        s, expanded = stack.pop()
        if s in done or s not in drivers:
            continue
        if expanded:
            done.add(s)
            order.append(s)
        else:
            stack.append((s, True))
            stack.extend((i, False) for i in drivers[s][2] if i not in done)
    leaves = sorted({i for s in order for i in drivers[s][2] if i not in drivers})
    uses = {}
    for s in order:
        for i in drivers[s][2]:
            uses[i] = uses.get(i, 0) + 1

    def gate_expr(s: str, sub) -> str:
        """Expression for gate output `s`, each input signal written as sub(signal).
        AND/OR/XOR-type cells are written n-ary; any other cell from its Liberty
        function, pins replaced by their signals."""
        _, master, _, pins = drivers[s]
        cell = d.cells[master]
        ins = [sub(pins[p]) for p in cell.inputs if p in pins]
        op = {"AND": " * ", "NAND": " * ", "OR": " + ", "NOR": " + ", "XOR": " ^ ", "XNOR": " ^ "}
        if cell.logic == "BUF":
            return ins[0]
        if cell.logic == "NOT":
            return f"~({ins[0]})"
        if cell.logic in op:
            joined = "(" + op[cell.logic].join(ins) + ")"
            return "~" + joined if cell.logic in ("NAND", "NOR", "XNOR") else joined
        out = next(p for p in cell.outputs if pins.get(p) == s)
        token = {"!": "~", "*": " * ", "+": " + ", "^": " ^ "}
        return "(" + re.sub(r"[A-Za-z_]\w*|[!*+^]",  # one pass: substituted text is not rescanned
                            lambda m: token.get(m[0]) or (sub(pins[m[0]]) if m[0] in pins else m[0]),
                            cell.functions[out]) + ")"

    size = {}  # length the fully substituted expression would have
    for s in order:
        size[s] = 4 + sum(size.get(i, len(i)) + 3 for i in drivers[s][2])
    flat = size[name] <= MAX_FLAT_CHARS
    keep = set() if flat else {s for s in order if uses.get(s, 0) > 1 and s != name}

    expr = {}
    for s in order:
        expr[s] = gate_expr(s, lambda i: i if i in keep else expr.get(i, i))
    gate_equations = [f"{drivers[s][0]} ({drivers[s][1].upper()}): {s} = {gate_expr(s, lambda i: i)}"
                      for s in order]
    result.update(driver_type="combinational", cone_gate_count=len(order),
                  equation=f"{signal} = {expr[name]}",
                  primary_inputs_used=[s for s in leaves if s in pis],
                  state_inputs=[s for s in leaves if s in qs],
                  primary_input_only=all(s in pis or s in ("1'b0", "1'b1") for s in leaves),
                  gate_equations=gate_equations[:inline_limit])
    if keep:
        result["shared_terms"] = {s: expr[s] for s in order if s in keep}
        result["note"] = (f"fully substituted equation would be about {size[name]} characters; "
                          "signals used more than once are kept as named terms (see shared_terms)")
    if inline_limit is not None and len(gate_equations) > inline_limit:
        result["file_path"] = utils.write_list(out_dir, f"equation_{signal}",
                                               gate_equations + ["", result["equation"]])
    return result
