"""Helpers shared by several queries and transforms (analysis.py and later ones).

A helper used by only one function stays nested inside that function instead.

Connectivity helpers work on signal (net) names: edges go through combinational
gates only, so flip-flops cut every walk.  Names in, names out: no OpenDB
objects leak out.
"""
from pathlib import Path

from .design import CONSTANT, Design

Gates = dict[str, tuple[list[str], list[str]]]  # gate name -> (input signals, output signals)
LIST_LIMIT = 50  # longest list of names returned inline in a query result


# -- reachability ----------------------------------------------------------------

def fanin_signals(d: Design, name: str) -> list[str]:
    """Input signals of the combinational gate driving `name` ([] at a port, flop or constant)."""
    drv = d.net(name).getFirstOutput()
    if drv is None or not d.is_gate(drv.getInst()):
        return []
    return [it.getNet().getName() for it in drv.getInst().getITerms()
            if it.getIoType() == "INPUT" and it.getNet()]


def reach_forward(d: Design, sources: set[str], avoid: str | None = None) -> tuple[Gates, set[str]]:
    """Gates reachable from `sources` and the signals they reach (sources included).

    Never enters `avoid`: paths stop there (a gate's other outputs still count).
    """
    gates, reached = {}, set(sources) - {avoid}
    stack = sorted(reached)  # sorted: same order on every run
    while stack:
        for it in d.net(stack.pop()).getITerms():
            inst = it.getInst()
            if it.getIoType() != "INPUT" or not d.is_gate(inst) or inst.getName() in gates:
                continue
            outs = [o.getNet().getName() for o in inst.getITerms()
                    if o.isOutputSignal() and o.getNet() and o.getNet().getName() != avoid]
            if not outs:
                continue
            gates[inst.getName()] = ([i.getNet().getName() for i in inst.getITerms()
                                      if i.getIoType() == "INPUT" and i.getNet()], outs)
            for out in outs:
                if out not in reached:
                    reached.add(out)
                    stack.append(out)
    return gates, reached


def reach_backward(d: Design, sinks: set[str], avoid: str | None = None) -> set[str]:
    """All signals with a combinational path to one of `sinks` (sinks included), not via `avoid`."""
    seen = set(sinks) - {avoid}
    stack = list(seen)
    while stack:
        for s in fanin_signals(d, stack.pop()):
            if s != avoid and s not in seen:
                seen.add(s)
                stack.append(s)
    return seen


def topo_order(gates: Gates, sources: set[str], reached: set[str]) -> list[str]:
    """Gate names ordered so each gate comes after the gates feeding it.

    Inputs outside `reached` are ignored.  Raises on a combinational loop.
    """
    readers = {}
    for g, (ins, _) in gates.items():
        for n in ins:
            readers.setdefault(n, []).append(g)
    pending = {g: sum(n in reached for n in ins) for g, (ins, _) in gates.items()}
    order, queue = [], sorted(s for s in sources if s in reached)
    while queue:
        for g in readers.get(queue.pop(), []):
            pending[g] -= 1
            if pending[g] == 0:
                order.append(g)
                queue.extend(out for out in gates[g][1] if out not in sources)
    if len(order) != len(gates):
        raise ValueError("combinational loop: some gates never got all their inputs")
    return order


# -- endpoint sets ------------------------------------------------------------------

# Class keywords usable instead of a signal name (sources / sinks of a path).
SOURCE_CLASSES = {
    "ANY_PI": "every primary-input bit",
    "ANY_DFF_Q": "every flip-flop Q output",
    "ANY_START": "ANY_PI + ANY_DFF_Q + constant nets",
    "ANY_GATE_OUTPUT": "every combinational gate output",
    "ANY_SIGNAL": "every driven signal: ANY_PI + ANY_DFF_Q + ANY_GATE_OUTPUT",
}
SINK_CLASSES = {
    "ANY_PO": "every primary-output bit",
    "ANY_DFF_D": "every flip-flop D input",
    "ANY_END": "ANY_PO + ANY_DFF_D",
}


def endpoints(d: Design, spec: str, role: str) -> list[tuple[str, str]]:
    """[(label, signal)] for a signal name or a class keyword; role is "source" or "sink".

    The label names the endpoint in results: the port bit, `<dff>/D`, or the signal.
    """
    classes = SOURCE_CLASSES if role == "source" else SINK_CLASSES
    if spec not in SOURCE_CLASSES and spec not in SINK_CLASSES:
        return [(spec, d.net(spec).getName())]  # a signal name (error if unknown)
    if spec not in classes:
        raise ValueError(f"{spec} cannot be a {role}; use one of {', '.join(classes)}")

    found = []
    if spec in ("ANY_PI", "ANY_START", "ANY_SIGNAL", "ANY_PO", "ANY_END"):
        io = "INPUT" if role == "source" else "OUTPUT"
        found += [(bt.getName(), bt.getNet().getName()) for bt in d.block.getBTerms()
                  if bt.getIoType() == io and bt.getNet()]
    if spec in ("ANY_DFF_Q", "ANY_START", "ANY_SIGNAL", "ANY_DFF_D", "ANY_END"):
        for inst in d.block.getInsts():  # flip-flop outputs (Q, QN) / data inputs (D, ...)
            cell = d.cell(inst)
            pins = cell.outputs if role == "source" else cell.data_inputs
            for pin in pins if cell.kind == "sequential" else []:
                net = inst.findITerm(pin).getNet()
                if net:
                    found.append((f"{inst.getName()}/{pin}", net.getName()))
    if spec in ("ANY_GATE_OUTPUT", "ANY_SIGNAL"):
        for inst in d.block.getInsts():
            for it in inst.getITerms() if d.is_gate(inst) else []:
                if it.isOutputSignal() and it.getNet():
                    found.append((it.getNet().getName(), it.getNet().getName()))
    if spec == "ANY_START":
        found += [(n.getName(), n.getName()) for n in d.block.getNets()
                  if d.signal(n) != n.getName()]  # constants: tie-cell outputs, 1'b0 / 1'b1
    return found


# -- longest paths and path graphs ---------------------------------------------------

def longest_from(d: Design, sources: set[str]) -> dict[str, tuple[int, str | None]]:
    """{signal: (depth, previous signal)} for every signal reachable from `sources`.

    Sources have depth 0; a gate output is 1 + the deepest of its reachable
    inputs.  Follow `previous` back (trace_path) to recover a path.
    """
    gates, reached = reach_forward(d, sources)
    best = {s: (0, None) for s in sources}
    for g in topo_order(gates, sources, reached):
        ins, outs = gates[g]
        depth, prev = max((best[n][0], n) for n in ins if n in best)
        for out in outs:
            if out not in best or depth + 1 > best[out][0]:
                best[out] = (depth + 1, prev)
    return best


def trace_path(best: dict[str, tuple[int, str | None]], signal: str) -> list[str]:
    """Signals from the source to `signal`, following longest_from's back-pointers."""
    path = [signal]
    while best[path[-1]][1] is not None:
        path.append(best[path[-1]][1])
    return path[::-1]


def path_graph(d: Design, sources: set[str], sinks: set[str],
               avoid: str | None = None) -> tuple[dict[str, list[str]], list[str]]:
    """The part of the circuit lying on some source->sink path, avoiding `avoid`.

    Returns ({signal: [next signals]}, signals in topological order).  Every
    signal in it is reachable from a source and leads to a sink.
    """
    gates, reached = reach_forward(d, sources, avoid)
    on_path = reached & reach_backward(d, sinks, avoid)
    order = sorted(s for s in sources if s in on_path)
    succ = {s: [] for s in on_path}
    for g in topo_order(gates, sources, reached):
        ins, outs = gates[g]
        for out in outs:
            if out not in on_path:
                continue
            if out not in sources:
                order.append(out)
            for n in dict.fromkeys(ins):  # a gate reading the same signal twice is one step
                if n in on_path:
                    succ[n].append(out)
    return succ, order


# -- cones --------------------------------------------------------------------------

def cone_gates(d: Design, signal: str, direction: str = "fanin") -> tuple[set[str], dict]:
    """Gate names in the transitive fanin or fanout cone of `signal`, plus how it was resolved.

    fanin : gates feeding `signal`, stopping at primary inputs, flip-flop outputs and
            constants.  If `signal` is a flip-flop output, the cone of that flop's data
            inputs (its next-state logic: D, and SI/SE of a scan flop) is used:
            {"through_dff": <flop>, "resolved_signal": <its D net>}.
    fanout: gates reachable downstream; flip-flops reached are included, not crossed.
    """
    resolved = {"resolved_signal": d.net(signal).getName()}  # error if unknown
    if direction == "fanin":
        drv = d.net(signal).getFirstOutput()
        starts = {resolved["resolved_signal"]}
        if drv is not None and d.cell(drv.getInst()).kind == "sequential":
            flop = drv.getInst()
            nets = [flop.findITerm(p).getNet() for p in d.cell(flop).data_inputs]
            starts = {n.getName() for n in nets if n}
            resolved = {"resolved_signal": nets[0].getName() if nets and nets[0] else None,
                        "through_dff": flop.getName()}
        if not starts:
            return set(), resolved
        gates = set()
        for s in reach_backward(d, starts):
            drv = d.net(s).getFirstOutput()
            if drv is not None and d.is_gate(drv.getInst()):
                gates.add(drv.getInst().getName())
        return gates, resolved
    if direction == "fanout":
        gates, reached = reach_forward(d, {signal})
        found = set(gates)
        for s in reached:  # flip-flops reading any reached signal end the cone
            for it in d.net(s).getITerms():
                if it.getIoType() == "INPUT" and d.cell(it.getInst()).kind == "sequential":
                    found.add(it.getInst().getName())
        return found, resolved
    raise ValueError("direction must be 'fanin' or 'fanout'")


# -- netlist view, gates and result files -------------------------------------------------------------

def netlist_view(d: Design) -> tuple[dict, dict]:
    """Plain-Python view of the netlist, built once per query.

    Use it when an algorithm visits signals many times (patterns, proofs):
    dictionary lookups instead of one OpenDB call per step.  Cells are classified
    from OpenDB (sequential or not, pin directions), not by name.

    drivers: {output signal: (cell, master, [input signals], {pin: signal})} for every
             non-sequential cell output;
    flops:   {cell: {pin: signal}} for every sequential cell.
    Constants appear as 1'b0 / 1'b1; unconnected pins are left out.
    """
    drivers, flops = {}, {}
    for inst in d.block.getInsts():
        iterms = [it for it in inst.getITerms() if it.getNet()]
        pins = {it.getMTerm().getName(): d.signal(it.getNet()) for it in iterms}
        if inst.getMaster().isSequential():
            flops[inst.getName()] = pins
            continue
        ins = [d.signal(it.getNet()) for it in iterms if it.getIoType() == "INPUT"]
        for it in iterms:
            if it.getIoType() == "OUTPUT" and d.signal(it.getNet()) not in CONSTANT.values():
                drivers[d.signal(it.getNet())] = (inst.getName(), inst.getMaster().getName(), ins, pins)
    return drivers, flops


def depends_on(drivers: dict, signal: str, target: str, stop: set[str] = frozenset()) -> bool:
    """Does `signal` depend on `target` through combinational gates?

    Walks back from `signal` over a netlist_view `drivers` map; never passes a
    signal in `stop` (e.g. other flip-flop outputs).  Constants and undriven
    signals end the walk.
    """
    seen, stack = set(), [signal]
    while stack:
        s = stack.pop()
        if s == target:
            return True
        if s not in seen and s not in stop and s in drivers:
            seen.add(s)
            stack.extend(drivers[s][2])
    return False


def trace_buf_not(d: Design, drivers: dict, signal: str) -> tuple[str, bool]:
    """Follow `signal` back through buffers / inverters to where the chain starts.

    Returns (source signal, inverted?): inverted is True for an odd number of NOTs.
    E.g. y = not(not(x)) -> ("x", False).  Stops at any other gate, a flip-flop,
    a port or a constant (and on a loop of BUF/NOTs).
    """
    inverted, seen = False, set()
    kind = lambda s: d.cells[drivers[s][1]].kind
    while signal in drivers and signal not in seen and kind(signal) in ("buffer", "inverter"):
        seen.add(signal)
        inverted ^= kind(signal) == "inverter"
        signal = drivers[signal][2][0]
    return signal, inverted


def gate_info(d: Design, inst) -> dict:
    """{"gate", "type", "pins": {pin: signal}} for one instance (pins in library order)."""
    return {"gate": inst.getName(), "type": inst.getMaster().getName().upper(),
            "pins": {it.getMTerm().getName(): d.signal(it.getNet()) if it.getNet() else None
                     for it in inst.getITerms() if it.getMTerm().getSigType() not in ("POWER", "GROUND")}}


def write_list(out_dir: str | Path, stem: str, lines: list[str]) -> str:
    """Write one item per line to <out_dir>/<stem>.txt (brackets made file-safe); return the path."""
    path = Path(out_dir) / (stem.replace("[", "_").replace("]", "") + ".txt")
    path.write_text("".join(f"{line}\n" for line in lines))
    return str(path)
