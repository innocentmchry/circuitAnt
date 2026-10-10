"""Fanout of signals: loads driven, and the maximum over a class of signals."""
from pathlib import Path

from .. import utils
from ..design import Design
from ..utils import LIST_LIMIT


def get_fanout(d: Design, signal: str, inline_limit: int | None = 10, out_dir: str | Path = ".") -> dict:
    """Fanout = number of input pins (gate inputs, flip-flop D/CK/RN/SN) a signal drives.

    signal: a signal name, a gate name (its output signal is used), or a keyword
    (ANY_PI, ANY_DFF_Q, ANY_GATE_OUTPUT, ANY_SIGNAL).
    One signal: {"fanout", "gate_count", "loads" (first `inline_limit` gates with the
    pins used), "file_path" (all, if more)}.  Keyword: {"signals", "max_fanout",
    "at_max", "tie_count"} over every signal of that class.
    """
    def load_pins(net) -> dict[str, list[str]]:  # gate -> its pins on this net
        pins = {}
        for it in net.getITerms():
            if it.getIoType() == "INPUT":
                pins.setdefault(it.getInst().getName(), []).append(it.getMTerm().getName())
        return pins

    if signal in utils.SOURCE_CLASSES:
        fanout = {s: sum(map(len, load_pins(d.net(s)).values()))
                  for _, s in utils.endpoints(d, signal, "source") if d.net(s).getSigType() == "SIGNAL"}
        top = max(fanout.values(), default=0)
        at_max = sorted(s for s, n in fanout.items() if n == top)
        return {"signal": signal, "signals": len(fanout), "max_fanout": top,
                "at_max": at_max[:LIST_LIMIT], "tie_count": len(at_max)}

    result = {"signal": signal}
    net = d.block.findNet(signal) or d.block.findNet(d.escaped(signal))
    if net is None and (d.block.findInst(signal) or d.block.findInst(d.escaped(signal))):  # a gate: its output
        outs = [it.getNet() for it in d.inst(signal).getITerms() if it.isOutputSignal() and it.getNet()]
        if len(outs) > 1:
            raise ValueError(f"gate {signal} drives several signals ({', '.join(n.getName() for n in outs)}); "
                             "ask for one of them")
        net = outs[0] if outs else None
        result["output_signal"] = net.getName() if net else None
    if net is None:
        d.net(signal)  # raise "signal not found"
    pins = load_pins(net)
    loads = [{"gate": g, "type": d.inst(g).getMaster().getName().upper(), "pins": p}
             for g, p in sorted(pins.items())]
    result.update(fanout=sum(len(p) for p in pins.values()), gate_count=len(loads),
                  loads=loads[:inline_limit])
    if inline_limit is not None and len(loads) > inline_limit:
        result["file_path"] = utils.write_list(
            out_dir, f"fanout_{signal}",
            [f"{x['gate']} ({x['type']}) {','.join(x['pins'])}" for x in loads])
    return result
