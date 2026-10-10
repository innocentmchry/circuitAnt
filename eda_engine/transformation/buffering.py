"""Buffer insertion: limit how many loads a signal drives, or give loads their own buffer.

A load is an input pin of a gate or flip-flop (as in get_fanout); output ports
are not loads.  Buffers are the library's buffer cell (Design.buffer_cell).
"""
from .. import edit
from ..design import Design


def insert_buffers_for_fanout(d: Design, max_fanout: int, net_name: str | None = None) -> dict:
    """Insert buffer trees so no signal drives more than `max_fanout` loads, using the
    fewest buffers: groups of `max_fanout` loads move behind a new buffer, which is
    itself a load of the signal, until the signal is within the limit.
    net_name: only this signal; default every signal except constants.
    {"max_fanout", "buffers_inserted", "nets_buffered"}.
    """
    if max_fanout < 2:
        raise ValueError("max_fanout must be >= 2; buffers cannot reduce fanout to 1")
    buf, a, y = d.buffer_cell()
    nets = [d.net(net_name)] if net_name else [n for n in d.block.getNets() if d.signal(n) == n.getName()]
    inserted = buffered = 0
    for net in nets:
        loads = [it for it in net.getITerms() if it.isInputSignal()]
        if len(loads) <= max_fanout:
            continue
        buffered += 1
        while len(loads) > max_fanout:
            group, loads = loads[:max_fanout], loads[max_fanout:]
            out = edit.new_net(d, f"{net.getName()}_fo")
            for it in group:
                it.disconnect()
                it.connect(out)
            loads.append(edit.add_gate(d, buf, {a: net, y: out}, f"{net.getName()}_fobuf").findITerm(a))
            inserted += 1
    result = {"max_fanout": max_fanout, "buffers_inserted": inserted, "nets_buffered": buffered}
    return {"net_name": net_name, **result} if net_name else result


def insert_dedicated_buffers_for_loads(d: Design, net_name: str) -> dict:
    """Give every load of `net_name` its own buffer: afterwards the signal drives only
    the new buffers, one per former load.  {"net_name", "buffers_inserted"}.
    """
    net = d.net(net_name)
    buf, a, y = d.buffer_cell()
    loads = [it for it in net.getITerms() if it.isInputSignal()]
    for it in loads:
        out = edit.new_net(d, f"{net_name}_ded")
        it.disconnect()
        it.connect(out)
        edit.add_gate(d, buf, {a: net, y: out}, f"{net_name}_dedbuf")
    return {"net_name": net_name, "buffers_inserted": len(loads)}
