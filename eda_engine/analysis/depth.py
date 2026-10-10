"""Logic depth: longest paths and per-endpoint depth / cone-size statistics.

A path is a sequence of signals (net names) through combinational gates;
its depth is the number of gates.
"""
from .. import utils
from ..design import Design
from ..utils import LIST_LIMIT

COMPARE = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b,
           "<=": lambda a, b: a <= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


def get_max_depth(d: Design, source: str, sink: str) -> dict:
    """Longest path from `source` to `sink`, each a signal name or class keyword.

    {"depth", "path", "start", "end", "tie_count"}; depth -1 if no path.  start/end
    name the endpoints: a port bit, `<dff>/Q` / `<dff>/D`, or the signal.  If two
    plain signals have no path in the given order, the other order is tried
    (only one can exist), so "between A and B" works either way round.
    """
    def longest(src: str, snk: str) -> dict:
        starts = {s: label for label, s in utils.endpoints(d, src, "source")}
        best = utils.longest_from(d, set(starts))
        reached = [(best[s][0], label, s) for label, s in utils.endpoints(d, snk, "sink")
                   if s in best]
        if not reached:
            return {"source": src, "sink": snk, "depth": -1, "path": [],
                    "note": "no combinational path"}
        depth = max(r[0] for r in reached)
        ties = [r for r in reached if r[0] == depth]
        path = utils.trace_path(best, ties[0][2])
        return {"source": src, "sink": snk, "depth": depth, "path": path,
                "start": starts[path[0]], "end": ties[0][1], "tie_count": len(ties)}

    result = longest(source, sink)
    plain = source not in utils.SOURCE_CLASSES and sink not in utils.SINK_CLASSES
    if result["depth"] < 0 and plain and source != sink:
        reverse = longest(sink, source)
        if reverse["depth"] >= 0:
            reverse.update(source=source, sink=sink, note=(
                f"no path from {source} to {sink}; reported the path from {sink} to {source}"))
            return reverse
    return result


def get_endpoint_stats(d: Design, sink: str = "ANY_PO", metric: str = "depth",
                       operator: str | None = None, threshold: int | None = None) -> dict:
    """A value for every endpoint in `sink`, summarised.

    metric "depth": logic depth from any start point; "cone_size": gates in its
    fanin cone (see get_cone).  {"endpoints", "max_value", "at_max", "tie_count",
    "no_path"}, plus {"count", "matching"} for value <operator> <threshold>.
    Name lists are capped at LIST_LIMIT (the counts are always complete).
    """
    ends = utils.endpoints(d, sink, "sink")
    if metric == "depth":
        best = utils.longest_from(d, {s for _, s in utils.endpoints(d, "ANY_START", "source")})
        values = {label: best[s][0] for label, s in ends if s in best}
    elif metric == "cone_size":
        values = {label: len(utils.cone_gates(d, s)[0]) for label, s in ends}
    else:
        raise ValueError("metric must be 'depth' or 'cone_size'")
    result = {"sink": sink, "metric": metric, "endpoints": len(ends),
              "no_path": len(ends) - len(values)}
    if values:
        top = max(values.values())
        at_max = [label for label, v in values.items() if v == top]
        result.update(max_value=top, at_max=at_max[:LIST_LIMIT], tie_count=len(at_max))
    if operator:
        compare = COMPARE.get(operator)
        if compare is None:
            raise ValueError(f"operator must be one of {', '.join(COMPARE)}")
        matching = [label for label, v in values.items() if compare(v, threshold)]
        result.update(operator=operator, threshold=threshold, count=len(matching),
                      matching=matching[:LIST_LIMIT])
    return result
