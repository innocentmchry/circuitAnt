"""Enumerating, counting and checking combinational paths."""
from pathlib import Path

from .. import utils
from ..design import Design

PATH_FILE_LIMIT = 100000  # write at most this many paths to a file


def find_all_paths(d: Design, source: str, sink: str, avoid: str | None = None,
                   max_depth: int | None = None, inline_limit: int | None = 5,
                   out_dir: str | Path = ".") -> dict:
    """All combinational paths from `source` to `sink` (signal names or class keywords).

    {"count", "exists", "paths" (first `inline_limit`), "file_path" (all, if more)}.
    `avoid`: no path may pass through this signal.  `max_depth`: only paths with
    at most this many gates (0 = the source signal is itself a sink).  Each listed
    path is {"start", "end", "path"}: endpoint labels and the signals along it.
    """
    starts = {s: label for label, s in utils.endpoints(d, source, "source")}
    ends = {}
    for label, s in utils.endpoints(d, sink, "sink"):
        ends[s] = f"{ends[s]}, {label}" if s in ends else label
    if avoid is not None:
        d.net(avoid)  # raise a clear error for an unknown name
    succ, order = utils.path_graph(d, set(starts), set(ends), avoid)

    # Count paths in one topological pass: count[s][length] = paths reaching s.
    # Lengths are only kept apart when max_depth needs them.
    count = {s: {0: 1} for s in starts if s in succ}
    for s in order:
        for length, n in count.get(s, {}).items():
            for nxt in succ[s]:
                step = length + 1 if max_depth is not None else 0
                if max_depth is None or step <= max_depth:
                    count.setdefault(nxt, {}).setdefault(step, 0)
                    count[nxt][step] += n
    total = sum(sum(count.get(s, {}).values()) for s in ends)

    def walk():  # yields every path, depth-first, without recursion
        for start in starts:
            if start not in succ:
                continue
            stack = [[start]]
            while stack:
                path = stack.pop()
                if path[-1] in ends:
                    yield {"start": starts[start], "end": ends[path[-1]], "path": path}
                if max_depth is None or len(path) - 1 < max_depth:
                    stack.extend(path + [n] for n in reversed(succ[path[-1]]))

    result = {"source": source, "sink": sink, "count": total, "exists": total > 0, "paths": []}
    if avoid is not None:
        result["avoid"] = avoid
        if avoid in starts or avoid in ends:
            result["note"] = f"{avoid} is itself a source or sink, so every path through it is excluded"
    if max_depth is not None:
        result["max_depth"] = max_depth
    paths = walk()
    for _ in range(total if inline_limit is None else min(total, inline_limit)):
        result["paths"].append(next(paths))
    if inline_limit is not None and total > inline_limit:
        if total > PATH_FILE_LIMIT:
            result["note"] = f"{total} paths; too many to list (limit {PATH_FILE_LIMIT})"
        else:
            safe = lambda s: s.replace("[", "_").replace("]", "")
            f = Path(out_dir) / f"paths_{safe(source)}_to_{safe(sink)}.txt"
            with open(f, "w") as fh:
                for i, p in enumerate(walk(), 1):
                    fh.write(f"{i}. {p['start']} -> {p['end']}: {' -> '.join(p['path'])}\n")
            result["file_path"] = str(f)
    return result


def find_cut_points(d: Design, source: str = "ANY_PI", sink: str = "ANY_PO",
                    signal: str | None = None) -> dict:
    """Points every path must pass through.

    Without `signal` (source and sink are signal names): the gates lying on every
    source->sink path -- removing any one disconnects them (articulation points),
    ordered from source to sink.  {"path_exists", "count", "gates", "signals"}.
    With `signal` (source/sink may be class keywords, default ANY_PI -> ANY_PO):
    whether some source and some sink are connected only through `signal`.
    {"is_on_path", "is_cut", "cut_pairs" (up to 10 (source, sink) examples)}.
    """
    if signal is not None:
        d.net(signal)  # raise a clear error for an unknown name
        starts = {s: label for label, s in utils.endpoints(d, source, "source")}
        ends = {s: label for label, s in utils.endpoints(d, sink, "sink")}
        upstream = [s for s in starts if s in utils.reach_backward(d, {signal})]
        downstream = set(ends) & utils.reach_forward(d, {signal})[1]
        pairs = []
        for s in sorted(upstream):
            reached = utils.reach_forward(d, {s}, avoid=signal)[1]
            pairs += [(starts[s], ends[t]) for t in sorted(downstream - reached)]
            if len(pairs) >= 10:
                break
        return {"source": source, "sink": sink, "signal": signal,
                "is_on_path": bool(upstream and downstream), "is_cut": bool(pairs),
                "cut_pairs": [list(p) for p in pairs[:10]]}

    if source in utils.SOURCE_CLASSES or sink in utils.SINK_CLASSES:
        raise ValueError("listing cut points needs two signal names; to test one "
                         "signal against classes of endpoints, pass `signal`")
    d.net(source), d.net(sink)  # raise a clear error for unknown names
    result = {"source": source, "sink": sink}
    if source == sink:
        return {**result, "path_exists": True, "count": 0, "gates": [], "signals": []}
    succ, order = utils.path_graph(d, {source}, {sink})
    if source not in succ:
        return {**result, "path_exists": False, "count": 0, "gates": [], "signals": []}

    # paths_to[s] * paths_from[s] = paths through s; s is on every path iff that is all of them.
    paths_to = {source: 1}
    for s in order:
        for nxt in succ[s]:
            paths_to[nxt] = paths_to.get(nxt, 0) + paths_to.get(s, 0)
    paths_from = {sink: 1}
    for s in reversed(order):
        if s != sink:
            paths_from[s] = sum(paths_from[n] for n in succ[s])
    total = paths_to[sink]
    steps = {source: 0}  # fewest gates from the source, for ordering
    for s in order:
        for nxt in succ[s]:
            steps[nxt] = min(steps.get(nxt, steps[s] + 1), steps[s] + 1)
    cuts = sorted((steps[s], d.net(s).getFirstOutput().getInst().getName(), s)
                  for s in order if s != source and paths_to[s] * paths_from[s] == total)
    return {**result, "path_exists": True, "count": len(cuts),
            "gates": [g for _, g, _ in cuts], "signals": [s for _, _, s in cuts]}
