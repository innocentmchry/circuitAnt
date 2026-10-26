"""Compare runs: tool choice against the predicted tools, LLM calls, time, tokens and cost, per run.

    python3 experiments/compare_runs.py <run id> [<run id> ...] [--cases 51-100]

A run id names debugging/run_analysis_<run>.csv and debugging/run_info_<run>.json (developer mode).
Tool agreement uses debugging/predicted_run_analysis.csv -- an expectation, not ground truth (see
debugging/README.md): "exact" = the same tools in the same order, "covers" = every expected tool was
called (extra calls allowed).  Requests are split into session ones (set_testcase_name, read_design,
write_design: always offered directly) and the rest, where the access modes differ.
"""
import csv
import json
import sys
from pathlib import Path

DEBUG = Path(__file__).resolve().parent.parent / "debugging"
SESSION = {"set_testcase_name", "read_design", "write_design"}


def tools(cell: str) -> list[str]:
    return [t.strip() for t in cell.split(";") if t.strip()]


def summarize(run: str, predicted: dict, cases: tuple[int, int] | None) -> dict:
    info_path = DEBUG / f"run_info_{run}.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    rows = [r for r in csv.DictReader(open(DEBUG / f"run_analysis_{run}.csv"))
            if not cases or cases[0] <= int(r["testcase_number"][4:]) <= cases[1]]
    out = {"run": run, "mode": (info.get("tools") or {}).get("access", "?"), "model": info.get("model", "?"),
           "testcases": len({r["testcase_number"] for r in rows}), "requests": len(rows)}
    for part, keep in (("session", lambda e: set(e) <= SESSION | {"(history)"} and set(e) & SESSION),
                       ("other", lambda e: not (set(e) <= SESSION | {"(history)"} and set(e) & SESSION))):
        sel = [(r, predicted.get((r["testcase_number"], r["prompt"]))) for r in rows]
        sel = [(r, e) for r, e in sel if e is not None and keep(e)]
        n = len(sel) or 1
        called = [tools(r["tool_called"]) for r, _ in sel]
        out[part] = {
            "requests": len(sel),
            "exact": sum(c == e for c, (_, e) in zip(called, sel)) / n,
            "covers": sum(set(e) - {"(history)"} <= set(c) or (e == ["(history)"] and c == ["(history)"])
                          for c, (_, e) in zip(called, sel)) / n,
            "llm_calls": sum(int(r["llm_calls"]) for r, _ in sel) / n,
            "seconds": (sum(float(r["seconds"]) for r, _ in sel) / n) if sel and sel[0][0].get("seconds") else None,
            "input_tokens": sum(int(r["input_tokens"]) for r, _ in sel) / n,
            "cached": sum(int(r["cached_input_tokens"]) for r, _ in sel) / n,
            "output_tokens": sum(int(r["output_tokens"]) for r, _ in sel) / n,
            "graph_ops": sum(len(tools(r.get("graph_ops", ""))) for r, _ in sel) / n}
    costs = [r.get("cost_billed_usd") for r in rows]  # the cost columns exist only with prices set
    out["cost_billed_usd"] = sum(float(c) for c in costs) if costs and all(costs) else None
    out["cost_list_usd"] = sum(float(r["cost_list_usd"]) for r in rows) if out["cost_billed_usd"] is not None else None
    out["input_tokens_total"] = sum(int(r["input_tokens"]) for r in rows)
    out["seconds_total"] = sum(float(r["seconds"]) for r in rows) if rows and rows[0].get("seconds") else None
    return out


def main() -> None:
    args = sys.argv[1:]
    cases = None
    if "--cases" in args:
        lo, hi = args[args.index("--cases") + 1].split("-")
        cases = (int(lo), int(hi))
        del args[args.index("--cases"):args.index("--cases") + 2]
    if not args:
        sys.exit(__doc__)
    predicted = {(r["testcase_number"], r["prompt"]): tools(r["tool_called"])
                 for r in csv.DictReader(open(DEBUG / "predicted_run_analysis.csv"))}
    results = [summarize(run, predicted, cases) for run in args]
    print(f"{'run':28} {'mode':6} {'model':18} {'cases':>5} {'reqs':>5} | {'part':7} {'reqs':>4} {'exact':>6} "
          f"{'covers':>6} {'calls':>5} {'secs':>5} {'in_tok':>7} {'cached':>7} {'out_tok':>7} {'g_ops':>5}")
    for r in results:
        for part in ("session", "other"):
            p = r[part]
            head = f"{r['run']:28} {r['mode']:6} {r['model']:18} {r['testcases']:>5} {r['requests']:>5}" \
                if part == "session" else " " * 67
            secs = f"{p['seconds']:>5.1f}" if p["seconds"] is not None else f"{'-':>5}"  # older runs: no column
            print(f"{head} | {part:7} {p['requests']:>4} {p['exact']:>6.0%} {p['covers']:>6.0%} {p['llm_calls']:>5.2f} "
                  f"{secs} {p['input_tokens']:>7.0f} {p['cached']:>7.0f} {p['output_tokens']:>7.0f} {p['graph_ops']:>5.2f}")
        cost = (f"billed ${r['cost_billed_usd']:.4f}, list ${r['cost_list_usd']:.4f}"
                if r["cost_billed_usd"] is not None else "no prices in the config")
        total = f"{r['seconds_total']:.0f} s" if r["seconds_total"] is not None else "not recorded"
        print(f"{'':67} | total input tokens {r['input_tokens_total']:,}; total time {total}; {cost}")


if __name__ == "__main__":
    main()
