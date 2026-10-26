"""How often the temporary keyword search_tools ranks a prompt's expected tool near the top (no LLM).

search_tools is off by default (tools.graph.search) and is to be replaced by embedding-based search;
this script measures the keyword version, as a baseline for that.

    ./openroad_python.sh experiments/search_recall.py [--misses]

Expected tools come from debugging/predicted_run_analysis.csv (an expectation, not ground truth).
Session tools and history answers are left out (the session tools are always offered directly).
Reported separately for testcases 1-50 (seen while writing knowledge.ttl) and 51-100 (held out:
look, never tune on them).
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.tool_access.graph.build import build  # noqa: E402
from agent.tool_access.graph.operations import Operations  # noqa: E402

SESSION = {"set_testcase_name", "read_design", "write_design"}


def main() -> None:
    ops = Operations(build())
    rows = list(csv.DictReader(open(ROOT / "debugging/predicted_run_analysis.csv")))
    for label, lo, hi in (("testcases 1-50 (seen)", 1, 50), ("testcases 51-100 (held out)", 51, 100)):
        hits, total, misses = {1: 0, 3: 0, 5: 0}, 0, []
        for r in rows:
            expected = {t.strip() for t in r["tool_called"].split(";")} - SESSION - {"(history)"}
            if not expected or not lo <= int(r["testcase_number"][4:]) <= hi:
                continue
            ranked = [x["tool"] for x in ops.search(r["prompt"], limit=5)]
            total += 1
            for k in hits:
                hits[k] += expected <= set(ranked[:k])
            if not expected <= set(ranked[:3]):
                misses.append((r["testcase_number"], sorted(expected), ranked[:3], r["prompt"]))
        print(f"{label}: {total} prompts; expected tool(s) in top 1: {hits[1] / total:.0%}, "
              f"top 3: {hits[3] / total:.0%}, top 5: {hits[5] / total:.0%}")
        if "--misses" in sys.argv and lo == 1:  # only the seen half, so nothing is tuned on the held-out half
            for case, exp, got, prompt in misses:
                print(f"   {case} expected {exp} got {got}: {prompt[:100]}")


if __name__ == "__main__":
    main()
