"""Which embedding model should the agent's cache use?  Offline, no LLM.

    ./openroad_python.sh experiments/semantic_cache/model_comparison.py   (needs sentence-transformers)

A. Replay: the recorded requests of testcases 1-50 (data/benchmark_traces.jsonl), in order, as the
   live cache sees them: each request is compared with every earlier one, the most similar is the
   candidate, it is a hit when the similarity reaches the threshold (and, in the "check" variant,
   the opposite-word check passes), and every request is cached after it.  A hit is right when the
   candidate's first tool is the tool the LLM used first.  Similarities are not comparable across
   models, so each model is swept over its own range; the number to compare is the hit rate at a
   given share of right hits (98% is what all-MiniLM-L6-v2 reaches at 0.80).
B. Probe (data/opposite_probe.json): hand-made triples of a request, a paraphrase with other names
   (should match) and an opposite (tool_changes: the opposite needs another tool, so it must not
   match; otherwise the same tool, so matching it is harmless).

Results: printed, and experiments/results/semantic_cache_models/results.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from experiments.semantic_cache.validation.polarity import check_negative_constraints  # noqa: E402

MODELS = {  # name -> text prefix the model expects
    "all-MiniLM-L6-v2": "",
    "BAAI/bge-base-en-v1.5": "",
    "sentence-transformers/all-mpnet-base-v2": "",
    "intfloat/e5-base-v2": "query: ",
}
TESTCASES = (1, 50)  # tuning set; 51-100 are held out
PRECISIONS = (0.95, 0.98, 0.99)
OUT = HERE.parent / "results" / "semantic_cache_models"


def replay(sim: np.ndarray, tools: list[str], passes: np.ndarray, check: bool) -> list[tuple]:
    """(threshold, hit rate, right share) for each threshold, from the best earlier match of each request."""
    n = len(tools)
    best = [None] + [int(np.argmax(sim[i, :i])) for i in range(1, n)]
    score = np.array([-1.0] + [sim[i, best[i]] for i in range(1, n)])
    right = np.array([False] + [tools[best[i]] == tools[i] for i in range(1, n)])
    allowed = passes if check else np.ones(n, bool)
    curve = []
    for t in np.unique(np.round(score[1:], 3)):
        hit = (score >= t) & allowed
        if hit.sum():
            curve.append((float(t), hit.sum() / n, right[hit].sum() / hit.sum()))
    return curve


def best_at(curve: list[tuple], precision: float) -> tuple | None:
    """The threshold with the most hits whose right share is at least precision."""
    ok = [c for c in curve if c[2] >= precision]
    return max(ok, key=lambda c: c[1]) if ok else None


def main() -> None:
    from sentence_transformers import SentenceTransformer
    traces = [json.loads(line) for line in open(HERE / "data" / "benchmark_traces.jsonl")]
    traces = [t for t in traces if TESTCASES[0] <= int(t["testcase"][4:]) <= TESTCASES[1] and t["tool_calls"]]
    prompts = [t["prompt"] for t in traces]
    tools = [t["tool_calls"][0]["tool"] for t in traces]
    n = len(prompts)
    # the opposite-word check between each request and its best earlier match depends on the model's
    # choice of match, so it is computed per model below
    probe = json.loads((HERE / "data" / "opposite_probe.json").read_text())
    results = {}
    for name, prefix in MODELS.items():
        model = SentenceTransformer(name)
        started = time.perf_counter()
        emb = model.encode([prefix + p for p in prompts], normalize_embeddings=True, batch_size=32)
        ms = (time.perf_counter() - started) / n * 1000
        sim = emb @ emb.T
        passes = np.array([True] + [check_negative_constraints(prompts[i], prompts[int(np.argmax(sim[i, :i]))])[0]
                                    for i in range(1, n)])
        curves = {check: replay(sim, tools, passes, check) for check in (False, True)}
        rows = []
        for item in probe:
            a, p, o = model.encode([prefix + item[k] for k in ("anchor", "paraphrase", "opposite")],
                                   normalize_embeddings=True)
            rows.append({"note": item["note"], "tool_changes": item["tool_changes"],
                         "paraphrase": float(a @ p), "opposite": float(a @ o)})
        results[name] = {"ms_per_request": ms, "dims": int(emb.shape[1]), "probe": rows,
                         "at": {f"{check}_{p}": best_at(curves[check], p) for check in (False, True)
                                for p in PRECISIONS},
                         "unrelated_median": float(np.median(sim[np.triu_indices(n, 1)]))}
        print(f"\n== {name}  ({emb.shape[1]} dims, {ms:.1f} ms/request, median similarity of any two "
              f"requests {results[name]['unrelated_median']:.2f})")
        for check in (False, True):
            for p in PRECISIONS:
                b = results[name]["at"][f"{check}_{p}"]
                print(f"   opposite check {'on ' if check else 'off'}  right >= {p:.0%}: " +
                      (f"threshold {b[0]:.3f}  hit rate {b[1]:.1%}  right {b[2]:.1%}" if b else "never reached"))
        print("   probe (similarity to paraphrase / to opposite):")
        for r in rows:
            mark = "ok " if r["paraphrase"] > r["opposite"] else "BAD"
            print(f"     {mark if r['tool_changes'] else 'n/a'} {r['paraphrase']:.3f} / {r['opposite']:.3f}  "
                  f"{'tool changes' if r['tool_changes'] else 'same tool   '}  {r['note']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n{n} requests of testcases {TESTCASES[0]}-{TESTCASES[1]}; results in {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
