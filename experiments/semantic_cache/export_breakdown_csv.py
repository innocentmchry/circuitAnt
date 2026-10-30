#!/usr/bin/env python3
"""Export full execution breakdown CSV across all testcase prompts.

This script parses the request execution logs from an experiment run and produces
a prompt-by-prompt CSV spreadsheet detailing:
  - Exact user prompt
  - Cache decision (HIT vs. MISS/FALLBACK)
  - Execution reason
  - Agreement with reference LLM plan
  - Tokens and latency saved
  - Tool calls proposed by cache vs. reference LLM calls

Usage:
    python experiments/semantic_cache/export_breakdown_csv.py
    python experiments/semantic_cache/export_breakdown_csv.py --run-dir experiments/semantic_cache/results/E6_1793206174
    python experiments/semantic_cache/export_breakdown_csv.py --output my_breakdown.csv
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path


def find_latest_run_dir(results_dir: str = "experiments/semantic_cache/results") -> str:
    """Find the latest experiment run directory containing request_logs.jsonl (prioritizing E6 full hybrid)."""
    e6_candidates = [d for d in glob.glob(os.path.join(results_dir, "E6_*")) if os.path.exists(os.path.join(d, "request_logs.jsonl"))]
    if e6_candidates:
        e6_candidates.sort(key=os.path.getmtime, reverse=True)
        return e6_candidates[0]

    pattern = os.path.join(results_dir, "E*")
    candidates = []
    for d in glob.glob(pattern):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "request_logs.jsonl")):
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No valid experiment run directories found in {results_dir}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def export_breakdown_csv(run_dir: str, output_csv: str) -> None:
    log_path = os.path.join(run_dir, "request_logs.jsonl")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Missing request logs at: {log_path}")

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"[WARNING] No records found in {log_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    try:
        f_out = open(output_csv, "w", newline="", encoding="utf-8")
    except PermissionError:
        base, ext = os.path.splitext(output_csv)
        output_csv = f"{base}_latest{ext}"
        print(f"[NOTE] Target file is open in another program. Writing to: {output_csv}")
        f_out = open(output_csv, "w", newline="", encoding="utf-8")

    with f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "request_id",
            "testcase",
            "prompt",
            "decision_type",
            "accepted",
            "reason",
            "agreement",
            "tokens_saved",
            "baseline_tokens",
            "cost_saved_usd",
            "latency_saved_sec",
            "proposed_tools",
            "reference_tools",
        ])

        for r in records:
            prop_str = "; ".join([f"{tc['tool']}({tc.get('args', {})})" for tc in r.get("proposed_tool_calls", [])])
            ref_str = "; ".join([f"{tc['tool']}({tc.get('args', {})})" for tc in r.get("reference_tool_calls", [])])
            base_tok = r.get("baseline_prompt_tokens", 0) + r.get("baseline_completion_tokens", 0)

            writer.writerow([
                r.get("request_id"),
                r.get("testcase"),
                r.get("prompt"),
                r.get("decision_type"),
                r.get("accepted"),
                r.get("reason"),
                r.get("agreement"),
                r.get("total_tokens_saved"),
                base_tok,
                f"{r.get('cost_saved_usd', 0.0):.5f}",
                f"{r.get('latency_saved_sec', 0.0):.3f}",
                prop_str,
                ref_str,
            ])

    # Summary Statistics
    total = len(records)
    hits = sum(1 for r in records if r.get("accepted"))
    misses = total - hits
    agreements = sum(1 for r in records if r.get("accepted") and r.get("agreement") is True)
    disagreements = sum(1 for r in records if r.get("accepted") and r.get("agreement") is False)
    tokens_saved = sum(r.get("total_tokens_saved", 0) for r in records)
    cost_saved = sum(r.get("cost_saved_usd", 0.0) for r in records)
    testcases = sorted(list(set(r.get("testcase") for r in records)))

    print("\n" + "=" * 65)
    print("Execution Breakdown CSV Export Complete")
    print("=" * 65)
    print(f"Run Source:          {run_dir}")
    print(f"Output CSV:          {output_csv}")
    print(f"Total Requests:      {total}")
    print(f"Testcases Covered:   {len(testcases)} ({testcases[0]} to {testcases[-1]})")
    print(f"Cache Hits:          {hits} ({hits/total*100:.2f}%)")
    print(f"Cache Misses:        {misses} ({misses/total*100:.2f}%)")
    if hits > 0:
        print(f"Agreement Rate:      {agreements}/{hits} ({agreements/hits*100:.2f}%)")
        print(f"False-Hit Proxy:     {disagreements}/{hits} ({disagreements/hits*100:.2f}%)")
    print(f"Total Tokens Saved:  {tokens_saved:,}")
    print(f"Cost Saved (USD):    ${cost_saved:.4f}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export prompt-by-prompt cache execution breakdown CSV.")
    parser.add_argument(
        "--run-dir",
        default="",
        help="Path to experiment run directory containing request_logs.jsonl (defaults to latest E6 run).",
    )
    parser.add_argument(
        "--output",
        default="experiments/semantic_cache/results/full_execution_breakdown_100cases.csv",
        help="Target path for generated CSV file.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir:
        run_dir = find_latest_run_dir()

    export_breakdown_csv(run_dir, args.output)


if __name__ == "__main__":
    main()
