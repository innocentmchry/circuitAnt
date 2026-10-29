import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_all_experiment_metrics(results_dir: str = "experiments/semantic_cache/results") -> List[Dict[str, Any]]:
    """Loads metrics.json from all run directories in results_dir, keeping latest per experiment."""
    metric_files = glob.glob(os.path.join(results_dir, "*", "metrics.json"))
    runs_by_exp: Dict[str, Dict[str, Any]] = {}
    for mf in sorted(metric_files, key=os.path.getmtime):
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
                exp_id = data.get("experiment_id", "")
                if exp_id:
                    runs_by_exp[exp_id] = data
        except Exception:
            pass
    # Sort in order: E0, E1, E2, E3, E4, E5, E6
    exp_order = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    sorted_runs = []
    for exp in exp_order:
        if exp in runs_by_exp:
            sorted_runs.append(runs_by_exp[exp])
    for k, v in runs_by_exp.items():
        if k not in exp_order:
            sorted_runs.append(v)
    return sorted_runs


def generate_ablation_table(
    runs: List[Dict[str, Any]],
    output_md: str = "experiments/semantic_cache/results/ablation_comparison.md",
    output_csv: str = "experiments/semantic_cache/results/ablation_comparison.csv",
) -> str:
    """Generate standardized ablation comparison Markdown and CSV tables."""
    os.makedirs(os.path.dirname(output_md), exist_ok=True)

    headers = [
        "Experiment",
        "Strategy",
        "Hit Rate (%)",
        "LLM Avoidance (%)",
        "Agreement (%)",
        "False Hit Proxy (%)",
        "Tokens Saved",
        "Token Red. (%)",
        "Cost Saved ($)",
        "Latency Red. (%)",
        "Cache Size",
    ]

    rows = []
    for r in runs:
        exp_id = r.get("experiment_id", "-")
        strategy = r.get("strategy", "-")
        hit_rate = f"{r.get('hit_rate_pct', 0.0):.1f}%"
        avoidance = f"{r.get('llm_planning_avoidance_pct', 0.0):.1f}%"
        agreement = f"{r.get('reference_agreement_pct', 0.0):.1f}%"
        false_hit = f"{r.get('false_hit_proxy_rate_pct', 0.0):.1f}%"
        tokens_saved = f"{r.get('total_tokens_saved', 0):,}"
        token_red = f"{r.get('token_reduction_pct', 0.0):.1f}%"
        cost_saved = f"${r.get('estimated_cost_saved_usd', 0.0):.4f}"
        lat_red = f"{r.get('latency_reduction_pct', 0.0):.1f}%"
        cache_size = str(r.get("final_cache_size", 0))

        rows.append([
            exp_id, strategy, hit_rate, avoidance, agreement,
            false_hit, tokens_saved, token_red, cost_saved,
            lat_red, cache_size
        ])

    # Build Markdown table
    md_lines = [
        "# circuitAnt Semantic Tool-Call Caching: Ablation Study Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")

    md_content = "\n".join(md_lines) + "\n"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Build CSV table
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return md_content


if __name__ == "__main__":
    runs = load_all_experiment_metrics()
    if runs:
        md = generate_ablation_table(runs)
        print(md)
    else:
        print("No completed experiment runs found in results directory.")
