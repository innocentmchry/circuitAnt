#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.semantic_cache.data.loader import build_benchmark_dataset
from experiments.semantic_cache.evaluation.analyze import generate_ablation_table, load_all_experiment_metrics
from experiments.semantic_cache.evaluation.runner import ExperimentRunner


def run_single_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ensure dataset exists
    ds_path = cfg.get("dataset_path", "experiments/semantic_cache/data/benchmark_traces.jsonl")
    if not os.path.isabs(ds_path):
        ds_path = str((REPO_ROOT / ds_path).resolve())
    if not os.path.exists(ds_path):
        print(f"Building benchmark dataset traces -> {ds_path}")
        build_benchmark_dataset(output_path=ds_path, max_cases=999)

    print(f"\n=======================================================")
    print(f"Running Experiment {cfg.get('experiment_id')}: {cfg.get('strategy')}")
    print(f"Config: {config_path}")
    print(f"=======================================================")

    runner = ExperimentRunner(cfg)
    metrics = runner.run_replay(ds_path)

    print("\n--- Completed Experiment Run ---")
    print(f"Run ID: {metrics.get('run_id')}")
    print(f"Total Requests: {metrics.get('total_requests')}")
    print(f"Cache Hits: {metrics.get('cache_hits')} ({metrics.get('hit_rate_pct')}%)")
    print(f"Agreements: {metrics.get('agreements')} ({metrics.get('reference_agreement_pct')}%)")
    print(f"False-Hit Proxy: {metrics.get('disagreements')} ({metrics.get('false_hit_proxy_rate_pct')}%)")
    print(f"Tokens Saved: {metrics.get('total_tokens_saved'):,} ({metrics.get('token_reduction_pct')}%)")
    print(f"Cost Saved (USD): ${metrics.get('estimated_cost_saved_usd'):.4f} ({metrics.get('estimated_cost_reduction_pct')}%)")
    print(f"Latency Reduction: {metrics.get('latency_reduction_pct')}%")
    print(f"Cache Size: {metrics.get('final_cache_size')}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="CircuitAnt Semantic Cache Experiment Runner")
    ap.add_argument("-config", help="Path to experiment config YAML")
    ap.add_argument("--all", action="store_true", help="Run all ablation experiments E0 through E6")
    args = ap.parse_args()

    if args.all:
        configs = [
            "experiments/semantic_cache/configs/e0_baseline.yaml",
            "experiments/semantic_cache/configs/e1_exact.yaml",
            "experiments/semantic_cache/configs/e2_semantic.yaml",
            "experiments/semantic_cache/configs/e3_semantic_rerank.yaml",
            "experiments/semantic_cache/configs/e4_semantic_structural.yaml",
            "experiments/semantic_cache/configs/e5_semantic_both.yaml",
            "experiments/semantic_cache/configs/e6_full_hybrid.yaml",
        ]
        for c in configs:
            run_single_config(c)

        print("\n\n=======================================================")
        print("All Ablations Completed. Generating Comparison Report...")
        print("=======================================================")
        runs = load_all_experiment_metrics()
        table_md = generate_ablation_table(runs)
        print("\n" + table_md)
    elif args.config:
        run_single_config(args.config)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
