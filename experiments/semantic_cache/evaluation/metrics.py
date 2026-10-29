from typing import Any, Dict, List


TOOL_DEFAULTS = {
    "get_cone": {"direction": "fanin"},
    "get_max_depth": {"source": "ANY_START", "sink": "ANY_END"},
    "get_endpoint_stats": {"sink": "ANY_PO", "metric": "depth"},
    "find_cut_points": {"source": "ANY_PI", "sink": "ANY_PO"},
}


import os


def canonicalize_args(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize arguments according to Tools method signatures and defaults."""
    c = dict(args)
    if tool in TOOL_DEFAULTS:
        for k, v in TOOL_DEFAULTS[tool].items():
            if k not in c and v is not None:
                c[k] = v
    # Normalize circuit endpoint aliases
    if "source" in c and isinstance(c["source"], str):
        s_lower = c["source"].strip().lower()
        if s_lower in ("register", "reg", "dff", "flip-flop", "flip_flop"):
            c["source"] = "ANY_DFF_Q"
        elif s_lower in ("pi", "primary_input", "inputs"):
            c["source"] = "ANY_PI"
        elif s_lower in ("start", "any"):
            c["source"] = "ANY_START"

    if "sink" in c and isinstance(c["sink"], str):
        s_lower = c["sink"].strip().lower()
        if s_lower in ("register", "reg", "dff", "flip-flop", "flip_flop"):
            c["sink"] = "ANY_DFF_D"
        elif s_lower in ("po", "primary_output", "outputs"):
            c["sink"] = "ANY_PO"
        elif s_lower in ("end", "any"):
            c["sink"] = "ANY_END"

    if tool == "get_max_depth" and c.get("source") and c.get("sink") and c["source"] == c["sink"] and not c["source"].startswith("ANY_"):
        # When an output signal was extracted to both source and sink, fanin starts at ANY_START
        c["source"] = "ANY_START"

    # Normalize paths and strings: resolve redundant ./ or .\ segments
    for k, v in list(c.items()):
        if isinstance(v, str):
            if "path" in k or "file" in k or "/" in v or "\\" in v:
                norm_p = os.path.normpath(v).replace("\\", "/")
                if tool == "write_design" and "path" in k:
                    c[k] = os.path.basename(norm_p)
                else:
                    c[k] = norm_p
            elif v.startswith("./") or v.startswith(".\\"):
                c[k] = v[2:]
        if v is None:
            del c[k]
    return c


OPTIONAL_HELPERS = {"filter_unused", "simplify_netlist"}

def _strip_optional(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of the plan without optional‑helper calls."""
    return [call for call in plan if call.get("tool") not in OPTIONAL_HELPERS]

def plans_agree(plan_a: List[Dict[str, Any]], plan_b: List[Dict[str, Any]]) -> bool:
    """Check whether two tool plans are identical after removing optional helpers.

    The function first discards any calls whose tool name is in ``OPTIONAL_HELPERS``.
    Then it performs a strict tool‑name and canonicalised‑argument comparison.
    This allows cached plans that contain harmless preprocessing steps (e.g.,
    ``filter_unused``) to still be counted as agreeing with the baseline.
    """
    a = _strip_optional(plan_a)
    b = _strip_optional(plan_b)
    if len(a) != len(b):
        return False
    for call_a, call_b in zip(a, b):
        tool_a = call_a.get("tool")
        tool_b = call_b.get("tool")
        if tool_a != tool_b:
            return False
        args_a = canonicalize_args(tool_a, call_a.get("args", {}))
        args_b = canonicalize_args(tool_b, call_b.get("args", {}))
        if args_a != args_b:
            return False
    return True


def compute_metrics(request_records: List[Dict[str, Any]], total_requests: int) -> Dict[str, Any]:
    """Compute comprehensive caching and ablation metrics from request logs."""
    if total_requests == 0:
        return {}

    hits = sum(1 for r in request_records if r.get("accepted"))
    misses = total_requests - hits

    hit_rate = (hits / total_requests) * 100.0
    avoidance_rate = hit_rate

    # ---- Agreement handling with multiple reference plans ----
    agreements = 0
    disagreements = 0
    for r in request_records:
        if not r.get("accepted"):
            continue
        # If record has precomputed agreement flag without explicit tool plans
        if "agreement" in r and r.get("agreement") is not None and not r.get("reference_tool_calls") and not r.get("reference_plans"):
            if r.get("agreement") is True:
                agreements += 1
            else:
                disagreements += 1
            continue

        cached_plan = r.get("proposed_tool_calls", [])
        ref_plans = r.get("reference_plans")
        if not ref_plans:
            # fallback to single reference field for backward compatibility
            ref_plans = [r.get("reference_tool_calls", [])]
        if any(plans_agree(cached_plan, ref) for ref in ref_plans if ref):
            agreements += 1
        else:
            disagreements += 1
    # ----------------------------------------------------------

    agreement_rate = (agreements / hits * 100.0) if hits > 0 else 0.0
    false_hit_proxy_rate = (disagreements / hits * 100.0) if hits > 0 else 0.0

    # Tokens and cost
    total_prompt_tokens_saved = sum(r.get("prompt_tokens_saved", 0) for r in request_records)
    total_completion_tokens_saved = sum(r.get("completion_tokens_saved", 0) for r in request_records)
    total_tokens_saved = total_prompt_tokens_saved + total_completion_tokens_saved

    baseline_prompt_tokens = sum(r.get("baseline_prompt_tokens", 0) for r in request_records)
    baseline_completion_tokens = sum(r.get("baseline_completion_tokens", 0) for r in request_records)
    baseline_total_tokens = baseline_prompt_tokens + baseline_completion_tokens

    token_reduction_pct = (total_tokens_saved / baseline_total_tokens * 100.0) if baseline_total_tokens > 0 else 0.0

    # Costs (gpt-4o-mini pricing: $0.15/1M prompt, $0.60/1M completion)
    cost_saved_usd = (total_prompt_tokens_saved * 0.15 / 1e6) + (total_completion_tokens_saved * 0.60 / 1e6)
    baseline_cost_usd = (baseline_prompt_tokens * 0.15 / 1e6) + (baseline_completion_tokens * 0.60 / 1e6)
    cost_reduction_pct = (cost_saved_usd / baseline_cost_usd * 100.0) if baseline_cost_usd > 0 else 0.0

    # Latencies
    baseline_latencies = [r.get("baseline_latency_sec", 0.0) for r in request_records]
    actual_latencies = [r.get("actual_latency_sec", 0.0) for r in request_records]
    overhead_latencies = [r.get("cache_overhead_sec", 0.0) for r in request_records]

    mean_baseline_lat = sum(baseline_latencies) / len(baseline_latencies) if baseline_latencies else 0.0
    mean_actual_lat = sum(actual_latencies) / len(actual_latencies) if actual_latencies else 0.0
    mean_overhead_lat = sum(overhead_latencies) / len(overhead_latencies) if overhead_latencies else 0.0

    lat_reduction_pct = ((mean_baseline_lat - mean_actual_lat) / mean_baseline_lat * 100.0) if mean_baseline_lat > 0 else 0.0

    final_cache_size = request_records[-1].get("cache_size_after", 0) if request_records else 0

    return {
        "total_requests": total_requests,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": round(hit_rate, 2),
        "llm_planning_avoidance_pct": round(avoidance_rate, 2),
        "agreements": agreements,
        "disagreements": disagreements,
        "reference_agreement_pct": round(agreement_rate, 2),
        "false_hit_proxy_rate_pct": round(false_hit_proxy_rate, 2),
        "total_prompt_tokens_saved": total_prompt_tokens_saved,
        "total_completion_tokens_saved": total_completion_tokens_saved,
        "total_tokens_saved": total_tokens_saved,
        "token_reduction_pct": round(token_reduction_pct, 2),
        "estimated_cost_saved_usd": round(cost_saved_usd, 5),
        "estimated_cost_reduction_pct": round(cost_reduction_pct, 2),
        "mean_baseline_latency_sec": round(mean_baseline_lat, 4),
        "mean_actual_latency_sec": round(mean_actual_lat, 4),
        "mean_cache_overhead_sec": round(mean_overhead_lat, 4),
        "latency_reduction_pct": round(lat_reduction_pct, 2),
        "final_cache_size": final_cache_size,
    }
