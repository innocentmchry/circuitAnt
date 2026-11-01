import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_log_file(log_path: str) -> Optional[Dict[str, Any]]:
    """Parse one circuitAnt request log file."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return None

    # Testcase name
    m_case = re.search(r"Case:\s*(\w+)", txt)
    case_name = m_case.group(1) if m_case else ""

    # Request number
    m_req = re.search(r"Request #(\d+)", txt)
    req_num = int(m_req.group(1)) if m_req else 0

    # User input
    m_user = re.search(r"USER INPUT[^\n]*\n(.*?)\n\s*└", txt, re.DOTALL)
    prompt = m_user.group(1).strip() if m_user else ""
    if not prompt:
        return None

    # Context summary if available
    m_ctx = re.search(r"── CONTEXT ──\s*\n(.*?)\n\s*─────────────", txt, re.DOTALL)
    context_str = m_ctx.group(1).strip() if m_ctx else ""

    # Model name
    m_model = re.search(r"Model:\s*([\w\-.]+)", txt)
    model_name = m_model.group(1) if m_model else "gpt-4o-mini"

    # Raw JSON responses to extract tool calls and token counts
    raw_blocks = re.findall(r"RAW JSON:\s*\n(.*?)\n\s*│ ───", txt, re.DOTALL)
    tool_calls: List[Dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for block in raw_blocks:
        clean_json = "\n".join(re.sub(r"^\s*│\s?", "", line) for line in block.splitlines())
        try:
            data = json.loads(clean_json)
            usage = data.get("usage", {})
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    if fn_name:
                        tool_calls.append({"tool": fn_name, "args": args})
        except Exception:
            pass

    # Extract latency
    latencies = [float(m.group(1)) for m in re.finditer(r"elapsed:\s*([\d.]+)s", txt)]
    total_latency = sum(latencies) if latencies else 1.0

    # Check execution error
    status = "error" if "FAILED" in txt or "[ERROR]" in txt else "success"

    request_id = f"{case_name}_r{req_num:03d}"
    return {
        "request_id": request_id,
        "testcase": case_name,
        "request_index": req_num,
        "prompt": prompt,
        "context": context_str,
        "model": model_name,
        "tool_calls": tool_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_sec": total_latency,
        "status": status,
    }


def build_benchmark_dataset(
    logs_dir: str = "debugging/logs",
    output_path: str = "experiments/semantic_cache/data/benchmark_traces.jsonl",
    max_cases: int = 999,
) -> List[Dict[str, Any]]:
    """Build standardized benchmark traces from authentic circuitAnt logs across all 100 testcases."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.isdir(logs_dir):
        all_logs = sorted(glob.glob(os.path.join(logs_dir, "*", "*_request_*.log")))
        if not all_logs:
            all_logs = sorted(glob.glob(os.path.join(logs_dir, "*_request_*.log")))
    else:
        all_logs = sorted(glob.glob(logs_dir))

    traces_by_req: Dict[str, Dict[str, Any]] = {}

    for log_file in all_logs:
        parsed = parse_log_file(log_file)
        if not parsed or not parsed.get("tool_calls"):
            continue
        case = parsed["testcase"]
        if not re.match(r"^test\d+$", case):
            continue
        traces_by_req[parsed["request_id"]] = parsed

    traces = list(traces_by_req.values())
    # Sort traces by testcase and request index to maintain chronological order
    traces.sort(key=lambda t: (t["testcase"], t["request_index"]))

    unique_cases = sorted(list(set(t["testcase"] for t in traces)))
    if max_cases < len(unique_cases):
        allowed_cases = set(unique_cases[:max_cases])
        traces = [t for t in traces if t["testcase"] in allowed_cases]

    with open(output_path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    return traces


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    traces = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces
