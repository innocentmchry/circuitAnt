"""Developer-mode conversation logging — writes human-readable logs of LLM conversations.

format_netlist() prints a short summary of the OpenDB design, not a full dump.

Only used when ``developer: mode: true`` in config.yaml.  When the flag is
off, ``NoopLogger`` is used instead (zero runtime overhead).  In developer mode
RunAnalysis also records which tools each request actually called, in
debugging/run_analysis_<run id>.csv (the run id of the log folder, logs/<run id>/),
and write_run_info() the settings the run used, in debugging/run_info_<run id>.json.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOGS_DIR: Optional[Path] = None


def run_logs_dir(base_dir: Optional[str | Path] = None) -> Path:
    """Create (if needed) and return this run's log folder, logs/<run id>/ under
    base_dir (default: debugging/).  The run id is $CIRCUITANT_RUN -- run.sh sets one
    for all the testcases it runs -- or else the time this process started."""
    global _LOGS_DIR
    if _LOGS_DIR is not None:
        return _LOGS_DIR

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    run_id = os.environ.get("CIRCUITANT_RUN") or datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOGS_DIR = Path(base_dir) / "logs" / run_id
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return _LOGS_DIR


def sanitize_label(label: str) -> str:
    """Return a filesystem-safe version of *label*."""
    safe = re.sub(r"[^a-zA-Z0-9_\-.]", "_", label)
    return safe.strip("_") or "unknown"


def extract_case_name(user_message: str) -> str:
    """Try to extract a testcase name from a user message.

    Looks for patterns like:  case name is test001.  or  case name is 'test8'
    Falls back to an empty string.
    """
    m = re.search(
        r"case\s+name\s+(?:is\s+)?['\"]?(\w+)",
        user_message,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)
    return ""


def display_lines(text: str, width: int = 100) -> List[str]:
    """Break message text into readable lines for the log (display only).

    Paragraph-length lines (e.g. the system prompt) get one sentence per line;
    other long lines are wrapped, continuing under their own indentation.  The
    text sent to the model is unchanged.
    """
    lines: List[str] = []
    for line in text.splitlines() or [""]:
        if len(line) <= width:
            lines.append(line)
            continue
        indent = re.match(r"\s*(- )?", line).group(0)
        hang = " " * len(indent)
        parts = (re.split(r"(?<=[.;:?!])\s+(?=[A-Z])", line)
                 if len(line) > 4 * width else [line])
        for part in parts:
            lines.extend(textwrap.wrap(part, width, subsequent_indent=hang))
    return lines


def format_netlist(design: Any) -> str:
    """Summary of the design as loaded into OpenDB (not a full dump).

    Args:
        design: An eda_engine.Design.

    Returns:
        Multi-line string: counts, ports, cell usage and a few sample instances.
    """
    block = design.block
    insts = block.getInsts()
    nets = block.getNets()
    lines: List[str] = []

    lines.append(f"top module     : {design.top}  ({design.path})")
    lines.append(f"instances      : {len(insts)}")
    lines.append(f"nets           : {len(nets)}")
    port_list = design.port_list()
    lines.append(f"ports          : {len(port_list)} declared, {len(block.getBTerms())} bits")
    for direction in ("input", "output", "inout"):
        ports = [n if m is None else f"{n}[{m}:{l}]"
                 for n, d, m, l in port_list if d == direction]
        if ports:
            lines.append(f"  {direction:6s} ({len(ports)}): {', '.join(ports)}")

    usage: Dict[str, int] = {}
    for inst in insts:
        name = inst.getMaster().getName()
        usage[name] = usage.get(name, 0) + 1
    lines.append("cells by type  : " + ", ".join(f"{n} {c}" for n, c in sorted(usage.items())))

    consts = [f"{n.getName()}={design.signal(n)}" for n in nets if n.getSigType() != "SIGNAL"]
    lines.append(f"constant nets  : {', '.join(consts) if consts else 'none'}")

    lines.append("sample instances:")
    for inst in insts[:3]:
        pins = ", ".join(f"{it.getMTerm().getName()}={design.signal(it.getNet()) if it.getNet() else '-'}"
                         for it in inst.getITerms())
        lines.append(f"  {inst.getName()} ({inst.getMaster().getName()}): {pins}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real conversation logger (used in developer mode)
# ---------------------------------------------------------------------------

class ConversationLogger:
    """Writes a human-readable log of a single stdin→LLM conversation.

    One instance per request.  Use as a context manager::

        with ConversationLogger(idx, logs_dir, model, label="test8") as log:
            log.log_user_input(...)
            ...
    """

    def __init__(
        self,
        request_index: int,
        logs_dir: Path,
        model: str,
        label: str = "",
    ) -> None:
        self._index = request_index
        self._model = model
        self._start_time = time.monotonic()

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_label = sanitize_label(label) if label else ""
        if safe_label:
            filename = logs_dir / f"{safe_label}_request_{request_index:04d}_{ts}.log"
        else:
            filename = logs_dir / f"request_{request_index:04d}_{ts}.log"
        self._fh: TextIO = open(filename, "w", encoding="utf-8")

        self._fh.write("═" * 72 + "\n")
        self._fh.write(f"Request #{request_index} — {ts} UTC\n")
        if label:
            self._fh.write(f"Case: {label}\n")
        self._fh.write(f"Model: {model}\n")
        self._fh.write("═" * 72 + "\n\n")

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_user_input(self, text: str) -> None:
        """Record the original user message."""
        self._fh.write("┌─ USER INPUT ─────────────────────────────\n")
        self._fh.write(f"{text.rstrip()}\n")
        self._fh.write("└──────────────────────────────────────────\n\n")

    def log_llm_prompt(self, messages: List[Dict[str, Any]], turn: int) -> None:
        """Log the full prompt sent to the LLM for a given turn."""
        self._fh.write(f"┌─ LLM PROMPT (turn {turn}) ───────────────\n")
        for i, m in enumerate(messages):
            role = m.get("role", "?")
            self._fh.write(f"│ [{i}] role: {role}\n")
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    self._fh.write(f"│     tool_call: {fn.get('name', '?')}\n")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        args_str = json.dumps(args, indent=6)
                    except (json.JSONDecodeError, TypeError):
                        args_str = str(fn.get("arguments", "{}"))
                    self._fh.write("│     arguments:\n")
                    for line in args_str.splitlines():
                        self._fh.write(f"│       {line}\n")
            else:
                content = m.get("content") or ""
                try:
                    parsed = json.loads(content)
                    content_str = json.dumps(parsed, indent=6)
                    self._fh.write("│     content (JSON):\n")
                    for line in content_str.splitlines():
                        self._fh.write(f"│       {line}\n")
                except (json.JSONDecodeError, TypeError):
                    for line in display_lines(content):
                        self._fh.write(f"│     {line}\n")
        self._fh.write("└──────────────────────────────────────────\n\n")

    def log_llm_response(self, response: Any, turn: int, elapsed: float = 0.0) -> None:
        """Log the LLM's response — raw JSON first, then summary.
        
        Args:
            response: The model's reply (llm.Reply).
            turn: Turn number within this request.
            elapsed: Seconds this LLM call took (wall-clock).
        """
        msg = response
        finish = response.finish_reason

        self._fh.write(f"┌─ LLM RESPONSE (turn {turn}) ─────────────\n")
        self._fh.write("│ RAW JSON:\n")
        try:
            raw_str = json.dumps(response.raw, indent=6, ensure_ascii=False)
        except (AttributeError, TypeError):
            raw_str = str(response)
        for line in raw_str.splitlines():
            self._fh.write(f"│   {line}\n")

        self._fh.write("│ ─────────────────────────────────\n")
        self._fh.write("│ SUMMARY:\n")
        self._fh.write(f"│   finish_reason: {finish}\n")

        if msg.content:
            self._fh.write(f"│   text: {json.dumps(msg.content)}\n")

        if msg.tool_calls:
            self._fh.write(f"│   tool_calls: {len(msg.tool_calls)}\n")
            for tc in msg.tool_calls:
                self._fh.write(f"│     → {tc.name}\n")
                self._fh.write(f"│       args: {tc.arguments}\n")

        self._fh.write(
            f"│   tokens: prompt={response.prompt_tokens} (cached={response.cached_tokens}"
            f"  cache_write={response.cache_write_tokens})  "
            f"completion={response.completion_tokens}  "
            f"total={response.prompt_tokens + response.completion_tokens}\n"
        )

        if elapsed > 0:
            self._fh.write(f"│   elapsed: {elapsed:.2f}s\n")

        self._fh.write("└──────────────────────────────────────────\n\n")

    def log_tool_result(self, tool_name: str, result_str: str) -> None:
        """Log the result of a dispatched tool call."""
        self._fh.write(f"  ── TOOL RESULT: {tool_name} ──\n")
        try:
            parsed = json.loads(result_str)
            result_str = json.dumps(parsed, indent=4)
        except (json.JSONDecodeError, TypeError):
            pass
        for line in result_str.splitlines():
            self._fh.write(f"  {line}\n")
        self._fh.write("  ──────────────────────────────\n\n")

    def log_netlist(self, netlist: Any) -> None:
        """Log a summary of the design after a read_design call."""
        self._fh.write("  ── NETLIST LOADED ──\n")
        summary = format_netlist(netlist)
        for line in summary.splitlines():
            self._fh.write(f"  {line}\n")
        self._fh.write("  ────────────────────\n\n")

    def log_tools_offered(self, names: List[str]) -> None:
        """The tools the model can call in this request (as offered with every LLM call)."""
        self._fh.write(f"  ── TOOLS OFFERED ({len(names)}) ──\n  {', '.join(names)}\n\n")

    def log_trace(self, lines: List[str]) -> None:
        """What happened, one line per LLM call: the tool calls (and what they ran) or the answer."""
        self._fh.write("┌─ TRACE (one line per LLM call) ──────────\n")
        for line in lines:
            self._fh.write(f"│ {line}\n")
        self._fh.write("└──────────────────────────────────────────\n\n")

    def log_final_response(self, text: str) -> None:
        """Log the final text response returned to the user."""
        self._fh.write("┌─ FINAL RESPONSE ─────────────────────────\n")
        for line in text.splitlines():
            self._fh.write(f"│ {line}\n")
        self._fh.write("└──────────────────────────────────────────\n\n")

    def log_usage(self, usage: Any, costs: Any = None) -> None:
        """Log the request's model usage, summed over its LLM calls (llm.Usage), and
        (list, billed) cost in USD when prices are set."""
        self._fh.write("┌─ REQUEST USAGE (all LLM calls) ──────────\n")
        self._fh.write(f"│ llm_calls: {usage.llm_calls}\n")
        self._fh.write(f"│ input_tokens: {usage.input_tokens} (cached={usage.cached_input_tokens}"
                       f"  cache_write={usage.cache_write_tokens})\n")
        self._fh.write(f"│ output_tokens: {usage.output_tokens}\n")
        self._fh.write(f"│ chars: input={usage.input_chars}  output={usage.output_chars}\n")
        if costs:
            self._fh.write(f"│ cost_usd: list={costs[0]:.6f}  billed={costs[1]:.6f}\n")
        self._fh.write("└──────────────────────────────────────────\n\n")

    def close(self) -> None:
        """Close the log file."""
        if not self._fh.closed:
            elapsed = time.monotonic() - self._start_time
            self._fh.write("═" * 72 + "\n")
            self._fh.write(f"END OF REQUEST — total time: {elapsed:.2f}s\n")
            self._fh.write("═" * 72 + "\n")
            self._fh.close()

    def __enter__(self) -> "ConversationLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# No-op logger (production mode — zero overhead)
# ---------------------------------------------------------------------------

class NoopLogger:
    """Drop-in replacement for ConversationLogger that does nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_user_input(self, text: str) -> None:
        pass

    def log_llm_prompt(self, messages: List[Dict[str, Any]], turn: int) -> None:
        pass

    def log_llm_response(self, response: Any, turn: int, elapsed: float = 0.0) -> None:
        pass

    def log_tool_result(self, tool_name: str, result_str: str) -> None:
        pass

    def log_netlist(self, netlist: Any) -> None:
        pass

    def log_final_response(self, text: str) -> None:
        pass

    def log_tools_offered(self, names: List[str]) -> None:
        pass

    def log_trace(self, lines: List[str]) -> None:
        pass

    def log_usage(self, usage: Any, costs: Any = None) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "NoopLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Tools actually called, per request
# ---------------------------------------------------------------------------

class RunAnalysis:
    """run_analysis_<run id>.csv: for every request, the tools the agent actually called
    (columns as predicted_run_analysis.csv, plus the arguments), how long the request took
    and the model usage summed over its LLM calls (see llm.Usage).  The cost columns exist
    only when prices are set in the config.  One file per run -- run.sh's testcases share
    one -- next to this file.  Several calls in one request are joined with " ; " in call
    order; no call at all is "(history)".  A testcase run again in the same run replaces its
    earlier rows; rows stay grouped by testcase, in prompt order."""

    COLUMNS = ["testcase_number", "prompt", "tool_called", "tool_input_structure", "graph_ops",
               "llm_calls", "seconds", "input_tokens", "cached_input_tokens", "output_tokens",
               "input_chars", "output_chars"]
    COST_COLUMNS = ["cost_list_usd", "cost_billed_usd"]

    def __init__(self, run_id: str, costs: bool = False) -> None:
        """costs: write the cost columns (prices are set in the config)."""
        self.path = Path(__file__).resolve().parent / f"run_analysis_{run_id}.csv"
        self.columns = self.COLUMNS + (self.COST_COLUMNS if costs else [])
        self.started: set = set()  # testcases begun in this run

    def record(self, case: str, prompt: str, calls: List[tuple], usage: Any, costs: Any = None,
               ops: List[tuple] = (), seconds: float = 0.0) -> None:
        """calls: [(tool name, {argument: value})] of the engine tools, in the order they were made;
        usage: llm.Usage; costs: (list, billed) USD or None; ops: the other tool calls, as calls
        (e.g. tool knowledge graph navigation; none in access mode "all"); seconds: wall-clock
        time from the request to its answer (LLM calls and tools)."""
        rows = []
        if self.path.exists():  # by column name, so a file written with other columns still merges
            with open(self.path, newline="") as fh:
                rows = list(csv.DictReader(fh))
        if case not in self.started:  # a new run of this testcase: drop its old rows
            rows = [r for r in rows if r["testcase_number"] != case]
            self.started.add(case)
        args = lambda a: ", ".join(f"{k}={v!r}".replace("'", '"') for k, v in a.items())
        row = {"testcase_number": case, "prompt": prompt,
               "tool_called": " ; ".join(name for name, _ in calls) or "(history)",
               "tool_input_structure": " ; ".join(f"{name}({args(a)})" for name, a in calls),
               "graph_ops": " ; ".join(f"{name}({args(a)})" for name, a in ops),
               "llm_calls": usage.llm_calls, "seconds": f"{seconds:.2f}",
               "input_tokens": usage.input_tokens, "cached_input_tokens": usage.cached_input_tokens,
               "output_tokens": usage.output_tokens, "input_chars": usage.input_chars,
               "output_chars": usage.output_chars}
        if costs:
            row.update(cost_list_usd=f"{costs[0]:.6f}", cost_billed_usd=f"{costs[1]:.6f}")
        rows.append(row)
        rows.sort(key=lambda r: r["testcase_number"])  # stable: prompt order within a testcase
        with open(self.path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, self.columns, quoting=csv.QUOTE_ALL, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# The settings of a run
# ---------------------------------------------------------------------------

def write_run_info(run_id: str, info: Dict[str, Any]) -> Path:
    """debugging/run_info_<run id>.json: the settings a run used (info) and the source version
    (git commit; dirty = uncommitted changes), so a run's CSV and logs can be traced to them.
    Written by the run's first process; run.sh's testcases share one run id."""
    path = Path(__file__).resolve().parent / f"run_info_{run_id}.json"
    if path.exists():
        return path
    git = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True,
                                       cwd=Path(__file__).resolve().parent).stdout.strip()
    info = {"run": run_id, "started": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD") or None, "git_dirty": bool(git("status", "--porcelain")),
            **info}
    path.write_text(json.dumps(info, indent=2) + "\n")
    return path
