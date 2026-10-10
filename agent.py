"""LLM agent: natural-language requests -> engine tool calls -> answer.

The tools the model can call, and their generated schemas, are in tools.py.
"""
import inspect
import json
import time
from pathlib import Path

from openai import OpenAI

from debugging.developer import ConversationLogger, NoopLogger, RunAnalysis, extract_case_name, run_logs_dir
from tools import Tools, schema

MAX_TOOL_ROUNDS = 20
MAX_RESULT_CHARS = 20000
CONTEXT_HISTORY_LIMIT = 20  # earlier requests summarised into each new request

# Context handling (history summary between requests) follows cada1067 (src/agent.py).
SYSTEM_PROMPT = (
    "You are an expert EDA assistant. You analyse and transform a gate-level Verilog netlist "
    "through the tools provided. Requests arrive one at a time; for each, call the tools it needs, "
    "in order, then answer concisely in natural language with what was found or what was changed.\n"
    "- Use tools for every number or structural fact about the design; never estimate.\n"
    "- Answer from the context history only when it states the exact value asked (for example how many "
    "gates the previous transformation removed or added); otherwise, and for any question about the "
    "design as it is now, call a tool.\n"
    "- Use the signal, gate and file names exactly as the request gives them.\n"
    "- When a tool result includes file_path, report that exact path along with the count.\n"
    "- Transformation tools preserve functionality by construction; call check_equivalence when the "
    "request asks to verify, check or prove equivalence with the original design.\n"
    "- Do not discuss scoring or evaluation."
)


class Agent:
    """Answers one request at a time, as in cada1067.

    Each request starts a fresh conversation: [system prompt, "Current state:
    <summary of earlier requests>", request].  After it, the request and its tool
    results are condensed into one history line for the next summary.
    """

    def __init__(self, api_key: str, model: str, developer: dict | None = None,
                 design: dict | None = None, output_dir: str | Path = ".") -> None:
        """developer: the config's `developer` section ({"mode": bool, "log_dir": path});
        design: its `design` section (how netlists are read, see design_io.read_design);
        output_dir: where testcase logs, written designs and result lists go."""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tools = Tools(design, output_dir)
        self.functions = {name: fn for name, fn in inspect.getmembers(self.tools, inspect.ismethod)
                          if not name.startswith("_")}
        self.schemas = [schema(fn) for fn in self.functions.values()]
        self.requests = 0
        self.case = ""
        self.history = []  # one line per earlier request that called tools

        # Developer mode: one human-readable log file per request.
        developer = developer or {}
        self.developer_mode = bool(developer.get("mode"))
        self.run_analysis = None  # in developer mode: the tools each request actually called
        if self.developer_mode:
            self.logs_dir = run_logs_dir(developer.get("log_dir"))
            self.run_analysis = RunAnalysis(self.logs_dir.name)  # same run id as the log folder
            self.Logger = ConversationLogger
        else:
            self.logs_dir = Path(".")  # unused
            self.Logger = NoopLogger

    # -- context between requests ---------------------------------------------

    def build_state_summary(self) -> str:
        """Testcase, loaded netlist and the history of earlier requests."""
        parts = []
        if self.case:
            parts.append(f"testcase='{self.case}'")
        if self.tools.design:
            parts.append(f"original_netlist='{Path(self.tools.design.path).resolve()}'")
        history = self.history[-CONTEXT_HISTORY_LIMIT:]
        if history:
            parts.append("operations so far in order:")
            if len(self.history) > len(history):
                parts.append(f"  - ... {len(self.history) - len(history)} earlier operations omitted")
            parts.extend(f"  - {item}" for item in history)
        return "\n".join(parts) if parts else "(no state)"

    def _remember(self, request: str, tool_summaries: list[str]) -> None:
        """Condense one request's tool calls into one history line."""
        if not tool_summaries:
            return
        prompt = " ".join(request.split())
        if len(prompt) > 120:
            prompt = prompt[:117] + "..."
        counts = {}
        for summary in tool_summaries:
            counts[summary] = counts.get(summary, 0) + 1
        details = [f"{s} ({n}x)" if n > 1 else s for s, n in counts.items()]
        self.history.append(f"{prompt} => " + "; ".join(details))
        del self.history[:-CONTEXT_HISTORY_LIMIT]

    # -- one request -----------------------------------------------------------

    def ask(self, request: str, context_summary: str = "") -> str:
        self.requests += 1
        self.case = extract_case_name(request) or self.tools.case or self.case
        with self.Logger(self.requests, self.logs_dir, self.model, label=self.case) as log:
            log.log_user_input(request)
            if context_summary and self.developer_mode:
                log._fh.write(f"  ── CONTEXT ──\n  {context_summary}\n  ─────────────\n\n")
            tool_summaries, calls = [], []
            answer = self._answer(request, context_summary, tool_summaries, calls, log)
            self._remember(request, tool_summaries)
            if self.run_analysis:  # the testcase name is known once set_testcase_name has run
                self.run_analysis.record(self.tools.case or self.case, request, calls)
            log.log_final_response(answer)
            return answer

    def _answer(self, request: str, context_summary: str, tool_summaries: list[str], calls: list, log) -> str:
        """The answer to one request; fills tool_summaries (history lines) and calls ((name, arguments))."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context_summary:
            messages.append({"role": "system", "content": f"Current state: {context_summary}"})
        messages.append({"role": "user", "content": request})
        for turn in range(1, MAX_TOOL_ROUNDS + 1):
            log.log_llm_prompt(messages, turn)
            start = time.monotonic()
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=self.schemas)
            log.log_llm_response(response, turn, time.monotonic() - start)
            reply = response.choices[0].message
            messages.append(reply.model_dump(exclude_none=True))
            if not reply.tool_calls:
                return reply.content or ""
            for call in reply.tool_calls:
                try:
                    calls.append((call.function.name, json.loads(call.function.arguments or "{}")))
                except json.JSONDecodeError:
                    calls.append((call.function.name, {"arguments": call.function.arguments}))
                result = self._run(call.function)
                log.log_tool_result(call.function.name, result)
                tool_summaries.append(_summarize(call.function.name, result))
                if call.function.name == "read_design" and '"error"' not in result:
                    log.log_netlist(self.tools.design)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        return "[ERROR] too many tool calls without an answer"

    def _run(self, call) -> str:
        """JSON {"result": ...} or {"error": ...}, as in cada1067."""
        try:
            value = self.functions[call.name](**json.loads(call.arguments or "{}"))
            if isinstance(value, dict) and value.get("file_path"):  # a full list written to a file:
                path = Path(value["file_path"])  # name it after the testcase and request, so it is unique
                unique = path.with_name(f"{self.tools.case or 'design'}_r{self.requests:02d}_{path.name}")
                path.replace(unique)
                value["file_path"] = str(unique)
            result = {"result": value}
        except Exception as exc:  # report tool errors to the model, not the user
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result)[:MAX_RESULT_CHARS]


def _summarize(tool_name: str, result_str: str) -> str:
    """One-line history entry for a tool result (cada1067's wording)."""
    try:
        result = json.loads(result_str)
    except json.JSONDecodeError:
        return f"{tool_name} completed"
    if "error" in result:
        return f"{tool_name} FAILED: {result['error'][:2000]}"
    data = result.get("result", result)
    if tool_name == "read_design":
        return "loaded design via read_design"
    if tool_name == "write_design":
        return "wrote design via write_design"
    if tool_name == "set_testcase_name":
        return "testcase initialized via set_testcase_name"
    if tool_name == "count_gates":
        parts = ", ".join(f"{n}x {t}" for t, n in sorted(data["breakdown"].items()) if n > 0)
        return f"count_gates: {data['total_gates']} total gates ({parts})"
    if tool_name == "get_max_depth":
        summary = f"get_max_depth: depth={data.get('depth')}"
        if data.get("path"):
            summary += f" ({data['path'][0]} -> {data['path'][-1]})"
        return summary
    if tool_name == "get_endpoint_stats":
        summary = (f"get_endpoint_stats: {data.get('sink')} max {data.get('metric')}="
                   f"{data.get('max_value')} at {', '.join(data.get('at_max', [])[:3])}")
        if data.get("operator"):
            summary += (f"; {data['count']} with {data['metric']} "
                        f"{data['operator']} {data['threshold']}")
        return summary
    if tool_name == "get_cone":
        summary = f"get_cone: {data.get('direction')} cone of {data.get('signal')}"
        if data.get("shared_with"):
            summary += f" shared with {data['shared_with']}"
        summary += f" has {data.get('count')} gates {data.get('by_type')}"
        if data.get("through_dff"):
            summary += f" (via {data['through_dff']}.D)"
        return summary
    if tool_name == "get_fanout":
        if "max_fanout" in data:
            return (f"get_fanout: {data['signal']} max fanout={data['max_fanout']} at "
                    f"{', '.join(data['at_max'][:3])}")
        return f"get_fanout: {data.get('signal')} drives {data.get('fanout')} pins on {data.get('gate_count')} gates"
    if tool_name == "find_gates":
        filters = ", ".join(f"{k}={data[k]}" for k in ("gate_type", "connected_to", "pin", "name") if k in data)
        return f"find_gates({filters}): {data.get('count')} gates"
    if tool_name == "get_ports":
        return "get_ports: " + "; ".join(
            f"{k}: {v['ports']} ports / {v['bits']} bits ("
            + ", ".join(p["name"] + (p["range"] or "") for p in v["list"][:20])
            + (", ..." if len(v["list"]) > 20 else "") + ")"
            for k, v in data.items() if isinstance(v, dict))
    if tool_name == "find_cut_points":
        if "is_cut" in data:
            return f"find_cut_points: {data['signal']} is_cut={data['is_cut']} ({data['source']} -> {data['sink']})"
        return f"find_cut_points: {data.get('count')} gates on every path {data.get('source')} -> {data.get('sink')}"
    if tool_name == "find_inverter_pairs":
        return f"find_inverter_pairs: {data.get('count')} pairs"
    if tool_name == "find_duplicates":
        return (f"find_duplicates: {data.get('count')} duplicate groups, {data.get('removable_gates')} gates "
                f"removable at this point (merging can expose more)")
    if tool_name == "find_enable_hold":
        return f"find_enable_hold: {data.get('count')} flip-flops {data.get('classification_breakdown')}"
    if tool_name == "derive_boolean_equation":
        eq = data.get("equation", "")
        return f"derive_boolean_equation: {eq[:150]}{'...' if len(eq) > 150 else ''}"
    if tool_name == "check_equivalence":
        return f"check_equivalence: {data.get('status')} ({data.get('method')})"
    if tool_name == "check_signal_equivalence":
        return f"check_signal_equivalence: {data.get('a')} vs {data.get('b')} equivalent={data.get('equivalent')}"
    if tool_name == "check_symmetry":
        return (f"check_symmetry: {data.get('signal')} in {data.get('input_a')}, {data.get('input_b')} "
                f"symmetric={data.get('symmetric')}")
    if tool_name == "find_all_paths":
        summary = f"find_all_paths: {data.get('count')} paths from {data.get('source')} to {data.get('sink')}"
        if data.get("avoid"):
            summary += f" avoiding {data['avoid']}"
        if data.get("max_depth") is not None:
            summary += f" with depth <= {data['max_depth']}"
        if data.get("file_path"):
            summary += f"; full list in {data['file_path']}"
        return summary
    if tool_name in ("rename_gate", "rename_wire"):
        return f"{tool_name}: '{data.get('old_name')}' → '{data.get('new_name')}'"
    if tool_name == "remove_dangling_gates":
        return f"remove_dangling_gates: {data.get('gates_removed', 0)} gates removed"
    if tool_name == "collapse_inverter_pairs":
        return (f"collapse_inverter_pairs: {data.get('pairs_collapsed', 0)} pairs collapsed, "
                f"{data.get('gates_removed', 0)} gates removed")
    if tool_name == "merge_duplicate_gates":
        return (f"merge_duplicate_gates: {data.get('gates_merged', 0)} gates removed (merged into identical "
                f"gates), total gates {data.get('gates_before')} -> {data.get('gates_after')}")
    if tool_name == "simplify_gates_with_constant_inputs":
        return (f"simplify_gates_with_constant_inputs: simplified {data.get('simplified_gates', 0)} "
                f"{data.get('gate_type', '?')} gate(s); replacements={data.get('replacement_breakdown', {})}")
    if tool_name == "insert_buffers_for_fanout":
        return f"insert_buffers_for_fanout: {data.get('buffers_inserted', 0)} buffers"
    if tool_name == "insert_dedicated_buffers_for_loads":
        return (f"insert_dedicated_buffers_for_loads: "
                f"{data.get('buffers_inserted', 0)} buffers on {data.get('net_name')}")
    if tool_name == "reduce_depth":
        return (f"reduce_depth: depth {data.get('depth_before')} -> {data.get('depth_after')}, "
                f"gates {data.get('gates_before')} -> {data.get('gates_after')}")
    if tool_name == "remap_gates":
        scope = f" in cone of {data['signal']}" if data.get("signal") else ""
        return (f"remap_gates({','.join(data.get('allowed', []))}){scope}: {data.get('gates_replaced')} gates "
                f"remapped, total {data.get('gates_before')} -> {data.get('gates_after')}")
    if tool_name == "replace_gate_type":
        scope = f" in cone of {data['signal']}" if data.get("signal") else ""
        return (f"replace_gate_type: {data.get('replaced')} {data.get('gate_type')} gates{scope} replaced by "
                f"{','.join(data.get('allowed', []))}; removed {data.get('removed')}, added {data.get('added')}; "
                f"total gates {data.get('gates_before')} -> {data.get('gates_after')}")
    return f"{tool_name} completed"
