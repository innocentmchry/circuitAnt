"""Manager coordinating plan cache matching, execution, answer synthesis, and learning."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .constants import MUTATING_TOOLS
from .executor import PlanExecutor
from .generalize import PlanTemplate, trace_to_template
from .matcher import MatchResult, PlanMatcher
from .store import PlanStore
from .trace import ExecutionTrace, ToolCallRecord


class PlanCacheManager:
    """Coordinates plan cache matching, execution, answer synthesis, and learning."""

    def __init__(self, config: dict[str, Any], client: Any, tools: Any, model: str = "gpt-4o-mini") -> None:
        self.config = config
        self.client = client
        self.tools = tools
        self.model = model

        self.enabled = bool(config.get("enabled", False))
        db_path = config.get("path", "output/plan_cache.sqlite")
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.similarity_threshold = float(config.get("similarity_threshold", 0.85))
        self.max_failures_before_drop = int(config.get("max_failures_before_drop", 3))

        self.store = PlanStore(db_path)
        self.matcher = PlanMatcher(self.store, embedding_fn=self.get_embedding)

    def get_embedding(self, text: str) -> list[float]:
        """Fetch embedding for text using OpenAI API client."""
        resp = self.client.embeddings.create(input=text, model=self.embedding_model)
        return resp.data[0].embedding

    def try_execute(
        self,
        request: str,
        context_summary: str = "",
        log: Any = None,
        summarize_fn: Any = None,
    ) -> tuple[bool, str, list[str], list[tuple[str, dict[str, Any]]], ExecutionTrace | None]:
        """Try matching and executing a cached plan template.

        Returns (hit, answer, tool_summaries, calls, trace_or_none).
        """
        if not self.enabled or self.tools.design is None:
            return False, "", [], [], None

        # Exclude session initialization commands
        req_lower = request.lower()
        if "read_design" in req_lower or "set_testcase_name" in req_lower:
            return False, "", [], [], None

        t_start = time.monotonic()
        match_res = self.matcher.find_match(
            request, self.tools, similarity_threshold=self.similarity_threshold
        )
        if not match_res.matched or match_res.template is None or match_res.template_id is None:
            if log and hasattr(log, "_fh") and not log._fh.closed:
                log._fh.write(f"  ── PLAN CACHE MISS ──\n  Reason: {match_res.reason}\n  ─────────────────────\n\n")
            return False, "", [], [], None

        # Matched template
        tpl = match_res.template
        tid = match_res.template_id

        if log and hasattr(log, "_fh") and not log._fh.closed:
            log._fh.write(
                f"  ── PLAN CACHE HIT ──\n"
                f"  Template ID: {tid} (sim: {match_res.similarity:.3f})\n"
                f"  Intent: {tpl.masked_intent}\n"
                f"  Tools: {tpl.tool_sequence}\n"
                f"  ────────────────────\n\n"
            )

        # Execute template steps
        tool_start = time.monotonic()
        success, tool_records, err_msg = PlanExecutor.execute(tpl, match_res.filled_slots, self.tools)
        tool_time = time.monotonic() - tool_start

        if not success:
            # Treat execution failure as miss, drop if too many failures
            dropped = self.store.record_failure(tid, max_failures_before_drop=self.max_failures_before_drop)
            if log and hasattr(log, "_fh") and not log._fh.closed:
                drop_info = " (template dropped)" if dropped else ""
                log._fh.write(
                    f"  ── PLAN CACHE EXECUTION FAILED ──\n"
                    f"  Error: {err_msg}{drop_info}; falling back to LLM agent.\n"
                    f"  ─────────────────────────────────\n\n"
                )
            return False, "", [], [], None

        # Plan succeeded: record success
        self.store.record_success(tid)

        # Format tool summaries and calls for conversation history
        tool_summaries: list[str] = []
        calls: list[tuple[str, dict[str, Any]]] = []
        for rec in tool_records:
            calls.append((rec.name, rec.arguments))
            if log and hasattr(log, "log_tool_result"):
                log.log_tool_result(rec.name, json.dumps(rec.result))
            if summarize_fn:
                tool_summaries.append(summarize_fn(rec.name, json.dumps({"result": rec.result})))

        # Single small LLM call to word the natural language answer
        llm_start = time.monotonic()
        ans_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an expert EDA assistant. The tools for the user request have already been executed. "
                    "Answer concisely in natural language with what was found or changed, following the exact format:\n"
                    "- Use the signal, gate and file names exactly as given.\n"
                    "- State numbers and counts directly.\n"
                    "- Do not mention that tools were cached or pre-executed."
                ),
            }
        ]
        if context_summary:
            ans_prompt.append({"role": "system", "content": f"Current state: {context_summary}"})
        ans_prompt.append({"role": "user", "content": request})
        results_summary = json.dumps(
            [{"tool": r.name, "arguments": r.arguments, "result": r.result} for r in tool_records]
        )
        ans_prompt.append({"role": "system", "content": f"Executed tool results:\n{results_summary}"})

        resp = self.client.chat.completions.create(model=self.model, messages=ans_prompt)
        llm_time = time.monotonic() - llm_start
        answer = resp.choices[0].message.content or ""

        usage = getattr(resp, "usage", None)
        p_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        c_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        t_tok = getattr(usage, "total_tokens", 0) if usage else 0

        trace = ExecutionTrace(
            request=request,
            tool_calls=tool_records,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            wall_clock_time=time.monotonic() - t_start,
            llm_time=llm_time,
            tool_time=tool_time,
            success=True,
            answer=answer,
            context_summary=context_summary,
        )

        return True, answer, tool_summaries, calls, trace

    def record_and_learn(self, request: str, trace: ExecutionTrace) -> None:
        """Generalize and store a new template after a successful uncached run."""
        if not self.enabled or not trace.success or not trace.tool_calls:
            return

        # Do not cache session commands
        for tc in trace.tool_calls:
            if tc.name in ("read_design", "set_testcase_name"):
                return

        template, err = trace_to_template(trace)
        if template is None:
            return

        try:
            emb = self.get_embedding(template.masked_intent)
            self.store.save_or_update(template, emb)
        except Exception:
            # Embedding or storage failure during learning should not fail user request
            pass
