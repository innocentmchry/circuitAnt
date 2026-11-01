import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .entry import ToolCall
from .store import SemanticCacheStore
from .models.embeddings import EmbeddingModel
from .models.reliability import ReliabilityModel
from .models.reranker import CrossEncoderReranker
from .validation.constants import OPPOSITE_PAIRS
from .validation.structural import StructuralValidator, check_negative_constraints


class AgentCallWrapper:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


class AgentCacheManager:
    """Live execution caching layer for circuitAnt Agent."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        if not self.enabled:
            return

        self.store = SemanticCacheStore()
        self.structural_validator = StructuralValidator() if self.config.get("structural_validation_enabled", True) else None

        self.embedding_model = None
        if self.config.get("embedding_enabled", True):
            m_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            self.embedding_model = EmbeddingModel(model_name=m_name)

        self.reranker = None
        if self.config.get("reranker_enabled", True):
            m_name = self.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2")
            self.reranker = CrossEncoderReranker(model_name=m_name)

        self.reliability_model = None
        if self.config.get("reliability_enabled", True):
            conf_thresh = self.config.get("confidence_threshold", 0.65)
            self.reliability_model = ReliabilityModel(confidence_threshold=conf_thresh)

    def try_execute_cached_plan(
        self,
        agent: Any,
        request: str,
        context_summary: str,
        tool_summaries: List[str],
        calls: List[Tuple[str, dict]],
        log: Any,
    ) -> Tuple[bool, Optional[str]]:
        """Attempt to retrieve, validate, and execute a cached plan.
        
        Returns: (is_cache_hit, final_answer)
        """
        if not self.enabled:
            return False, None

        all_entries = self.store.get_all_entries()
        if not all_entries:
            return False, None

        testcase = agent.tools.case or agent.case or ""

        # Exact match check first
        match = self.store.get_exact_match(request)
        if match:
            candidate_calls = match.tool_calls
            if match.testcase == testcase:
                return self._execute_calls(agent, candidate_calls, tool_summaries, calls, log, request, context_summary)

        # Semantic retrieval
        if not self.embedding_model:
            return False, None

        query_emb, _ = self.embedding_model.encode(request)
        candidates_with_emb = [(e.entry_id, e.embedding) for e in all_entries if e.embedding]
        top_k = self.config.get("retrieval_top_k", 5)
        cos_thresh = self.config.get("cosine_threshold", 0.82)

        retrieved = self.embedding_model.retrieve_top_k(
            query_emb, candidates_with_emb, top_k=top_k, threshold=cos_thresh
        )
        if not retrieved:
            return False, None

        # Select Top-1 Candidate
        best_cand_id, best_sim, _ = retrieved[0]
        best_entry = self.store.entries[best_cand_id]

        # 1. Polarity and negative constraint check
        pass_neg, neg_reason = check_negative_constraints(request, best_entry.original_request)
        if not pass_neg:
            return False, None

        if not best_entry.tool_calls:
            return False, None

        target_tool = best_entry.tool_calls[0].tool
        tool_func = getattr(agent.tools, target_tool, None) if hasattr(agent, "tools") else None

        # 2. Light LLM argument extraction (~100 tokens, no regex)
        extracted_args = self._extract_args_with_llm(agent, target_tool, tool_func, request)
        if extracted_args is None:
            return False, None

        # 3. Generalized schema and signature validation
        if self.structural_validator:
            valid, err = self.structural_validator.validate_schema(target_tool, extracted_args, tool_func=tool_func)
            if not valid:
                return False, None

        # 4. Construct bound call and execute safely
        bound_calls = [ToolCall(tool=target_tool, args=extracted_args)]
        return self._execute_calls(agent, bound_calls, tool_summaries, calls, log, request, context_summary)

    def _extract_args_with_llm(
        self, agent: Any, tool_name: str, tool_func: Optional[Any], request: str
    ) -> Optional[Dict[str, Any]]:
        """Lightweight LLM call (~100 tokens) to extract arguments for a single target tool."""
        import inspect
        sig_str = ""
        doc_str = ""
        if tool_func is not None:
            try:
                sig_str = str(inspect.signature(tool_func))
                doc_str = inspect.getdoc(tool_func) or ""
            except Exception:
                pass

        prompt = (
            f"You are a strict argument extractor for an EDA tool.\n"
            f"Tool: {tool_name}{sig_str}\n"
            + (f"Description: {doc_str}\n" if doc_str else "")
            + f"User Request: \"{request}\"\n\n"
            f"Instructions:\n"
            f"1. Extract the arguments required for '{tool_name}' from the user request as a JSON object.\n"
            f"2. If the user request does not match this tool, or if required arguments are missing, return {{\"reject\": true}}.\n"
            f"3. Return ONLY valid JSON (no commentary, no markdown ticks)."
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            if hasattr(agent, "client") and hasattr(agent.client, "complete"):
                reply = agent.client.complete(messages, [])
                content = reply.content or "{}"
            elif hasattr(agent, "client") and hasattr(agent.client, "chat"):
                resp = agent.client.chat.completions.create(
                    model=getattr(agent, "model", "gpt-4o-mini"),
                    messages=messages,
                    temperature=0.0,
                    max_tokens=150,
                )
                content = resp.choices[0].message.content or "{}"
            else:
                return None

            content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(content)
            if not isinstance(data, dict) or data.get("reject"):
                return None
            return data
        except Exception:
            return None

    def _execute_calls(
        self,
        agent: Any,
        tool_calls: List[ToolCall],
        tool_summaries: List[str],
        calls: List[Tuple[str, dict]],
        log: Any,
        request: str,
        context_summary: str,
    ) -> Tuple[bool, Optional[str]]:
        try:
            from agent.agent import _summarize
        except ImportError:
            from agent import _summarize
        executed_results = []

        for tc in tool_calls:
            arg_str = json.dumps(tc.args)
            if hasattr(agent, "access") and hasattr(agent.access, "execute"):
                outcome = agent.access.execute(tc.tool, arg_str)
                result_str = outcome.result
            elif hasattr(agent, "_run"):
                wrapper = AgentCallWrapper(name=tc.tool, arguments=tc.args)
                result_str = agent._run(wrapper)
            else:
                fn = getattr(agent.tools, tc.tool, None)
                if fn:
                    try:
                        res = fn(**tc.args)
                        result_str = json.dumps({"result": res})
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                else:
                    return False, None

            log.log_tool_result(tc.tool, result_str)

            # Check if tool execution failed
            if '"error"' in result_str:
                # Cache execution error -> fallback to LLM
                return False, None

            calls.append((tc.tool, tc.args))
            tool_summaries.append(_summarize(tc.tool, result_str))
            executed_results.append((tc.tool, result_str))

        # Synthesis of response: call agent's model with tool results to synthesize concise answer
        messages = [
            {"role": "system", "content": "You are an expert EDA assistant. Answer concisely based on tool results."},
            {"role": "user", "content": request},
        ]
        for name, res in executed_results:
            messages.append({"role": "tool", "tool_call_id": f"cached_{name}", "content": res})

        try:
            if hasattr(agent.client, "complete"):
                reply = agent.client.complete(messages, [])
                answer = reply.content or ""
            else:
                resp = agent.client.chat.completions.create(model=agent.model, messages=messages)
                answer = resp.choices[0].message.content or ""
        except Exception:
            answer = "; ".join(tool_summaries)

        return True, answer

    def update_on_success(self, request: str, testcase: str, calls: List[Tuple[str, dict]], context: str) -> None:
        """Insert verified successful LLM plan into the live cache."""
        if not self.enabled or not calls:
            return

        tool_calls = [ToolCall(tool=name, args=args) for name, args in calls]
        emb = None
        if self.embedding_model:
            emb, _ = self.embedding_model.encode(request)

        req_id = f"live_{int(time.time()*1000)}"
        self.store.insert(
            request_id=req_id,
            testcase=testcase,
            original_request=request,
            tool_calls=tool_calls,
            netlist_id=testcase,
            context_summary=context,
            embedding=emb,
        )
