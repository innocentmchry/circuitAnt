from pathlib import Path
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.cache import CacheDecision, CandidateMatch, ToolCall, SemanticCacheStore
from agent.cache.models.embeddings import EmbeddingModel
from agent.cache.models.reliability import ReliabilityModel
from agent.cache.models.reranker import CrossEncoderReranker
from agent.cache.validation.constants import OPPOSITE_PAIRS
from agent.cache.validation.structural import StructuralValidator, check_negative_constraints
from ..data.loader import load_dataset
from .logger import ExperimentLogger
from .metrics import compute_metrics, plans_agree


def _load_tools_ast_metadata() -> Dict[str, Dict[str, Any]]:
    """Dynamically parses agent/tools.py using AST to extract signatures, types, and docstrings."""
    import ast
    for path_cand in ("agent/tools.py", "../../agent/tools.py", "c:/Users/lavan/Desktop/Sem7/BTP/ICCAD2026/circuitAnt/agent/tools.py"):
        pth = Path(path_cand)
        if pth.exists():
            try:
                tree = ast.parse(pth.read_text(encoding="utf-8"))
                tools_cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tools"][0]
                meta = {}
                for m in tools_cls.body:
                    if isinstance(m, ast.FunctionDef) and not m.name.startswith("_"):
                        params = []
                        for a in m.args.args:
                            if a.arg == "self":
                                continue
                            ann = ast.unparse(a.annotation) if a.annotation else ""
                            params.append(f"{a.arg}: {ann}" if ann else a.arg)
                        doc = ast.get_docstring(m) or ""
                        meta[m.name] = {
                            "params": params,
                            "doc": doc,
                        }
                return meta
            except Exception:
                pass
    return {}


class ExperimentRunner:
    """Independent runner for deterministic offline replay and ablation evaluation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_id = config.get("run_id") or f"{config.get('experiment_id', 'exp')}_{int(time.time())}"
        self.output_dir = config.get("results_dir", "experiments/semantic_cache/results")

        # Seed
        seed = config.get("random_seed", 42)
        random.seed(seed)

        # Logger
        self.logger = ExperimentLogger(self.run_id, self.output_dir, self.config)

        # Cache store
        capacity = config.get("cache_capacity")
        self.store = SemanticCacheStore(capacity=capacity, event_callback=self.logger.log_cache_event)

        # Models
        self.embedding_model = None
        if config.get("embedding_enabled", False):
            m_name = config.get("embedding_model", "all-MiniLM-L6-v2")
            self.embedding_model = EmbeddingModel(model_name=m_name)

        self.reranker = None
        if config.get("reranker_enabled", False):
            m_name = config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2")
            self.reranker = CrossEncoderReranker(model_name=m_name)

        self.structural_validator = None
        if config.get("structural_validation_enabled", False):
            self.structural_validator = StructuralValidator()

        self.reliability_model = None
        if config.get("reliability_enabled", False):
            conf_thresh = config.get("confidence_threshold", 0.65)
            self.reliability_model = ReliabilityModel(confidence_threshold=conf_thresh)

        # Light LLM Argument Extractor
        self.tools_metadata = _load_tools_ast_metadata()
        self.light_llm_enabled = bool(config.get("light_llm_enabled", False))
        self.openai_client = None
        if self.light_llm_enabled:
            api_key = self._load_api_key()
            if api_key:
                try:
                    from openai import OpenAI
                    self.openai_client = OpenAI(api_key=api_key)
                except Exception as e:
                    print(f"[WARNING] Could not initialize OpenAI client for Light LLM: {e}")


    def _load_api_key(self) -> Optional[str]:
        for env_path in ("../../.env", ".env", "c:/Users/lavan/Desktop/Sem7/BTP/ICCAD2026/circuitAnt/.env"):
            p = Path(env_path)
            if p.exists():
                raw = p.read_text(encoding="utf-8", errors="ignore").strip()
                if "=" in raw:
                    for line in raw.splitlines():
                        if "OPENAI_API_KEY" in line:
                            return line.partition("=")[2].strip("'\" \r\n")
                elif raw.startswith("sk-"):
                    return raw
        return os.environ.get("OPENAI_API_KEY")

    def _call_light_llm_extract_plan(self, candidate_tool_calls: List[ToolCall], request_prompt: str) -> Tuple[Optional[List[ToolCall]], int]:
        """Live OpenAI API call to extract arguments for the entire candidate plan with docstrings."""
        if not self.openai_client or not candidate_tool_calls:
            return None, 0

        # Build tools description with docstrings and type annotations
        tool_descs = []
        for idx, tc in enumerate(candidate_tool_calls, 1):
            t_name = tc.tool
            meta = self.tools_metadata.get(t_name, {})
            param_list = meta.get("params", [])
            param_str = f"({', '.join(param_list)})" if param_list else "()"
            doc_str = meta.get("doc", "")
            first_line = doc_str.strip().splitlines()[0] if doc_str else ""
            tool_descs.append(f"{idx}. {t_name}{param_str}\n   Docstring: {first_line}")

        tools_block = "\n".join(tool_descs)
        is_multi = len(candidate_tool_calls) > 1

        sys_prompt = (
            f"You are a strict, expert argument extractor for an EDA execution plan.\n"
            f"Plan Tools to Call:\n{tools_block}\n\n"
            f"User Request: \"{request_prompt}\"\n\n"
            f"Instructions:\n"
            f"1. Extract arguments for the tool(s) from the user request.\n"
            f"2. Pay close attention to parameter types, names, and docstrings.\n"
            f"   (e.g., outputs are 'sink', inputs/flip-flops are 'source', 'direction' is 'fanin' or 'fanout').\n"
            f"3. If the request contradicts the tools or required info is missing, return {{\"reject\": true}}.\n"
            + (f"4. Return strictly a JSON list of objects: [{{\"tool\": \"...\", \"args\": {{...}}}}, ...]" if is_multi else
               f"4. Return strictly a JSON object of arguments: {{\"param\": value, ...}}")
        )

        try:
            import json
            model = self.config.get("light_llm_model", "gpt-4o-mini")
            resp = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": sys_prompt}],
                temperature=0.0,
                max_tokens=180,
            )
            tokens_used = resp.usage.total_tokens if resp.usage else 70
            raw_content = resp.choices[0].message.content or "{}"
            raw_content = raw_content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw_content)

            if isinstance(data, dict) and data.get("reject"):
                return None, tokens_used

            bound_plan: List[ToolCall] = []
            if is_multi:
                if not isinstance(data, list):
                    return None, tokens_used
                for item in data:
                    t = item.get("tool")
                    a = item.get("args", {})
                    bound_plan.append(ToolCall(tool=t, args=a))
            else:
                target_tool = candidate_tool_calls[0].tool
                args = data.get("args", data) if isinstance(data, dict) else {}
                bound_plan.append(ToolCall(tool=target_tool, args=args))

            return bound_plan, tokens_used
        except Exception:
            return None, 0

    def _is_safe_exact_reuse(self, tool_calls: List[ToolCall], target_testcase: str) -> bool:
        """Check whether tool calls have no stale netlist-dependent arguments."""
        import re
        from ..validation.constants import CLASS_KEYWORDS
        for tc in tool_calls:
            for k, v in tc.args.items():
                if isinstance(v, str):
                    if "testcase/test" in v and target_testcase not in v:
                        return False
                    if re.search(r"\b[ng]\d+", v) and v not in CLASS_KEYWORDS:
                        return False
        return True

    def _evaluate_request(self, trace: Dict[str, Any]) -> Tuple[CacheDecision, Dict[str, float]]:
        """Evaluate a single request against the current cache state."""
        prompt = trace["prompt"]
        testcase = trace["testcase"]
        request_id = trace["request_id"]

        latencies = {
            "embedding": 0.0,
            "retrieval": 0.0,
            "rerank": 0.0,
            "validation": 0.0,
            "reliability": 0.0,
        }

        strategy = self.config.get("strategy", "original_llm")

        # Strategy E0: Original LLM (No Caching)
        if strategy == "original_llm" or self.config.get("experiment_id") == "E0":
            return CacheDecision(
                accepted=False,
                decision_type="MISS",
                reason="baseline_e0_caching_disabled",
            ), latencies

        # Strategy E1: Exact-Match Caching
        if strategy == "exact" or (not self.config.get("embedding_enabled", False) and self.config.get("exact_enabled", False)):
            t0 = time.perf_counter()
            match = self.store.get_exact_match(prompt)
            lat = time.perf_counter() - t0
            latencies["retrieval"] = lat

            if match:
                # Conservative exact reuse check: ensure testcase context matches or tool arguments are netlist-independent
                is_safe = (match.testcase == testcase) or self._is_safe_exact_reuse(match.tool_calls, testcase)
                if is_safe:
                    cand = CandidateMatch(
                        entry=match,
                        cosine_similarity=1.0,
                        rank=1,
                        validation_passed=True,
                        bound_tool_calls=match.tool_calls,
                    )
                    return CacheDecision(
                        accepted=True,
                        decision_type="HIT",
                        reason="exact_match_found",
                        selected_candidate=cand,
                        tool_calls=match.tool_calls,
                        candidate_scores=[cand.to_dict()],
                    ), latencies
                else:
                    return CacheDecision(
                        accepted=False,
                        decision_type="FALLBACK",
                        reason=f"exact_match_context_mismatch_{match.testcase}_vs_{testcase}",
                    ), latencies
            return CacheDecision(
                accepted=False,
                decision_type="MISS",
                reason="exact_match_not_found",
            ), latencies

        # Embedding Retrieval (E2 - E6)
        query_emb = None
        if self.embedding_model:
            query_emb, lat_emb = self.embedding_model.encode(prompt)
            latencies["embedding"] = lat_emb

        all_entries = self.store.get_all_entries()
        if not all_entries:
            return CacheDecision(
                accepted=False,
                decision_type="MISS",
                reason="cache_empty",
            ), latencies

        # 1. Cosine similarity candidate retrieval
        t_ret = time.perf_counter()
        candidates_with_emb = [(e.entry_id, e.embedding) for e in all_entries if e.embedding]
        top_k = self.config.get("retrieval_top_k", 5)
        cos_thresh = self.config.get("cosine_threshold", 0.85)

        retrieved = self.embedding_model.retrieve_top_k(
            query_emb, candidates_with_emb, top_k=top_k, threshold=0.0
        )
        latencies["retrieval"] = time.perf_counter() - t_ret

        if not retrieved:
            return CacheDecision(
                accepted=False,
                decision_type="MISS",
                reason="no_candidates_retrieved",
            ), latencies

        # Build candidate matches
        candidate_matches: List[CandidateMatch] = []
        for cand_id, sim, rank in retrieved:
            entry = self.store.entries[cand_id]
            candidate_matches.append(CandidateMatch(
                entry=entry,
                cosine_similarity=sim,
                rank=rank,
            ))

        # 2. Cross-Encoder Reranking if enabled (E3, E5, E6)
        if self.reranker and self.config.get("reranker_enabled", False):
            cand_texts = [cm.entry.original_request for cm in candidate_matches]
            scores, lat_rerank = self.reranker.score_pairs(prompt, cand_texts)
            latencies["rerank"] = lat_rerank
            for cm, score in zip(candidate_matches, scores):
                cm.cross_encoder_score = score
            # Rerank candidates by cross-encoder score
            candidate_matches.sort(key=lambda cm: cm.cross_encoder_score if cm.cross_encoder_score is not None else -999.0, reverse=True)
            for idx, cm in enumerate(candidate_matches, 1):
                cm.rank = idx

        # Filter by threshold: E2/E3 without structural validation requires cosine >= threshold
        best_candidate = candidate_matches[0]
        candidate_scores_log = [cm.to_dict() for cm in candidate_matches]

        # 3. Generalized Light LLM Strategy (E7)
        if self.light_llm_enabled:
            # Check similarity threshold on Top-1
            if best_candidate.cosine_similarity < cos_thresh:
                return CacheDecision(
                    accepted=False,
                    decision_type="MISS",
                    reason=f"cosine_similarity_{best_candidate.cosine_similarity:.3f}_below_threshold_{cos_thresh}",
                    candidate_scores=candidate_scores_log,
                ), latencies

            # Polarity / negative constraint check
            pass_neg, neg_reason = check_negative_constraints(prompt, best_candidate.entry.original_request)
            if not pass_neg:
                return CacheDecision(
                    accepted=False,
                    decision_type="FALLBACK",
                    reason=f"negative_constraint_rejection: {neg_reason}",
                    candidate_scores=candidate_scores_log,
                ), latencies

            if not best_candidate.entry.tool_calls:
                return CacheDecision(
                    accepted=False,
                    decision_type="FALLBACK",
                    reason="candidate_has_no_tool_calls",
                    candidate_scores=candidate_scores_log,
                ), latencies

            t_llm = time.perf_counter()
            bound_plan, tokens_spent = self._call_light_llm_extract_plan(best_candidate.entry.tool_calls, prompt)
            latencies["validation"] = time.perf_counter() - t_llm
            latencies["light_llm_tokens"] = tokens_spent

            if bound_plan is None:
                return CacheDecision(
                    accepted=False,
                    decision_type="FALLBACK",
                    reason="light_llm_rejected_or_unextractable",
                    candidate_scores=candidate_scores_log,
                ), latencies

            # Schema validation for all calls in plan
            if self.structural_validator:
                all_valid = True
                for tc in bound_plan:
                    valid, err = self.structural_validator.validate_schema(tc.tool, tc.args)
                    if not valid:
                        all_valid = False
                        break
                if not all_valid:
                    return CacheDecision(
                        accepted=False,
                        decision_type="FALLBACK",
                        reason="schema_validation_failed",
                        candidate_scores=candidate_scores_log,
                    ), latencies

            best_candidate.validation_passed = True
            best_candidate.bound_tool_calls = bound_plan
            best_candidate.extracted_params = bound_plan[0].args if bound_plan else {}

            return CacheDecision(
                accepted=True,
                decision_type="HIT",
                reason="generalized_light_llm_hit",
                selected_candidate=best_candidate,
                tool_calls=best_candidate.bound_tool_calls,
                candidate_scores=candidate_scores_log,
            ), latencies

        # 3. Structural Validation & Parameter Substitution if enabled (E4, E5, E6)
        if self.structural_validator and self.config.get("structural_validation_enabled", False):
            t_val = time.perf_counter()
            validated_candidates = []
            for cm in candidate_matches:
                success, bound_calls, reason, extracted_params = self.structural_validator.instantiate_template(
                    cm.entry.tool_calls,
                    cm.entry.original_request,
                    prompt,
                    testcase_name=testcase,
                )
                cm.validation_passed = success
                cm.rejection_reason = reason
                cm.bound_tool_calls = bound_calls
                cm.extracted_params = extracted_params
                if success:
                    validated_candidates.append(cm)
            latencies["validation"] = time.perf_counter() - t_val

            if not validated_candidates:
                return CacheDecision(
                    accepted=False,
                    decision_type="FALLBACK",
                    reason=f"structural_validation_failed: {best_candidate.rejection_reason or 'no_valid_template'}",
                    candidate_scores=candidate_scores_log,
                ), latencies
            best_candidate = validated_candidates[0]
        else:
            # Strategies without structural validation (E2, E3):
            # Check negative constraints (opposite words)
            pass_neg, neg_reason = check_negative_constraints(prompt, best_candidate.entry.original_request)
            if not pass_neg:
                return CacheDecision(
                    accepted=False,
                    decision_type="FALLBACK",
                    reason=f"negative_constraint_rejection: {neg_reason}",
                    candidate_scores=candidate_scores_log,
                ), latencies

            # Safe exact argument reuse only when arguments are netlist-compatible
            best_candidate.bound_tool_calls = best_candidate.entry.tool_calls
            best_candidate.validation_passed = True

        # Check similarity threshold
        if best_candidate.cosine_similarity < cos_thresh:
            return CacheDecision(
                accepted=False,
                decision_type="MISS",
                reason=f"cosine_similarity_{best_candidate.cosine_similarity:.3f}_below_threshold_{cos_thresh}",
                candidate_scores=candidate_scores_log,
            ), latencies

        # Check reranker threshold if enabled
        if self.reranker and self.config.get("reranker_enabled", False):
            rerank_thresh = self.config.get("reranker_threshold", -5.0)
            if best_candidate.cross_encoder_score is not None and best_candidate.cross_encoder_score < rerank_thresh:
                return CacheDecision(
                    accepted=False,
                    decision_type="MISS",
                    reason=f"cross_encoder_score_{best_candidate.cross_encoder_score:.3f}_below_threshold_{rerank_thresh}",
                    candidate_scores=candidate_scores_log,
                ), latencies

        # 4. Reliability-Based Acceptance if enabled (E6)
        if self.reliability_model and self.config.get("reliability_enabled", False):
            features = self.reliability_model.extract_features(
                cosine_sim=best_candidate.cosine_similarity,
                cross_encoder_score=best_candidate.cross_encoder_score,
                rank=best_candidate.rank,
                param_count=len(best_candidate.extracted_params),
                structural_passed=best_candidate.validation_passed,
            )
            prob, lat_rel = self.reliability_model.predict_probability(features)
            latencies["reliability"] = lat_rel
            best_candidate.reliability_score = prob

            if not self.reliability_model.is_reliable(prob):
                return CacheDecision(
                    accepted=False,
                    decision_type="FALLBACK",
                    reason=f"reliability_score_{prob:.3f}_below_confidence_threshold_{self.reliability_model.confidence_threshold}",
                    candidate_scores=candidate_scores_log,
                ), latencies

        # Accept candidate plan!
        return CacheDecision(
            accepted=True,
            decision_type="HIT",
            reason="confidence_accepted",
            selected_candidate=best_candidate,
            tool_calls=best_candidate.bound_tool_calls,
            candidate_scores=candidate_scores_log,
        ), latencies

    def run_replay(self, dataset_path: str) -> Dict[str, Any]:
        """Run sequential replay on benchmark traces."""
        traces = load_dataset(dataset_path)
        request_records: List[Dict[str, Any]] = []

        total_traces = len(traces)
        for idx, trace in enumerate(traces, 1):
            if idx % 50 == 0 or idx == total_traces:
                print(f"Replay Progress: [{idx}/{total_traces}] requests evaluated...")
            req_id = trace["request_id"]
            prompt = trace["prompt"]
            testcase = trace["testcase"]
            ref_calls = trace.get("tool_calls", [])
            base_prompt_tokens = trace.get("prompt_tokens", 0)
            base_completion_tokens = trace.get("completion_tokens", 0)
            base_total_tokens = trace.get("total_tokens", 0)
            base_latency = trace.get("latency_sec", 1.0)

            # Evaluate against cache
            t_start = time.perf_counter()
            decision, latencies = self._evaluate_request(trace)
            cache_overhead = time.perf_counter() - t_start

            # Check agreement with LLM reference if HIT
            agreement = None
            proposed_calls_dict = []
            if decision.accepted and decision.tool_calls:
                proposed_calls_dict = [tc.to_dict() for tc in decision.tool_calls]
                agreement = plans_agree(proposed_calls_dict, ref_calls)
                self.store.record_hit(decision.selected_candidate.entry.entry_id, req_id)
            else:
                self.store.record_miss(req_id, decision.reason)

            # Fallback handling & Dynamic Cache Insertion
            if not decision.accepted:
                # LLM planning is invoked. We insert the reference plan into cache!
                ref_tool_objs = [ToolCall.from_dict(tc) for tc in ref_calls]
                emb = None
                if self.embedding_model:
                    emb, _ = self.embedding_model.encode(prompt)

                self.store.insert(
                    request_id=req_id,
                    testcase=testcase,
                    original_request=prompt,
                    tool_calls=ref_tool_objs,
                    netlist_id=testcase,
                    context_summary=trace.get("context", ""),
                    embedding=emb,
                )

            # Savings calculation
            if decision.accepted:
                # LLM planning call avoided!
                spent_tokens = int(latencies.get("light_llm_tokens", 0))
                prompt_tokens_saved = max(0, base_prompt_tokens - spent_tokens)
                completion_tokens_saved = base_completion_tokens
                total_tokens_saved = max(0, base_total_tokens - spent_tokens)
                actual_lat = cache_overhead + latencies.get("validation", 0.0)
                lat_saved = max(0.0, base_latency - actual_lat)
                cost_saved = (prompt_tokens_saved * 0.15 / 1e6) + (completion_tokens_saved * 0.60 / 1e6)
            else:
                prompt_tokens_saved = 0
                completion_tokens_saved = 0
                total_tokens_saved = 0
                actual_lat = base_latency + cache_overhead
                lat_saved = 0.0
                cost_saved = 0.0

            record = {
                "experiment_id": self.config.get("experiment_id"),
                "run_id": self.run_id,
                "request_id": req_id,
                "testcase": testcase,
                "prompt": prompt,
                "decision_type": decision.decision_type,
                "accepted": decision.accepted,
                "reason": decision.reason,
                "agreement": agreement,
                "selected_candidate_id": decision.selected_candidate.entry.entry_id if decision.selected_candidate else None,
                "proposed_tool_calls": proposed_calls_dict,
                "reference_tool_calls": ref_calls,
                "baseline_latency_sec": base_latency,
                "actual_latency_sec": actual_lat,
                "cache_overhead_sec": cache_overhead,
                "latency_saved_sec": lat_saved,
                "baseline_prompt_tokens": base_prompt_tokens,
                "baseline_completion_tokens": base_completion_tokens,
                "prompt_tokens_saved": prompt_tokens_saved,
                "completion_tokens_saved": completion_tokens_saved,
                "total_tokens_saved": total_tokens_saved,
                "cost_saved_usd": cost_saved,
                "latencies": latencies,
                "cache_size_after": len(self.store.entries),
            }
            request_records.append(record)

            # Log candidates and request
            if decision.candidate_scores:
                self.logger.log_candidates(req_id, decision.candidate_scores)
            self.logger.log_request(record)

        # Compute aggregate metrics
        metrics = compute_metrics(request_records, len(traces))
        metrics["experiment_id"] = self.config.get("experiment_id")
        metrics["strategy"] = self.config.get("strategy")
        metrics["run_id"] = self.run_id

        self.logger.finish(metrics)
        return metrics
