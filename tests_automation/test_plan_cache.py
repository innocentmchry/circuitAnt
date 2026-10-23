"""Unit tests for semantic plan cache (offline, no external network calls)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agent import Agent  # noqa: E402
from plan_cache.generalize import PlanTemplate, trace_to_template  # noqa: E402
from plan_cache.masking import mask_request  # noqa: E402
from plan_cache.store import PlanStore  # noqa: E402
from plan_cache.trace import ExecutionTrace, ToolCallRecord  # noqa: E402


class TestTraceSerialization(unittest.TestCase):
    def test_tool_call_record(self):
        rec = ToolCallRecord(name="count_gates", arguments={}, result={"total_gates": 10}, elapsed=0.012)
        d = rec.to_dict()
        rec2 = ToolCallRecord.from_dict(d)
        self.assertEqual(rec.name, rec2.name)
        self.assertEqual(rec.arguments, rec2.arguments)
        self.assertEqual(rec.result, rec2.result)
        self.assertAlmostEqual(rec.elapsed, rec2.elapsed)

    def test_execution_trace(self):
        trace = ExecutionTrace(
            request="how many gates are in the design",
            tool_calls=[
                ToolCallRecord(name="count_gates", arguments={}, result={"total_gates": 10}, elapsed=0.05)
            ],
            prompt_tokens=120,
            completion_tokens=25,
            total_tokens=145,
            wall_clock_time=0.45,
            llm_time=0.38,
            tool_time=0.05,
            success=True,
            answer="There are 10 gates.",
            context_summary="testcase='test10'",
        )
        json_str = trace.to_json()
        trace2 = ExecutionTrace.from_json(json_str)
        self.assertEqual(trace.request, trace2.request)
        self.assertEqual(len(trace2.tool_calls), 1)
        self.assertEqual(trace2.tool_calls[0].name, "count_gates")
        self.assertEqual(trace2.prompt_tokens, 120)
        self.assertEqual(trace2.completion_tokens, 25)
        self.assertEqual(trace2.total_tokens, 145)
        self.assertTrue(trace2.success)
        self.assertEqual(trace2.answer, "There are 10 gates.")


class TestAgentTraceRecording(unittest.TestCase):
    def _create_mock_agent(self, plan_cache_enabled=False):
        agent = Agent(
            api_key="mock-key",
            model="mock-model",
            plan_cache={"enabled": plan_cache_enabled},
        )
        agent.client = MagicMock()
        return agent

    def test_trace_recording_disabled_by_default(self):
        agent = self._create_mock_agent(plan_cache_enabled=False)

        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content = "Test answer"
        mock_msg.model_dump.return_value = {"role": "assistant", "content": "Test answer"}

        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 50
        mock_resp.usage.completion_tokens = 10
        mock_resp.usage.total_tokens = 60

        agent.client.chat.completions.create.return_value = mock_resp

        ans = agent.ask("Hello")
        self.assertEqual(ans, "Test answer")
        self.assertIsNone(agent.last_trace)

    def test_trace_recording_enabled(self):
        agent = self._create_mock_agent(plan_cache_enabled=True)

        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "count_gates"
        tc.function.arguments = "{}"

        msg1 = MagicMock()
        msg1.tool_calls = [tc]
        msg1.content = None
        msg1.model_dump.return_value = {"role": "assistant", "tool_calls": [{"id": "call_1"}]}

        choice1 = MagicMock()
        choice1.message = msg1
        choice1.finish_reason = "tool_calls"

        resp1 = MagicMock()
        resp1.choices = [choice1]
        resp1.usage.prompt_tokens = 80
        resp1.usage.completion_tokens = 15
        resp1.usage.total_tokens = 95

        msg2 = MagicMock()
        msg2.tool_calls = None
        msg2.content = "Total gates: 42"
        msg2.model_dump.return_value = {"role": "assistant", "content": "Total gates: 42"}

        choice2 = MagicMock()
        choice2.message = msg2
        choice2.finish_reason = "stop"

        resp2 = MagicMock()
        resp2.choices = [choice2]
        resp2.usage.prompt_tokens = 110
        resp2.usage.completion_tokens = 20
        resp2.usage.total_tokens = 130

        agent.client.chat.completions.create.side_effect = [resp1, resp2]
        agent.tools.count_gates = MagicMock(return_value={"total_gates": 42, "breakdown": {}})
        agent.functions["count_gates"] = agent.tools.count_gates

        ans = agent.ask("how many gates are in the netlist")
        self.assertEqual(ans, "Total gates: 42")
        self.assertIsNotNone(agent.last_trace)
        trace = agent.last_trace
        self.assertEqual(trace.request, "how many gates are in the netlist")
        self.assertEqual(trace.prompt_tokens, 190)
        self.assertEqual(trace.completion_tokens, 35)
        self.assertEqual(trace.total_tokens, 225)
        self.assertTrue(trace.success)
        self.assertEqual(trace.answer, "Total gates: 42")
        self.assertEqual(len(trace.tool_calls), 1)
        self.assertEqual(trace.tool_calls[0].name, "count_gates")
        self.assertEqual(trace.tool_calls[0].result, {"total_gates": 42, "breakdown": {}})
        self.assertGreater(trace.wall_clock_time, 0.0)


class TestMasking(unittest.TestCase):
    def test_mask_signal_and_number(self):
        res = mask_request("what is the depth from n12 to n25[0] with max depth 15")
        self.assertIn("<SIGNAL:0>", res.masked_text)
        self.assertIn("<SIGNAL:1>", res.masked_text)
        self.assertIn("<NUMBER:0>", res.masked_text)
        self.assertEqual(res.slots["<SIGNAL:0>"], "n12")
        self.assertEqual(res.slots["<SIGNAL:1>"], "n25[0]")
        self.assertEqual(res.slots["<NUMBER:0>"], 15)

    def test_mask_gate_type_and_path(self):
        res = mask_request("how many NAND gates are written to output/result.txt")
        self.assertIn("<GATE_TYPE:0>", res.masked_text)
        self.assertIn("<PATH:0>", res.masked_text)
        self.assertEqual(res.slots["<GATE_TYPE:0>"], "NAND")
        self.assertEqual(res.slots["<PATH:0>"], "output/result.txt")

    def test_mask_preserve_keywords(self):
        res = mask_request("depth from ANY_PI to ANY_PO")
        self.assertIn("ANY_PI", res.masked_text)
        self.assertIn("ANY_PO", res.masked_text)


class TestGeneralization(unittest.TestCase):
    def test_linear_plan_with_dataflow_reference(self):
        """Multi-step plan with dataflow references:
        Step 0: get_endpoint_stats(sink="ANY_PO", metric="depth")
                returns {"at_max": ["n25[16]"], "max_value": 53}
        Step 1: get_cone(signal="n25[16]", direction="fanin")
        Verify step 1's signal argument becomes $step_0.result.at_max[0]
        """
        trace = ExecutionTrace(
            request="find the deepest primary output and get its fanin cone",
            tool_calls=[
                ToolCallRecord(
                    name="get_endpoint_stats",
                    arguments={"sink": "ANY_PO", "metric": "depth"},
                    result={"at_max": ["n25[16]"], "max_value": 53},
                    elapsed=0.1,
                ),
                ToolCallRecord(
                    name="get_cone",
                    arguments={"signal": "n25[16]", "direction": "fanin"},
                    result={"count": 120},
                    elapsed=0.15,
                ),
            ],
            prompt_tokens=200,
            completion_tokens=40,
            total_tokens=240,
            success=True,
            answer="Deepest PO is n25[16] with depth 53 and fanin cone size 120.",
        )

        template, err = trace_to_template(trace)
        self.assertIsNone(err)
        self.assertIsNotNone(template)
        self.assertEqual(len(template.steps), 2)
        # Step 0 args
        self.assertEqual(template.steps[0]["args"]["sink"], "ANY_PO")
        self.assertEqual(template.steps[0]["args"]["metric"], "depth")
        # Step 1 args: signal must be a reference to step 0
        self.assertEqual(template.steps[1]["args"]["signal"], "$step_0.result.at_max[0]")
        self.assertEqual(template.steps[1]["args"]["direction"], "fanin")
        self.assertFalse(template.mutates_design)
        self.assertEqual(template.tool_sequence, "get_endpoint_stats,get_cone")

    def test_non_cacheable_looping_trace(self):
        """Trace with repeating consecutive tool calls (per-item repetition) is rejected."""
        trace = ExecutionTrace(
            request="get cone for each output",
            tool_calls=[
                ToolCallRecord(name="get_cone", arguments={"signal": "n1"}, result={"count": 5}),
                ToolCallRecord(name="get_cone", arguments={"signal": "n2"}, result={"count": 8}),
                ToolCallRecord(name="get_cone", arguments={"signal": "n3"}, result={"count": 12}),
            ],
            success=True,
        )
        template, err = trace_to_template(trace)
        self.assertIsNone(template)
        self.assertIn("Per-item repetition detected", err)

    def test_non_cacheable_cycle_loop(self):
        """Trace with repeating tool cycle [A, B, A, B] is rejected."""
        trace = ExecutionTrace(
            request="query alternately",
            tool_calls=[
                ToolCallRecord(name="count_gates", arguments={}, result={"total": 10}),
                ToolCallRecord(name="get_fanout", arguments={"signal": "n1"}, result={"fanout": 2}),
                ToolCallRecord(name="count_gates", arguments={}, result={"total": 10}),
                ToolCallRecord(name="get_fanout", arguments={"signal": "n2"}, result={"fanout": 3}),
            ],
            success=True,
        )
        template, err = trace_to_template(trace)
        self.assertIsNone(template)
        self.assertIn("Loop detected with repeating tool pattern", err)


class TestPlanStore(unittest.TestCase):
    def test_store_and_retrieve_and_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_file = Path(tmp) / "test_cache.sqlite"
            store = PlanStore(db_file)

            template = PlanTemplate(
                masked_intent="how many <GATE_TYPE:0> gates are in the design",
                steps=[{"step": 0, "tool": "find_gates", "args": {"gate_type": "<GATE_TYPE:0>"}}],
                slot_schema={"<GATE_TYPE:0>": "gate_type"},
                mutates_design=False,
                tool_sequence="find_gates",
            )
            emb = [0.1, 0.2, 0.3]

            tid = store.save_or_update(template, emb)
            self.assertGreaterEqual(tid, 1)

            # Retrieve
            all_tpls = store.get_all_templates()
            self.assertEqual(len(all_tpls), 1)
            t_id, t_obj, t_emb = all_tpls[0]
            self.assertEqual(t_id, tid)
            self.assertEqual(t_obj.masked_intent, template.masked_intent)
            self.assertEqual(t_emb, emb)

            # Save duplicate with same tool sequence and slot schema -> should update use_count instead of inserting new
            tid2 = store.save_or_update(template, emb)
            self.assertEqual(tid, tid2)
            all_tpls_after = store.get_all_templates()
            self.assertEqual(len(all_tpls_after), 1)
            self.assertEqual(all_tpls_after[0][1].use_count, 1)

            # Record failures until drop
            dropped = store.record_failure(tid, max_failures_before_drop=3)
            self.assertFalse(dropped)
            dropped = store.record_failure(tid, max_failures_before_drop=3)
            self.assertFalse(dropped)
            dropped = store.record_failure(tid, max_failures_before_drop=3)
            self.assertTrue(dropped)

            # Check that it was dropped from DB
            self.assertEqual(len(store.get_all_templates()), 0)


class TestOppositeWords(unittest.TestCase):
    def test_opposite_words_detection(self):
        from plan_cache.matcher import check_opposite_words

        # Agreeing pairs
        ok, _ = check_opposite_words("what is the maximum depth", "find the maximum depth of design")
        self.assertTrue(ok)

        # Opposite mismatch: max vs min
        ok, reason = check_opposite_words("what is the minimum depth", "what is the maximum depth")
        self.assertFalse(ok)
        self.assertIn("Opposite word mismatch", reason)

        # Opposite mismatch: count vs list
        ok, reason = check_opposite_words("list all NAND gates", "count all NAND gates")
        self.assertFalse(ok)
        self.assertIn("Opposite word", reason)


class TestExecutor(unittest.TestCase):
    def test_resolve_reference_path(self):
        from plan_cache.executor import resolve_reference_path

        earlier = [
            {"breakdown": {"NAND": 5}, "signals": ["n12", "n25[0]"]},
            {"critical_path": ["n12", "n13", "n14"]},
        ]

        # Dot key
        self.assertEqual(resolve_reference_path("$step_0.result.breakdown.NAND", earlier), 5)
        # Array index
        self.assertEqual(resolve_reference_path("$step_0.result.signals[1]", earlier), "n25[0]")
        self.assertEqual(resolve_reference_path("$step_1.result.critical_path[0]", earlier), "n12")

        # Invalid index or key
        with self.assertRaises(IndexError):
            resolve_reference_path("$step_2.result", earlier)
        with self.assertRaises(KeyError):
            resolve_reference_path("$step_0.result.nonexistent", earlier)


class TestMatcherAndValidation(unittest.TestCase):
    def test_design_slot_validation(self):
        from tools import Tools
        from plan_cache.matcher import PlanMatcher
        from eda_engine import read_design

        TEST10 = HERE / "test10.v"
        tools = Tools()
        tools.design = read_design(TEST10)

        with tempfile.TemporaryDirectory() as tmp:
            store = PlanStore(Path(tmp) / "cache.sqlite")
            tpl = PlanTemplate(
                masked_intent="cone of <SIGNAL:0>",
                steps=[{"step": 0, "tool": "get_cone", "args": {"signal": "<SIGNAL:0>"}}],
                slot_schema={"<SIGNAL:0>": "signal"},
                mutates_design=False,
                tool_sequence="get_cone",
            )
            dummy_emb = [1.0, 0.0]
            store.save_or_update(tpl, dummy_emb)

            matcher = PlanMatcher(store, embedding_fn=lambda text: [1.0, 0.0])

            # Valid signal: n12 exists in test10
            res = matcher.find_match("cone of n12", tools, similarity_threshold=0.8)
            self.assertTrue(res.matched)
            self.assertEqual(res.filled_slots["<SIGNAL:0>"], "n12")

            # Non-existent signal: n999999 does NOT exist in test10
            res_bad = matcher.find_match("cone of n999999", tools, similarity_threshold=0.8)
            self.assertFalse(res_bad.matched)
            self.assertIn("not found in design", res_bad.reason)


class TestEndToEndAgentHitAndMiss(unittest.TestCase):
    def test_learn_and_hit_cached_plan(self):
        from eda_engine import read_design
        TEST10 = HERE / "test10.v"

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "cache.sqlite"
            agent = Agent(
                api_key="mock-key",
                model="gpt-4o-mini",
                plan_cache={"enabled": True, "path": str(db_path), "similarity_threshold": 0.8},
            )
            # Load design
            agent.tools.design = read_design(TEST10)

            # Mock OpenAI client embeddings and completions
            agent.client = MagicMock()
            agent.client.embeddings.create.return_value.data = [MagicMock(embedding=[0.5, 0.5])]
            agent.plan_cache.client = agent.client

            # Request 1: Normal LLM run (generates tool call, answers, records & learns)
            tc = MagicMock()
            tc.id = "call_1"
            tc.function.name = "count_gates"
            tc.function.arguments = "{}"

            msg1 = MagicMock()
            msg1.tool_calls = [tc]
            msg1.content = None
            msg1.model_dump.return_value = {"role": "assistant", "tool_calls": [{"id": "call_1"}]}

            choice1 = MagicMock(message=msg1, finish_reason="tool_calls")
            resp1 = MagicMock(choices=[choice1], usage=MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120))

            msg2 = MagicMock(tool_calls=None, content="Total gates is 4836.")
            msg2.model_dump.return_value = {"role": "assistant", "content": "Total gates is 4836."}
            choice2 = MagicMock(message=msg2, finish_reason="stop")
            resp2 = MagicMock(choices=[choice2], usage=MagicMock(prompt_tokens=140, completion_tokens=25, total_tokens=165))

            agent.client.chat.completions.create.side_effect = [resp1, resp2]

            ans1 = agent.ask("how many gates are in the circuit")
            self.assertEqual(ans1, "Total gates is 4836.")

            # Verify that the template was learned into store
            all_tpls = agent.plan_cache.store.get_all_templates()
            self.assertEqual(len(all_tpls), 1)

            # Request 2: Plan Cache Hit!
            # Single LLM call to word the answer directly from executed tool results
            word_msg = MagicMock(tool_calls=None, content="There are 4836 total gates.")
            word_choice = MagicMock(message=word_msg, finish_reason="stop")
            word_resp = MagicMock(choices=[word_choice], usage=MagicMock(prompt_tokens=50, completion_tokens=10, total_tokens=60))

            agent.client.chat.completions.create.side_effect = [word_resp]

            ans2 = agent.ask("what is the total gate count of the circuit")
            self.assertEqual(ans2, "There are 4836 total gates.")
            # Trace should reflect the cache hit
            self.assertIsNotNone(agent.last_trace)
            self.assertEqual(len(agent.last_trace.tool_calls), 1)
            self.assertEqual(agent.last_trace.tool_calls[0].name, "count_gates")


if __name__ == "__main__":
    unittest.main()
