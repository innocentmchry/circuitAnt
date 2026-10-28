import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

# Mock platform-specific EDA C++ bindings if running offline / on Windows
if 'openroad' not in sys.modules:
    sys.modules['openroad'] = MagicMock()
if 'odb' not in sys.modules:
    sys.modules['odb'] = MagicMock()

from agent.cache import CacheDecision, CacheEntry, CandidateMatch, ToolCall, SemanticCacheStore
from agent.cache.models.embeddings import cosine_similarity
from agent.cache.models.reliability import ReliabilityModel
from agent.cache.validation.constants import OPPOSITE_PAIRS
from agent.cache.validation.extractor import extract_entities
from agent.cache.validation.structural import StructuralValidator, check_negative_constraints
from experiments.semantic_cache.evaluation.metrics import compute_metrics, plans_agree
from experiments.semantic_cache.evaluation.runner import ExperimentRunner


class TestSemanticCacheFramework(unittest.TestCase):

    def test_tool_call_and_cache_entry(self):
        tc = ToolCall(tool="count_gates", args={})
        self.assertEqual(tc.signature_str(), "count_gates()")

        tc2 = ToolCall(tool="find_gates", args={"gate_type": "NAND"})
        self.assertEqual(tc2.signature_str(), "find_gates(gate_type='NAND')")

        entry = CacheEntry(
            entry_id="test_001",
            request_id="req_001",
            testcase="test001",
            original_request="count all gates",
            normalized_request="count all gates",
            context_summary="",
            tool_calls=[tc],
            netlist_id="test001",
        )
        d = entry.to_dict()
        self.assertEqual(d["entry_id"], "test_001")
        self.assertEqual(len(d["tool_calls"]), 1)

    def test_cache_store_operations(self):
        events = []
        def on_event(ev_type, data):
            events.append((ev_type, data))

        store = SemanticCacheStore(capacity=2, event_callback=on_event)
        tc = ToolCall(tool="count_gates", args={})

        e1 = store.insert(
            request_id="r1",
            testcase="test001",
            original_request="Count all the gates",
            tool_calls=[tc],
        )
        self.assertEqual(len(store.entries), 1)
        self.assertIsNotNone(store.get_exact_match("Count all the gates"))
        self.assertIsNotNone(store.get_exact_match("  Count   all the gates  "))

        # Snapshot and restore
        snap = store.snapshot()
        store.clear()
        self.assertEqual(len(store.entries), 0)
        store.restore(snap)
        self.assertEqual(len(store.entries), 1)

        # Capacity eviction
        e2 = store.insert("r2", "test001", "req 2", [tc])
        e3 = store.insert("r3", "test001", "req 3", [tc])
        self.assertEqual(len(store.entries), 2)
        # Check that INVALIDATION event was fired
        inval_events = [ev for ev in events if ev[0] == "INVALIDATION"]
        self.assertEqual(len(inval_events), 1)

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)

        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

    def test_negative_constraints(self):
        # Fanin vs Fanout
        passed, reason = check_negative_constraints(
            "What is the fanin logic cone of n7[28]?",
            "What is the fanout of n7[28]?"
        )
        self.assertFalse(passed)
        self.assertIn("Opposite word mismatch", reason)

        # Max vs Min
        passed2, reason2 = check_negative_constraints(
            "Compute maximum depth",
            "Compute minimum depth"
        )
        self.assertFalse(passed2)

        # Non-clashing requests
        passed3, _ = check_negative_constraints(
            "Count all gates in this design",
            "Please count all gates in the netlist"
        )
        self.assertTrue(passed3)

    def test_entity_extraction(self):
        prompt = "Does output n7[59] depend on input n6[0]? Also count NAND gates in testcase/test001/test001.v"
        entities = extract_entities(prompt)
        self.assertIn("n7[59]", entities.signals)
        self.assertIn("n6[0]", entities.signals)
        self.assertIn("NAND", entities.gate_types)
        self.assertIn("testcase/test001/test001.v", entities.paths)

    def test_structural_validator(self):
        validator = StructuralValidator()

        # Schema validation
        valid, err = validator.validate_schema("find_gates", {"gate_type": "NAND"})
        self.assertTrue(valid)

        invalid, err = validator.validate_schema("find_gates", {"invalid_param": 123})
        self.assertFalse(invalid)

        # Parameter binding
        cand_calls = [ToolCall(tool="find_gates", args={"gate_type": "NAND"})]
        success, bound, reason, params = validator.instantiate_template(
            cand_calls,
            cand_request="List 5 NAND gates in this design.",
            new_request="List 5 OR gates in this design.",
            testcase_name="test002",
        )
        self.assertTrue(success)
        self.assertEqual(bound[0].args["gate_type"], "OR")

    def test_reliability_model(self):
        model = ReliabilityModel(confidence_threshold=0.60)
        feats = model.extract_features(
            cosine_sim=0.92,
            cross_encoder_score=2.5,
            rank=1,
            param_count=1,
            structural_passed=True,
        )
        self.assertEqual(len(feats), 5)
        prob, _ = model.predict_probability(feats)
        self.assertGreater(prob, 0.5)

    def test_metrics_calculation(self):
        sample_records = [
            {
                "accepted": True,
                "agreement": True,
                "prompt_tokens_saved": 1000,
                "completion_tokens_saved": 50,
                "baseline_prompt_tokens": 1000,
                "baseline_completion_tokens": 50,
                "baseline_latency_sec": 1.5,
                "actual_latency_sec": 0.05,
                "cache_overhead_sec": 0.05,
                "cache_size_after": 1,
            },
            {
                "accepted": False,
                "agreement": None,
                "prompt_tokens_saved": 0,
                "completion_tokens_saved": 0,
                "baseline_prompt_tokens": 1200,
                "baseline_completion_tokens": 60,
                "baseline_latency_sec": 1.8,
                "actual_latency_sec": 1.82,
                "cache_overhead_sec": 0.02,
                "cache_size_after": 2,
            },
        ]
        metrics = compute_metrics(sample_records, 2)
        self.assertEqual(metrics["total_requests"], 2)
        self.assertEqual(metrics["cache_hits"], 1)
        self.assertEqual(metrics["hit_rate_pct"], 50.0)
        self.assertEqual(metrics["reference_agreement_pct"], 100.0)
        self.assertGreater(metrics["token_reduction_pct"], 0.0)

    def test_agent_live_cache_integration(self):
        import sys
        from unittest.mock import MagicMock
        sys.modules['openroad'] = MagicMock()
        sys.modules['odb'] = MagicMock()
        from agent import Agent

        # Test Agent with cache disabled
        agent_no_cache = Agent("dummy", "gpt-4o-mini", cache_config={"enabled": False})
        self.assertIsNone(agent_no_cache.cache_manager)

        # Test Agent with cache enabled
        agent_cached = Agent("dummy", "gpt-4o-mini", cache_config={"enabled": True, "strategy": "exact"})
        self.assertIsNotNone(agent_cached.cache_manager)
        self.assertTrue(agent_cached.cache_manager.enabled)


if __name__ == "__main__":
    unittest.main()

