"""Live verification test for Dynamic Tool Synthesis in circuitAnt."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from agent.synthesis.kg_context import KGContextExtractor
from agent.synthesis.synthesizer import DynamicToolSynthesizer
from agent.tools import Tools
from eda_engine import read_design, verification
from main import load_config


class TestDynamicToolSynthesis(unittest.TestCase):
    def setUp(self):
        self.config, self.key = load_config("config.yaml")
        self.tools = Tools(output_dir="./output_temp")
        self.extractor = KGContextExtractor(self.tools)
        self.synthesizer = DynamicToolSynthesizer(api_key=self.key, model="gpt-4o", tools_instance=self.tools)

    def tearDown(self):
        # Clean up temporary test files
        for tool_name in ["insert_dedicated_buffers_for_loads", "collapse_inverter_pairs"]:
            p = Path("agent/dynamic_tools") / f"{tool_name}.py"
            if p.exists():
                p.unlink(missing_ok=True)

    def test_01_kg_context_bundle(self):
        bundle = self.extractor.build_context_bundle(
            category="transformation",
            intent="collapse back-to-back inverter pairs into direct wires",
            suggested_name="collapse_inverter_pairs"
        )
        self.assertIn("TRANSFORMATION", bundle)
        self.assertIn("edit.remove_gate", bundle)
        self.assertIn("edit.merge_nets", bundle)
        print("\n[OK] KG Context Bundle generated successfully.")

    def test_02_synthesis_optimization_buffering(self):
        print("\n[INFO] Synthesizing 'insert_dedicated_buffers_for_loads' (Optimization)...")
        result = self.synthesizer.synthesize_and_register(
            category="optimization",
            intent="Insert a BUF gate on the specified signal so that every load of the signal is fed through its own dedicated buffer.",
            suggested_name="insert_dedicated_buffers_for_loads",
            expected_inputs="net_name: Signal",
            expected_output_fields="net_name, buffers_inserted"
        )
        self.assertEqual(result.get("status"), "SUCCESS", f"Synthesis failed: {result.get('error')}")
        self.assertTrue(hasattr(self.tools, "insert_dedicated_buffers_for_loads"))

        d_path = "testcase/test039/test039.v"
        if Path(d_path).exists():
            self.tools.read_design(d_path)
            res = self.tools.insert_dedicated_buffers_for_loads(net_name="n140")
            print(f"[INFO] Executed synthesized tool. Result: {res}")
            self.assertIn("buffers_inserted", res)
            count = len(res["buffers_inserted"]) if isinstance(res.get("buffers_inserted"), list) else res.get("buffers_inserted", 0)
            self.assertGreater(count, 0)

            cec = self.tools.check_equivalence()
            print(f"[INFO] Formal ABC CEC Verification Status: {cec.get('status')}")
            self.assertEqual(cec.get("status"), "PASS", "Transformed design must be formally equivalent!")
            print("[OK] Synthesized optimization tool passed ABC CEC verification!")

    def test_03_synthesis_transformation_inverters(self):
        print("\n[INFO] Synthesizing 'collapse_inverter_pairs' (Transformation)...")
        result = self.synthesizer.synthesize_and_register(
            category="transformation",
            intent="Find all pairs of back-to-back inverters (NOT followed by NOT) and collapse them into direct wires. Repeat until none remain.",
            suggested_name="collapse_inverter_pairs",
            expected_inputs="none",
            expected_output_fields="pairs_collapsed, gates_removed"
        )
        self.assertEqual(result.get("status"), "SUCCESS", f"Synthesis failed: {result.get('error')}")
        self.assertTrue(hasattr(self.tools, "collapse_inverter_pairs"))

        d_path = "testcase/test098/test098.v"
        if Path(d_path).exists():
            self.tools.read_design(d_path)
            try:
                res = self.tools.collapse_inverter_pairs()
                print(f"[INFO] Initial run of collapse_inverter_pairs: {res}")
            except Exception as exc:
                print(f"[WARN] Initial execution encountered: {exc}. Triggering KTCE Mutation pass...")
                mut_res = self.synthesizer.mutate_and_reregister(
                    category="transformation",
                    intent="Find all pairs of back-to-back inverters (NOT followed by NOT) and collapse them into direct wires. Repeat until none remain.",
                    suggested_name="collapse_inverter_pairs",
                    error_feedback=str(exc)
                )
                self.assertEqual(mut_res.get("status"), "SUCCESS")
                self.tools.read_design(d_path)
                res = self.tools.collapse_inverter_pairs()
                print(f"[INFO] Post-Mutation run of collapse_inverter_pairs: {res}")

            cec = self.tools.check_equivalence()
            print(f"[INFO] Formal ABC CEC Verification Status on test098: {cec.get('status')}")
            self.assertEqual(cec.get("status"), "PASS", "Collapsed inverter design must be formally equivalent!")
            print("[OK] Synthesized transformation tool passed ABC CEC verification on held-out test098!")


if __name__ == "__main__":
    unittest.main()
