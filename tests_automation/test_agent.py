"""Test automation for the agent loop (no LLM): a scripted model makes the tool calls.

    circuitAnt/openroad_python.sh circuitAnt/tests_automation/test_agent.py

Checks the safeguards against a model that repeats itself: an identical call is refused (unless the
design changed since), the request stops after MAX_STUCK_TURNS LLM calls in a row whose tool calls all
fail -- repeats or refusals -- (but not for parallel copies within one call, nor for an error the model
then fixes), a graph-mode run_tool that only returned a description is not a repeat, and setting the
current testcase name again keeps its log.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from agent import Agent  # noqa: E402
from agent.agent import MAX_STUCK_TURNS  # noqa: E402
from agent.llm import Reply, ToolCall  # noqa: E402

NETLIST = str(HERE / "test10.v")


class Script:
    """A model that makes the given turns: a list of (tool, arguments) calls, or a text answer."""

    def __init__(self, turns: list):
        self.turns = list(turns)

    def complete(self, messages, tools):
        turn = self.turns.pop(0) if self.turns else "done"
        if isinstance(turn, str):
            return Reply(turn, [], "end_turn", 1, 0, 0, 1, {})
        return Reply("", [ToolCall(f"c{len(self.turns)}{i}", n, json.dumps(a)) for i, (n, a) in enumerate(turn)],
                     "tool_use", 1, 0, 0, 1, {})


class AgentLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="circuitant_agent_")
        self.records, self.agents = [], []

    def tearDown(self):
        for agent in self.agents:
            if agent.tools.log:
                agent.tools.log.close()
        self.tmp.cleanup()

    def agent(self, turns: list, access: dict | None = None) -> Agent:
        agent = Agent("unused", "scripted", output_dir=self.tmp.name, access=access)
        agent.run_analysis = self  # record() below collects what the run analysis would write
        agent.client = Script(turns)
        self.agents.append(agent)
        return agent

    def record(self, case, prompt, calls, usage, costs, ops, seconds):
        self.records.append({"calls": calls, "ops": ops, "llm_calls": usage.llm_calls})

    def test_identical_call_refused_and_request_stopped(self):
        agent = self.agent([[("read_design", {"path": NETLIST})], "loaded"]
                           + [[("count_gates", {})]] * (MAX_STUCK_TURNS + 5))
        agent.ask(f"load {NETLIST}")
        answer = agent.ask("count")
        self.assertIn("stopped", answer)
        last = self.records[-1]
        self.assertEqual(last["calls"], [("count_gates", {})])  # ran once
        self.assertEqual(len(last["ops"]), MAX_STUCK_TURNS)  # then refused
        self.assertEqual(last["llm_calls"], MAX_STUCK_TURNS + 1)  # stopped, far below MAX_TOOL_ROUNDS

    def test_new_refused_calls_every_turn_stop_the_request(self):
        names = [[("set_testcase_name", {"name": f"made_up_{i}"})] for i in range(10)]
        agent = self.agent(names)
        self.assertIn("stopped", agent.ask("What is the fanout of primary input n1 (reset)?"))
        self.assertEqual(self.records[-1]["llm_calls"], MAX_STUCK_TURNS)
        self.assertIsNone(agent.tools.case)  # nothing was renamed

    def test_an_error_the_model_fixes_does_not_stop_the_request(self):
        agent = self.agent([[("read_design", {"path": NETLIST})], "loaded",
                            [("count_gates", {"design": "x"})], [("count_gates", {"all": True})],
                            [("count_gates", {})], "4836 gates"])  # two wrong argument sets, then right
        agent.ask(f"load {NETLIST}")
        self.assertEqual(agent.ask("count the gates"), "4836 gates")
        self.assertEqual(self.records[-1]["llm_calls"], 4)

    def test_parallel_copies_in_one_call_do_not_stop_the_request(self):
        agent = self.agent([[("read_design", {"path": NETLIST})], "loaded",
                            [("count_gates", {})] * 6, "4836 gates"])
        agent.ask(f"load {NETLIST}")
        self.assertEqual(agent.ask("count"), "4836 gates")  # the model sees the refusals and answers
        self.assertEqual(len(self.records[-1]["ops"]), 5)

    def test_same_call_allowed_after_the_design_changed(self):
        agent = self.agent([[("read_design", {"path": NETLIST})], "loaded",
                            [("count_gates", {})], [("remove_dangling_gates", {})], [("count_gates", {})], "done"])
        agent.ask(f"load {NETLIST}")
        agent.ask("count, clean up, count again")
        self.assertEqual([c for c, _ in self.records[-1]["calls"]],
                         ["count_gates", "remove_dangling_gates", "count_gates"])
        self.assertEqual(self.records[-1]["ops"], [])

    def test_graph_run_tool_returning_the_description_is_not_a_repeat(self):
        run = ("run_tool", {"name": "count_gates", "arguments": {}})
        agent = self.agent([[("read_design", {"path": NETLIST})], "loaded", [run], [run], "done"],
                           access={"access": "graph", "graph": {"core": ["read_design"]}})
        agent.ask(f"load {NETLIST}")
        agent.ask("count")
        self.assertEqual(self.records[-1]["calls"], [("count_gates", {})])  # first: description; second: runs
        self.assertEqual(self.records[-1]["ops"], [run])

    def test_setting_the_same_testcase_again_keeps_its_log(self):
        agent = self.agent([[("set_testcase_name", {"name": "t1"})], "set"])  # (the request names t1)
        agent.ask("This is the beginning of a new testcase. The case name is t1.")
        agent.tools.log.write("#RESPONSE 1\nkept\n#END 1\n")
        agent.tools.log.flush()
        agent.client = Script([[("set_testcase_name", {"name": "t1"})], "again"])
        agent.ask("Start testcase t1 again.")
        self.assertIn("kept", (Path(self.tmp.name) / "t1.log").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
