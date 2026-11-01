"""Test automation for the agent loop (no LLM): a scripted model makes the tool calls.

    circuitAnt/openroad_python.sh circuitAnt/tests_automation/test_agent.py

Checks the safeguards against a model that repeats itself: an identical call is refused (unless the
design changed since), the request stops after MAX_STUCK_TURNS LLM calls in a row whose tool calls all
fail -- repeats or refusals -- (but not for parallel copies within one call, nor for an error the model
then fixes), a graph-mode run_tool that only returned a description is not a repeat, and setting the
current testcase name again keeps its log.  And the cache (agent/cache/): a hit takes a short first turn
with only the cached tools, then the normal loop; a refused or failing first turn falls back to it; a
miss sends what an agent without the cache sends.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from agent import Agent  # noqa: E402
from agent.agent import CACHE_PROMPT, MAX_STUCK_TURNS, SYSTEM_PROMPT  # noqa: E402
from agent.cache.manager import AgentCacheManager  # noqa: E402
from agent.cache.models.embeddings import EmbeddingModel  # noqa: E402
from agent.cache.store import SemanticCacheStore  # noqa: E402
from agent.llm import Reply, ToolCall  # noqa: E402

NETLIST = str(HERE / "test10.v")


class Script:
    """A model that makes the given turns: a list of (tool, arguments) calls, or a text answer."""

    def __init__(self, turns: list):
        self.turns = list(turns)
        self.sent = []  # (messages, tool names) of every call, as sent

    def complete(self, messages, tools):
        self.sent.append((json.loads(json.dumps(messages)), [t["function"]["name"] for t in tools]))
        turn = self.turns.pop(0) if self.turns else "done"
        if isinstance(turn, str):
            return Reply(turn, [], "end_turn", 1, 0, 0, 1, {})
        return Reply("", [ToolCall(f"c{len(self.turns)}{i}", n, json.dumps(a)) for i, (n, a) in enumerate(turn)],
                     "tool_use", 1, 0, 0, 1, {})


class WordEmbedding(EmbeddingModel):
    """Words hashed into a vector: similar wording, similar vector (no model to load)."""

    def encode(self, text):
        vector = [0.0] * 64
        for word in text.lower().split():
            vector[sum(map(ord, word)) % 64] += 1.0
        return vector, 0.0


def cache(path: Path | None = None) -> AgentCacheManager:
    """A cache with WordEmbedding instead of a real model; path: its file (None: none)."""
    manager = AgentCacheManager.__new__(AgentCacheManager)
    manager.store, manager.embedding_model, manager.threshold = SemanticCacheStore(), WordEmbedding(), 0.8
    manager.path = path
    manager.load()
    return manager


class AgentLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="circuitant_agent_")
        self.records, self.agents = [], []

    def tearDown(self):
        for agent in self.agents:
            if agent.tools.log:
                agent.tools.log.close()
        self.tmp.cleanup()

    def agent(self, turns: list, access: dict | None = None, cached: bool | Path = False) -> Agent:
        agent = Agent("unused", "scripted", output_dir=self.tmp.name, access=access)
        agent.run_analysis = self  # record() below collects what the run analysis would write
        agent.client = Script(turns)
        if cached:  # True: a cache of its own; a path: the cache file shared with other agents
            agent.cache_manager = cache(None if cached is True else cached)
        self.agents.append(agent)
        return agent

    def record(self, case, prompt, calls, usage, costs, ops, seconds, cache=None):
        self.records.append({"calls": calls, "ops": ops, "llm_calls": usage.llm_calls,
                             "cache": cache.status() if cache else None})

    def test_identical_call_refused_and_request_stopped(self):
        agent = self.agent([[("read_design", {"path": NETLIST})], "loaded"]
                           + [[("find_gates", {})]] * (MAX_STUCK_TURNS + 5))
        agent.ask(f"load {NETLIST}")
        answer = agent.ask("count")
        self.assertIn("stopped", answer)
        last = self.records[-1]
        self.assertEqual(last["calls"], [("find_gates", {})])  # ran once
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
                            [("find_gates", {"design": "x"})], [("find_gates", {"all": True})],
                            [("find_gates", {})], "4836 gates"])  # two wrong argument sets, then right
        agent.ask(f"load {NETLIST}")
        self.assertEqual(agent.ask("count the gates"), "4836 gates")
        self.assertEqual(self.records[-1]["llm_calls"], 4)

    def test_parallel_copies_in_one_call_do_not_stop_the_request(self):
        agent = self.agent([[("read_design", {"path": NETLIST})], "loaded",
                            [("find_gates", {})] * 6, "4836 gates"])
        agent.ask(f"load {NETLIST}")
        self.assertEqual(agent.ask("count"), "4836 gates")  # the model sees the refusals and answers
        self.assertEqual(len(self.records[-1]["ops"]), 5)

    def test_same_call_allowed_after_the_design_changed(self):
        agent = self.agent([[("read_design", {"path": NETLIST})], "loaded",
                            [("find_gates", {})], [("remove_dangling_gates", {})], [("find_gates", {})], "done"])
        agent.ask(f"load {NETLIST}")
        agent.ask("count, clean up, count again")
        self.assertEqual([c for c, _ in self.records[-1]["calls"]],
                         ["find_gates", "remove_dangling_gates", "find_gates"])
        self.assertEqual(self.records[-1]["ops"], [])

    def test_graph_run_tool_returning_the_description_is_not_a_repeat(self):
        run = ("run_tool", {"name": "find_gates", "arguments": {}})
        agent = self.agent([[("read_design", {"path": NETLIST})], "loaded", [run], [run], "done"],
                           access={"access": "graph", "graph": {"core": ["read_design"]}})
        agent.ask(f"load {NETLIST}")
        agent.ask("count")
        self.assertEqual(self.records[-1]["calls"], [("find_gates", {})])  # first: description; second: runs
        self.assertEqual(self.records[-1]["ops"], [run])

    def test_setting_the_same_testcase_again_keeps_its_log(self):
        agent = self.agent([[("set_testcase_name", {"name": "t1"})], "set"])  # (the request names t1)
        agent.ask("This is the beginning of a new testcase. The case name is t1.")
        agent.tools.log.write("#RESPONSE 1\nkept\n#END 1\n")
        agent.tools.log.flush()
        agent.client = Script([[("set_testcase_name", {"name": "t1"})], "again"])
        agent.ask("Start testcase t1 again.")
        self.assertIn("kept", (Path(self.tmp.name) / "t1.log").read_text())

    def test_booleans_written_as_text_are_converted(self):
        # graph-mode run_tool arguments are not type-checked: collapse="false" must not collapse
        patterns = str(HERE / "patterns.v")
        agent = self.agent([[("read_design", {"path": patterns})], "loaded",
                            [("find_inverter_pairs", {"collapse": "false"})],
                            [("find_inverter_pairs", {"collapse": "maybe"})],
                            [("find_inverter_pairs", {"collapse": "True"})], "done"])
        agent.ask(f"load {patterns}")
        agent.ask("find the inverter pairs, then collapse them")
        results = [json.loads(m["content"]) for m in agent.client.sent[-1][0] if m["role"] == "tool"]
        self.assertIn("count", results[0]["result"])  # reported, design unchanged
        self.assertIn("must be true or false", results[1]["error"])
        self.assertIn("pairs_collapsed", results[2]["result"])  # collapsed

    # -- the cache ----------------------------------------------------------------

    def cached_agent(self, last_turns: list, access: dict | None = None, load=None) -> Agent:
        """An agent with a cache that has seen "count the gates" (-> find_gates) and is asked a
        similar request next, which the model answers with last_turns."""
        first = [[load or ("read_design", {"path": NETLIST})], "loaded"]
        if access:  # graph mode: describe find_gates (run_tool returns its description), then run it
            run = ("run_tool", {"name": "find_gates", "arguments": {}})
            counted = [[run], [run], "4836 gates"]
        else:
            counted = [[("find_gates", {})], "4836 gates"]
        agent = self.agent(first + counted + last_turns, access=access, cached=True)
        agent.ask(f"load {NETLIST}")
        agent.ask("count the gates")
        self.assertEqual(self.records[-1]["cache"], "miss")
        return agent

    def test_cache_hit_takes_a_short_turn_then_the_normal_loop(self):
        agent = self.cached_agent([[("find_gates", {})], "4836 gates again"])
        self.assertEqual(agent.ask("count all the gates"), "4836 gates again")
        last = self.records[-1]
        self.assertEqual((last["cache"], last["llm_calls"], last["calls"]), ("hit", 2, [("find_gates", {})]))
        (short, short_tools), (full, _) = agent.client.sent[-2:]
        self.assertEqual(short_tools, ["find_gates"])  # only the cached tool
        self.assertEqual(short[0]["content"].split("\n")[0], CACHE_PROMPT.split("\n")[0])
        self.assertEqual([m["role"] for m in short], ["system", "user"])  # no history
        self.assertTrue(full[0]["content"].startswith(SYSTEM_PROMPT))  # turn 2: the full prompt,
        self.assertEqual(full[-2]["tool_calls"][0]["function"]["name"], "find_gates")  # with turn 1's call

    def test_cache_tools_refused_fall_back_to_the_normal_loop(self):
        agent = self.cached_agent(["find_gates does not fit", [("find_gates", {})], "4836 gates"])
        self.assertEqual(agent.ask("count all the gates"), "4836 gates")
        last = self.records[-1]
        self.assertEqual((last["cache"], last["llm_calls"]), ("refused", 3))
        full = agent.client.sent[-2][0]  # a normal first turn, without the refused short turn
        self.assertTrue(full[0]["content"].startswith(SYSTEM_PROMPT))
        self.assertEqual([m["role"] for m in full], ["system", "user"])

    def test_cache_turn_error_is_kept_and_only_successful_calls_are_cached(self):
        agent = self.cached_agent([[("find_gates", {"design": "x"})], [("find_gates", {})], "4836 gates"])
        self.assertEqual(agent.ask("count all the gates"), "4836 gates")
        self.assertEqual(self.records[-1]["llm_calls"], 3)
        self.assertIn('"error"', agent.client.sent[-2][0][-1]["content"])  # turn 2 sees turn 1's error
        stored = agent.cache_manager.store.get_exact_match("count all the gates")
        self.assertEqual([(c.tool, c.args) for c in stored.tool_calls], [("find_gates", {})])

    def test_graph_cache_hit_is_written_as_run_tool_and_runs(self):
        agent = self.cached_agent([[("find_gates", {})], "done"],
                                  access={"access": "graph", "graph": {"core": ["read_design"]}})
        agent.ask("count all the gates")
        last = self.records[-1]
        self.assertEqual((last["cache"], last["llm_calls"], last["calls"]), ("hit", 2, [("find_gates", {})]))
        call = agent.client.sent[-1][0][-2]["tool_calls"][0]["function"]
        self.assertEqual((call["name"], json.loads(call["arguments"])),
                         ("run_tool", {"name": "find_gates", "arguments": {}}))

    def test_cache_file_is_shared_with_the_next_process(self):
        shared = Path(self.tmp.name) / "cache.json"
        first = self.agent([[("read_design", {"path": NETLIST})], "loaded", [("find_gates", {})], "4836"],
                           cached=shared)  # one testcase's process ...
        first.ask(f"load {NETLIST}")
        first.ask("count the gates")
        second = self.agent([[("read_design", {"path": NETLIST})], "loaded", [("find_gates", {})], "4836"],
                            cached=shared)  # ... and the next one's: it starts with the first one's cache
        self.assertEqual(len(second.cache_manager.store.entries), 2)
        second.ask(f"load {NETLIST}")
        self.assertEqual((self.records[-1]["cache"], self.records[-1]["llm_calls"]), ("hit", 2))
        second.ask("count all the gates")
        self.assertEqual((self.records[-1]["cache"], self.records[-1]["llm_calls"]), ("hit", 2))

    def test_cache_miss_sends_what_an_agent_without_cache_sends(self):
        turns = [[("read_design", {"path": NETLIST})], "loaded", [("find_gates", {})], "4836 gates"]
        plain, cached = self.agent(list(turns)), self.agent(list(turns), cached=True)
        for agent in (plain, cached):
            agent.ask(f"load {NETLIST}")
            agent.ask("how many gates are there")
        self.assertEqual(plain.client.sent, cached.client.sent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
