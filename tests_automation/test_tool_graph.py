"""Test automation for the tool knowledge graph (no LLM): the graph must say what tools.py says,
and access mode "graph" must offer it correctly.

    circuitAnt/openroad_python.sh circuitAnt/tests_automation/test_tool_graph.py

Checks that every tool is in the graph with its category, parameters, kinds, allowed values and
examples, that it agrees with the tool schemas the model gets in tools.access "all", and that
knowledge.ttl names only tools and kinds that exist.
"""
import inspect
import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rdflib import Literal  # noqa: E402
from rdflib.namespace import RDF, RDFS, SKOS  # noqa: E402

from agent.tool_access.graph.build import CA, CT, build, summary  # noqa: E402
from agent.tool_access import make_access  # noqa: E402
from agent.tools import Tools, schema  # noqa: E402

TOOLS = {name: fn for name, fn in inspect.getmembers(Tools, inspect.isfunction) if not name.startswith("_")}
PREFIXES = """PREFIX ca: <https://w3id.org/circuitant/ontology#> PREFIX ct: <https://w3id.org/circuitant/tools#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""


class ToolGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = build()

    def query(self, q: str) -> set:
        return {tuple(str(x) for x in row) for row in self.g.query(PREFIXES + q)}

    def test_every_tool_and_only_tools(self):
        tools = list(self.g.subjects(RDF.type, CA.Tool))
        self.assertEqual({str(self.g.value(t, RDFS.label)) for t in tools}, set(TOOLS))
        self.assertEqual(len(tools), len(TOOLS))

    def test_categories_follow_the_engine_packages(self):
        rows = self.query("""SELECT ?tool ?pkg WHERE { ?t a ca:Tool ; rdfs:label ?tool ; ca:inCategory ?c .
                             OPTIONAL { ?c skos:broader/rdfs:label ?pkg } }""")
        package = {tool: pkg for tool, pkg in rows}
        self.assertEqual(len(package), len(TOOLS))  # one category each
        self.assertEqual(package["get_cone"], "Analysis")
        self.assertEqual(package["remap_gates"], "Transformation")
        self.assertEqual(package["check_symmetry"], "Verification")
        self.assertEqual({t for t, p in package.items() if p == "None"},
                         {"set_testcase_name", "read_design", "write_design"})  # Session: no package
        for topic in self.g.objects(None, CA.inCategory):  # every category is named and described
            self.assertIsNotNone(self.g.value(topic, RDFS.label))
            self.assertTrue(str(self.g.value(topic, CA.description) or ""), topic)

    def test_design_flags(self):
        changes = {t for (t,) in self.query("SELECT ?t WHERE { ?x ca:changesDesign true ; rdfs:label ?t }")}
        self.assertIn("reduce_depth", changes)
        self.assertIn("read_design", changes)
        self.assertNotIn("check_equivalence", changes)
        needs = {t for (t,) in self.query("SELECT ?t WHERE { ?x ca:requiresDesign false ; rdfs:label ?t }")}
        self.assertEqual(needs, {"set_testcase_name", "read_design"})

    def test_summaries_are_one_sentence(self):
        for name, fn in TOOLS.items():
            text = str(self.g.value(CT[name], CA.summary))
            self.assertTrue(text.endswith("."), name)
            self.assertTrue(inspect.getdoc(fn).replace("\n", " ").startswith(text.rstrip(".")), name)
        self.assertEqual(summary("Rename a gate (e.g. g1 -> g2).  More."), "Rename a gate (e.g. g1 -> g2).")

    def test_parameters_agree_with_the_schemas(self):
        for name, fn in TOOLS.items():
            spec = schema(getattr(Tools(), name))["function"]["parameters"]  # as the agent builds it
            params = {str(self.g.value(p, RDFS.label)): p for p in self.g.objects(CT[name], CA.hasParameter)}
            self.assertEqual(set(params), set(spec["properties"]), name)
            for pname, p in params.items():
                self.assertEqual(self.g.value(p, CA.required).toPython(), pname in spec["required"], (name, pname))
                values = {str(v) for v in self.g.objects(p, CA.allowedValue)}
                self.assertEqual(values, set(spec["properties"][pname].get("enum", [])), (name, pname))
            positions = sorted(self.g.value(p, CA.position).toPython() for p in params.values())
            self.assertEqual(positions, list(range(1, len(params) + 1)), name)
        self.assertEqual(str(self.g.value(CT["get_max_depth.source"], CA.default)), "ANY_START")
        self.assertIsNone(self.g.value(CT["get_cone.shared_with"], CA.default))  # None is no default

    def test_kinds_and_keywords(self):
        self.assertEqual(self.query("""SELECT DISTINCT ?tool WHERE { ?t ca:hasParameter/ca:ofKind/ca:hasKeyword
                                       ct:ANY_PI ; rdfs:label ?tool }"""),
                         {("get_max_depth",), ("find_cut_points",), ("find_all_paths",), ("get_fanout",)})
        self.assertEqual(str(self.g.value(CT.ANY_PI, CA.description)), "all primary inputs")
        self.assertEqual(str(self.g.value(CT["get_cone.shared_with"], CA.ofKind)), str(CT.Signal))

    def test_examples_and_alternatives(self):
        bindings = self.query("""SELECT ?param ?value WHERE { ct:get_max_depth ca:hasExample ?e .
            ?e rdfs:label "register-to-register depth" ; ca:hasBinding ?b .
            ?b ca:forParameter/rdfs:label ?param ; ca:value ?value }""")
        self.assertEqual(bindings, {("source", "ANY_DFF_Q"), ("sink", "ANY_DFF_D")})
        for name, fn in TOOLS.items():
            self.assertEqual(len(list(self.g.objects(CT[name], CA.hasExample))), len(getattr(fn, "examples", [])))
            for other in getattr(fn, "instead", {}):
                self.assertIn((CT[other], CA.confusableWith, CT[name]), self.g)  # both ways

    def test_examples_use_placeholders_not_netlist_names(self):
        # Examples must generalise: X, A, B, g, T ... -- never a signal or gate of some netlist (n4042, g462).
        netlist_name = re.compile(r"(?<!')\b[a-z]+\d+(\[\d+\])?\b")  # not Verilog constants (1'b1)
        for name, fn in TOOLS.items():
            for phrase, args in getattr(fn, "examples", []):
                for text in [phrase, *map(str, args.values())]:
                    self.assertIsNone(netlist_name.search(text), (name, text))

    def test_knowledge_names_only_existing_tools_and_kinds(self):
        for prop in (CA.answeredBy, CA.followedBy):
            for s, o in self.g.subject_objects(prop):
                self.assertIn((o, RDF.type, CA.Tool), self.g, f"{s} {prop} {o}")
        for kind in self.g.subjects(CA.denotes, None):
            self.assertIn((kind, RDF.type, CA.ValueKind), self.g, kind)

    def test_every_tool_reachable_from_a_concept(self):
        answered = {str(self.g.value(t, RDFS.label)) for t in self.g.objects(None, CA.answeredBy)}
        self.assertEqual(set(TOOLS) - answered, set())

    def test_without_knowledge(self):
        g = build(knowledge=False)
        self.assertEqual(len(set(g.subjects(RDF.type, CA.Concept))), 0)
        self.assertEqual(len(set(g.subjects(RDF.type, CA.Tool))), len(TOOLS))
        self.assertNotIn((CT.reduce_depth, CA.followedBy, CT.check_equivalence), g)
        self.assertIn((CT.get_max_depth, CA.changesDesign, Literal(False)), g)


class GraphMode(unittest.TestCase):
    """Access mode "graph": what the model is offered, and each graph operation."""

    @classmethod
    def setUpClass(cls):
        cls.access = make_access({"access": "graph", "graph": {"sparql": True}}, Tools())

    def setUp(self):
        self.access.begin(1, "a request")

    def call(self, operation: str, **args) -> dict:
        return json.loads(self.access.execute(operation, json.dumps(args)).result)

    def test_offered_and_catalog(self):
        self.access.begin(2, "a request")
        first = {s["function"]["name"] for s in self.access.offered()}
        self.assertEqual(first, {"describe", "sparql"})  # run_tool only once something is described
        self.call("describe", tools="get_cone")
        offered = {s["function"]["name"] for s in self.access.offered()}
        self.assertEqual(offered, {"describe", "run_tool", "sparql"})  # default: no core tools, no search
        again = self.call("describe", tools="get_cone, get_fanout")["result"]
        self.assertEqual(again["get_cone"], "already described above in this request")  # not sent twice
        self.assertIn("arguments", again["get_fanout"])
        self.access.begin(3, "next request")
        self.assertIs(self.access.offered(), self.access.first_schemas)  # a new request starts without run_tool
        self.assertNotIn("search_tools", self.access.system_text)
        self.assertIn("error", self.call("search_tools", request="fanout"))  # off: not answered
        session = make_access({"access": "graph", "graph": {"core": ["read_design"]}}, Tools())
        self.assertIn("read_design", {s["function"]["name"] for s in session.offered()})
        self.assertNotIn("\nread_design\n", session.system_text)  # offered directly: not in the catalog
        others = set(TOOLS)
        catalog = self.access.system_text.split("Catalog:\n")[1].splitlines()
        self.assertEqual(catalog, sorted(others))  # names only, one per line
        plain = make_access({"access": "graph"}, Tools()).system_text
        self.assertFalse(plain.count("sparql"))
        summaries = make_access({"access": "graph", "graph": {"catalog": "summaries"}}, Tools()).system_text
        for name in others:
            self.assertIn(f"\n{name} -- ", summaries, name)  # name -- summary
        self.assertGreater(len(summaries), 3 * len(plain.split("Catalog:")[1]))
        with self.assertRaises(ValueError):
            make_access({"access": "graph", "graph": {"catalog": "short"}}, Tools())

    def test_describe(self):
        info = self.call("describe", tools="get_max_depth, get_cones")["result"]
        depth = info["get_max_depth"]
        self.assertEqual(depth["arguments"]["source"]["default"], "ANY_START")
        self.assertEqual(depth["arguments"]["source"]["keywords"]["ANY_PI"], "all primary inputs")
        self.assertIn("find_all_paths", depth["use_instead"])
        self.assertIn({"request": "register-to-register depth",
                       "arguments": {"source": "ANY_DFF_Q", "sink": "ANY_DFF_D"}}, depth["examples"])
        self.assertEqual(info["get_cones"]["did_you_mean"][0], "get_cone")
        cone = self.call("describe", tools="get_cone")["result"]["get_cone"]
        self.assertEqual(cone["arguments"]["direction"]["allowed"], ["fanin", "fanout"])
        self.assertIn("find_gates", cone["easily_confused_with"])  # stated on find_gates' side only
        self.assertIn("check_equivalence", self.call("describe", tools="reduce_depth")["result"]["reduce_depth"]
                      ["usually_followed_by"])

    def test_search(self):  # temporary (tools.graph.search, off by default)
        searching = make_access({"access": "graph", "graph": {"search": True}}, Tools())
        searching.begin(1, "a request")
        self.assertIn("search_tools", searching.system_text)
        call = lambda op, **a: json.loads(searching.execute(op, json.dumps(a)).result)
        found = call("search_tools", request="transitive fanout cone of X")["result"]
        self.assertIn("arguments", found[0]["description"])  # the best matches come described ...
        self.assertNotIn("description", found[-1])
        run = searching.execute("run_tool", json.dumps({"name": found[0]["tool"], "arguments": {"signal": "x"}}))
        self.assertTrue(run.ran)  # ... so run_tool may follow at once
        top = lambda text: [r["tool"] for r in call("search_tools", request=text)["result"]]
        self.assertEqual(top("transitive fanout cone of X")[0], "get_cone")
        self.assertEqual(top("shorten the critical path")[0], "reduce_depth")
        self.assertIn("check_symmetry", top("are two inputs of X interchangeable"))

    def test_run_tool_only_after_describe(self):
        run = lambda: self.access.execute("run_tool", json.dumps({"name": "get_ports", "arguments": {}}))
        self.access.begin(7, "how many ports")
        first = run()  # not described in this request: returns the description, runs nothing
        self.assertIsNone(first.engine_call)
        self.assertFalse(first.ran)
        self.assertIn("arguments", json.loads(first.result)["result"]["description"])
        self.assertEqual(run().engine_call, ("get_ports", {}))  # now it runs
        self.access.begin(8, "how many ports")
        self.assertIsNone(run().engine_call)  # a new request must read it again
        self.access.begin(9, "how many ports")
        self.access.execute("describe", json.dumps({"tools": "get_ports"}))
        self.assertEqual(run().engine_call, ("get_ports", {}))  # described first: runs at once
        self.access.begin(10, "testcase t1")
        guess = self.call("run_tool", name="set_testcase_name", arguments={"testcase_name": "t1"})["result"]
        self.assertIn("unknown argument 'testcase_name'", guess["your_arguments"])  # says what to fix
        self.assertIn("name (required)", guess["your_arguments"])
        wrong = self.call("run_tool", name="set_testcase_name", arguments={"testcase_name": "t1"})
        self.assertIn("missing required argument 'name'", wrong["error"])  # clear, not a TypeError

    def test_session_tools_only_for_what_the_request_names(self):
        self.access.begin(3, "Which output has the largest fanin logic cone?")
        self.call("describe", tools="set_testcase_name, read_design")
        refused = self.call("run_tool", name="set_testcase_name", arguments={"name": "fanin_cone_test"})
        self.assertIn("does not name", refused["error"])
        self.assertIn("does not name",
                      self.call("run_tool", name="read_design", arguments={"path": "testcase/t1/t1.v"})["error"])
        self.access.begin(4, "This is the beginning of a new testcase. The case name is t1.")
        self.call("describe", tools="set_testcase_name")
        self.assertNotIn("error", self.call("run_tool", name="set_testcase_name", arguments={"name": "t1"}))

    def test_run_tool_and_sparql(self):
        self.assertIn("did you mean", self.call("run_tool", name="get_cones", arguments={})["error"])
        self.assertIn("get_fanout", self.call("describe", tools="functions.get_fanout, 'get_ports'")["result"])
        self.assertIn("get_ports", self.call("describe", tools="functions.get_fanout, 'get_ports'")["result"])
        self.access.execute("describe", json.dumps({"tools": "count_gates"}))
        outcome = self.access.execute("run_tool", json.dumps({"name": "count_gates", "arguments": {}}))
        self.assertEqual(outcome.engine_call, ("count_gates", {}))  # recorded as the engine tool
        self.assertIn("no design loaded", json.loads(outcome.result)["error"])
        rows = self.call("sparql", query="SELECT ?n WHERE { ct:get_cone ca:inCategory/rdfs:label ?n }")
        self.assertEqual(rows["result"]["rows"], [["Cones"]])
        self.assertIn("error", self.call("sparql", query="INSERT DATA { ct:x ct:y ct:z }"))  # no updates
        self.assertIn("only SELECT", self.call("sparql", query="CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")["error"])
        self.assertIn("error", self.call("sparql", query="SELECT WHERE {"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
