"""Access mode "graph": the model reaches the tools through the tool knowledge graph.

The model is offered the graph operations describe and run_tool (and, if enabled, sparql and the
temporary search_tools) and, if configured, some core tools directly (by default none).  The other tools are named in
a catalog in the system prompt (names only, or with one-line summaries); the model
describes the candidates it needs, several at once -- arguments, examples, alternatives, what
usually follows -- picks one and calls it through run_tool.  A
tool runs only once its description has been read in the request: until something is described,
run_tool is not even offered (instructions alone did not stop models from guessing arguments from
the catalog line), and a run_tool of a tool not yet described returns its description instead of
running.  The system prompt is the same for every request and the tool list is one of two fixed
lists (before / after the first describe), so the provider's prompt cache keeps working.

Options (config tools.graph):
  core       tools offered directly (default none: every tool through the graph; offering the session
             tools directly made GPT-4o-mini call them unasked, e.g. for "input n1 (reset)")
  catalog    "names" (default): the catalog lists tool names only; "summaries": each with its summary
  sparql     also offer read-only SPARQL over the graph (default false)
  search     also offer search_tools (default false).  TEMPORARY, do not use: keyword matching against
             hand-written synonyms, which follows the wording it was written for; to be replaced by
             embedding-based search (needed once the catalog no longer fits in the prompt)
  knowledge  include knowledge.ttl: concepts and synonyms, followedBy (default true)
"""
import json

from agent.tool_access.direct import DirectAccess, Outcome, argument_problems
from agent.tool_access.graph.build import build
from agent.tool_access.graph.operations import Operations, close
from agent.tools import Tools

CORE = []
SEARCH_DESCRIBED = 3  # search_tools (temporary, off by default) returns the descriptions of its best matches

USAGE = """

The netlist tools are kept in a tool knowledge graph; their names are listed in the catalog below
({listing}).  For a request: (1) call describe with every catalog name that may fit, all in one call;
(2) pick the tool that fits from the descriptions; (3) call run_tool with its name and arguments.{search}{sparql}

Catalog:
{catalog}"""

SPARQL_USAGE = """  sparql answers read-only SELECT / ASK queries over the graph:
classes ca:Tool, ca:Category, ca:Parameter, ca:ValueKind, ca:Keyword, ca:Example, ca:Concept; every node
has rdfs:label and ca:description; ca:summary, ca:inCategory (skos:broader for the package),
ca:hasParameter (ca:required, ca:default, ca:ofKind, ca:allowedValue), ca:hasKeyword, ca:hasExample,
ca:confusableWith, ca:followedBy, ca:changesDesign, ca:answeredBy (Concept -> Tool, synonyms in
skos:altLabel).  Tools are ct:<name>."""

OPERATIONS = [
    {"type": "function", "function": {
        "name": "describe",
        "description": "Step 1 for any catalog tool: everything the tool knowledge graph knows about some tools -- "
                       "what each does and returns, its arguments (meaning, required, default, allowed values, "
                       "keywords), example requests with their arguments, tools to use instead and when, and "
                       "tools that usually follow.  Give every candidate name in one call.",
        "parameters": {"type": "object", "properties": {
            "tools": {"type": "string", "description": "tool names from the catalog, comma-separated"}},
            "required": ["tools"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "run_tool",
        "description": "Run a catalog tool with its arguments; returns the tool's result.  Only after describe "
                       "returned the tool's description in this request: otherwise it returns the description "
                       "and runs nothing.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "the tool name"},
            "arguments": {"type": "object", "description": "the tool's arguments by name ({} for none)"}},
            "required": ["name", "arguments"], "additionalProperties": False}}},
]

# TEMPORARY -- do not use (offered only with tools.graph.search: true).  Keyword matching against the
# hand-written synonyms of knowledge.ttl: it finds what its word lists anticipate, follows the wording
# they were written from, and understands neither meaning nor direction.  Kept only until it is
# replaced by embedding-based search, which a catalog too large for the prompt will need.
SEARCH = {"type": "function", "function": {
    "name": "search_tools",
    "description": "Tools of the knowledge graph that best match a request (through EDA concepts and their "
                   "synonyms, examples and summaries); the best matches come with their full descriptions, "
                   "as from describe.",
    "parameters": {"type": "object", "properties": {
        "request": {"type": "string", "description": "the request, or the part a tool is needed for"}},
        "required": ["request"], "additionalProperties": False}}}

SPARQL = {"type": "function", "function": {
    "name": "sparql",
    "description": "Run a read-only SPARQL SELECT or ASK query over the tool knowledge graph (prefixes ca:, ct:, "
                   "rdfs:, skos: are predefined); returns the rows.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"], "additionalProperties": False}}}


class GraphAccess:
    def __init__(self, tools: Tools, options: dict | None = None) -> None:
        options = options or {}
        self.direct = DirectAccess(tools)
        self.core = list(options.get("core", CORE))
        unknown = set(self.core) - set(self.direct.functions)
        if unknown:
            raise ValueError(f"tools.graph.core: no such tools {sorted(unknown)}")
        self.ops = Operations(build(knowledge=options.get("knowledge", True)))
        self.use_sparql = bool(options.get("sparql", False))
        self.use_search = bool(options.get("search", False))  # temporary; see SEARCH
        catalog = options.get("catalog", "names")
        if catalog not in ("names", "summaries"):
            raise ValueError(f"tools.graph.catalog must be names or summaries, not {catalog!r}")
        self.schemas = ([s for s in self.direct.schemas if s["function"]["name"] in self.core]
                        + OPERATIONS + ([SEARCH] if self.use_search else []) + ([SPARQL] if self.use_sparql else []))
        self.first_schemas = [s for s in self.schemas if s["function"]["name"] != "run_tool"]  # before describe
        self.system_text = USAGE.format(
            catalog=self.ops.catalog(leave_out=set(self.core), summaries=catalog == "summaries"),
            listing="names only" if catalog == "names" else "each with a one-line summary",
            search="\nUse search_tools when no name seems to fit." if self.use_search else "",
            sparql=("\n" + SPARQL_USAGE) if self.use_sparql else "")
        self.described = set()  # tools whose description was read in the current request

    def offered(self) -> list[dict]:
        """Without run_tool until a tool has been described in the request (so describe comes first)."""
        return self.schemas if self.described else self.first_schemas

    def begin(self, request: int, text: str) -> None:
        self.direct.begin(request, text)
        self.described = set()

    def execute(self, name: str, arguments: str) -> Outcome:
        if name in self.core:
            return self.direct.execute(name, arguments)
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return Outcome(error(f"arguments are not valid JSON: {exc}"), None)
        if name == "run_tool":
            tool = self.tool_name(str(args.get("name", "")))
            if tool not in self.direct.functions:
                return Outcome(error(f"no tool {tool!r}; did you mean {close(tool, set(self.direct.functions))}?"),
                               None)
            tool_args = args.get("arguments") or {}
            if tool not in self.core and tool not in self.described:
                self.described.add(tool)
                answer = {"not_run": f"{tool} was not run: this is its description (it counts as describe -- do "
                                     "not call describe for it); call run_tool again with arguments that "
                                     "follow it", "description": self.ops.describe([tool])[tool]}
                wrong = argument_problems(self.direct.functions[tool], tool_args)
                if wrong:
                    answer["your_arguments"] = "; ".join(wrong)
                return Outcome(result(answer), None, label=f"{tool}: NOT RUN, description returned", ran=False)
            # Run exactly as access "all" would; history and the run analysis record the engine tool.
            return Outcome(self.direct.run(tool, json.dumps(tool_args)), (tool, tool_args), label=tool)
        if name == "describe":
            names = [self.tool_name(n) for n in str(args.get("tools", "")).split(",") if n.strip()]
            new = [n for n in names if n not in self.described]
            answer = self.ops.describe(new) if new else {}
            for n in names:
                if n not in new:  # its description is already above: not sent again
                    answer[n] = "already described above in this request"
            self.described.update(names)
            return Outcome(result(answer), None, label=f"described {', '.join(new) or '(nothing new)'}")
        if name == "search_tools" and self.use_search:
            found = self.ops.search(str(args.get("request", "")))
            for match in found[:SEARCH_DESCRIBED]:  # the best matches come described, so run_tool can follow
                match["description"] = self.ops.describe([match["tool"]])[match["tool"]]
                self.described.add(match["tool"])
            return Outcome(result(found), None, label="found " + ", ".join(f["tool"] for f in found))
        if name == "sparql" and self.use_sparql:
            answer = self.ops.sparql(str(args.get("query", "")))
            return Outcome(json.dumps(answer) if "error" in answer else result(answer), None)
        return Outcome(error(f"no tool {name!r}: call catalog tools through run_tool"), None)

    def tool_name(self, name: str) -> str:
        """A tool name as the model wrote it, made exact: quotes and spaces removed, and a namespace
        prefix dropped when the rest is a tool (models write e.g. "functions.get_cone")."""
        name = name.strip().strip("'\"` ")
        if name not in self.direct.functions and name.rsplit(".", 1)[-1] in self.direct.functions:
            name = name.rsplit(".", 1)[-1]
        return name


def result(value) -> str:
    return json.dumps({"result": value})


def error(message: str) -> str:
    return json.dumps({"error": message})
