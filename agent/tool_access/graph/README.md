# Tool knowledge graph

The tools in `agent/tools.py` as an RDF graph, for the `graph` tool access mode: the model
finds the tools a request needs by querying the graph instead of receiving every tool
schema with every LLM call.

| File | What it is | Written by |
|---|---|---|
| `ontology.ttl` | the vocabulary: classes and properties | hand |
| `knowledge.ttl` | knowledge that is not in the code: concepts and synonyms (`answeredBy`), `followedBy`, what each kind `denotes` | hand, reviewed by a domain expert |
| `build.py` | builds the graph: the two files above + the facts generated from `agent/tools.py` and `eda_engine/` | code |
| `operations.py` | the questions asked of the graph, each a fixed SPARQL query: catalog, describe, search, free read-only SPARQL | code |
| `access.py` | access mode `graph`: what the model is offered and how its calls are carried out | code |

## Access mode `graph`

The model is offered the graph operations `describe` and `run_tool` (and `sparql` with
`tools.graph.sparql: true`); every tool, the session tools too, is reached through them.
`tools.graph.core` can offer some tools directly instead, but offering the session tools directly
made GPT-4o-mini call them unasked (they were the only "real" tools in view): on test001 the
request "What is the fanout of primary input n1 (reset)?" failed in 3 of 6 runs (it renamed the
testcase instead), and in none of 4 runs with `core: []`, which costs one more LLM call per session
request. Session tool calls are also *grounded* -- refused unless the request names the testcase
or file (`@tool(grounded=...)`) -- so an unasked call cannot rename the testcase or reload or
overwrite the design.
The system prompt gets a short usage note and the **catalog**: the names of every other tool,
one per line (with `tools.graph.catalog: summaries`, each with its one-line summary). The
model describes every candidate that may fit in one `describe` call, picks one from the
descriptions and runs it with `run_tool`. Tool names are taken leniently (quotes, and a namespace
prefix such as `functions.get_cone` that GPT models add, are dropped). The tool list and system
prompt are the same for every request, so prompt caching keeps working.

| Operation | Returns |
|---|---|
| `describe(tools)` | (a tool already described in the request gets "already described above", not its description again) per tool: description, arguments (meaning, required, default, allowed values, keywords), examples with arguments, `use_instead`, `easily_confused_with`, `usually_followed_by`, `changes_design` |
| `run_tool(name, arguments)` | **offered only once a tool has been described in the request** (the first LLM call of a request sees only `describe`; instructions alone did not stop models from skipping it). The tool's result, run exactly as in mode `all`. **Only once the tool's description was read in this request**: otherwise it returns the description (and what is wrong with the arguments given, if anything) and runs nothing (without this, models guessed arguments from the catalog line: wrong argument names, `source` for `sink`, a fanin cone for a fanout one) |
| `search_tools(request)` | **temporary, off by default (`tools.graph.search`), do not use**: keyword matching -- concepts whose label or synonym occurs in the request, then shared words. It finds only what the hand-written synonyms anticipate and follows the wording they were written from; to be replaced by embedding-based search, which a catalog too large for the prompt will need |
| `sparql(query)` | rows of a read-only SELECT / ASK query (anything else is refused) |

History lines and `tool_called` in the run analysis record the engine tools run (`get_cone`,
not `run_tool`); the graph operations go to `graph_ops`.

```bash
./openroad_python.sh agent/tool_access/graph/build.py out.ttl   # the whole graph as Turtle, to read
./openroad_python.sh tests_automation/test_tool_graph.py        # the graph agrees with agent/tools.py
```

Generated facts can't drift from the code: change a docstring, a `Param` or a `@tool`
and the graph follows. Only `knowledge.ttl` is maintained by hand.

## The vocabulary

Prefixes: `ca:` the ontology, `ct:` the facts (tools, parameters, kinds, keywords,
categories, concepts). Standard vocabularies where they fit: RDFS (`rdfs:label`), OWL
(class and property declarations), SKOS (category hierarchy, concept synonyms). Every
node is named with `rdfs:label` and described with `ca:description`.

```
ca:Category ──skos:broader──▶ ca:Category          Analysis ▶ Depth, Cones, Paths, ...
     ▲                                              Transformation ▶ Buffering, Cleanup, Naming, Resynthesis
     │ ca:inCategory                                Verification ▶ Equivalence, Signals;  Session
ca:Tool ──ca:hasParameter──▶ ca:Parameter ──ca:ofKind──▶ ca:ValueKind ──ca:hasKeyword──▶ ca:Keyword
  │  ca:summary, ca:description     ca:position, ca:required,          (Signal, StartPoint, ...)  (ANY_PI, ...)
  │  ca:changesDesign, ca:requiresDesign  ca:default, ca:datatype,     ──ca:denotes──▶ ca:Net, ca:GateInstance, ...
  │  ca:engineFunction              ca:allowedValue                         (design objects: future design graph)
  ├──ca:hasExample──▶ ca:Example ──ca:hasBinding──▶ ca:Binding (ca:forParameter, ca:value)
  ├──ca:hasAlternative──▶ ca:Alternative (ca:useTool, ca:when)
  ├──ca:confusableWith──▶ ca:Tool        (symmetric, from the alternatives)
  └──ca:followedBy──▶ ca:Tool            (hand-written)
ca:Concept ──ca:answeredBy──▶ ca:Tool    (hand-written; synonyms in skos:altLabel)
```

| Fact | Comes from |
|---|---|
| tool, `summary`, `description` | method name; docstring (first sentence; whole) |
| category, its description | the engine package and file the method calls (`analysis/depth.py` → Analysis ▶ Depth); the file's docstring. Methods that call no engine package are Session |
| parameters: position, required, default, datatype | the signature and type hints |
| parameter description, allowed values, kind | `Param` in the type hint |
| keywords (`ANY_PI`, ...) and their meanings | `Param(keywords=...)` of the kind; one meaning per keyword |
| examples with bindings; alternatives | `@tool(examples=..., instead=...)` |
| `changesDesign` | a Transformation, or `read_design` |
| `requiresDesign` | the method calls `self._d()` |
| concepts, synonyms, `followedBy`, `denotes` | `knowledge.ttl` |

## Questions the graph answers that flat tool schemas cannot

```sparql
# Which tools accept ANY_PI?  (through the shared kind, whichever tool it is)
SELECT DISTINCT ?tool ?param WHERE {
  ?t ca:hasParameter ?p ; rdfs:label ?tool .
  ?p rdfs:label ?param ; ca:ofKind/ca:hasKeyword/rdfs:label "ANY_PI" }

# Which tools answer "critical path"?  (concept synonyms)
SELECT ?tool WHERE { ?c skos:altLabel ?l ; ca:answeredBy/rdfs:label ?tool .
  FILTER(CONTAINS(LCASE(?l), "critical path")) }

# After a tool that changes the design, what usually follows?
SELECT ?tool ?next WHERE { ?t ca:changesDesign true ; rdfs:label ?tool ; ca:followedBy/rdfs:label ?next }
```
