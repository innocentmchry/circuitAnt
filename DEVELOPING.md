# Developing circuitAnt

How to find and fix a wrong answer, where each kind of change goes, and the rules
the code follows. For what the project is and how to run it, see [README.md](README.md).

## 1. When an answer is wrong: find which layer is at fault

Every wrong answer comes from one of four places. Work through them in order.

1. **Find the request.** With developer mode on, `debugging/run_analysis_<run>.csv`
   has one row per request: the tools called and their exact arguments. Compare
   with `debugging/predicted_run_analysis.csv` (the tool each prompt is expected to
   need — an expectation, not ground truth; see `debugging/README.md`).
2. **Read its log.** `debugging/logs/<run>/<case>_request_<n>_<time>.log` shows
   the tools offered, the context (history) the model saw, every model turn, every
   tool call and the full tool result; the **TRACE** at its end summarises the request
   in one line per LLM call (start there).
3. **Decide which layer is wrong:**

| Symptom | Layer | Where to fix |
|---|---|---|
| The model called the wrong tool, or no tool (e.g. `find_gates` for a cone question) | tool description | the method's docstring, `@tool` examples or `instead` in `agent/tools.py` |
| Right tool, wrong arguments (e.g. PI→PO for "depth of the design") | argument description / defaults | the parameter's `Param` or the `@tool` example's arguments in `agent/tools.py`; give the common case a default argument |
| No tool called, and the answer was made up from incomplete history | history summary | the tool's line in `_summarize` (`agent/agent.py`): it must state complete facts |
| Right tool and arguments, but the tool result is wrong | engine | the function in `eda_engine/` — and add a test |

4. **Fix generically.** Never add a rule for one testcase or one prompt. A fix must
   read as a general statement about the tool or the design ("for the gates *in a
   cone* use get_cone"), not "for test036 call get_cone".
5. **Check it.** Re-run the affected testcases (several times for tool-choice
   problems: the model is not deterministic), and run the test automation.

What *not* to fix in the system prompt: tool-specific routing ("for X use tool Y")
belongs in that tool's docstring; the system prompt only holds rules that apply
to every request. Recovery from tool errors is deliberately not in the prompt
(a separate component will handle it).

### Fixing an engine function

* Reproduce it in a script (`./openroad_python.sh script.py`) on the smallest
  netlist that shows it.
* Fix the function; add a test in `tests_automation/test_engine.py` (small netlists
  live in `tests_automation/`).
* For a transformation, the test must also check `check_equivalence` is PASS and
  that the written netlist reads back.
* Before trusting a new result, check it independently: against an independent
  implementation, with a random simulation, with a theoretical
  bound (e.g. the minimum buffer count), and with a **negative control** — a
  deliberately broken design must be reported as broken.

## 2. Where things go

```
design.py, design_io.py        the Design and reading/writing        (bottom layer)
utils.py                        shared netlist queries
edit.py                         shared netlist edits
blif.py                         BLIF and running ABC
   ▲
   ├── analysis/          queries          may use: utils, design
   ├── verification/      proofs (ABC)     may use: utils, blif, design_io
   └── transformation/    edits            may use: analysis, verification, utils, edit, blif
   ▲   (eda_engine/ never imports from agent/)
agent/                          the LLM side
  tools.py                      one method per agent tool, forwarding to the engine
  tool_access/                  how the model reaches those tools (config tools.access)
    graph/                      the tool knowledge graph, built from tools.py + knowledge.ttl
  llm.py                        the chat-model clients (OpenAI, Claude)
  agent.py                      the LLM loop, system prompt, history lines
```

Knowledge about the circuit (what a netlist contains, how to query or change it) goes in
`eda_engine/`; knowledge about how the model uses the tools (descriptions, examples,
synonyms, which tool to use when) goes in `agent/`. A graph of the design itself would be
an engine exporter; the tool knowledge graph is in `agent/tool_access/graph/`.

| You are adding... | It goes in |
|---|---|
| a new question about the design | a function in the `analysis/` file whose topic it shares (or a new topic file), exported in `analysis/__init__.py` |
| a new change to the design | a function in the matching `transformation/` file, exported in `transformation/__init__.py` |
| a new proof | `verification/` |
| a helper used by **one** function | nested inside that function |
| a query helper used by **two or more** files | `utils.py` (only once the second user exists) |
| an OpenDB editing primitive | `edit.py` |
| anything about cells (what a cell type computes, its pins) | the `Cell` table in `design.py` |
| anything specific to the ICCAD format | `eda_engine/iccad/` only |
| a new netlist format | a new package next to `iccad/` and `generic/`, dispatched from `design_io.py` |
| a new agent tool | a method in `Tools` (`agent/tools.py`) + its history line in `_summarize` (`agent/agent.py`) + its concept in `knowledge.ttl` |
| a new way of offering the tools to the model (e.g. a tool knowledge graph) | a new mode in `agent/tool_access/`, registered in `MODES` |
| a new model provider | a client class in `agent/llm.py`, registered in `PROVIDERS` |
| knowledge for the tool graph that is not in the code (concepts, synonyms, which tool comes next) | `agent/tool_access/graph/knowledge.ttl` (never testcase names) |
| an experiment script (comparing runs, dumping what the model sees) | `experiments/`; its outputs in `experiments/results/` (not in git) |

### Adding an agent tool

1. Write the engine function (above) and test it.
2. Add a method to `Tools` in `agent/tools.py` that forwards to it. The method **is** the
   tool: its name is the tool name, and everything the model is told about it is
   generated from the method (`schema()` for tool access `all`; a tool knowledge
   graph uses the same facts). Keep that knowledge structured, not in prose:
   * **Docstring**: what the tool does and returns. The **first sentence is the
     summary** that stands on its own (it is what a catalog of tools shows; it ends
     at a period followed by two spaces or a line break, so "e.g. g12" is safe).
   * **Parameters**: types `str`, `int`, `bool`, optionally `| None`. A list is passed
     as a comma-separated string and split in the method (see `remap_gates`).
     A parameter without a default is required; give the common case a default
     (e.g. `get_max_depth()` = the whole design).
   * **What each argument is** goes in its type hint, not the docstring:
     `Annotated[str, Param("description", values=(...))]`. Use a shared kind
     (`Signal`, `Gate`, `StartPoint`, `EndPoint`, `GateType`, `GateTypes`, `FilePath`)
     whenever the argument is one; add a kind (with `kind="..."`) when a new kind of
     value is used by more than one tool. Give `values` only for a fixed set.
     Reserved values that name a class of objects go in `keywords=(("ANY_PI", "all
     primary inputs"), ...)`, with one meaning per keyword everywhere (the graph test
     checks it).
   * **The requests it answers** go in `@tool(examples=...)`: phrases, each with the
     arguments to use when they are not obvious (`("register-to-register depth",
     {"source": "ANY_DFF_Q", "sink": "ANY_DFF_D"})`). Use placeholders (X, A, B, g, T),
     never names from a testcase (`test_tool_graph.py` fails on netlist names), and
     write the intent of a kind of request, not a copy of one prompt.
   * **What it is not for** goes in `@tool(instead={"other_tool": "when"})`, in both
     tools of an easily confused pair.
   * If the tool writes long lists, pass `out_dir=self.out_dir`.
   * Check what the model now receives: `./openroad_python.sh experiments/dump_tool_schemas.py`
     (also gives each tool's token count).
3. Add its history line to `_summarize` in `agent/agent.py`: one line with the complete
   facts a later question may ask about (totals before and after, names, not just
   a count). Incomplete history lines make the model invent the rest.
4. In `agent/tool_access/graph/knowledge.ttl`: link it from the concept(s) whose
   requests it answers (`ca:answeredBy`), adding a concept with its synonyms if none
   fits, and add `ca:followedBy` if another tool usually comes next. Everything else
   about the tool reaches the graph from step 2 by itself. Run
   `tests_automation/test_tool_graph.py`: it fails if a tool is reachable from no concept.
5. Add the prompt pattern to the predicted CSV if you track one.

### Reusing code from another layer

* A transformation may call an analysis query directly. Pass `inline_limit=None`
  to get the complete list (and no file) instead of the report the agent sees.
* If two files need the same logic, move it to `utils.py` (queries) or `edit.py`
  (edits) — not before the second user exists, and never into a "misc" or
  "helpers" file.
* When you move code, update both users in the same change and run the test automation.

## 3. Design rules

### Use standard tools; don't reinvent them

* The netlist lives in **OpenDB**, read and linked by **OpenROAD/OpenSTA**, and
  written by OpenROAD. Ports, cells, pins and connectivity are asked from OpenDB.
* Cell facts come from the libraries: pins from LEF, functions and flip-flop pin
  roles from Liberty (through OpenSTA).
* Logic optimisation and proofs use **ABC**; logic moves between OpenDB and ABC
  through **OpenROAD's BLIF writer and reader** (`rmp::create_blif` /
  `blif_dump` / `blif_read`).
* Write our own parser or writer only where no standard tool does the job, keep it
  minimal, and document why at the top of the file. The current exceptions:
  * the ICCAD reader/writer (`iccad/`): the contest format is primitives, which
    standard tools don't read or write that way;
  * the module name of a flattened netlist (`generic/reader.py`): OpenROAD needs it
    to link and cannot list modules before linking;
  * evaluating a Liberty function into a truth table (`Cell.truth_tables`):
    OpenSTA does not expose evaluation;
  * the BLIF written for equivalence (`blif.write_blif`): flip-flop cut points must
    be named after the flip-flop so two designs line up even when the nets around a
    flip-flop changed. (For resynthesis, OpenROAD's writer and reader are used.)
* Try the standard route first and test it on real netlists before deciding it
  doesn't work (e.g. ABC's Verilog reader failed on 31 of 100 testcases; that is
  why equivalence uses BLIF).

### Generic first: cells are known by what they are, not by their names

* Classify cells with the `Cell` table (`design.py`): `kind` (sequential, buffer,
  inverter, constant0/1, combinational, other), `logic` (AND, NAND, …, AOI21, FA —
  from truth tables, any library), `swappable` inputs, flip-flop `data_inputs` (pins
  with a setup check) and `inverted_outputs` (QN).
* Never compare cell names (`"NAND"`, `"DFF"`) or pin names (`"Q"`, `"D"`, `"A"`,
  `"Y"`) outside `iccad/`. Ask the cell table instead.
* Never assume one output per gate (`getFirstOutput()` on an instance): flip-flops
  have Q and QN, adders have CO and S.
* Look names up with `Design.inst()` / `Design.net()`: they accept both `r3_reg[0]`
  and OpenDB's escaped form.
* Every new feature should be checked on a generic netlist (`examples/`, Nangate45)
  as well as an ICCAD one.

### Files and functions

* Public functions are complete queries or transformations, each called by an agent
  tool (or by another layer as described above).
* A helper used by one function is nested inside it, even if long.
* No `_private` module-level functions; no misc/helpers files. If nothing fits, a
  new topic file, even for one function.
* Results are plain dicts. Long lists: the first `inline_limit` items inline, the
  full list written to a file (`file_path`).
* Match the surrounding code: type hints, short docstrings that say what is
  returned, comments only where something is not obvious.
* Keep it simple: no extra options, layers or configuration until something needs
  them. Measure before optimising (the constant-net cache stayed because the
  alternative was 20× slower). (`agent/tool_access/` is such a layer: it exists because the
  ways of offering tools are compared against each other.)

### Editing the netlist

* All OpenDB edits go through `edit.py`: unique, Verilog-safe names; a net on a
  port keeps its name; `Design.constant_nets` stays correct. The one exception is
  resynthesis, where OpenROAD's BLIF reader edits the netlist itself and
  `Design.scan_constants()` is called afterwards.
* Results of `analysis` and `utils.netlist_view` are snapshots: after an edit,
  query again.
* Transformations edit in place and return counts (removed, added, before/after);
  they do not verify themselves — the agent calls `check_equivalence` when asked.
* Flip-flops are never removed or renamed: equivalence is checked at flip-flop
  boundaries with flip-flops matched by instance name.

### Formats

* ICCAD-specific code and data live only in `eda_engine/iccad/` (the primitive
  cells, the primitive ↔ cell translation, the exact module header).
* The generic reader only calls OpenROAD; it expects a **flattened** netlist (one
  module, the top). Ports come from OpenSTA (`Design.port_list()`);
  `Design.source_header` is kept only by readers whose writer must reproduce it.

### The agent

* The system prompt (`agent/agent.py`) is short and generic: rules that hold for every
  request. No testcase-specific or tool-specific routing there.
* Tool choice is steered by tool docstrings and argument defaults; answers from
  history are made reliable by complete history lines.
* The model answers from history only when the history states the exact value;
  "now" questions call a tool.
* Tool errors are reported to the model as `{"error": ...}`; how to recover is left
  to a future component, not the prompt.
* A call identical to one already made in the request (same tool and arguments) is
  refused unless the design changed since, and the request stops after
  `MAX_STUCK_TURNS` consecutive LLM calls whose tool calls all fail (repeats, refusals,
  errors; an error the model then fixes resets the count): models otherwise loop until
  `MAX_TOOL_ROUNDS` (seen: 8 parallel `set_testcase_name` calls per turn for 20 turns,
  and a new refused testcase name in every turn).
* Tools that reset the session (`set_testcase_name`, `read_design`, `write_design`)
  declare `@tool(grounded=(...))`: the call is refused unless the request itself names
  that argument (the testcase, the file name). Models called them unasked -- renaming
  the testcase (its responses then went to another log), reloading the design (undoing
  every change), writing over the input netlist.
* Calls with unknown or missing argument names are refused before the tool runs, with
  the tool's arguments listed (models guess names such as `file_path` for `path`).
* `tests_automation/test_agent.py` checks these safeguards with a scripted model.
* How the model reaches the tools is the access mode (`agent/tool_access/`); the engine,
  `agent/tools.py`, the providers and the history are the same in every mode, so modes can be
  compared. A mode may add generic usage text to the system prompt (e.g. how to browse a
  tool knowledge graph), never routing for particular requests.
* History lines and the run analysis's `tool_called` record only engine tools, in every
  mode; other tool calls (navigation) go to `graph_ops`.

### Outputs and logs

* Everything a run produces goes to `output_dir` (testcase logs, netlists, result
  lists) or `debugging/logs/<run>/` (OpenROAD log, request logs); never into the
  source tree.
* Result-list files are named `<case>_r<request>_<name>` so runs and requests never
  overwrite each other; a run's logs and tool-call CSV share its run id.
* Tests run in a temporary directory and leave nothing behind.

### Experiments

* Compare modes, models or providers by separate runs, one at a time, each repeated
  (the model is not deterministic, and Claude takes no temperature setting), on
  testcases not used to tune what is being compared: testcases 1-50 may inform tool
  descriptions and `knowledge.ttl`; 51-100 are held out (look at results, never tune on
  them). One config per arm in `experiments/configs/` (`CIRCUITANT_CONFIG`), differing
  only in what is compared; name runs with `CIRCUITANT_RUN`; summarise with
  `experiments/compare_runs.py`.
* Every developer-mode run writes `debugging/run_info_<run>.json` (settings and git
  commit); keep it with the run's CSV.
* A restructuring that should not change behaviour is checked by comparing the request
  payloads before and after (same requests, same tool results, byte-identical messages
  and tools), not only by re-running testcases.

## 4. Process

* Discuss a change before implementing it when it changes the design (new
  dependencies, new file structure, behaviour visible to the model).
* Test on real testcases, not only on small hand-written netlists, and include
  negative controls.
* After any change: `./openroad_python.sh tests_automation/test_engine.py` must pass; after
  a change to `agent/tools.py` or the tool graph, `tests_automation/test_tool_graph.py`
  too; after a change to the agent loop, `tests_automation/test_agent.py`; for agent changes, re-run the affected testcases with `./run.sh` and check the
  run's CSV and logs.

## 5. Known limitations

* The resynthesis tools depend on OpenROAD's BLIF commands, which OpenROAD labels
  "exposed for testing".
* ABC cannot model tri-state cells; equivalence on designs using them reports
  ABC's error.
* ABC needs an inverter to map; NAND-only / NOR-only results use the standard
  circuits in `resynthesis.DECOMPOSITIONS`.
* Very large designs: ABC's depth optimisation took ~160 s on 213k gates (timeout
  260 s).
* Escaped instance names are shown as OpenDB stores them (`r3_reg\[0\]`).
* The model (`gpt-4o-mini` by default) does not always choose the same tool for the
  same question; check fixes with repeated runs.
