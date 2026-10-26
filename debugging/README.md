# debugging/

Tools for checking what the agent does. See [../DEVELOPING.md](../DEVELOPING.md),
section 1, for how to use them when an answer is wrong.

| File | What it is |
|---|---|
| `developer.py` | developer-mode logging: per-request logs and the tool-call CSV |
| `logs/<run>/` | one folder per run: `openroad.log` and, in developer mode, one log per request (not in git) |
| `run_analysis_<run>.csv` | developer mode: the tools each request **actually** called, with their arguments, and its model usage (not in git; see below) |
| `run_info_<run>.json` | developer mode: the settings the run used (provider, model, `tools`, design format) and the git commit, so a CSV can be traced to its setup |
| `predicted_run_analysis.csv` | the tool each ICCAD testcase prompt is **expected** to need |

## `predicted_run_analysis.csv` is an expectation, not ground truth

It was written from the prompt wording (one row per prompt line of the 100 ICCAD
testcases), not taken from a verified run, and it records *which tool* — not the
answer. Read it as "the tool we expect", for spotting surprising tool choices:

* **Other choices can be equally correct.** "How many NAND gates are in the design
  now?" can be answered by `count_gates` or `find_gates`; "What is the total gate
  count now?" by `count_gates` or from history when the previous answer stated it.
  Redundant gates (test017) can be found with `find_duplicates` +
  `merge_duplicate_gates` when there are no dangling gates.
* **Extra calls are fine.** A run may also call `check_equivalence` after a
  transformation, or a `find_*` query before one; the file lists only the tools the
  answer needs.
* `(history)` means the answer is expected from an earlier result in the same
  testcase (e.g. "How many dangling gates were removed?"), with no tool call.
* Several expected calls in one request are joined with ` ; ` in call order, as in
  `run_analysis_<run>.csv`.

So a difference between the two files is a prompt to look at the run's log, not
proof of a mistake; and agreement is not proof the answer is right — check the
answer itself.

## Navigation, token and cost columns of `run_analysis_<run>.csv`

Each request is a loop of LLM calls, one per tool round, and every call sends the whole
conversation again (tool schemas, system prompt, history summary, request, earlier tool
calls and results). The columns are summed over all of a request's calls:

| Column | Meaning |
|---|---|
| `graph_ops` | tool calls that ran no engine tool (e.g. tool knowledge graph navigation); empty in `tools.access: all`. `tool_called` lists only engine tools in every mode, so it stays comparable with `predicted_run_analysis.csv` |
| `llm_calls` | LLM calls in the request (tool rounds + the final answer) |
| `seconds` | wall-clock time from the request to its answer: all LLM calls and tool runs |
| `input_tokens` | everything sent, **cached or not**, the same for every provider |
| `cached_input_tokens` | the part read from the provider's prompt cache (billed cheaper; answers unchanged) |
| `output_tokens` | tool calls and text the model wrote |
| `input_chars`, `output_chars` | the same as text characters: tokenizer-free, comparable across models |
| `cost_list_usd` | every token at the full price, as with no cache: the fair cross-model cost |
| `cost_billed_usd` | with the cache prices (Claude cache writes at their premium) |

Prices come from `<provider>.price` in `config.yaml`; without them the run has no cost
columns at all. A request that fails part-way still gets a row with what it spent. Each request
log ends with the same totals (`REQUEST USAGE`); its per-call lines show where they come from.
