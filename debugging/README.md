# debugging/

Tools for checking what the agent does. See [../DEVELOPING.md](../DEVELOPING.md),
section 1, for how to use them when an answer is wrong.

| File | What it is |
|---|---|
| `developer.py` | developer-mode logging: per-request logs and the tool-call CSV |
| `logs/<run>/` | one folder per run: `openroad.log` and, in developer mode, one log per request (not in git) |
| `run_analysis_<run>.csv` | developer mode: the tools each request **actually** called, with their arguments (not in git) |
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
