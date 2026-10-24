"""Answer natural-language netlist requests, one per stdin line.

    circuitAnt/circuitant -config circuitAnt/config.yaml < testcase/test10/prompt.txt

Each request gets a block on stdout (and in <testcase>.log once the testcase
name is set):

    #RESPONSE <n>
    <answer>
    #END <n>

Settings come from the YAML config (see config.yaml).  With developer mode on,
each request is logged to circuitAnt/debugging/logs/<run>/<case>_request_<n>_<time>.log,
and the tools each request called are recorded
in circuitAnt/debugging/run_analysis_<run>.csv.

stdout carries only the #RESPONSE blocks: everything else written to it
(OpenROAD's INFO/WARNING/ERROR messages, stray prints) goes to openroad.log in the
run's log folder, debugging/logs/<run>/ (one folder per run.sh run, see run.sh).
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agent import Agent  # noqa: E402
from debugging.developer import run_logs_dir  # noqa: E402


def load_config(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    try:
        config = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        sys.exit(f"ERROR: config file not found: {path}")
    openai_cfg = config.get("openai", {})

    # API key: inline, else a key file (relative to the config), else the env.
    key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if openai_cfg.get("api_key_file"):
        key_file = (path.parent / openai_cfg["api_key_file"]).resolve()
        if key_file.exists():
            key = key_file.read_text().strip()
    if not key:
        sys.exit("ERROR: no OpenAI API key (set openai.api_key, openai.api_key_file "
                 "or OPENAI_API_KEY)")
    return config, key


def reserve_stdout(log_path: str | Path):  # -> text stream for responses
    """Return a private stream for responses; send the real stdout to log_path.

    OpenROAD's C++ logger writes straight to file descriptor 1, so the
    redirection is done at the descriptor level, not just on sys.stdout.
    """
    sys.stdout.flush()
    responses = os.fdopen(os.dup(1), "w")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)  # one log for a run's testcases
    os.dup2(fd, 1)
    os.close(fd)
    return responses


def design_config(config: dict, config_path: str | Path) -> dict:
    """The `design` section, with lef / liberty paths made absolute (relative to the config file)."""
    design = dict(config.get("design") or {})
    base = Path(config_path).resolve().parent
    for key in ("lef", "liberty"):
        if key in design:
            files = design[key] if isinstance(design[key], list) else [design[key]]
            design[key] = [str((base / f).resolve()) for f in files]
    return design


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-assisted netlist exploration on OpenDB.")
    ap.add_argument("-config", default=str(HERE / "config.yaml"), help="YAML config file")
    args = ap.parse_args()
    config, key = load_config(args.config)
    out = reserve_stdout(run_logs_dir(config.get("developer", {}).get("log_dir")) / "openroad.log")

    model = config.get("openai", {}).get("model", "gpt-5.4-mini")
    output_dir = Path(".")  # without `output_dir`, outputs go to the current directory
    if config.get("output_dir"):
        output_dir = Path(args.config).resolve().parent / config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
    agent = Agent(key, model, config.get("developer", {}), design_config(config, args.config), output_dir)

    n = 0
    context = ""  # summary of earlier requests, passed into the next one
    for line in sys.stdin:
        request = line.strip()
        if not request:
            continue
        n += 1
        try:
            answer = agent.ask(request, context)
        except Exception as exc:
            answer = f"[ERROR] {type(exc).__name__}: {exc}"
        block = f"#RESPONSE {n}\n{answer}\n#END {n}\n"
        out.write(block)
        out.flush()
        if agent.tools.log:
            agent.tools.log.write(block)
            agent.tools.log.flush()
        context = agent.build_state_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
