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
from debugging.developer import run_logs_dir, write_run_info  # noqa: E402


# Per provider: the names its key may have in the key file or the environment.
KEY_NAMES = {"openai": ("OPENAI_API_KEY",), "claude": ("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")}
DEFAULT_MODELS = {"openai": "gpt-5.4-mini", "claude": "claude-haiku-4-5"}


def read_key_file(path: Path) -> dict[str, str]:
    """KEY=value lines (quotes optional, # comments skipped); a file holding only a bare key gives {"": key}."""
    keys = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if not sep:
            keys[""] = line
            continue
        keys[name.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return keys


def load_config(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    try:
        config = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        sys.exit(f"ERROR: config file not found: {path}")
    provider = config.setdefault("provider", "openai")
    if provider not in KEY_NAMES:
        sys.exit(f"ERROR: unknown provider {provider!r} (use one of {', '.join(KEY_NAMES)})")
    section = config.get(provider) or {}

    # API key: inline, else the key file (relative to the config), else the env.
    names = KEY_NAMES[provider]
    key = section.get("api_key")
    key_file = config.get("api_key_file") or section.get("api_key_file")
    if not key and key_file and (path.parent / key_file).exists():
        keys = read_key_file((path.parent / key_file).resolve())
        key = next((keys[n] for n in names if keys.get(n)), None)
        if not key and provider == "openai":
            key = keys.get("")  # old format: the file is just the OpenAI key
    key = key or next((os.environ[n] for n in names if os.environ.get(n)), None)
    if not key:
        sys.exit(f"ERROR: no {provider} API key (set {' or '.join(names)} in {key_file or 'api_key_file'}, "
                 f"{provider}.api_key, or the environment)")
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

    provider = config["provider"]
    model = (config.get(provider) or {}).get("model", DEFAULT_MODELS[provider])
    output_dir = Path(".")  # without `output_dir`, outputs go to the current directory
    if config.get("output_dir"):
        output_dir = (Path(args.config).resolve().parent / config["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    section = config.get(provider) or {}
    agent = Agent(key, model, config.get("developer", {}), design_config(config, args.config), output_dir,
                  provider, section.get("cache", True), section.get("price"), config.get("tools"),
                  cache_config=config.get("cache"))
    if agent.run_analysis:  # developer mode: record what this run used (no keys)
        write_run_info(agent.logs_dir.name, {
            "config": str(Path(args.config).resolve()), "provider": provider, "model": model,
            provider: {k: v for k, v in section.items() if k != "api_key"},
            "tools": config.get("tools") or {"access": "all"}, "design": config.get("design") or {}})

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
