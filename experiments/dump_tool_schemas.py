"""Dump what the model receives: the tool schemas and one full request.

    ./openroad_python.sh experiments/dump_tool_schemas.py [request log]

Writes, into results/ next to this file:
  tool_schemas_openai.json   the tools exactly as agent/agent.py sends them (OpenAI format)
  tool_schemas_claude.json   the same tools as agent/llm.py converts them for Claude
  tool_schemas.md            one section per tool, with its Claude token count
  request_example_openai.json  call 1 and call 2 of the given request log, as agent/agent.py sends
                               them to OpenAI (default: the newest test003 request 10 log)
  request_example_claude.json  the same two calls as agent/llm.py converts them for Claude
Token counts use Anthropic's free count_tokens endpoint (needs the Claude key).
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "results"
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import anthropic  # noqa: E402
from agent import Agent  # noqa: E402
from agent.agent import SYSTEM_PROMPT  # noqa: E402
from agent.llm import _to_claude  # noqa: E402
from main import read_key_file  # noqa: E402

MODEL = "claude-haiku-4-5"


def claude_tools(schemas: list[dict]) -> list[dict]:
    return [{"name": t["function"]["name"], "description": t["function"]["description"] or "",
             "input_schema": t["function"]["parameters"]} for t in schemas]


def request_from_log(path: Path) -> list[dict]:
    """[call 1 messages, call 2 messages] (OpenAI format) of a one-tool-round request log."""
    log = path.read_text().splitlines()
    text = "\n".join(log) + "\n"
    request = log[log.index("┌─ USER INPUT ─────────────────────────────") + 1]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if "  ── CONTEXT ──" in log:
        i = log.index("  ── CONTEXT ──")
        j = log.index("  ─────────────", i)
        messages.append({"role": "system", "content": "Current state: " + "\n".join([log[i + 1][2:]] + log[i + 2:j])})
    messages.append({"role": "user", "content": request})
    name = re.search(r"│     tool_call: (\w+)", text).group(1)
    args = re.search(r"│     arguments:\n((?:│       .*\n)+)", text).group(1).replace("│       ", "")
    call_id = re.search(r'"id": "((?:toolu|call)_\w+)"', text).group(1)
    result = re.search(r"role: tool\n│     content \(JSON\):\n((?:│       .*\n)+)", text).group(1).replace("│       ", "")
    call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(json.loads(args))}}]}
    return [messages, messages + [call, {"role": "tool", "tool_call_id": call_id,
                                         "content": json.dumps(json.loads(result))}]]


def main() -> None:
    HERE.mkdir(exist_ok=True)
    keys = read_key_file(ROOT / ".env")
    key = keys.get("CLAUDE_API_KEY") or keys.get("ANTHROPIC_API_KEY")
    agent = Agent("unused", MODEL, provider="claude")  # only for its tool schemas
    schemas = agent.access.offered()
    tools = claude_tools(schemas)
    (HERE / "tool_schemas_openai.json").write_text(json.dumps(schemas, indent=2) + "\n")
    (HERE / "tool_schemas_claude.json").write_text(json.dumps(tools, indent=2) + "\n")

    client = anthropic.Anthropic(api_key=key)
    msgs = [{"role": "user", "content": "hi"}]
    count = lambda **kw: client.messages.count_tokens(model=MODEL, messages=msgs, **kw).input_tokens
    base, total = count(), count(tools=tools)
    alone = {t["name"]: count(tools=[t]) - base for t in tools}
    preamble = round((sum(alone.values()) - (total - base)) / (len(tools) - 1))  # added once per request

    lines = [f"# Tool schemas the model receives ({len(tools)} tools)", "",
             f"Claude tokens ({MODEL}): all tools = {total - base}, of which ~{preamble} are the tool-use",
             "preamble Claude adds once; the rest are the schemas below (sent again on every LLM call).", "",
             "| tool | tokens | required | optional |", "|---|---|---|---|"]
    for t in sorted(tools, key=lambda t: -alone[t["name"]]):
        props, req = t["input_schema"]["properties"], t["input_schema"]["required"]
        lines.append(f"| `{t['name']}` | {alone[t['name']] - preamble} | {', '.join(req) or '–'} | "
                     f"{', '.join(p for p in props if p not in req) or '–'} |")
    for t in tools:  # in the order the model receives them
        props, req = t["input_schema"]["properties"], t["input_schema"]["required"]
        lines += ["", f"## `{t['name']}`  (~{alone[t['name']] - preamble} tokens)", "",
                  "Arguments: " + (", ".join(f"`{p}`: {v['type']}{'' if p in req else ' (optional)'}"
                                             for p, v in props.items()) or "none"), "", t["description"]]
    (HERE / "tool_schemas.md").write_text("\n".join(lines) + "\n")

    logs = sorted(ROOT.glob("debugging/logs/*/test003_request_0010_*.log"))
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else (logs[-1] if logs else None)
    if log:
        openai, claude = {"source_log": str(log)}, {"source_log": str(log)}
        for n, messages in enumerate(request_from_log(log), 1):
            openai[f"call_{n}"] = {"model": "<openai.model>", "messages": messages,
                                   "tools": f"<the {len(tools)} tools in tool_schemas_openai.json>"}
            system, converted = _to_claude(messages, cache=True)
            claude[f"call_{n}"] = {"model": MODEL, "max_tokens": 16000, "cache_control": {"type": "ephemeral"},
                                   "tools": f"<the {len(tools)} tools in tool_schemas_claude.json>",
                                   "system": system, "messages": converted}
        (HERE / "request_example_openai.json").write_text(json.dumps(openai, indent=2) + "\n")
        (HERE / "request_example_claude.json").write_text(json.dumps(claude, indent=2) + "\n")
    print(f"wrote {HERE}: {len(tools)} tools, {total - base} tokens (preamble ~{preamble})")


if __name__ == "__main__":
    main()
