"""Mode "all": every public Tools method is offered to the model as a tool and run directly."""
import inspect
import json
import typing
from dataclasses import dataclass
from pathlib import Path

from agent.tools import Tools, parameter, schema

MAX_RESULT_CHARS = 20000


@dataclass
class Outcome:
    """What one tool call did."""
    result: str                           # JSON {"result": ...} or {"error": ...}, given back to the model
    engine_call: tuple[str, dict] | None  # the Tools method run and its arguments; None for a step that
                                          # ran none (e.g. browsing a tool knowledge graph)
    label: str = ""                       # what happened, for the request log's trace (default: the call)
    ran: bool = True                      # False: the call did nothing (so repeating it is not a repeat)


class DirectAccess:
    """The model sees every tool, with every LLM call, and calls them by name."""

    system_text = ""  # nothing to add to the system prompt

    def __init__(self, tools: Tools, options: dict | None = None) -> None:
        self.tools = tools
        self.functions = {name: fn for name, fn in inspect.getmembers(tools, inspect.ismethod)
                          if not name.startswith("_")}
        self.schemas = [schema(fn) for fn in self.functions.values()]
        self.request, self.request_text = 0, ""

    def begin(self, request: int, text: str) -> None:
        """A new request: its number (for file names) and its text (grounded arguments must occur in it)."""
        self.request, self.request_text = request, text

    def offered(self) -> list[dict]:
        """The tool schemas sent with the next LLM call (here: always all of them)."""
        return self.schemas

    def focused(self, names: list[str]) -> list[dict]:
        """The schemas of the named tools only (the short first turn of a cache hit)."""
        return [s for s in self.schemas if s["function"]["name"] in names]

    def as_call(self, name: str, arguments: str) -> tuple[str, str]:
        """A call of a focused tool as the full tool list makes it (here: the same call)."""
        return name, arguments

    def execute(self, name: str, arguments: str) -> Outcome:
        """Run one tool call of the current request (arguments: its JSON text)."""
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = None
        return Outcome(self.run(name, arguments), (name, args if args is not None else {"arguments": arguments}))

    def run(self, name: str, arguments: str) -> str:
        """JSON {"result": ...} or {"error": ...} of a Tools method."""
        try:
            args = json.loads(arguments or "{}")
            fn = self.functions[name]
            wrong = argument_problems(fn, args)
            if not wrong:
                args, wrong = typed_arguments(fn, args)
            if wrong:
                return json.dumps({"error": f"{name} not run: " + "; ".join(wrong)})
            for arg in getattr(fn, "grounded", ()):  # must be named by the request itself
                value = args.get(arg)
                if value and Path(str(value)).name.lower() not in self.request_text.lower():
                    return json.dumps({"error": f"{name} refused: the request does not name {value!r}. "
                                                f"{name} is only for a request that names its {arg}; the current "
                                                f"testcase and loaded design are in the context."})
            value = fn(**args)
            if isinstance(value, dict) and value.get("file_path"):  # a full list written to a file:
                path = Path(value["file_path"])  # name it after the testcase and request, so it is unique
                unique = path.with_name(f"{self.tools.case or 'design'}_r{self.request:02d}_{path.name}")
                path.replace(unique)
                value["file_path"] = str(unique)
            result = {"result": value}
        except Exception as exc:  # report tool errors to the model, not the user
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result)[:MAX_RESULT_CHARS]


def argument_problems(fn, args: dict) -> list[str]:
    """What is wrong with the argument names of a call: unknown or missing required ones, with the
    tool's arguments listed so the model can correct them ([] if nothing)."""
    params = inspect.signature(fn).parameters
    listing = ", ".join(f"{n} (required)" if p.default is p.empty else n for n, p in params.items()) or "none"
    problems = [f"unknown argument {a!r}" for a in args if a not in params]
    problems += [f"missing required argument {n!r}" for n, p in params.items()
                 if p.default is p.empty and n not in args]
    return problems + [f"its arguments are: {listing}"] if problems else []


def typed_arguments(fn, args: dict) -> tuple[dict, list[str]]:
    """The arguments with the types the tool declares, and what cannot be converted ([] if nothing).  A
    boolean or integer written as text ("true", "4") is converted: in graph mode run_tool's arguments are
    a free object whose types nothing enforces, and any non-empty text is true in Python, so
    collapse="False" would collapse."""
    hints = typing.get_type_hints(fn, include_extras=True)
    typed, problems = dict(args), []
    for name, value in args.items():
        base, _ = parameter(hints[name])
        if value is None or base is str:
            continue
        if base is bool:
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                typed[name] = value.strip().lower() == "true"
            elif not isinstance(value, bool):
                problems.append(f"{name} must be true or false, not {value!r}")
        elif base is int:
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                typed[name] = int(value)
            elif isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"{name} must be an integer, not {value!r}")
    return typed, problems
