"""How the model reaches the engine tools: which tools it is offered, and how a call is run.

Each mode is a class with
  system_text                       text added to the system prompt ("" for none)
  offered() -> [tool schema]        the tools sent with the next LLM call (the same list object for the
                                    same tools: each distinct list is a stable, cacheable prompt prefix)
  begin(request, text)              a new request: its number and text
  execute(name, arguments) -> Outcome   run one tool call of the current request
The agent, the engine and tools.py are the same in every mode, so modes can be compared.

Modes (config `tools.access`):
  all    every Tools method as a tool (direct.py)
  graph  core tools + operations on the tool knowledge graph (graph/access.py)
"""
from agent.tool_access.direct import DirectAccess, Outcome
from agent.tools import Tools


def graph_access(tools: Tools, options: dict | None = None):
    from agent.tool_access.graph.access import GraphAccess  # imported only when used (needs rdflib)
    return GraphAccess(tools, options)


MODES = {"all": DirectAccess, "graph": graph_access}


def make_access(config: dict | None, tools: Tools):
    """The access mode named by the config's `tools` section ({"access": mode, <mode>: options})."""
    config = config or {}
    mode = config.get("access", "all")
    if mode not in MODES:
        raise ValueError(f"unknown tools.access {mode!r} (use one of {', '.join(MODES)})")
    return MODES[mode](tools, config.get(mode))
