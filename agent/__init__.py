"""The LLM side of circuitAnt: everything that talks to a model about the engine.

    agent.py         the LLM loop: system prompt, history summaries, tool calls
    llm.py           the chat-model clients (OpenAI, Claude) behind one interface
    tools.py         the engine operations offered as tools (one Tools method each) + schema()
    tool_access/     how the model reaches those tools (config tools.access)

eda_engine/ (the netlist engine) knows nothing about this package.
"""
from agent.agent import Agent
