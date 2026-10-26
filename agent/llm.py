"""Chat-model clients: OpenAI and Claude behind one interface.

The agent keeps its conversation in the OpenAI chat format (system / user /
assistant with tool_calls / tool messages) and tool schemas in the OpenAI
format (tools.schema).  Each client sends that conversation to its model and
returns a Reply; the Claude client translates both ways.

Token counts are reported the same way for both providers: prompt_tokens is
everything sent, cached or not; cached_tokens is the part the provider read from
its prompt cache (billed cheaper).  OpenAI caches long prompt prefixes on its
own; Claude caches only when asked (ClaudeClient's `cache`).
"""
import json
from dataclasses import dataclass, field

MAX_TOKENS = 16000  # Claude needs an explicit output limit


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, as OpenAI returns it


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str
    prompt_tokens: int       # everything sent, cached or not
    cached_tokens: int       # the part read from the prompt cache
    cache_write_tokens: int  # Claude: the part written to the cache (billed at a premium)
    completion_tokens: int
    raw: dict = field(repr=False)  # the provider's full response, for the logs

    def message(self) -> dict:
        """The reply as an OpenAI-format assistant message, to append to the conversation."""
        message = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls]
        return message

    def output_chars(self) -> int:
        """Characters the model wrote: text plus tool-call names and arguments."""
        return len(self.content) + sum(len(c.name) + len(c.arguments) for c in self.tool_calls)


@dataclass
class Usage:
    """One request's model usage, summed over its LLM calls (one per tool round).

    Every call sends the whole conversation again, so input_tokens counts the
    earlier turns once per later call.  The *_chars counts measure the same thing
    independently of any tokenizer: text of the messages and tool schemas sent,
    text and tool calls received."""
    llm_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    input_chars: int = 0
    output_chars: int = 0

    def add(self, reply: Reply, input_chars: int) -> None:
        self.llm_calls += 1
        self.input_tokens += reply.prompt_tokens
        self.cached_input_tokens += reply.cached_tokens
        self.cache_write_tokens += reply.cache_write_tokens
        self.output_tokens += reply.completion_tokens
        self.input_chars += input_chars
        self.output_chars += reply.output_chars()

    def costs(self, price: dict | None) -> tuple[float, float] | None:
        """(list, billed) cost in USD, or None without prices.  price: USD per 1M tokens,
        {input, output, cached_input, cache_write}; the last two default to input.
        List cost bills every input token at the full price (as with no cache);
        billed cost applies the cache prices."""
        if not price:
            return None
        full = price["input"]
        cached, write = price.get("cached_input", full), price.get("cache_write", full)
        uncached = self.input_tokens - self.cached_input_tokens - self.cache_write_tokens
        output = self.output_tokens * price["output"]
        listed = self.input_tokens * full + output
        billed = uncached * full + self.cached_input_tokens * cached + self.cache_write_tokens * write + output
        return listed / 1e6, billed / 1e6


def message_chars(messages: list[dict]) -> int:
    """Characters of text in OpenAI-format messages: contents, tool-call names and arguments."""
    return sum(len(m.get("content") or "")
               + sum(len(c["function"]["name"]) + len(c["function"]["arguments"]) for c in m.get("tool_calls", []))
               for m in messages)


class OpenAIClient:
    def __init__(self, api_key: str, model: str, cache: bool = True) -> None:
        """cache: no effect; OpenAI caches long prompt prefixes automatically."""
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict], tools: list[dict]) -> Reply:
        response = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools)
        choice = response.choices[0]
        usage = response.usage
        details = usage and usage.prompt_tokens_details
        return Reply(
            content=choice.message.content or "",
            tool_calls=[ToolCall(c.id, c.function.name, c.function.arguments or "{}")
                        for c in choice.message.tool_calls or []],
            finish_reason=choice.finish_reason or "?",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            cached_tokens=(details.cached_tokens or 0) if details else 0,
            cache_write_tokens=0,
            completion_tokens=usage.completion_tokens if usage else 0,
            raw=response.model_dump(mode="json"))


class ClaudeClient:
    def __init__(self, api_key: str, model: str, cache: bool = True) -> None:
        """cache: use Claude's prompt cache.  The tool schemas and the first system
        message (the fixed system prompt) are cached for all requests, and each
        call's whole conversation for the next tool round.  Prefixes under the
        model's minimum (4096 tokens on Haiku 4.5) are silently not cached."""
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cache = cache

    def complete(self, messages: list[dict], tools: list[dict]) -> Reply:
        system, converted = _to_claude(messages, self.cache)
        extra = {"cache_control": {"type": "ephemeral"}} if self.cache else {}  # moves with the conversation
        response = self.client.messages.create(
            model=self.model, max_tokens=MAX_TOKENS, system=system, messages=converted,
            tools=[{"name": t["function"]["name"], "description": t["function"]["description"] or "",
                    "input_schema": t["function"]["parameters"]} for t in tools], **extra)
        usage = response.usage
        read, write = usage.cache_read_input_tokens or 0, usage.cache_creation_input_tokens or 0
        return Reply(
            content="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[ToolCall(b.id, b.name, json.dumps(b.input))
                        for b in response.content if b.type == "tool_use"],
            finish_reason=response.stop_reason or "?",
            prompt_tokens=usage.input_tokens + read + write,  # input_tokens leaves the cached part out
            cached_tokens=read,
            cache_write_tokens=write,
            completion_tokens=usage.output_tokens,
            raw=response.model_dump(mode="json"))


def _to_claude(messages: list[dict], cache: bool = False) -> tuple[list[dict], list[dict]]:
    """OpenAI-format messages -> (Claude system prompt blocks, Claude messages).

    System messages become the system prompt, one block each; with cache, the
    first (fixed) one ends a cached prefix.  The results of one assistant turn's
    tool calls go back in a single user message."""
    system, out = [], []
    for m in messages:
        role = m["role"]
        if role == "system":
            system.append({"type": "text", "text": m["content"]})
        elif role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                        "input": json.loads(c["function"]["arguments"] or "{}")}
                       for c in m.get("tool_calls", [])]
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            result = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
    if cache and system:
        system[0]["cache_control"] = {"type": "ephemeral"}
    return system, out


PROVIDERS = {"openai": OpenAIClient, "claude": ClaudeClient}
