"""Provider-neutral types shared by every model client.

The Scout loop, the single-step RAG system, and the suite judges all speak this
vocabulary. Conversations are Chat Completions-shaped message lists
(``system`` / ``user`` / ``assistant`` with ``tool_calls`` / ``tool``); each
client translates them to its provider's wire format on every call, so the loop
never branches on the provider.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Usage:
    """Token counters for one or more model calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCall:
    """One function call requested by the model. ``arguments`` is raw JSON text."""

    id: str
    name: str
    arguments: str


@dataclass
class HostedSearch:
    """Web-search activity the model provider executed inside one generation.

    ``sources`` holds one ``{"url", "title", "text"}`` dict per distinct URL the
    provider surfaced (search results, opened pages, or answer citations).
    """

    num_searches: int = 0
    queries: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Generation:
    """The normalized outcome of one model call.

    ``cost_usd`` covers tokens only, priced from ``harness.llm.pricing``; it is
    ``None`` when the model has no known price. Hosted-search fees are priced
    separately by the caller from ``hosted_search.num_searches``.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    hosted_search: HostedSearch | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    reasoning: list[str] = field(default_factory=list)
    raw: Any = None


class ContextWindowExceeded(RuntimeError):
    """Raised by a client when the prompt no longer fits the model's context window."""
