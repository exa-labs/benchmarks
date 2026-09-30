"""Model-client interface and the provider-prefixed factory.

Models are named ``<provider>/<model>`` (``openai/gpt-5.6-luna``,
``anthropic/claude-sonnet-5``). Each provider client reads its standard SDK
environment variables (``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``,
``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``), so any compatible gateway
works without code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from harness.llm.types import Generation

SUPPORTED_PROVIDERS = ("openai", "anthropic")


class ModelClient(ABC):
    """One provider model behind the harness's provider-neutral call shape."""

    provider: str
    # Whether `generate` enforces `response_schema` natively (structured outputs).
    supports_response_schema: bool = False

    def __init__(self, model: str) -> None:
        provider, _, name = model.partition("/")
        if provider != self.provider or not name:
            raise ValueError(
                f"{type(self).__name__} expects '{self.provider}/<model>', got {model!r}"
            )
        self.model = model
        self.model_name = name

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        hosted_web_search: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> Generation:
        """Run one model call.

        ``messages`` is a Chat Completions-shaped conversation. ``tools`` are flat
        function definitions (``{"name", "description", "parameters"}``).
        ``hosted_web_search`` enables the provider's own web-search tool with the
        given provider options (``{}`` for defaults). ``response_schema``
        (``{"name", "schema"}``) constrains the reply to JSON on clients that
        support it and is ignored elsewhere. Raises
        ``ContextWindowExceeded`` when the prompt does not fit.
        """


def provider_of(model: str) -> str:
    """Return the provider prefix of a model name, validating it."""
    provider, separator, name = model.partition("/")
    if not separator or not name:
        raise ValueError(
            f"model names are provider-prefixed, e.g. 'openai/gpt-5.6-luna'; got {model!r}"
        )
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported model provider {provider!r}; expected one of {SUPPORTED_PROVIDERS}"
        )
    return provider


def create_client(model: str, **options: Any) -> ModelClient:
    """Build the client for a provider-prefixed model name."""
    provider = provider_of(model)
    if provider == "openai":
        from harness.llm.openai import OpenAIClient

        return OpenAIClient(model, **options)
    from harness.llm.anthropic import AnthropicClient

    return AnthropicClient(model, **options)
