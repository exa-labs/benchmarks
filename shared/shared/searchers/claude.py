import os

from .base import Searcher, SearchResult

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


class ClaudeWebFetchSearcher(Searcher):
    name = "claude"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 16384,
        tool_version: str = "web_fetch_20250910",
    ):
        if anthropic is None:
            raise ImportError("anthropic package required for ClaudeWebFetchSearcher")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.model = model
        self.max_tokens = max_tokens
        self.tool_version = tool_version
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        raise NotImplementedError("ClaudeWebFetchSearcher only supports extract")

    async def extract(self, url: str, query: str | None = None) -> list[SearchResult]:
        if self.tool_version == "web_fetch_20250910":
            prompt = f"Fetch the content at {url}"
        else:
            prompt = (
                f"Fetch {url} and extract the content most relevant to this question: {query}"
                if query
                else f"Fetch the content at {url}"
            )

        tool_def: dict = {"type": self.tool_version, "name": "web_fetch"}
        if self.tool_version == "web_fetch_20250910":
            tool_def["max_uses"] = 1

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=[tool_def],
            messages=[{"role": "user", "content": prompt}],
        )

        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "web_fetch_tool_result":
                inner = getattr(block, "content", None)
                if isinstance(inner, list):
                    for sub in inner:
                        if hasattr(sub, "text"):
                            text_parts.append(sub.text)
                elif isinstance(inner, str):
                    text_parts.append(inner)
            elif hasattr(block, "text"):
                text_parts.append(block.text)

        text = "\n".join(text_parts)

        return [
            SearchResult(
                url=url,
                title="",
                text=text,
                metadata={"text_length": len(text)},
            )
        ]
