import os

from openai import AsyncOpenAI

from .base import Searcher, SearchResult


class PerplexitySearcher(Searcher):
    name = "perplexity"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "sonar",
    ):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY required")

        self.pplx_model = model
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai",
        )

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        response = await self._client.chat.completions.create(
            model=self.pplx_model,
            messages=[{"role": "user", "content": query}],
        )

        answer = response.choices[0].message.content or ""
        citations = getattr(response, "citations", []) or []

        results = []
        for i, url in enumerate(citations):
            results.append(
                SearchResult(
                    url=url if isinstance(url, str) else str(url),
                    title="",
                    text=answer if i == 0 else "",
                    metadata={"rank": i},
                )
            )

        if not results and answer:
            results.append(
                SearchResult(url="", title="", text=answer, metadata={"rank": 0})
            )

        return results
