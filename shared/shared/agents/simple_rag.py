import json
from dataclasses import asdict, dataclass
from datetime import datetime

from openai import AsyncOpenAI

from ..searchers.base import SearchResult

SIMPLE_RAG_SYSTEM_PROMPT = """\
You are a tabula rasa answer-synthesis engine. You have NO internal knowledge
about the world — no facts, no opinions, no training data to draw from. The ONLY
information that exists for you is the search results provided below.

Today's date: {current_date}

Rules (absolute, no exceptions):
1. You MUST answer using ONLY information explicitly present in the search results.
2. You MUST NOT use any knowledge from your training data, even if you "know" the answer.
3. If the search results do not contain enough information to answer the question,
   reply exactly: "I don't know"
4. Do NOT hedge with "based on my knowledge" or similar — you have no knowledge.
5. Keep your answer short and concise (1-2 sentences max).
6. Prefer information from earlier (higher-ranked) results — they are more relevant.
7. If you cite a fact, it must be traceable to a specific search result.
8. Use today's date to resolve relative time references (e.g. "yesterday", "last week", "3 days ago")."""

SIMPLE_RAG_USER_PROMPT = """\
Question: {question}

Search results:
{search_results}

Answer the question using ONLY the search results above. If they don't contain the answer, say "I don't know"."""


@dataclass
class RAGCitation:
    url: str
    title: str
    text: str


@dataclass
class RAGResult:
    answer: str
    citations: list[RAGCitation]


class SimpleRAGAgent:
    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: str | None = None,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt or SIMPLE_RAG_SYSTEM_PROMPT
        self._client = AsyncOpenAI(api_key=api_key)

    async def synthesize(self, question: str, search_results: list[SearchResult]) -> RAGResult:
        citations = [
            RAGCitation(url=r.url, title=r.title, text=r.text or "\n".join(r.highlights))
            for r in search_results
            if r.text or r.highlights
        ]

        formatted_results = "\n".join(
            json.dumps(asdict(c)) for c in citations
        )

        current_date = datetime.now().strftime("%Y-%m-%d")
        system = self.system_prompt.format(current_date=current_date)
        user = SIMPLE_RAG_USER_PROMPT.format(
            question=question,
            search_results=formatted_results,
        )

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        answer = response.choices[0].message.content or "I don't know"

        return RAGResult(answer=answer, citations=citations)
