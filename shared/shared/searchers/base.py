from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    url: str = ""
    title: str = ""
    text: str = ""
    highlights: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Searcher(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        pass

    async def extract(self, url: str, query: str | None = None) -> list[SearchResult]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support URL extraction")
