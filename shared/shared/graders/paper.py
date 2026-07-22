"""Publication-identity retrieval grader.

Scores search results against a single gold publication described by
``{doi, title}``. Each result is reduced to candidate publication identifiers and
matched to the gold publication in priority order:

1. DOI (explicit field, URL, or text)
2. Fuzzy title similarity (word-level Jaccard >= threshold)

This makes heterogeneous searchers — academic APIs, general web search, or
agents that return arbitrary URLs — comparable apples-to-apples on publication
identity rather than content format. Grading is fully deterministic (no LLM).
"""

import re

from ..searchers import SearchResult
from .base import GradeResult

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;\"'>\]]+", re.IGNORECASE)
TITLE_MATCH_THRESHOLD = 0.70


def _normalize_doi(doi: str) -> str:
    """Lowercase and strip trailing punctuation from a DOI."""
    return doi.lower().rstrip(".,;:)")


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for fuzzy matching."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def _title_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two normalized titles."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _extract_dois(*texts: str) -> set[str]:
    dois: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in DOI_RE.finditer(text):
            dois.add(_normalize_doi(match.group(0)))
    return dois


class PaperRetrievalGrader:
    """Deterministic publication-identity matcher for a single search result.

    Returns ``is_match = 1.0`` when the result identifies the gold publication by
    DOI or fuzzy title, else ``0.0``. The benchmark records the rank of the
    first match per query, from which R@1/R@5/R@10 and MRR fall out.
    """

    def __init__(self, title_match_threshold: float = TITLE_MATCH_THRESHOLD):
        self.title_match_threshold = title_match_threshold

    def match(self, result: SearchResult, gold_paper: dict) -> tuple[bool, str | None]:
        """Return ``(matched, method)`` for one result against the gold publication."""
        gold_doi = _normalize_doi(gold_paper["doi"]) if gold_paper.get("doi") else ""
        gold_title = _normalize_title(gold_paper.get("title", "")) if gold_paper.get("title") else ""

        url = result.url or ""
        title = result.title or ""
        text = result.content or ""
        explicit_doi = str(result.metadata.get("doi", "")) if result.metadata else ""

        if gold_doi:
            cand = _extract_dois(explicit_doi, url, text)
            if explicit_doi:
                cand.add(_normalize_doi(explicit_doi))
            if gold_doi in cand:
                return True, "doi"

        if gold_title and title:
            if _title_similarity(gold_title, _normalize_title(title)) >= self.title_match_threshold:
                return True, "title"

        return False, None

    def grade(self, result: SearchResult, gold_paper: dict) -> GradeResult:
        matched, method = self.match(result, gold_paper)
        return GradeResult(
            scores={"is_match": 1.0 if matched else 0.0},
            details={"match_method": method},
        )
