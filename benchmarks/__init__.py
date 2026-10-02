"""Name-to-suite registry for every benchmark the harness can run."""

from __future__ import annotations

from benchmarks.browsecomp import BrowseComp
from benchmarks.company_findall import CompanyFindAll
from benchmarks.dsqa import DeepSearchQA
from benchmarks.exa import (
    CompanyRAG,
    CompanyRetrieval,
    People,
    Publication,
    PublicationToT,
    WebCodeHighlights,
    WebCodeRAG,
)
from benchmarks.frames import Frames
from benchmarks.swechatsearches import SWEChatSearches
from benchmarks.widesearch import WideSearch
from harness.suite import Suite

SUITES: dict[str, type[Suite]] = {
    suite.name: suite
    for suite in (
        BrowseComp,
        Frames,
        DeepSearchQA,
        WideSearch,
        People,
        CompanyRetrieval,
        CompanyRAG,
        CompanyFindAll,
        Publication,
        PublicationToT,
        SWEChatSearches,
        WebCodeRAG,
        WebCodeHighlights,
    )
}


def list_suites() -> list[str]:
    """Names of every registered suite, sorted."""
    return sorted(SUITES)


def get_suite(name: str) -> Suite:
    """Instantiate a suite by name; loading its data happens later in ``Suite.load``."""
    try:
        return SUITES[name]()
    except KeyError:
        raise ValueError(f"unknown suite {name!r}; available: {', '.join(list_suites())}") from None
