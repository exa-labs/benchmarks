"""Name-to-suite registry for every benchmark the harness can run."""

from __future__ import annotations

from harness.suites.base import Suite
from harness.suites.browsecomp import BrowseComp
from harness.suites.dsqa import DeepSearchQA
from harness.suites.frames import Frames
from harness.suites.local import (
    CompanyRAG,
    CompanyRetrieval,
    People,
    Publication,
    PublicationToT,
    WebCodeHighlights,
    WebCodeRAG,
)
from harness.suites.widesearch import WideSearch

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
        Publication,
        PublicationToT,
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
