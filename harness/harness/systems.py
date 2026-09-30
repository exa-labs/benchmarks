"""The systems catalog: parse ``systems.toml`` and build runnable systems.

A catalog entry resolves to a ``SystemSpec`` — a plain, hashable description of
everything that affects results (model, generation settings, budgets, search
backend and its request settings). The runner hashes the resolved spec into the
run directory name, so changing any of it starts a fresh run instead of mixing
results.
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.searchers import (
    BraveSearcher,
    ExaSearcher,
    ParallelSearcher,
    PerplexitySearcher,
    Searcher,
)

from harness.llm.clients import provider_of
from harness.rag import SingleStepRAG
from harness.scout import Scout, ScoutConfig
from harness.tools import SearchTool

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "systems.toml"
SYSTEM_KINDS = ("scout", "rag")
_SCOUT_FIELDS = {
    "reasoning_effort",
    "temperature",
    "max_output_tokens",
    "max_rounds",
    "max_tool_calls_per_round",
    "max_searches",
    "max_cost_usd",
    "max_no_tool_nudges",
    "budget_warning_fraction",
    "timeout_s",
}
_RAG_FIELDS = {"reasoning_effort", "max_output_tokens"}


def build_searcher(provider: str, options: dict[str, Any]) -> Searcher:
    """Construct a searcher from a catalog entry's provider and options."""
    options = dict(options)
    if provider == "exa":
        return ExaSearcher(search_type=options.pop("type", "auto"), **options)
    if provider == "brave":
        return BraveSearcher(**options)
    if provider == "parallel":
        return ParallelSearcher(**options)
    if provider == "perplexity":
        return PerplexitySearcher(**options)
    raise ValueError(f"unknown search provider {provider!r}")


@dataclass(frozen=True)
class SystemSpec:
    """A fully resolved system: everything that determines its behavior."""

    name: str
    kind: str
    model: str
    settings: dict[str, Any] = field(default_factory=dict)
    searcher: dict[str, Any] | None = None
    hosted_web_search: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "settings": self.settings,
            "searcher": self.searcher,
            "hosted_web_search": self.hosted_web_search,
        }


class Catalog:
    """Searcher and system definitions loaded from one TOML file."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.searchers: dict[str, dict[str, Any]] = data.get("searchers", {})
        self.defaults: dict[str, dict[str, Any]] = data.get("defaults", {})
        self.systems: dict[str, dict[str, Any]] = data.get("systems", {})

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CATALOG) -> Catalog:
        with open(path, "rb") as handle:
            return cls(tomllib.load(handle))

    def resolve(self, name: str, *, model: str | None = None) -> SystemSpec:
        """Resolve a system name (and optional model override) to a ``SystemSpec``."""
        if name not in self.systems:
            raise KeyError(f"unknown system {name!r}; known: {', '.join(sorted(self.systems))}")
        entry = copy.deepcopy(self.systems[name])
        kind = entry.pop("kind", None)
        if kind not in SYSTEM_KINDS:
            raise ValueError(f"system {name!r} has kind {kind!r}; expected one of {SYSTEM_KINDS}")
        merged = {**copy.deepcopy(self.defaults.get(kind, {})), **entry}
        base_model = merged.pop("model", None)
        resolved_model = model or base_model
        if not resolved_model:
            raise ValueError(f"system {name!r} has no model")
        provider_of(resolved_model)

        searcher_name = merged.pop("searcher", None)
        hosted = merged.pop("hosted_web_search", None)
        if kind == "rag" and (searcher_name is None or hosted is not None):
            raise ValueError(f"rag system {name!r} needs exactly one searcher and no hosted search")
        if kind == "scout" and (searcher_name is None) == (hosted is None):
            raise ValueError(f"scout system {name!r} needs either a searcher or hosted_web_search")
        if (
            hosted is not None
            and base_model
            and provider_of(resolved_model) != provider_of(base_model)
        ):
            raise ValueError(
                f"{name!r} uses {provider_of(base_model)}'s hosted search; the model must stay on that provider"
            )

        searcher = None
        if searcher_name is not None:
            if searcher_name not in self.searchers:
                raise KeyError(f"system {name!r} references unknown searcher {searcher_name!r}")
            searcher = {"name": searcher_name, **copy.deepcopy(self.searchers[searcher_name])}

        allowed = (_SCOUT_FIELDS if kind == "scout" else _RAG_FIELDS) | {"num_results"}
        unknown = set(merged) - allowed
        if unknown:
            raise ValueError(f"system {name!r} has unknown settings: {sorted(unknown)}")
        return SystemSpec(
            name=name if model is None else f"{name}@{resolved_model}",
            kind=kind,
            model=resolved_model,
            settings=merged,
            searcher=searcher,
            hosted_web_search=hosted,
        )


class System:
    """A runnable system: answers one prompt and returns a JSON-serializable record."""

    def __init__(self, spec: SystemSpec) -> None:
        self.spec = spec
        settings = dict(spec.settings)
        num_results = settings.pop("num_results", 10)
        self._searcher: Searcher | None = None
        if spec.searcher is not None:
            options = {k: v for k, v in spec.searcher.items() if k not in ("name", "provider")}
            self._searcher = build_searcher(spec.searcher["provider"], options)
        if spec.kind == "scout":
            tools = [SearchTool(self._searcher, num_results=num_results)] if self._searcher else []
            self._runner: Scout | SingleStepRAG = Scout(
                ScoutConfig(model=spec.model, **settings),
                tools=tools,
                hosted_web_search=spec.hosted_web_search,
            )
        else:
            assert self._searcher is not None
            self._runner = SingleStepRAG(
                self._searcher, spec.model, num_results=num_results, **settings
            )

    async def answer(self, prompt: str) -> dict[str, Any]:
        """Run the system on one prompt."""
        result = await self._runner.run(prompt)
        return result.to_dict()

    async def close(self) -> None:
        if self._searcher is not None:
            await self._searcher.close()
