"""Bundle canonical workspace data in wheels and source distributions.

Editable installs read the original files. Source distributions carry them inside
harness/, so subsequent wheel builds no longer depend on the repository layout.
"""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class WorkspaceDataHook(BuildHookInterface):
    def initialize(self, version, build_data):
        """Include workspace resources once, preserving self-contained source archives."""
        if version == "editable":
            return
        root = Path(self.root)
        sources = {
            "systems.toml": "systems.toml",
            "simple-people-benchmark/data/people/simple_people_search.jsonl": "data/people/simple_people_search.jsonl",
            "simple-company-benchmark/data/company/simple_company_search.jsonl": "data/company/simple_company_search.jsonl",
            "publication-benchmark/data/publication/publication_search.jsonl": "data/publication/publication_search.jsonl",
            "webcode-benchmark/data/rag/code_rag.jsonl": "data/rag/code_rag.jsonl",
            "webcode-benchmark/data/highlights/code_highlights.jsonl": "data/highlights/code_highlights.jsonl",
        }
        for source, relative in sources.items():
            destination = f"harness/{relative}"
            if (root / destination).is_file():
                continue
            source_path = root.parent / source
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing benchmark resource: {source_path}")
            build_data["force_include"][str(source_path)] = destination
