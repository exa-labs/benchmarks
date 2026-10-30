"""Load benchmark tasks and source revisions; no prompts, grading, or API execution.

Exa datasets are bundled beside this module. Public sources are pinned and cached
by data.sources; BrowseComp is decrypted only in memory.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.sources import SourceError, csv_rows, fetch_verified, hf_file, hf_snapshot, require_count
from harness.suite import Task

BROWSECOMP_SOURCE_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)
BROWSECOMP_SHA256 = "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"
BROWSECOMP_ROW_COUNT = 1266


def derive_key(password: str, length: int) -> bytes:
    """Repeat SHA-256(password) to ``length`` bytes, as simple-evals does."""
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt one base64 XOR-encrypted BrowseComp field."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key, strict=True)).decode()


def browsecomp_tasks(rows: list[dict[str, str]]) -> list[Task]:
    """Decrypt source rows into tasks; ids are the row index at the pinned hash."""
    return [
        Task(
            id=f"browsecomp-{index:04d}",
            problem=decrypt(row["problem"], row["canary"]),
            answer=decrypt(row["answer"], row["canary"]),
            metadata={"problem_topic": row["problem_topic"]},
        )
        for index, row in enumerate(rows)
    ]


def load_browsecomp() -> list[Task]:
    """Download (or reuse) the pinned CSV and decrypt every row in memory."""
    data = fetch_verified(BROWSECOMP_SOURCE_URL, BROWSECOMP_SHA256, "browse_comp_test_set.csv")
    rows = csv_rows(data)
    require_count("browsecomp", len(rows), BROWSECOMP_ROW_COUNT)
    return browsecomp_tasks(rows)


FRAMES_REPO_ID = "google/frames-benchmark"
FRAMES_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
FRAMES_FILENAME = "test.tsv"
FRAMES_ROW_COUNT = 824


def frames_tasks(rows: list[dict[str, str]]) -> list[Task]:
    """Build tasks; ids come from the upstream row index (the TSV's unnamed first column)."""
    return [
        Task(
            id=f"frames-{int(row['']):03d}",
            problem=row["Prompt"],
            answer=row["Answer"],
            metadata={
                "reasoning_types": row.get("reasoning_types", ""),
                "wiki_links": row.get("wiki_links", ""),
            },
        )
        for row in rows
    ]


def load_frames() -> list[Task]:
    """Fetch the pinned TSV through the Hugging Face cache."""
    rows = csv_rows(
        hf_file(FRAMES_REPO_ID, FRAMES_FILENAME, FRAMES_REVISION).read_bytes(), delimiter="\t"
    )
    require_count("frames", len(rows), FRAMES_ROW_COUNT)
    return frames_tasks(rows)


DSQA_REPO_ID = "google/deepsearchqa"
DSQA_REVISION = "b2623f8653065c2672de6d941fc5434cd652376c"
DSQA_FILENAME = "DSQA-full.csv"
DSQA_ROW_COUNT = 900


def dsqa_tasks(rows: list[dict[str, str]]) -> list[Task]:
    """Build tasks; ids are the row index at the pinned revision."""
    return [
        Task(
            id=f"dsqa-{index:03d}",
            problem=row["problem"],
            answer=row["answer"],
            metadata={
                "answer_type": row["answer_type"],
                "problem_category": row["problem_category"],
            },
        )
        for index, row in enumerate(rows)
    ]


def load_dsqa() -> list[Task]:
    """Fetch the pinned CSV through the Hugging Face cache."""
    rows = csv_rows(hf_file(DSQA_REPO_ID, DSQA_FILENAME, DSQA_REVISION).read_bytes())
    require_count("dsqa", len(rows), DSQA_ROW_COUNT)
    return dsqa_tasks(rows)


WIDESEARCH_REPO_ID = "ByteDance-Seed/WideSearch"
WIDESEARCH_REVISION = "6531a7e5b497d44c8912407e0cb3dc95bd98cc09"
WIDESEARCH_QUERIES_FILE = "widesearch.jsonl"
WIDESEARCH_GOLD_DIR = "widesearch_gold"
WIDESEARCH_ROW_COUNT = 200


def norm_column(column: Any) -> str:
    """Normalize a column name: lowercase with all spaces removed."""
    return str(column).strip().lower().replace(" ", "")


def gold_answer(gold_csv: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    """The task answer: gold CSV text plus the normalized evaluation spec."""
    return {
        "gold_csv": gold_csv,
        "required_columns": [norm_column(c) for c in evaluation["required"]],
        "unique_columns": [norm_column(c) for c in evaluation.get("unique_columns", [])],
        "eval_pipeline": {
            norm_column(column): spec
            for column, spec in evaluation.get("eval_pipeline", {}).items()
        },
    }


def load_gold_csv(path: Path, required: list[str], instance_id: str) -> str:
    """Read one gold CSV, keep its required columns, and return it as CSV text."""
    frame = pd.read_csv(path)
    frame.columns = [norm_column(column) for column in frame.columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceError(f"widesearch {instance_id}: gold CSV lacks required columns {missing}")
    return frame[required].to_csv(index=False)


def load_widesearch() -> list[Task]:
    """Fetch the pinned snapshot (queries and gold CSVs) through the Hugging Face cache."""
    root = hf_snapshot(
        WIDESEARCH_REPO_ID,
        WIDESEARCH_REVISION,
        [WIDESEARCH_QUERIES_FILE, f"{WIDESEARCH_GOLD_DIR}/*.csv"],
    )
    with (root / WIDESEARCH_QUERIES_FILE).open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    require_count("widesearch", len(items), WIDESEARCH_ROW_COUNT)
    tasks = []
    for item in items:
        instance_id = item["instance_id"]
        evaluation = item["evaluation"]
        if isinstance(evaluation, str):
            evaluation = json.loads(evaluation)
        required = [norm_column(c) for c in evaluation["required"]]
        gold_path = root / WIDESEARCH_GOLD_DIR / f"{instance_id}.csv"
        tasks.append(
            Task(
                id=instance_id,
                problem=item["query"],
                answer=gold_answer(load_gold_csv(gold_path, required, instance_id), evaluation),
                metadata={"language": item["language"]},
            )
        )
    return tasks


_DATASETS = {
    "people": "people.jsonl",
    "company": "company.jsonl",
    "publication": "publication.jsonl",
    "swechatsearches": "swechatsearches/searches.jsonl",
    "webcode-rag": "webcode/rag.jsonl",
    "webcode-highlights": "webcode/highlights.jsonl",
    "webcode-contents": "webcode/contents.jsonl",
    "webcode-e2e": "webcode/e2e.jsonl",
}


def dataset_path(dataset: str) -> Path:
    """Locate the bundled dataset identically in a checkout or installed wheel."""
    return Path(__file__).parent / _DATASETS[dataset]


def local_revision(dataset: str) -> str:
    """Identify a bundled dataset by its bytes, independent of checkout location."""
    return f"sha256:{hashlib.sha256(dataset_path(dataset).read_bytes()).hexdigest()}"


def load_rows(dataset: str) -> list[dict[str, Any]]:
    """Read a bundled JSONL export, including dataset-only Contents and E2E."""
    return [
        json.loads(line) for line in dataset_path(dataset).read_text().splitlines() if line.strip()
    ]


def load_local(dataset: str, track: str | None = None) -> list[Task]:
    """Build runnable Exa tasks from the canonical rows, optionally selecting a track."""
    return [
        Task(
            id=row.get("query_id", row.get("id")),
            problem=row.get("text", row.get("query", "")),
            answer=row.get("expected_answer", row.get("gold_paper")),
            metadata=row,
        )
        for row in load_rows(dataset)
        if track is None or row.get("track") == track
    ]
