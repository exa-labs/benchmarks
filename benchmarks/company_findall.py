"""Grade every returned company against all query criteria, with no evaluation row cap."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from benchmarks.graders.base import gather_judgments
from benchmarks.graders.findall import (
    HOLISTIC_DESCRIPTION,
    HOLISTIC_SYSTEM_PROMPT,
    format_candidate,
)
from benchmarks.graders.rubric import RubricGradeResult
from data import loaders
from harness.llm.judge import Judge
from harness.suite import Grade, Suite, Task


class CompanyFindAll(Suite):
    name = dataset = "company-findall"
    description = "Company FindAll: 300 synthetic GTM queries, grading every returned company"
    # Fleet normalization is computed by bench compare, not within one system.
    primary_metric = "num_passed"
    system_kinds = ("agent",)
    judge_settings = {"reasoning_effort": "low", "temperature": 0.0, "max_output_tokens": 2048}
    ratio_metrics = {"entity_precision": ("num_passed", "num_rows")}

    def failure_scores(self, result: dict[str, Any] | None) -> dict[str, float]:
        count = len((result or {}).get("rows", []))
        return {
            "num_rows": float(count),
            "num_passed": 0.0,
            "pass_rate": 0.0,
            "criteria_pass_rate": 0.0,
            "zero_entities": float(count == 0),
        }

    @property
    def revision(self) -> str:
        root = loaders.dataset_path(self.dataset).parent
        contract = b"".join(
            (root / name).read_bytes()
            for name in ("agent_instructions.txt", "output_schema.json", "parallel_output_spec.txt")
        )
        return f"{loaders.local_revision(self.dataset)}+agent-{hashlib.sha256(contract).hexdigest()[:12]}+grader-v2"

    def load(self) -> list[Task]:
        return [
            Task(id=row["query_id"], problem=row["query"], answer=row["criteria"])
            for row in loaders.load_rows(self.dataset)
        ]

    def agent_request(self) -> dict[str, Any]:
        root = loaders.dataset_path(self.dataset).parent
        return {
            "instructions": (root / "agent_instructions.txt").read_text(),
            "output_schema": json.loads((root / "output_schema.json").read_text()),
            "output_spec": (root / "parallel_output_spec.txt").read_text(),
        }

    async def grade_result(self, task: Task, result: dict[str, Any], judge: Judge) -> Grade:
        criteria = [c["description"] if isinstance(c, dict) else c for c in task.answer]
        expected = "Companies matching all of:\n" + "\n".join(f"- {c}" for c in criteria)
        expected += (
            "\n\nDo not include people, job postings, team pages, broad directories, "
            "listicles, unrelated companies, or companies missing any required criterion."
        )
        criterion = HOLISTIC_DESCRIPTION.format(
            entity_label="a company", query=task.problem, expected=expected
        )
        rows = result["rows"]
        judgments = await gather_judgments(
            *(
                judge.complete_json(
                    f"Query: {task.problem}\n\nCriterion: {criterion}\nResult: {format_candidate(row)}",
                    RubricGradeResult,
                    system=HOLISTIC_SYSTEM_PROMPT,
                )
                for row in rows
            )
        )
        passed = sum(verdict.score for verdict, _ in judgments)
        rate = passed / len(rows) if rows else 0.0
        return Grade(
            scores={
                "num_rows": float(len(rows)),
                "num_passed": float(passed),
                "pass_rate": rate,
                "criteria_pass_rate": rate,
                "zero_entities": float(not rows),
            },
            details={
                "candidates": [
                    {"rank": rank, **verdict.model_dump()}
                    for rank, (verdict, _) in enumerate(judgments, 1)
                ],
                "parse_error": result.get("parse_error"),
            },
        )
