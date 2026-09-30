"""The repository's retrieval and grounded-RAG suites, on the common runner.

Dataset files live under benchmarks/ and are included in wheels. Empty retrievals are scored as zero per query,
so providers cannot improve recall by returning nothing on difficult queries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.base import Grade, Suite, Task
from benchmarks.graders import (
    Citation,
    GroundedRAGGrader,
    PaperRetrievalGrader,
    PeopleGrader,
    RAGGrader,
    RetrievalGrader,
)
from benchmarks.graders.base import gather_judgments
from harness.llm.judge import Judge
from harness.searchers import SearchResult

_DATASETS = {
    "people": "people/data.jsonl",
    "company": "company/data.jsonl",
    "publication": "publication/data.jsonl",
    "webcode-rag": "webcode/data/rag.jsonl",
    "webcode-highlights": "webcode/data/highlights.jsonl",
}


def dataset_path(dataset: str) -> Path:
    """Locate the bundled dataset identically in a checkout or installed wheel."""
    return Path(__file__).parent / _DATASETS[dataset]


def retrieval_scores(matches: list[bool]) -> dict[str, float]:
    """Compute ranked metrics for one query, including empty result sets."""
    rank = next((i for i, match in enumerate(matches, 1) if match), None)
    return {
        "recall_at_1": float(rank == 1),
        "recall_at_5": float(rank is not None and rank <= 5),
        "recall_at_10": float(rank is not None and rank <= 10),
        "mrr": 1.0 / rank if rank else 0.0,
        "precision": sum(matches) / len(matches) if matches else 0.0,
    }


def result_rows(result: dict[str, Any]) -> list[SearchResult]:
    """Rehydrate the complete ranked results, retaining empty-text URL matches."""
    return [SearchResult(**row) for row in result["results"]]


class LocalSuite(Suite):
    dataset: str
    track: str | None = None

    @property
    def revision(self) -> str:
        return f"sha256:{hashlib.sha256(dataset_path(self.dataset).read_bytes()).hexdigest()}+grader-v2"

    def load(self) -> list[Task]:
        """Read the canonical JSONL file and select this suite's track."""
        rows = [
            json.loads(line)
            for line in dataset_path(self.dataset).read_text().splitlines()
            if line.strip()
        ]
        return [
            Task(
                id=row.get("query_id", row.get("id")),
                problem=row.get("text", row.get("query", "")),
                answer=row.get("expected_answer", row.get("gold_paper")),
                metadata=row,
            )
            for row in rows
            if self.track is None or row.get("track") == self.track
        ]


class RetrievalSuite(LocalSuite):
    primary_metric = "recall_at_10"
    system_kinds = ("search",)

    async def grade_result(self, task: Task, result: dict[str, Any], judge: Judge) -> Grade:
        """Judge each ranked result, then compute one metric record per query."""
        rows = result_rows(result)
        if self.dataset == "people":
            grader = PeopleGrader(judge)
            grades = await gather_judgments(*(grader.grade(task.problem, row) for row in rows))
        elif self.dataset == "company":
            grader = RetrievalGrader(judge)
            grades = await gather_judgments(
                *(
                    grader.grade(
                        task.problem,
                        row,
                        gold_homepage=task.metadata.get("gold_company_homepage"),
                        constraints=task.metadata.get("constraints"),
                    )
                    for row in rows
                )
            )
        else:
            grader = PaperRetrievalGrader()
            grades = [grader.grade(row, task.answer) for row in rows]
        return Grade(
            retrieval_scores([g.scores["is_match"] >= 1 for g in grades]),
            {"results": [{"rank": i, **g.scores, **g.details} for i, g in enumerate(grades, 1)]},
        )


class People(RetrievalSuite):
    name = dataset = "people"
    description = "People: 1,400 profile retrieval queries"


class CompanyRetrieval(RetrievalSuite):
    name = "company-retrieval"
    dataset = "company"
    track = "retrieval"
    description = "Company: 605 ranked retrieval queries"


class Publication(RetrievalSuite):
    name = "publication"
    dataset = "publication"
    track = "paper"
    requires_judge = False
    description = "Publication: 1,472 publication identity queries"


class PublicationToT(Publication):
    name = "publication-tot"
    track = "tot"
    description = "Publication: 394 tip-of-the-tongue queries"


class CompanyRAG(LocalSuite):
    name = "company-rag"
    dataset = "company"
    track = "rag"
    primary_metric = "accuracy"
    description = "Company: 234 factual questions, with numeric tolerance"

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        grade = await RAGGrader(judge).grade(task.problem, task.answer, response)
        return Grade({"accuracy": grade.scores["is_correct"]}, grade.details)


class WebCodeRAG(LocalSuite):
    name = dataset = "webcode-rag"
    primary_metric = "grounded"
    system_kinds = ("rag",)
    description = "WebCode: 307 grounded code-documentation questions"

    async def grade_result(self, task: Task, result: dict[str, Any], judge: Judge) -> Grade:
        """Score correctness and grounding using the evidence actually retrieved."""
        citations = [
            Citation(url=c["url"], title=c["title"], text=c["text"]) for c in result["citations"]
        ]
        grade = await GroundedRAGGrader(judge).grade(
            task.problem, task.answer, result["answer"], citations
        )
        return Grade(grade.scores, grade.details)


class WebCodeHighlights(WebCodeRAG):
    name = dataset = "webcode-highlights"
    system_kinds = ("extract-rag",)
    description = "WebCode: 250 questions over provided URLs"
