"""SWEChat Searches: Exa's retrieval benchmark derived from SALT-NLP/SWE-chat."""

from __future__ import annotations

from statistics import mean
from typing import Any

from benchmarks.graders import RubricResultGrader
from benchmarks.graders.base import gather_judgments
from data import loaders
from harness.llm.judge import Judge
from harness.searchers import SearchResult
from harness.suite import Grade, Suite, Task

COVERAGE_CUTOFFS = (1, 5, 10)


def coverage_scores(criterion_hits: list[list[bool]]) -> dict[str, float]:
    """Score ``covered_at_k``: the share of criteria some top-k result satisfies.

    ``criterion_hits[i][r]`` says whether the result at rank ``r + 1`` satisfies
    criterion ``i``. An empty result list covers nothing.
    """
    return {
        f"covered_at_{k}": mean(any(hits[:k]) for hits in criterion_hits) for k in COVERAGE_CUTOFFS
    }


class SWEChatSearches(Suite):
    """SWEChat Searches, derived by Exa from searches in SALT-NLP/SWE-chat sessions.

    Each query carries one to three criteria describing what its results should
    contain. A criterion is covered at k when any of the top k results satisfies it
    on the content the search API returned.
    """

    name = dataset = "swechatsearches"
    primary_metric = "covered_at_10"
    system_kinds = ("search",)
    description = "SWEChat Searches: 586 queries derived from SWE-chat, with Exa rubrics"

    @property
    def revision(self) -> str:
        return f"{loaders.local_revision(self.dataset)}+grader-v3"

    def load(self) -> list[Task]:
        """Load the bundled queries; keep criteria separate for the judge."""
        return [
            Task(id=row["id"], problem=row["query"], answer=row["criteria"])
            for row in loaders.load_rows(self.dataset)
        ]

    async def grade_result(self, task: Task, result: dict[str, Any], judge: Judge) -> Grade:
        """Judge every (criterion, top-10 result) pair, then score coverage per cutoff."""
        rows = [SearchResult(**row) for row in result["results"][: max(COVERAGE_CUTOFFS)]]
        grader = RubricResultGrader(judge)
        criteria = task.answer
        grades = await gather_judgments(
            *(
                grader.grade(task.problem, criterion["description"], row)
                for criterion in criteria
                for row in rows
            )
        )
        per_criterion = [grades[i * len(rows) : (i + 1) * len(rows)] for i in range(len(criteria))]
        return Grade(
            coverage_scores([[g.scores["score"] >= 1 for g in row] for row in per_criterion]),
            {
                "criteria": [
                    {
                        "id": criterion["id"],
                        "results": [
                            {"rank": rank, **g.scores, **g.details}
                            for rank, g in enumerate(criterion_grades, 1)
                        ],
                    }
                    for criterion, criterion_grades in zip(criteria, per_criterion, strict=True)
                ]
            },
        )
