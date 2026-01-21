"""
Metrics computation for company search benchmark.

Retrieval track metrics:
- R@1: % of queries where rank-1 result is correct
- R@10: % of queries with correct result in top 10
- Precision: % of results that are relevant

RAG track metrics:
- Accuracy: % of queries with correct extracted answer

Header author: Devin
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalMetrics:
    match: float
    recall_at_10: float
    precision: float
    num_queries: int


@dataclass
class RAGMetrics:
    accuracy: float
    num_queries: int


def compute_retrieval_metrics(grades: list[dict[str, Any]]) -> RetrievalMetrics:
    """Compute retrieval metrics from graded results."""
    by_query: dict[str, list[dict]] = {}
    for g in grades:
        qid = g["query_id"]
        if qid not in by_query:
            by_query[qid] = []
        by_query[qid].append(g)

    for qid in by_query:
        by_query[qid].sort(key=lambda x: x.get("rank", 0))

    per_query = []
    for qid, query_grades in by_query.items():
        first_match_rank = None
        for g in query_grades:
            if g.get("is_match", 0) >= 1.0:
                first_match_rank = g.get("rank", query_grades.index(g) + 1)
                break

        n_results = len(query_grades)
        relevances = [g.get("is_match", 0) for g in query_grades]

        match = 1.0 if first_match_rank == 1 else 0.0
        recall_10 = 1.0 if first_match_rank and first_match_rank <= 10 else 0.0
        n_matches = sum(1 for r in relevances if r >= 1.0)
        precision = n_matches / n_results if n_results > 0 else 0.0

        per_query.append({"match": match, "recall_10": recall_10, "precision": precision})

    n = len(per_query)
    if n == 0:
        return RetrievalMetrics(0, 0, 0, 0)

    return RetrievalMetrics(
        match=sum(m["match"] for m in per_query) / n,
        recall_at_10=sum(m["recall_10"] for m in per_query) / n,
        precision=sum(m["precision"] for m in per_query) / n,
        num_queries=n,
    )


def compute_rag_metrics(grades: list[dict[str, Any]]) -> RAGMetrics:
    """Compute RAG metrics from graded results."""
    if not grades:
        return RAGMetrics(accuracy=0.0, num_queries=0)

    correct = sum(1 for g in grades if g.get("is_correct", 0) >= 1.0)
    total = len(grades)

    return RAGMetrics(
        accuracy=correct / total if total > 0 else 0.0,
        num_queries=total,
    )
