from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalMetrics:
    recall_at_1: float  # R@1: % of queries where the first result is the gold publication
    recall_at_5: float  # R@5: % of queries with the gold publication in the top 5
    recall_at_10: float  # R@10: % of queries with the gold publication in the top 10
    mrr: float  # mean reciprocal rank of the first (and only) gold hit
    num_queries: int


def compute_retrieval_metrics(grades: list[dict[str, Any]]) -> RetrievalMetrics:
    by_query: dict[str, list[dict]] = {}
    for g in grades:
        by_query.setdefault(g["query_id"], []).append(g)

    per_query = []
    for query_grades in by_query.values():
        query_grades.sort(key=lambda x: x.get("rank", 0))

        first_match_rank = None
        for g in query_grades:
            if g.get("is_match", 0) >= 1.0:
                first_match_rank = g.get("rank")
                break

        per_query.append(
            {
                "r1": 1.0 if first_match_rank == 1 else 0.0,
                "r5": 1.0 if first_match_rank and first_match_rank <= 5 else 0.0,
                "r10": 1.0 if first_match_rank and first_match_rank <= 10 else 0.0,
                "rr": (1.0 / first_match_rank) if first_match_rank else 0.0,
            }
        )

    n = len(per_query)
    if n == 0:
        return RetrievalMetrics(0, 0, 0, 0, 0)

    return RetrievalMetrics(
        recall_at_1=sum(m["r1"] for m in per_query) / n,
        recall_at_5=sum(m["r5"] for m in per_query) / n,
        recall_at_10=sum(m["r10"] for m in per_query) / n,
        mrr=sum(m["rr"] for m in per_query) / n,
        num_queries=n,
    )
