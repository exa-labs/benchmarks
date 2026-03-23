from dataclasses import dataclass
from typing import Any


@dataclass
class ContentsMetrics:
    completeness: float = 0.0
    accuracy: float = 0.0
    structure: float = 0.0
    signal: float = 0.0
    composite_quality: float = 0.0
    det_paragraph_containment: float = 0.0
    det_heading_similarity: float = 0.0
    det_code_block_recall: float = 0.0
    det_table_recall: float = 0.0
    det_rouge_l: float = 0.0
    num_queries: int = 0


@dataclass
class GroundedRAGMetrics:
    groundedness: float = 0.0
    correctness: float = 0.0
    citation_precision: float = 0.0
    avg_citation_tokens: float = 0.0
    num_queries: int = 0


def compute_contents_metrics(grades: list[dict[str, Any]]) -> ContentsMetrics:
    if not grades:
        return ContentsMetrics()

    n = len(grades)
    metrics = ContentsMetrics(num_queries=n)

    score_keys = [
        "completeness", "accuracy", "structure", "composite_quality",
        "det_paragraph_containment", "det_heading_similarity",
        "det_code_block_recall", "det_table_recall", "det_rouge_l",
    ]

    for key in score_keys:
        values = [g.get(key, 0.0) for g in grades]
        setattr(metrics, key, sum(values) / n if values else 0.0)

    ratios = [g.get("det_length_ratio", 1.0) for g in grades]
    metrics.signal = sum(min(1.0, 1.0 / r) if r > 0 else 0.0 for r in ratios) / n

    return metrics


def compute_grounded_rag_metrics(grades: list[dict[str, Any]]) -> GroundedRAGMetrics:
    if not grades:
        return GroundedRAGMetrics()

    n = len(grades)
    return GroundedRAGMetrics(
        groundedness=sum(g.get("grounded", 0.0) for g in grades) / n,
        correctness=sum(g.get("score", 0.0) for g in grades) / n,
        citation_precision=sum(g.get("citation_precision", 0.0) for g in grades) / n,
        avg_citation_tokens=sum(g.get("avg_citation_tokens", 0.0) for g in grades) / n,
        num_queries=n,
    )
