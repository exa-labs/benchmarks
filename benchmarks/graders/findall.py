"""Holistic FindAll prompts and candidate formatting from the release source."""

import json
from typing import Any

HOLISTIC_SYSTEM_PROMPT = """You are a strict pointwise verifier for find-all benchmark rows.
Judge only the provided query, criterion, and candidate/result payload. Do not search the web, invent missing evidence, or substitute a different candidate.
Return score 1 when the candidate has a clear identity and the supplied payload supports the core requested criteria.
Return score 0 if identity, entity type, URL/evidence anchor, or a major required criterion is missing, contradicted, unsupported, inaccessible, or ambiguous.
Secondary or prioritization signals should not by themselves make an otherwise valid candidate incorrect.
Adjacent or synonymous titles are acceptable when the query asks for aligned roles, equivalent roles, or gives example titles rather than an exhaustive exact-title list.
When responsibilities, function, seniority, or operating scope are normally inherent to a verified current title at a verified organization, title plus organization may support that criterion. Do not apply this inference to credentials, certifications, named technologies, explicit dates, geography, revenue, funding, customers, or other concrete facts requiring direct evidence.
Do not confuse assets, AUM, deposits, or headcount with annual revenue unless the query explicitly allows that proxy.
Use 0 when uncertain. Return brief evidence-based reasoning and the binary score."""

HOLISTIC_DESCRIPTION = """You are verifying one candidate that should be {entity_label} for a find-all benchmark.

Original search query:
{query}

Expected criteria:
{expected}

Use only the candidate JSON, including retrieved content, snippets, attributes, primary URL, and evidence URLs already present in the payload. Do not search the web.
Return score 1 when the candidate is {entity_label} and public evidence supports the core requested criteria.
Return score 0 when the candidate is the wrong entity type, contradicted by public evidence, or missing a major required criterion.
Ignore minor formatting issues, prioritization preferences, and secondary signals that are not required by the query.
Return score 1 for a pass and score 0 for a fail."""


def compact_value(value: Any) -> Any:
    """Apply the source grader's per-string and per-list evidence bounds."""
    if isinstance(value, str):
        return value if len(value) <= 4000 else value[:4000].rstrip() + " ..."
    if isinstance(value, list):
        return [compact_value(item) for item in value[:12]]
    if isinstance(value, dict):
        return {
            str(key): compact_value(item)
            for key, item in value.items()
            if key not in {"raw", "html", "body", "content"}
        }
    return value


def format_candidate(row: dict[str, Any]) -> str:
    """Match BinaryPointwise's 2,048-token × 4 result-content character budget."""
    compact = {
        key: compact_value(value)
        for key, value in row.items()
        if key not in {"request_started_at_ms", "request_midpoint_at_ms", "request_finished_at_ms"}
    }
    content = json.dumps(compact, indent=2, sort_keys=True)[:8192]
    url = row.get("url") or row.get("canonical_url")
    title = row.get("title") or row.get("name")
    return f"URL: {url}\nTitle: {title}\n\n{content}"
