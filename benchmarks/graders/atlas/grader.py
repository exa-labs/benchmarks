"""The ATLAS grading pipeline (``grade_table``) and its LLM calls (``TableJudge``).

README.md in this package describes each step and what the scores mean.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from benchmarks.graders.atlas.matching import entity_identities, match_declared_names, names_entity
from benchmarks.graders.atlas.normalize import (
    Column,
    dedupe_columns,
    extract_table,
    is_blank,
    is_empty,
    key_text,
    middle_initial_equivalent,
    norm_column,
    normalize_cell,
    normalize_text,
    strip_links,
    text_equivalent,
    text_token,
    unwrap_answer,
)
from benchmarks.graders.atlas.prompts import (
    ALIAS_GUIDANCE,
    COLUMN_ALIGNMENT_PROMPT,
    DEFAULT_CRITERION,
    ENTITY_ALIGNMENT_PROMPT,
    EXCLUSION_GUIDANCE,
    JUDGE_EQUIVALENCE_RULES,
    JUDGE_GUIDANCE,
    JUDGE_PROMPT,
    VERSION_GUIDANCE,
)
from benchmarks.graders.base import JudgeSpend
from harness.llm.judge import Judge

SCORE_KEYS = (
    "row_precision",
    "row_recall",
    "row_f1",
    "item_precision",
    "item_recall",
    "item_f1",
    "discovery_precision",
    "discovery_recall",
    "discovery_f1",
    "complete_pass",
)
JUDGE_REVOTE_BATCH = 8
ALIGNMENT_ATTEMPTS = 3


@dataclass
class TableTask:
    """Everything the grader needs from one ATLAS task.

    Per-row annotations are aligned with ``rows``: ``accepted[i]`` and ``rejected[i]`` map a
    column to other values that count as right or wrong for that gold cell, and
    ``ungraded[i]`` lists the columns of row ``i`` that are not scored.
    """

    key_column: str
    columns: list[Column]
    rows: list[dict[str, str]]
    accepted: list[dict[str, list[str]]] = field(default_factory=list)
    rejected: list[dict[str, list[str]]] = field(default_factory=list)
    ungraded: list[list[str]] = field(default_factory=list)
    optional_entities: list[str] = field(default_factory=list)
    entity_aliases: dict[str, list[str]] = field(default_factory=dict)
    entity_exclusions: dict[str, list[str]] = field(default_factory=dict)
    alignment_note: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TableTask:
        """Build from one dataset record (the layout is documented in ``benchmarks/atlas.py``)."""
        names = [column["name"] for column in record["columns"]]
        accepted: list[dict[str, list[str]]] = [{} for _ in record["rows"]]
        rejected: list[dict[str, list[str]]] = [{} for _ in record["rows"]]
        for target, entries in (
            (accepted, record.get("accepted")),
            (rejected, record.get("rejected")),
        ):
            for entry in entries or []:
                target[entry["row"]].setdefault(entry["column"], []).extend(entry["values"])
        ungraded: list[list[str]] = [[] for _ in record["rows"]]
        for entry in record.get("ungraded") or []:
            ungraded[entry["row"]].append(entry["column"])
        return cls(
            key_column=record["key_column"],
            columns=[Column(c["name"], c["type"], c.get("criterion")) for c in record["columns"]],
            rows=[dict(zip(names, row, strict=True)) for row in record["rows"]],
            accepted=accepted,
            rejected=rejected,
            ungraded=ungraded,
            optional_entities=list(record.get("optional_entities") or []),
            entity_aliases={e["entity"]: e["aliases"] for e in record.get("entity_aliases") or []},
            entity_exclusions={
                e["entity"]: e["names"] for e in record.get("entity_exclusions") or []
            },
            alignment_note=record.get("alignment_note"),
        )


# ---------------------------------------------------------------------------
# LLM calls: alignment and cell verdicts
# ---------------------------------------------------------------------------


class AlignmentError(RuntimeError):
    """The aligner returned no usable mapping."""


class TableJudge:
    """The LLM side of grading: column and entity alignment, and cell verdicts.

    One instance serves a whole run; its verdict cache makes an identical
    (criterion, gold, response) pair score the same everywhere in the run.
    """

    def __init__(self, judge: Judge) -> None:
        self.judge = judge
        self._verdicts: dict[tuple[str, str, str], float] = {}

    async def align(self, prompt: str, spend: JudgeSpend) -> dict[str, str]:
        """A string-to-string mapping from the aligner, retried on malformed replies."""
        error: Exception | None = None
        for _ in range(ALIGNMENT_ATTEMPTS):
            try:
                reply = await self.judge.complete(prompt)
            except Exception as failure:  # noqa: BLE001 - a failed call is retried like a bad reply
                error = failure
                continue
            spend.add(reply)
            mapping = _last_json_object(reply.text) if reply.text else None
            if mapping is not None and all(
                isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
            ):
                return mapping
        raise AlignmentError(f"no valid alignment after {ALIGNMENT_ATTEMPTS} attempts") from error

    async def judge_cells(
        self, pairs: list[tuple[str, str]], criterion: str, spend: JudgeSpend
    ) -> list[tuple[float, str]]:
        """``(score, how)`` for each ``(response, gold)`` pair of one column.

        Deterministic equivalence and empty responses never reach the model. A single
        verdict is noisy and errs towards 0, so a 0 needs two further votes, in small
        batches, and the majority of the three decides.
        """
        results: dict[int, tuple[float, str]] = {}
        pending: list[int] = []
        for i, (response, gold) in enumerate(pairs):
            if middle_initial_equivalent(response, gold, criterion):
                results[i] = (1.0, "criterion allows middle initials")
            elif text_equivalent(response, gold):
                results[i] = (1.0, "normalized equality")
            elif is_empty(response) and not is_blank(gold):
                results[i] = (0.0, "empty response")
            elif (cached := self._verdicts.get((criterion, gold, response))) is not None:
                results[i] = (cached, "judge")
            else:
                pending.append(i)

        if pending:
            first = await self._vote([pairs[i] for i in pending], criterion, spend)
            votes = {i: [score] for i, score in zip(pending, first, strict=True)}
            doubtful = [i for i in pending if votes[i][0] < 1.0]
            for _ in range(2):
                for start in range(0, len(doubtful), JUDGE_REVOTE_BATCH):
                    chunk = doubtful[start : start + JUDGE_REVOTE_BATCH]
                    again = await self._vote([pairs[i] for i in chunk], criterion, spend)
                    for i, score in zip(chunk, again, strict=True):
                        votes[i].append(score)
            for i in pending:
                tally = votes[i]
                score = tally[0] if len(tally) == 1 else float(sum(v >= 1.0 for v in tally) >= 2)
                self._verdicts[(criterion, pairs[i][1], pairs[i][0])] = score
                results[i] = (score, "judge")
        return [results[i] for i in range(len(pairs))]

    async def _vote(
        self, pairs: list[tuple[str, str]], criterion: str, spend: JudgeSpend
    ) -> list[float]:
        """One judge call over ``(response, gold)`` pairs; a pair missing from the reply is 0."""
        listing = json.dumps(
            {
                f"idx_{k}": {"answer": gold, "response": response}
                for k, (response, gold) in enumerate(pairs)
            },
            ensure_ascii=False,
        )
        prompt = JUDGE_PROMPT.format(
            criterion=f"{criterion}\n\n{JUDGE_EQUIVALENCE_RULES}", response=listing
        )
        reply = await self.judge.complete(prompt)
        spend.add(reply)
        parsed = (_last_json_object(reply.text) if reply.text else None) or {}
        scores = []
        for k in range(len(pairs)):
            try:
                scores.append(float(parsed.get(f"idx_{k}", 0.0)))
            except (TypeError, ValueError):
                scores.append(0.0)
        return scores


def _last_json_object(text: str) -> dict | None:
    """The last ```json-fenced object in a reply, else its last bare top-level object."""
    for candidate in reversed(re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    decoder = json.JSONDecoder()
    last: dict | None = None
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        try:
            parsed, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(parsed, dict):
            last, index = parsed, end
        else:
            index += 1
    return last


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    index: int  # position in its table
    cells: dict[str, str]  # column -> value as written
    values: dict[str, str] = field(default_factory=dict)  # column -> value as scored
    name: str = ""  # entity name used to tell apart rows whose keys collide


async def grade_table(
    task: TableTask, response: str, judge: TableJudge, spend: JudgeSpend
) -> tuple[dict[str, float], dict[str, Any]]:
    """Grade one response table; returns ``(scores, details)``."""
    columns = {norm_column(c.name): c for c in task.columns}
    names = list(columns)
    key = norm_column(task.key_column)
    optional = set(task.optional_entities)
    details: dict[str, Any] = {}

    # 1. Read the response's table. A response without one scores 0.
    table = extract_table(unwrap_answer(response)) if response.strip() else None
    if table is None or not table.rows:
        return dict.fromkeys(SCORE_KEYS, 0.0), {"error": "no table in response"}

    # 2. Map the response's column names onto the task's (the LLM aligner runs only when
    #    they differ). A table still missing a required column scores 0.
    if not set(names) <= set(table.columns):
        prompt = COLUMN_ALIGNMENT_PROMPT.format(
            response_columns=table.columns, reference_columns=names
        )
        mapping = await judge.align(prompt, spend)
        details["column_alignment"] = mapping
        table.columns = [norm_column(mapping.get(c, c)) for c in table.columns]
        table.select(dedupe_columns(table.columns))
        if missing := sorted(set(names) - set(table.columns)):
            return dict.fromkeys(SCORE_KEYS, 0.0), {
                **details,
                "error": f"missing columns: {missing}",
            }

    # 3. One row per entity on each side: a repeated key keeps its first row.
    positions = [table.columns.index(c) for c in names]
    gold_rows = _first_per_key(
        [{norm_column(c): str(v) for c, v in row.items()} for row in task.rows], key
    )
    response_rows = _first_per_key(
        [dict(zip(names, (row[p] for p in positions))) for row in table.rows], key
    )

    # 4. Decide which gold entity each response row names: declared names and aliases
    #    first, then the LLM aligner for the rest.
    aligned = await _align_entities(task, gold_rows, response_rows, key, judge, spend, details)

    # 5. Normalize every value for comparison (numbers and dates to canonical tokens).
    for row in gold_rows:
        row.values = {c: normalize_cell(columns[c], row.cells[c], key=c == key) for c in names}
        row.name = row.cells[key]
    for row in response_rows:
        row.values = {c: normalize_cell(columns[c], row.cells[c]) for c in names if c != key}
        row.values[key] = key_text(aligned[row.index][0])
        row.name = aligned[row.index][1]

    # 6. Pair response rows with gold rows, one to one, on their normalized entity keys.
    matches = _join(gold_rows, response_rows, key)

    # 7. Score every graded cell of every matched row, one column at a time.
    ungraded = {
        row.index: ({norm_column(c) for c in _at(task.ungraded, row.index, [])} & set(names))
        - {key}
        for row in gold_rows
    }
    cell_scores: list[dict[str, float | None]] = [{key: 1.0} for _ in matches]
    notes: list[dict[str, str]] = [{} for _ in matches]
    for name in names:
        if name == key:
            continue
        for m, (score, note) in enumerate(
            await _score_column(task, columns[name], matches, ungraded, judge, spend)
        ):
            cell_scores[m][name] = score
            notes[m][name] = note

    # 8. Aggregate: precision, recall and F1 for rows, items and discovery, plus
    #    complete_pass (all defined in README.md).
    scores = {
        **_table_scores(
            gold_rows, response_rows, matches, cell_scores, ungraded, key, len(names), optional
        ),
        **_discovery_scores(gold_rows, response_rows, key, optional),
    }
    matched = {id(g) for g, _ in matches} | {id(r) for _, r in matches}
    details.update(
        matches=[
            {
                "gold_index": g.index,
                "response_index": r.index,
                "gold": g.cells,
                "response": r.cells,
                "scores": cell_scores[m],
                "notes": notes[m],
            }
            for m, (g, r) in enumerate(matches)
        ],
        missing=[
            {"gold_index": g.index, "cells": g.cells} for g in gold_rows if id(g) not in matched
        ],
        extra=[
            {"response_index": r.index, "cells": r.cells}
            for r in response_rows
            if id(r) not in matched
        ],
    )
    return {k: scores[k] for k in SCORE_KEYS}, details


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _at(items: list, index: int, default: Any) -> Any:
    return items[index] if index < len(items) else default


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 1e-9 else 0.0


def _first_per_key(rows: list[dict[str, str]], key: str) -> list[_Row]:
    seen: set[str] = set()
    kept = []
    for index, cells in enumerate(rows):
        if cells[key] not in seen:
            seen.add(cells[key])
            kept.append(_Row(index, cells))
    return kept


async def _align_entities(
    task: TableTask,
    gold_rows: list[_Row],
    response_rows: list[_Row],
    key: str,
    judge: TableJudge,
    spend: JudgeSpend,
    details: dict[str, Any],
) -> dict[int, tuple[str, str]]:
    """Per response row: the key to join on, and its name (the gold name when matched).

    A cell that exactly writes a gold name or declared alias matches it; a cell naming
    several entities, or an alias shared by several, stays unmatched; every other cell
    that is not already a gold key goes to the LLM aligner, which may not pair a gold
    entity with one of its audited exclusions.
    """
    gold_keys = [row.cells[key] for row in gold_rows]
    response_keys = [row.cells[key] for row in response_rows]
    declared, blocked = (
        match_declared_names(gold_keys, response_keys, task.entity_aliases)
        if task.entity_aliases
        else ({}, set())
    )
    mapping: dict[str, str] = {}
    pending = [
        k for k in response_keys if k not in declared and k not in blocked and k not in gold_keys
    ]
    if pending:
        prompt = ENTITY_ALIGNMENT_PROMPT.format(response=pending, reference=gold_keys)
        prompt += VERSION_GUIDANCE
        if task.entity_aliases:
            prompt += ALIAS_GUIDANCE.format(
                aliases=json.dumps(task.entity_aliases, ensure_ascii=False)
            )
        if task.entity_exclusions:
            prompt += EXCLUSION_GUIDANCE.format(
                exclusions=json.dumps(task.entity_exclusions, ensure_ascii=False)
            )
        if task.alignment_note:
            prompt += f"\n{task.alignment_note}"
        excluded = {
            gold: {normalize_text(name) for name in names}
            for gold, names in task.entity_exclusions.items()
        }
        mapping = {
            origin: target
            for origin, target in (await judge.align(prompt, spend)).items()
            if normalize_text(origin) not in excluded.get(target, set())
            and not (task.entity_aliases and (origin not in pending or target in declared.values()))
        }
    details["entity_alignment"] = {**declared, **mapping}
    if blocked:
        details["unresolved_entities"] = sorted(blocked)
    aligned = {}
    for row in response_rows:
        cell = row.cells[key]
        join_on = (
            f"__unresolved_{row.index}__"
            if cell in blocked
            else declared.get(cell, mapping.get(cell, cell))
        )
        # A matched row carries the gold name, so two rows naming one entity pair once.
        aligned[row.index] = (join_on, declared.get(cell, mapping.get(cell, cell)))
    return aligned


def _identities(
    gold: list[_Row], response: list[_Row], key: str
) -> tuple[list[tuple], list[tuple]]:
    gold_ids = entity_identities(
        [(r.values[key], r.name) for r in gold], [(r.values[key], "") for r in response]
    )
    response_ids = entity_identities(
        [(r.values[key], r.name) for r in response], [(r.values[key], "") for r in gold]
    )
    return gold_ids, response_ids


def _join(gold_rows: list[_Row], response_rows: list[_Row], key: str) -> list[tuple[_Row, _Row]]:
    """One-to-one ``(gold, response)`` row matches in gold order."""
    gold_ids, response_ids = _identities(gold_rows, response_rows, key)
    matches, taken = [], set()
    for gold, gold_id in zip(gold_rows, gold_ids):
        for row, row_id in zip(response_rows, response_ids):
            if row_id == gold_id and row.index not in taken:
                taken.add(row.index)
                matches.append((gold, row))
                break
    return matches


async def _score_column(
    task: TableTask,
    column: Column,
    matches: list[tuple[_Row, _Row]],
    ungraded: dict[int, set[str]],
    judge: TableJudge,
    spend: JudgeSpend,
) -> list[tuple[float | None, str]]:
    """``(score, note)`` for one column of every matched row; ``None`` marks ungraded.

    A response value equal to an audited rejected value scores 0. Number and date cells
    must equal the gold token exactly; text cells go through ``TableJudge.judge_cells``.
    A cell that fails against the gold can still pass against an accepted alternate.
    """
    name = norm_column(column.name)
    criterion = f"{column.criterion or DEFAULT_CRITERION}\n{JUDGE_GUIDANCE}"
    predicted = [strip_links(row.values[name]) for _, row in matches]
    scored: dict[int, tuple[float, str]] = {}
    alternates: dict[int, list[tuple[str, str | None]]] = {}
    to_judge: list[int] = []
    for m, (gold, row) in enumerate(matches):
        if name in ungraded[gold.index]:
            continue
        if (rejection := _rejected(task, gold.index, name, row.cells[name])) is not None:
            scored[m] = (0.0, f"rejected value {rejection!r}")
            continue
        alternates[m] = _alternates(task, gold.index, column)
        if column.type != "text":
            correct = predicted[m].lower() == gold.values[name].lower()
            scored[m] = (float(correct), "exact match" if correct else "mismatch")
            continue
        equal = next(
            (
                raw
                for raw, value in alternates[m]
                if (
                    is_blank(predicted[m])
                    if value is None
                    else text_equivalent(predicted[m], value)
                )
            ),
            None,
        )
        if equal is None:
            to_judge.append(m)
        else:
            scored[m] = (1.0, f"accepted alternate {equal!r}")
    pairs = [(predicted[m], matches[m][0].values[name]) for m in to_judge]
    scored.update(zip(to_judge, await judge.judge_cells(pairs, criterion, spend)))

    retry = [
        (m, raw, value)
        for m, values in alternates.items()
        if scored[m][0] < 1.0
        for raw, value in values
    ]
    judged = []
    for m, raw, value in retry:
        if value is None:
            if is_blank(predicted[m]):
                scored[m] = (1.0, "blank accepted")
        elif column.type == "text":
            judged.append((m, raw, value))
        elif predicted[m].lower() == value.lower():
            scored[m] = (1.0, f"accepted alternate {raw!r}")
    if judged:
        verdicts = await judge.judge_cells(
            [(predicted[m], value) for m, _, value in judged], criterion, spend
        )
        for (m, raw, _), (score, _) in zip(judged, verdicts):
            if score >= 1.0 and scored[m][0] < 1.0:
                scored[m] = (1.0, f"accepted alternate {raw!r}")
    return [scored.get(m, (None, "ungraded")) for m in range(len(matches))]


def _rejected(task: TableTask, gold_index: int, column: str, raw: str) -> str | None:
    """The audited wrong value this cell repeats, if any. Compared before lossy
    normalization, so qualifiers and punctuation still count."""
    for name, values in _at(task.rejected, gold_index, {}).items():
        if norm_column(name) == column:
            for value in values:
                if text_token(strip_links(value)) == text_token(strip_links(raw)):
                    return value
    return None


def _alternates(task: TableTask, gold_index: int, column: Column) -> list[tuple[str, str | None]]:
    """Accepted alternates for one gold cell as ``(written, normalized)``; ``None`` is an
    accepted blank."""
    return [
        (raw, normalize_cell(column, raw) if raw.strip() else None)
        for name, values in _at(task.accepted, gold_index, {}).items()
        if norm_column(name) == norm_column(column.name)
        for raw in map(str, values)
    ]


def _table_scores(
    gold_rows: list[_Row],
    response_rows: list[_Row],
    matches: list[tuple[_Row, _Row]],
    cell_scores: list[dict[str, float | None]],
    ungraded: dict[int, set[str]],
    key: str,
    width: int,
    optional: set[str],
) -> dict[str, float]:
    """Row and item precision, recall and F1, and ``complete_pass``."""
    counted = [m for m, (gold, _) in enumerate(matches) if gold.cells[key] not in optional]
    response_count = len(response_rows) - sum(gold.cells[key] in optional for gold, _ in matches)
    gold_count = sum(row.cells[key] not in optional for row in gold_rows)
    graded = [[s for s in cell_scores[m].values() if s is not None] for m in counted]
    correct_rows = sum(min(scores) for scores in graded)
    correct_cells = sum(sum(scores) for scores in graded)
    response_cells = response_count * width - sum(
        len(ungraded[matches[m][0].index]) for m in counted
    )
    gold_cells = gold_count * width - sum(
        len(ungraded[row.index]) for row in gold_rows if row.cells[key] not in optional
    )

    row_precision = correct_rows / response_count if response_count > 0 else 0.0
    row_recall = correct_rows / gold_count if gold_count > 0 else 0.0
    item_precision = correct_cells / response_cells if response_cells > 0 else 0.0
    item_recall = correct_cells / gold_cells if gold_cells > 0 else 0.0
    return {
        "row_precision": row_precision,
        "row_recall": row_recall,
        "row_f1": _f1(row_precision, row_recall),
        "item_precision": item_precision,
        "item_recall": item_recall,
        "item_f1": _f1(item_precision, item_recall),
        "complete_pass": float(row_precision == 1.0 and row_recall == 1.0),
    }


def _discovery_scores(
    gold_rows: list[_Row],
    response_rows: list[_Row],
    key: str,
    optional: set[str],
) -> dict[str, float]:
    """Precision, recall and F1 over the distinct entities named, ignoring their cells."""
    gold = [row for row in gold_rows if names_entity(row.values[key])]
    response = [row for row in response_rows if names_entity(row.values[key])]
    gold_ids, response_ids = _identities(gold, response, key)
    optional_ids = {i for i, row in zip(gold_ids, gold) if row.cells[key] in optional}
    shared = (set(gold_ids) & set(response_ids)) - optional_ids
    gold_entities = set(gold_ids) - optional_ids
    response_entities = set(response_ids) - optional_ids
    found = len(shared)
    precision = found / len(response_entities) if response_entities else 0.0
    recall = found / len(gold_entities) if gold_entities else 0.0
    return {
        "discovery_precision": precision,
        "discovery_recall": recall,
        "discovery_f1": _f1(precision, recall),
    }
