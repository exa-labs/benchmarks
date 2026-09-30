"""WideSearch: broad information-gathering tasks answered as one Markdown table.

Source: Hugging Face ``ByteDance-Seed/WideSearch`` (200 tasks, 100 English and 100 Chinese)
at a pinned commit: ``widesearch.jsonl`` plus one gold CSV per task under
``widesearch_gold/``. The system receives the task's ``query`` verbatim; it already states the
required columns and the ``````markdown`` output format.

Grading reproduces the official evaluator (WideSearch commit 9825ba7,
``src/evaluation/``) as ported by the grader Exa uses for its published comparisons:
extract the response table, align its columns and primary-key values to the gold with
an LLM, normalize cells with each column's ``preprocess`` steps, inner-join on the
primary key, and score every non-key cell with its column's metrics (exact_match,
url_match, in_match, number_near, date_near, llm_judge; the minimum across a column's
metrics). A row counts when all its cells pass. Per task this yields row-level and
item-level precision / recall / F1 and ``success_rate`` (the official ``score``: 1.0 when
the whole table matches). The primary metric is ``f1_by_row``, averaged over tasks.

Differences from the official evaluator, all following the grader Exa uses:

* the column/key alignment and cell-judge prompts are lightly edited versions of the
  official ones, and cell pairs are listed as ``idx_i: answer=..., response=...`` lines;
* an alignment reply that is not JSON is retried and then fails the grade (the
  official code silently uses an empty mapping); an unparseable cell-judge reply scores
  that column 0, as upstream does;
* duplicate column names are dropped (first kept) and only required columns are
  preprocessed;
* any other evaluator exception fails the grade instead of scoring 0. A response
  whose table cannot be parsed still scores 0, as upstream does.

Paper: https://arxiv.org/abs/2508.07999
"""

from __future__ import annotations

import json
import re
from io import StringIO
from typing import Any
from urllib.parse import urlparse

import dateparser
import pandas as pd

from benchmarks.graders.base import JudgeSpend, complete_parsed, is_blank
from data import loaders
from data.loaders import norm_column
from harness.llm.judge import Judge
from harness.suite import Grade, Suite, Task

SCORE_KEYS = (
    "success_rate",
    "precision_by_row",
    "recall_by_row",
    "f1_by_row",
    "precision_by_item",
    "recall_by_item",
    "f1_by_item",
)

# Adapted from https://github.com/ByteDance-Seed/WideSearch/blob/9825ba7/src/evaluation/metric_utils.py (MIT)
ENTITY_ALIGNMENT_PROMPT = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.

The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:

```json
{{
  "origin_str1": "transform_str1",
  "origin_str2": "transform_str2"
}}
```"""  # noqa: E501

# Adapted from https://github.com/ByteDance-Seed/WideSearch/blob/9825ba7/src/evaluation/metric_utils.py (MIT)
LLM_JUDGE_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria. Each answer and each response has an idx.

Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results.

Output in Markdown JSON format:
```json
{{
  "idx_0": score,
  "idx_1": score,
  ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results."""  # noqa: E501

DEFAULT_JUDGE_CRITERION = "Score 1 if the response matches the answer, 0 otherwise."


# ---------------------------------------------------------------------------
# Table extraction and column handling
# ---------------------------------------------------------------------------


def extract_table(response: str) -> pd.DataFrame | None:
    """Parse the response's first Markdown table the way the official evaluator does.

    Prefers a ``````markdown`` fence, otherwise the pipe-delimited block spanning the first
    to last ``|``. Returns ``None`` when there is no table. Raises
    ``pandas.errors.ParserError`` when the table is malformed.
    """
    tables = re.findall(r"```markdown(.*?)```", response, re.DOTALL)
    if not tables:
        pipes = [match.start() for match in re.finditer(r"\|", response)]
        if len(pipes) >= 4:
            start = response.rfind("\n", 0, pipes[0])
            start = 0 if start == -1 else start
            end = response.find("\n", pipes[-1])
            end = len(response) if end == -1 else end
            tables = re.findall(r"((?:\|.*\n?)+)", response[start:end])
    if not tables:
        return None

    lines = tables[0].strip().split("\n")
    lines[0] = lines[0].replace(" ", "").lower()
    kept = []
    for line in (line.strip() for line in lines):
        if set(line).issubset(set("|- :")) or "|" not in line:
            continue
        kept.append("|".join(part.strip() for part in line.split("|")))
    frame = pd.read_csv(StringIO("\n".join(kept)), sep="|")
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed")]


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and keep only the first of any duplicates."""
    frame = frame.copy()
    frame.columns = [norm_column(column) for column in frame.columns]
    return frame.loc[:, ~frame.columns.duplicated(keep="first")].copy()


def stringify(value: Any) -> str:
    """``astype(str)`` as pandas 2 did it, including missing nullable values as "nan"."""
    if value is pd.NA:
        return "nan"
    return str(value)


# ---------------------------------------------------------------------------
# Cell preprocessing and metrics (official registries)
# ---------------------------------------------------------------------------


def extract_number(content: str) -> str:
    """First number in the text (commas removed), or "NULL"."""
    numbers = re.findall(r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?", str(content).replace(",", ""))
    return numbers[0] if numbers else "NULL"


def norm_str(content: str) -> str:
    """Lowercase, trim, and drop spaces and asterisks."""
    return str(content).lower().strip().replace(" ", "").replace("*", "")


def norm_date(content: str) -> str:
    """Parse a date to YYYY-MM-DD, leaving unparseable text unchanged."""
    parsed = dateparser.parse(content, settings={"PREFER_DAY_OF_MONTH": "first"})
    return parsed.strftime("%Y-%m-%d") if parsed else str(content).strip()


PREPROCESSORS = {"extract_number": extract_number, "norm_str": norm_str, "norm_date": norm_date}


def exact_match(response: str, target: str) -> float:
    """Case-insensitive equality."""
    return 1.0 if response.lower() == target.lower() else 0.0


_URL = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def url_match(response: str, target: str) -> float:
    """Same set of URL hosts in both cells."""
    response_hosts = {urlparse(url).netloc for url in _URL.findall(response)}
    target_hosts = {urlparse(url).netloc for url in _URL.findall(target)}
    return 1.0 if response_hosts == target_hosts else 0.0


def in_match(response: str, target: str) -> float:
    """The response cell is a substring of the target cell."""
    return 1.0 if response in target else 0.0


def _parse_number(text: str) -> float | None:
    """Parse a float, reading a trailing percent sign as /100."""
    try:
        return float(text.replace("%", "")) / 100.0 if "%" in text else float(text)
    except (ValueError, TypeError):
        return None


def number_near(response: str, target: str, criterion: float) -> float:
    """Relative tolerance match; two identical non-numeric strings also match."""
    response_number = _parse_number(response)
    target_number = _parse_number(target)
    if response_number is None or target_number is None:
        both_text = response_number is None and target_number is None
        return 1.0 if both_text and response == target else 0.0
    return 1.0 if abs(response_number - target_number) <= abs(target_number) * criterion else 0.0


def _parse_date(text: str) -> Any:
    """dateparser with the official settings; any parser failure reads as no date."""
    try:
        return dateparser.parse(text, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:  # noqa: BLE001 - the official matcher treats parser errors as no date
        return None


def date_near(response: str, target: str) -> float:
    """Dates within 31 days; two unparseable values also match (official behavior)."""
    response_date = _parse_date(response)
    target_date = _parse_date(target)
    if response_date is None or target_date is None:
        return 1.0 if response_date is None and target_date is None else 0.0
    return 1.0 if abs((response_date - target_date).days) <= 31 else 0.0


def cell_metric(name: str, response: str, target: str, criterion: Any) -> float:
    """Score one cell pair with a non-LLM metric; unknown metric names score 0."""
    if name == "number_near":
        return number_near(response, target, criterion if criterion is not None else 0.1)
    if name == "date_near":
        return date_near(response, target)
    if name == "in_match":
        return in_match(response, target)
    if name == "exact_match":
        return exact_match(response, target)
    if name == "url_match":
        return url_match(response, target)
    return 0.0


# ---------------------------------------------------------------------------
# Judge reply parsing
# ---------------------------------------------------------------------------


def _last_json_object(text: str) -> dict | None:
    """The last top-level JSON object anywhere in free text."""
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


def parse_json_mapping(text: str) -> dict | None:
    """The last ```json-fenced object in a reply, else its last bare JSON object."""
    for candidate in reversed(re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return _last_json_object(text)


def _require_mapping(text: str) -> dict:
    """Parse an alignment reply, raising ``ValueError`` when it holds no JSON object."""
    parsed = parse_json_mapping(text) if text else None
    if parsed is None:
        raise ValueError(f"alignment reply has no JSON object: {text[:200]!r}")
    return parsed


def judge_scores(text: str, count: int) -> list[float] | None:
    """Per-pair 0/1 scores from a cell-judge reply; ``None`` when it has no JSON."""
    parsed = parse_json_mapping(text) if text else None
    if not parsed:
        return None
    scores = []
    for index in range(count):
        try:
            scores.append(float(parsed.get(f"idx_{index}", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 1e-9 else 0.0


def table_scores(
    success: float, true_rows: float, true_items: float, pred_rows: int, gold_rows: int, width: int
) -> dict[str, float]:
    """Row- and item-level precision / recall / F1 from true-positive counts.

    ``success_rate`` becomes 1.0 when all four precisions and recalls are 1.0, as in the
    official evaluator.
    """
    precision_by_row = true_rows / pred_rows if pred_rows else 0.0
    recall_by_row = true_rows / gold_rows if gold_rows else 0.0
    precision_by_item = true_items / (pred_rows * width) if pred_rows * width else 0.0
    recall_by_item = true_items / (gold_rows * width) if gold_rows * width else 0.0
    if min(precision_by_row, recall_by_row, precision_by_item, recall_by_item) == 1.0:
        success = 1.0
    return {
        "success_rate": success,
        "precision_by_row": precision_by_row,
        "recall_by_row": recall_by_row,
        "f1_by_row": _f1(precision_by_row, recall_by_row),
        "precision_by_item": precision_by_item,
        "recall_by_item": recall_by_item,
        "f1_by_item": _f1(precision_by_item, recall_by_item),
    }


def zero_scores() -> dict[str, float]:
    """Scores for a response with no usable table."""
    return dict.fromkeys(SCORE_KEYS, 0.0)


async def _align(
    judge: Judge, response_values: list[Any], reference: list[Any], spend: JudgeSpend, what: str
) -> dict:
    """Ask the judge to map response vocabulary onto the reference vocabulary."""
    prompt = ENTITY_ALIGNMENT_PROMPT.format(response=response_values, reference=reference)
    mapping, _ = await complete_parsed(judge, prompt, _require_mapping, spend, what=what)
    return mapping


async def grade_table(
    answer: dict[str, Any], response: str, judge: Judge, spend: JudgeSpend
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score one response table against the gold; returns (scores, details)."""
    required: list[str] = list(dict.fromkeys(answer["required_columns"]))
    unique: list[str] = list(dict.fromkeys(answer["unique_columns"]))
    pipeline: dict[str, dict[str, Any]] = answer["eval_pipeline"]
    details: dict[str, Any] = {}

    gold = normalize_columns(pd.read_csv(StringIO(answer["gold_csv"])))
    try:
        table = extract_table(response)
    except pd.errors.ParserError as error:
        details["error"] = f"response table did not parse: {error}"
        return zero_scores(), details
    if table is None or table.empty:
        details["error"] = "no table in response"
        return zero_scores(), details
    table = normalize_columns(table)

    if set(required) != set(table.columns):
        column_map = await _align(
            judge, list(table.columns), required, spend, "widesearch column alignment"
        )
        details["column_alignment"] = column_map
        table = normalize_columns(table.rename(columns=column_map))
        if set(required) != set(table.columns):
            details["error"] = (
                f"columns after alignment {sorted(map(str, table.columns))} "
                f"!= required {sorted(required)}"
            )
            return zero_scores(), details

    table = table[required].copy()
    gold = gold[required].copy()
    for column in required:
        gold_int = pd.api.types.is_integer_dtype(gold[column].dtype)
        gold_float = pd.api.types.is_float_dtype(gold[column].dtype)
        table_int = pd.api.types.is_integer_dtype(table[column].dtype)
        table_float = pd.api.types.is_float_dtype(table[column].dtype)
        if table_int and gold_float:
            table[column] = table[column].astype(float)
        elif table_float and gold_int:
            gold[column] = gold[column].astype(float)
        gold[column] = gold[column].apply(stringify)
        table[column] = table[column].apply(stringify)

    if unique:
        table = table.drop_duplicates(subset=unique).reset_index(drop=True)
        gold = gold.drop_duplicates(subset=unique).reset_index(drop=True)

    key_alignment: dict[str, dict] = {}
    for column in unique:
        metrics = pipeline.get(column, {}).get("metric", [])
        if "llm_judge" in metrics or "exact_match" in metrics:
            key_map = await _align(
                judge,
                table[column].tolist(),
                gold[column].tolist(),
                spend,
                f"widesearch key alignment ({column})",
            )
            key_alignment[column] = key_map
            if key_map:
                table[column] = table[column].apply(lambda value, m=key_map: m.get(value, value))
    details["key_alignment"] = key_alignment

    for column, spec in pipeline.items():
        if column not in required:
            continue
        for name in spec.get("preprocess", []):
            step = PREPROCESSORS.get(name)
            if step is None:
                continue
            table[column] = table[column].apply(step)
            gold[column] = gold[column].apply(step)

    success = 0.0
    if gold.shape == table.shape:
        gold_sorted = gold.sort_values(by=required).reset_index(drop=True)
        table_sorted = table.sort_values(by=required).reset_index(drop=True)
        if gold_sorted.equals(table_sorted):
            success = 1.0

    pred_rows, gold_rows, width = len(table), len(gold), len(required)
    details.update(response_rows=pred_rows, gold_rows=gold_rows)
    if not unique:
        return dict.fromkeys(SCORE_KEYS, success), details

    joined = pd.merge(gold, table, on=unique, how="inner", suffixes=("_gold", "_response"))
    details["matched_rows"] = len(joined)
    if joined.empty:
        return {**zero_scores(), "success_rate": success}, details

    cell_scores = pd.DataFrame(index=joined.index)
    unparsed_judge_columns = []
    for column in required:
        if column in unique:
            cell_scores[f"{column}_exact_match"] = 1.0
            continue
        spec = pipeline.get(column, {})
        criterion = spec.get("criterion")
        responses = joined[f"{column}_response"].tolist()
        targets = joined[f"{column}_gold"].tolist()
        for name in spec.get("metric", []):
            if name == "llm_judge":
                listing = "\n".join(
                    f"idx_{i}: answer={targets[i]}, response={responses[i]}"
                    for i in range(len(responses))
                )
                reply = await judge.complete(
                    LLM_JUDGE_PROMPT.format(
                        criterion=criterion or DEFAULT_JUDGE_CRITERION, response=listing
                    )
                )
                spend.add(reply)
                scores = judge_scores(reply.text, len(responses))
                if scores is None:
                    unparsed_judge_columns.append(column)
                    scores = [0.0] * len(responses)
            else:
                scores = [
                    cell_metric(name, str(r), str(t), criterion)
                    for r, t in zip(responses, targets, strict=True)
                ]
            cell_scores[f"{column}_{name}"] = pd.Series(scores, index=joined.index)
    if unparsed_judge_columns:
        details["unparsed_judge_columns"] = unparsed_judge_columns

    true_rows = float(cell_scores.min(axis=1).sum())
    true_items = float(cell_scores.sum().sum())
    return table_scores(success, true_rows, true_items, pred_rows, gold_rows, width), details


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class WideSearch(Suite):
    """ByteDance WideSearch with the official table evaluator."""

    name = "widesearch"
    description = "WideSearch: 200 table-building tasks (English and Chinese)"
    primary_metric = "f1_by_row"
    revision = f"{loaders.WIDESEARCH_REVISION}+grader-v1"

    def load(self) -> list[Task]:
        return loaders.load_widesearch()

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Score the response table against the task's gold table."""
        spend = JudgeSpend()
        if is_blank(response):
            return spend.grade(zero_scores(), {"empty_response": True})
        scores, details = await grade_table(task.answer, response, judge, spend)
        return spend.grade(scores, details)
