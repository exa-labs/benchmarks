"""Turning a response into comparable values: no LLM calls.

* reading the Markdown table out of a response;
* canonical tokens for number and date cells, so that equal values compare equal;
* deterministic text equivalence, applied before any cell reaches the judge.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from dateparser.date import DateDataParser

ColumnType = Literal["text", "number", "date"]


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def norm_column(name: Any) -> str:
    """Column identity: lowercase with every space removed."""
    return str(name).strip().lower().replace(" ", "")


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType
    criterion: str | None = None


# ---------------------------------------------------------------------------
# Reading the response table
# ---------------------------------------------------------------------------

# OpenAI hosted web search writes inline citations in private-use characters, as in
# "\ue200cite\ue202turn9view0\ue201". They are rendering hints, not answer text.
_CITATION_MARKER = re.compile(r"\s*\ue200[^\ue200\ue201]*\ue201")
_MARKER_CHARACTER = re.compile(r"[\ue200-\ue2ff]")
# A bare number that ends a sentence ("1995." before a citation marker).
_NUMBER_WITH_PERIOD = re.compile(r"[-+\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\.")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")


def _split_row(line: str) -> list[str]:
    r"""Split a pipe-table row; ``\|`` inside a cell is a literal pipe."""
    cells: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and line[index + 1 : index + 2] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current))
    return cells


def _is_separator(line: str) -> bool:
    return set(line.strip()).issubset(set("|- :"))


def _table_text(response: str) -> str | None:
    """The first fenced block that holds a pipe table, else the first bare pipe table."""
    for block in re.findall(r"```(?:markdown)?(.*?)```", response, re.DOTALL):
        lines = [line for line in block.strip().split("\n") if line.strip()]
        lines = [line for line in lines if not _is_separator(line)]
        if len(lines) >= 2 and "|" in lines[0]:
            return block.strip()
    pipes = [match.start() for match in re.finditer(r"\|", response)]
    if len(pipes) < 4:
        return None
    start = response.rfind("\n", 0, pipes[0])
    end = response.find("\n", pipes[-1])
    span = response[max(start, 0) : len(response) if end == -1 else end]
    tables = re.findall(r"((?:\|.*\n?)+)", span)
    return tables[0].strip() if tables else None


def _strip_markers(text: str) -> str:
    cleaned, markers = _CITATION_MARKER.subn("", text)
    cleaned, stray = _MARKER_CHARACTER.subn("", cleaned)
    return cleaned.strip() if markers or stray else text


def _clean_cell(text: str) -> str:
    cleaned = _strip_markers(text)
    if _NUMBER_WITH_PERIOD.fullmatch(cleaned.strip()):
        cleaned = cleaned.strip()[:-1]
    return cleaned


def dedupe_columns(columns: list[str]) -> list[int]:
    """Positions of the first occurrence of each normalized column name."""
    seen: set[str] = set()
    keep = []
    for position, column in enumerate(columns):
        if column not in seen:
            seen.add(column)
            keep.append(position)
    return keep


@dataclass
class ResponseTable:
    columns: list[str]
    rows: list[list[str]]

    def select(self, positions: list[int]) -> None:
        self.columns = [self.columns[p] for p in positions]
        self.rows = [[row[p] for p in positions] for row in self.rows]


def extract_table(response: str) -> ResponseTable | None:
    """Parse the response's table, with normalized and de-duplicated column names."""
    text = _table_text(response)
    if not text:
        return None
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [line for line in lines if not _is_separator(line)]
    if len(lines) < 2:
        return None
    headers = [norm_column(cell) for cell in _split_row(lines[0]) if cell.strip()]
    if not headers:
        return None
    rows = []
    for line in lines[1:]:
        cells = _split_row(line)
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        cells = [cell.strip() for cell in cells][: len(headers)]
        rows.append(cells + [""] * (len(headers) - len(cells)))
    columns = [norm_column(_strip_markers(header)) for header in headers]
    table = ResponseTable(columns, [[_clean_cell(cell) for cell in row] for row in rows])
    table.select(dedupe_columns(table.columns))
    return table


_ANSWER_KEYS = ("output", "answer", "text", "content", "markdown", "final_answer", "response")


def _payload_text(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return next((t for k in _ANSWER_KEYS if (t := _payload_text(payload.get(k)))), None)
    if isinstance(payload, list):
        parts = [t for item in payload if (t := _payload_text(item)) and t.strip()]
        return "\n".join(parts) or None
    return str(payload)


def unwrap_answer(response: str) -> str:
    """Some systems wrap the table in JSON, as in ``{"answer": "| a | b |..."}``."""
    if response.strip().startswith(("{", "[")):
        try:
            return _payload_text(json.loads(response.strip())) or response
        except json.JSONDecodeError:
            pass
    return response


def strip_links(text: str) -> str:
    """Grade the text of ``[text](url)``, not the markup."""
    return _MARKDOWN_LINK.sub(lambda match: match.group(1), text) if "](" in text else text


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

_NUMBER_TOKEN = "number:"
_DATE_TOKEN = "date:"
_TEXT_TOKEN = "text:"
_MISSING_TOKEN = "<missing>"
_MISSING_VALUES = frozenset(
    {
        "",
        "-",
        "\u2013",
        "\u2014",
        "na",
        "n/a",
        "none",
        "null",
        "not applicable",
        "not available",
        "unknown",
    }
)
_MISSING_COMPACT = frozenset(value.replace(" ", "") for value in _MISSING_VALUES)

_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?(?![\w.])"
)
_PERCENT_POINTS = re.compile(
    r"%|\bper ?cent(?:age)?s?\b|\b(?:rate|ratio|prevalence|turnout|share)\b", re.IGNORECASE
)
_NEGATION = re.compile(r"\b(?:no|not|never|cannot|can't|mustn't)\b", re.IGNORECASE)

MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_YEAR = re.compile(r"^\d{4}年?$", re.IGNORECASE)
_YEAR_MONTH = re.compile(
    rf"^(?:\d{{4}}(?:[-/.]\d{{1,2}}|年\d{{1,2}}月?)|{MONTH}[,\s]+\d{{4}}|\d{{4}}[,\s]+{MONTH})$",
    re.IGNORECASE,
)
_DAY = re.compile(
    rf"^(?:\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|"
    rf"\d{{4}}年\d{{1,2}}月\d{{1,2}}日?|"
    rf"{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH}[,]?\s+\d{{4}}|"
    rf"\d{{4}}\s+{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?)$",
    re.IGNORECASE,
)
_DATE_PARSER = DateDataParser(
    settings={
        "DATE_ORDER": "MDY",
        "PREFER_LOCALE_DATE_ORDER": False,
        "PREFER_DAY_OF_MONTH": "first",
        "PARSERS": ["absolute-time"],
        "RELATIVE_BASE": datetime(2000, 1, 1),
        "RETURN_AS_TIMEZONE_AWARE": False,
    }
)


def normalize_text(text: str) -> str:
    """NFKC, casefold, drop asterisks, collapse whitespace."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold().strip().replace("*", "")
    return re.sub(r"\s+", " ", normalized)


def text_token(text: str) -> str:
    normalized = normalize_text(text)
    return _MISSING_TOKEN if normalized in _MISSING_VALUES else f"{_TEXT_TOKEN}{normalized}"


def _number_value(text: str, percent_points: bool) -> Decimal | None:
    """The one number a cell states, as an exact decimal; ``None`` for anything else.

    A percent divides by 100 unless the column states percentages in points
    (``percent_points``), where ``4.1%`` and ``4.1`` are the same value.
    """
    text = unicodedata.normalize("NFKC", str(text)).strip().replace("\u2212", "-")
    matches = list(_NUMBER.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    token = match.group(0)
    is_percent = token.endswith("%") or bool(
        re.search(r"\bpercent\b", text[match.end() :], re.IGNORECASE)
    )
    try:
        value = Decimal(token.removesuffix("%").replace(",", ""))
    except InvalidOperation:
        return None
    if is_percent and not percent_points:
        value /= Decimal(100)
    return value


def normalize_number(text: str, percent_points: bool = False) -> str:
    """Canonical token for a number cell; text that is not one number stays text."""
    value = _number_value(text, percent_points)
    if value is None:
        return text_token(text)
    return f"{_NUMBER_TOKEN}{'0' if value == 0 else format(value.normalize(), 'f')}"


def _date_precision(text: str) -> Literal["year", "month", "day"] | None:
    text = unicodedata.normalize("NFKC", str(text)).strip()
    if _YEAR.fullmatch(text):
        return "year"
    if _YEAR_MONTH.fullmatch(text):
        return "month"
    if _DAY.fullmatch(text):
        return "day"
    return None


def normalize_date(text: str) -> str:
    """Canonical token for an absolute date at its stated precision (year, month or day)."""
    text = unicodedata.normalize("NFKC", str(text)).strip()
    precision = _date_precision(text)
    if precision is None:
        return text_token(text)
    parsed = _DATE_PARSER.get_date_data(text)
    if parsed.date_obj is None or parsed.period != precision:
        return text_token(text)
    date = parsed.date_obj
    value = {
        "year": f"{date.year:04d}",
        "month": f"{date.year:04d}-{date.month:02d}",
        "day": date.strftime("%Y-%m-%d"),
    }[precision]
    return f"{_DATE_TOKEN}{precision}:{value}"


def key_text(text: str) -> str:
    """Key cells, normalized: lowercase, no spaces or asterisks."""
    return str(text).lower().strip().replace(" ", "").replace("*", "")


def normalize_cell(column: Column, text: str, *, key: bool = False) -> str:
    if key:
        return key_text(text)
    if column.type == "number":
        percent_points = bool(
            _PERCENT_POINTS.search(norm_column(column.name))
            or _PERCENT_POINTS.search(column.criterion or "")
        )
        return normalize_number(text, percent_points)
    if column.type == "date":
        return normalize_date(text)
    return normalize_text(text)


def is_empty(value: str) -> bool:
    text = str(value).strip()
    return not text or text == _MISSING_TOKEN or text.casefold() == "nan"


def is_blank(value: str) -> bool:
    """Empty, or a missing-value marker such as "N/A" or "unknown"."""
    return is_empty(value) or normalize_text(value).replace(" ", "") in _MISSING_COMPACT


# ---------------------------------------------------------------------------
# Deterministic text equivalence (before the judge)
# ---------------------------------------------------------------------------


def _judge_norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold().replace("*", "")
    # A sign before a number is part of the value; other dashes are presentation.
    text = re.sub(r"(?<![\w.])[-\u2212\u2010-\u2015](?=\d)", " minus ", text)
    text = re.sub(r"[\u2010-\u2015_-]", " ", text)
    text = re.sub(r"[^\w\s%.]", " ", text)
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def _parts(text: str) -> list[str]:
    pieces = re.split(r"\s*(?:;|,|/|\band\b|&)\s*", unicodedata.normalize("NFKC", text).casefold())
    return sorted(part for part in map(_judge_norm, pieces) if part)


def text_equivalent(response: str, target: str) -> bool:
    """Equal after presentation normalization (case, punctuation, list order), or two
    missing-value markers."""
    if normalize_text(response) in _MISSING_VALUES and normalize_text(target) in _MISSING_VALUES:
        return True
    left, right = _judge_norm(response), _judge_norm(target)
    if not left or not right:
        return False
    if left == right:
        return True
    response_parts = _parts(response)
    return len(response_parts) > 1 and response_parts == _parts(target)


def _criterion_says(criterion: str, rule: str) -> bool:
    """Whether one sentence of the column criterion states ``rule`` without a negation."""
    return any(
        re.search(rule, sentence) and not _NEGATION.search(sentence)
        for sentence in re.split(r"[.!?]\s*|\n+", normalize_text(criterion))
    )


def middle_initial_equivalent(response: str, target: str, criterion: str) -> bool:
    """Where the criterion accepts names "with or without middle initials", match
    ``First M. Last`` to ``First Last``: one dotted initial, nothing else changed."""
    if not _criterion_says(criterion, r"\baccept\b.*\bwith or without middle\b.*\binitials?\b"):
        return False
    shorter, longer = sorted(
        [normalize_text(response).split(), normalize_text(target).split()], key=len
    )
    if not (2 <= len(shorter) <= 5 and len(longer) == len(shorter) + 1):
        return False
    if not re.fullmatch(r"[^\W\d_]\.", longer[1]) or longer[:1] + longer[2:] != shorter:
        return False
    word = r"[^\W\d_]+(?:[-'\u2019][^\W\d_]+)*"
    if not all(re.fullmatch(word + r"\.?", token) for token in shorter):
        return False
    if not re.fullmatch(word, shorter[0]) or len(shorter[0]) < 2:
        return False
    if shorter[0] in {"dr", "prof", "mr", "mrs", "ms", "sir", "dame", "lord", "lady"}:
        return False
    return len(shorter[-1].rstrip(".")) >= 2
