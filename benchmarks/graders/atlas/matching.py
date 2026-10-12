"""Matching response rows to gold entities: declared names, join keys, identities."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from benchmarks.graders.atlas.normalize import MONTH

# Typographic variants that never change a declared name: curly quotes, dashes, no-break space.
_TYPOGRAPHIC_FOLD = str.maketrans(
    {
        **dict.fromkeys("\u2018\u2019\u201a\u201b\u2032", "'"),
        **dict.fromkeys("\u201c\u201d\u201e\u2033", '"'),
        **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"),
        "\u00a0": " ",
    }
)
# What may separate distinct declared names in a cell that lists several guesses.
_GUESS_SEPARATORS = re.compile(r"(?:[\W_]|\b(?:and|or)\b)*")


def _name_form(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHIC_FOLD).casefold().split()
    )


def match_declared_names(
    gold_keys: list[str],
    response_keys: list[str],
    aliases: dict[str, list[str]],
) -> tuple[dict[str, str], set[str]]:
    """Map response cells that are exactly a gold name or one of its aliases.

    A cell may list names of one entity (``A, B`` or ``A (B)``). A cell that fits
    several gold rows, or lists names of different entities, is *blocked*: left
    unmatched rather than guessed. Exact canonical names claim their rows first; each
    gold row matches once. Returns ``(response -> gold key, blocked cells)``.
    """
    patterns = {}
    owners: dict[str, set[str]] = {}
    for key in gold_keys:
        names = {_name_form(name) for name in [key, *aliases.get(key, [])]}
        for name in names:
            owners.setdefault(name, set()).add(key)
        name = "(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + ")"
        sequence = name + r"(?:\s*[,;/|]\s*" + name + r")*"
        patterns[key] = re.compile(sequence + r"(?:\s*\(\s*" + sequence + r"\s*\))?")

    def lists_distinct_entities(text: str) -> bool:
        mentions = [
            (match.start(), match.end(), keys)
            for name, keys in owners.items()
            for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text)
        ]
        # A name inside a longer declared name is part of that name, not a second guess.
        kept = [
            (start, end, keys)
            for start, end, keys in mentions
            if not any(s < end and start < e and e - s > end - start for s, e, _ in mentions)
        ]
        if len(kept) < 2 or set.intersection(*(keys for _, _, keys in kept)):
            return False
        residue = list(text)
        for start, end, _ in kept:
            residue[start:end] = [" "] * (end - start)
        return _GUESS_SEPARATORS.fullmatch("".join(residue)) is not None

    mapping = {v: v for v in response_keys if v in gold_keys}
    claimed = set(mapping.values())
    blocked = set()
    for value in response_keys:
        if value in mapping:
            continue
        form = _name_form(value)
        candidates = [key for key, pattern in patterns.items() if pattern.fullmatch(form)]
        if len(candidates) == 1 and candidates[0] not in claimed:
            mapping[value] = candidates[0]
            claimed.add(candidates[0])
        elif candidates or lists_distinct_entities(form):
            blocked.add(value)
    return mapping, blocked


def join_key(value: str) -> str:
    """Conservative key normalization: accents, case, "&"/"and", a company suffix,
    a parenthetical month-day after a year, commas, periods and spaces."""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = text.casefold().replace("&", "and")
    text = re.sub(
        rf"^(?P<year>\d{{4}})\s*\({MONTH}\.?\s*\d{{1,2}}(?:st|nd|rd|th)?\.?\)\s*$",
        r"\g<year>",
        text,
    )
    text = re.sub(r"(?:,\s*)?\b(?:inc|ltd|llc|corp|co)\.?\s*$", "", text)
    return re.sub(r"\s+", "", re.sub(r"[,.]", "", text))


def entity_identities(keys: list[tuple[str, str]], others: list[tuple[str, str]]) -> list[tuple]:
    """Identity of each ``(normalized key, display)``: its ``join_key``, unless that join key
    repeats on either side, where the exact key keeps distinct rows distinct."""
    counts = Counter(join_key(value) for value, _ in keys)
    other_counts = Counter(join_key(value) for value, _ in others)
    identities = []
    for value, display in keys:
        normalized = join_key(value)
        ambiguous = counts[normalized] > 1 or other_counts[normalized] > 1
        identities.append(("exact", value, display) if ambiguous else ("normalized", normalized))
    return identities


_NOT_AN_ENTITY = frozenset({"", "nan", "none", "null", "<na>"})


def names_entity(value: str) -> bool:
    return str(value).strip().lower() not in _NOT_AN_ENTITY
