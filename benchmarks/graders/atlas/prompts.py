"""Prompts for the ATLAS aligner and cell judge.

The column-alignment, entity-alignment and judge prompts follow ByteDance-Seed/WideSearch
(MIT); the alias and exclusion guidance and the judge rules are ATLAS's own.
"""

COLUMN_ALIGNMENT_PROMPT = """Your task is to align column names from a response table to reference column names.

Response columns: {response_columns}
Reference columns: {reference_columns}

For each response column, find the matching reference column (semantically equivalent).
Output a JSON mapping from response column name to reference column name.
Only include columns that have a match. Output in this format:

```json
{{
  "response_col1": "reference_col1",
  "response_col2": "reference_col2"
}}
```"""  # noqa: E501

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

ALIAS_GUIDANCE = (
    "\nGrading-only names bound to each canonical entity: {aliases}"
    "\nMap only a single identifiable entity. Benign location or descriptive "
    "qualifiers may accompany a canonical name or alias. Multiple names in one "
    "cell are acceptable only when they all identify that same entity. Do not "
    "select one entity from distinct guesses or ignore a conflicting qualifier. "
    "A shared alias alone is ambiguous; additional context must actually "
    "disambiguate it. Leave unresolved cells unmapped."
)

EXCLUSION_GUIDANCE = (
    "\nAudited distinct entities that must never be paired with each gold key: {exclusions}"
    "\nThese exclusions apply only to the named pair. A response may still "
    "identify a different reference entity. Preserve identifying citations, "
    "instrument numbers, brands, and both ends of directed relationships."
)

# Adapted from https://github.com/ByteDance-Seed/WideSearch/blob/9825ba7/src/evaluation/metric_utils.py (MIT)
JUDGE_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria. Each answer and each response has an idx.

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

DEFAULT_CRITERION = "Score 1 if the response matches the answer, 0 otherwise."

JUDGE_GUIDANCE = (
    "Judge the fact, not surface wording, but obey the column criterion above as the complete "
    "definition of what the cell must contain. Score 1 for harmless presentation differences "
    "such as capitalization, punctuation, ordering of the same complete set, an article, or an "
    "unambiguous name/notation/unit conversion that preserves exactly the same fact. Score 0 if "
    "the response contradicts the answer; hedges between alternatives; or omits, broadens, or "
    "changes any condition, threshold, interval endpoint, subgroup, statutory/constitutional "
    "basis, dosage or chemical form, role, date, or other component required by the criterion. "
    "Never treat a qualifier as incidental merely because the shorter value is broadly related. "
    "Only ignore a substantive difference when the column criterion itself explicitly declares "
    "that precise equivalence acceptable."
)

JUDGE_EQUIVALENCE_RULES = """General equivalence rules (the column criterion above takes precedence where it is more specific):
- Score 1 when the response identifies the same person, organization, place or thing as the standard answer, however it is written: with or without middle names or middle initials; a common short form or nickname of a given name (Tom/Thomas, Jim/James); generational suffixes or honorifics and their punctuation (Jr., Jr, III); accents, transliteration or the native script; abbreviations or an added parenthetical abbreviation; an added country, city or parent-organization qualifier; a current or former official name of the same entity.
- Score 1 when the response names the standard answer's entity and only adds a sub-unit or detail (for example "X Department of Health, Food Program" for "X Department of Health"), unless the criterion asks for that sub-unit or level.
- Score 0 when the response names a different entity, a broader or narrower entity that the criterion distinguishes (for example it requires the exact legal name), omits a required part of a multi-part answer, or gives a different number, date or category.
- A response that is empty or says the value is unknown or not found scores 0 unless the standard answer is itself empty."""  # noqa: E501
