# ATLAS grader

[ATLAS](https://exa.ai/blog/atlas-benchmark) tasks ask for every entity that meets a set of
conditions, plus several attributes of each. A system answers with one Markdown table: one
row per entity, one column per attribute. This grader compares that table with the gold
table.

```bash
uv run bench run --suite atlas --system scout-exa-auto-highlights --limit 5
```

With the default Scout model (`openai/gpt-6-luna`), this system costs about $1 per task, most
of it Exa searches. `--model openai/gpt-6-astra` costs about $35 per task. Grading costs less
than a cent per task.

The dataset is gated on Hugging Face (`exa-ai/ATLAS`): accept its terms, then run
`hf auth login` or set `HF_TOKEN`.

## Metrics

Precision, recall and F1 are reported at three levels. Precision is the share of what the
response gave that is correct; recall is the share of the gold that the response got right;
F1 is their harmonic mean.

| Level | Correct means | Metrics |
|---|---|---|
| Row | A response row names a gold entity and every graded cell in it is correct | `row_precision`, `row_recall`, `row_f1` (primary) |
| Item | One cell is correct. A row's entity cell counts as correct when the row matches a gold entity, so finding an entity earns partial credit. | `item_precision`, `item_recall`, `item_f1` |
| Discovery | A named entity is a gold entity, whatever the other cells say | `discovery_precision`, `discovery_recall`, `discovery_f1` |

`complete_pass` is 1 when the whole table is right (every gold entity found, no extra rows,
every graded cell correct), else 0.

Each task scores 0 to 1 on every metric; a run reports the mean over tasks.

The gold marks two things as unscored: whole rows, and single cells inside rows.

- **Optional rows** (`optional_entities`) are entities the gold could not confirm as
  belonging to the answer, or as not belonging. Such a row is ignored entirely. A response
  that includes it gets no credit for the row and no penalty as an extra row, and none of
  its cells count. A response that omits it loses nothing.
- **Ungraded cells** (`ungraded`) are single values the gold could not settle, inside rows
  that are otherwise scored. Whatever the response writes there is not checked. The cell is
  left out of the item counts, and the row can be correct without it. The row and its other
  cells are scored as usual.

## How a response is graded

1. **Read the table.** Use the first fenced Markdown table, or else the first pipe table in
   the text. A table wrapped in JSON (`{"answer": "| ... |"}`) is unwrapped. Inline citation
   markers are removed. A response with no table scores 0.
2. **Map the columns.** Column names are compared without case or spaces. If the task's
   columns are not all present, an LLM maps the response's names onto them. A table that
   still lacks a column scores 0.
3. **One row per entity.** On each side, a repeated entity keeps its first row.
4. **Identify each row's entity.**
   - A cell that writes exactly a gold name or one of its declared aliases matches that
     entity.
   - A cell that names several entities, or uses an alias shared by several, stays
     unmatched instead of being guessed.
   - Every other name goes to an LLM aligner. The aligner sees the gold names, the aliases
     and the audited exclusions (names that must never match a given entity).
   - For products, models and software the version is part of the entity: a different
     version number, dated snapshot, size or variant does not match, so `Acme-2 Mini` is
     not `Acme-2`. A name that leaves out a version detail more than one gold entity could
     supply stays unmatched.
5. **Normalize values.**
   - **Numbers:** a cell must state exactly one number. `1,200` = `1200` and `5%` = `0.05`.
     A column measured in percentage points keeps `4.1%` = `4.1`.
   - **Dates** keep their stated precision. `March 5, 2021` = `2021-03-05`, but `2021`
     never equals a full date.
   - **Text** is compared without case, accents or extra whitespace.
6. **Pair rows.** Response rows are paired one to one with gold rows on their normalized
   entity names. When two response rows name the same gold entity, the first is paired and
   the other counts as an extra row. Two rows whose names collide after normalization stay
   distinct.
7. **Score cells.** Each graded cell of each paired row is scored 0 or 1:
   - A value the gold lists as **rejected** scores 0.
   - A **number or date** cell must equal the gold value exactly after normalization.
   - A **text** cell scores 1 when it equals the gold after normalizing case, punctuation
     and list order. Otherwise an LLM judge decides, following the column's grading
     criterion. A judge 0 is confirmed by two more votes and the majority wins; a 1
     stands.
   - A cell that fails against the gold value can still pass against one of the gold's
     **accepted alternates**.
8. **Aggregate** the cell scores into the metrics above.

The judge and aligner default to `openai/gpt-6-luna` (`--judge-model` overrides it). Within a
run, an identical (criterion, gold value, response value) pair always gets the same verdict.

## Code

| File | Contents |
|---|---|
| `grader.py` | The task model, the LLM judge and aligner, and the pipeline above (`grade_table`) |
| `normalize.py` | Reading the table and normalizing values. It makes no LLM calls. |
| `matching.py` | Entity matching: declared names, aliases and join keys |
| `prompts.py` | The aligner and judge prompts |

On the 12,034 answers behind the blog's numbers, this grader reproduces the scores of the
grader that produced them on all but one answer.

## Task format

Released records follow BrowseComp: only `id` and `canary` are plain text. `question` and
`gold` are XOR-encrypted with the record's canary string, and `data/loaders.py` decrypts
them in memory. Decrypted, `gold` holds:

| Field | Contents |
|---|---|
| `key_column` | The column that names each row's entity |
| `columns` | `{"name", "type", "criterion"}` in answer order. `type` is `text`, `number` or `date`; `criterion` is the column's grading rule. |
| `rows` | The gold table: one list of cells per entity, in `columns` order. `""` marks a cell with no value. |
| `accepted`, `rejected` | `{"row", "column", "values"}`: other values that count as right, or as wrong, for one gold cell |
| `ungraded` | `{"row", "column"}`: single gold cells that are not scored |
| `optional_entities` | Whole gold rows that are not scored |
| `entity_aliases` | `{"entity", "aliases"}`: other names that identify a gold entity |
| `entity_exclusions` | `{"entity", "names"}`: audited names that must never match it |
| `alignment_note` | An optional extra instruction for the entity aligner |
