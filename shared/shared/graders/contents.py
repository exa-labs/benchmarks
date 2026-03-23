import logging
import re

from pydantic import BaseModel, Field

from .base import BaseLLMGrader, GradeResult

logger = logging.getLogger(__name__)


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _paragraphs(markdown: str, min_len: int = 40) -> list[str]:
    raw = re.split(r"\n\s*\n", markdown)
    return [_normalize_ws(p) for p in raw if len(_normalize_ws(p)) >= min_len]


def paragraph_containment(golden: str, extracted: str) -> float:
    gold_paras = _paragraphs(golden)
    if not gold_paras:
        return 1.0
    ext_normalized = _normalize_ws(extracted)
    found = sum(1 for p in gold_paras if p in ext_normalized)
    return found / len(gold_paras)


def _extract_headings(markdown: str) -> list[str]:
    headings = []
    for line in markdown.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if m:
            level = len(m.group(1))
            text = _normalize_ws(m.group(2))
            if text:
                headings.append(f"{'#' * level} {text}")
    return headings


def heading_similarity(golden: str, extracted: str) -> float:
    g = set(_extract_headings(golden))
    e = set(_extract_headings(extracted))
    if not g and not e:
        return 1.0
    if not g or not e:
        return 0.0
    return len(g & e) / len(g | e)


def _count_fenced_blocks(markdown: str) -> int:
    return len(re.findall(r"^(?:```|~~~)", markdown, re.MULTILINE))


def _count_tables(markdown: str) -> int:
    table_rows = re.findall(r"^\|.+\|", markdown, re.MULTILINE)
    return max(0, len(table_rows) // 3)


def code_block_recall(golden: str, extracted: str) -> float:
    g = _count_fenced_blocks(golden)
    if g == 0:
        return 1.0
    e = _count_fenced_blocks(extracted)
    return min(e / g, 1.0)


def table_recall(golden: str, extracted: str) -> float:
    g = _count_tables(golden)
    if g == 0:
        return 1.0
    e = _count_tables(extracted)
    return min(e / g, 1.0)


def rouge_l(golden: str, extracted: str) -> float:
    g_tokens = golden.split()
    e_tokens = extracted.split()
    if not g_tokens or not e_tokens:
        return 0.0

    MAX_TOKENS = 10_000
    if len(g_tokens) > MAX_TOKENS:
        g_tokens = g_tokens[:MAX_TOKENS]
    if len(e_tokens) > MAX_TOKENS:
        e_tokens = e_tokens[:MAX_TOKENS]

    m, n = len(g_tokens), len(e_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if g_tokens[i - 1] == e_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs_len = prev[n]
    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_structural_metrics(golden: str, extracted: str) -> dict[str, float]:
    return {
        "paragraph_containment": round(paragraph_containment(golden, extracted), 3),
        "heading_similarity": round(heading_similarity(golden, extracted), 3),
        "code_block_recall": round(code_block_recall(golden, extracted), 3),
        "table_recall": round(table_recall(golden, extracted), 3),
        "rouge_l": round(rouge_l(golden, extracted), 3),
    }


class ContentsEvalResult(BaseModel):
    completeness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well the extraction targets the golden content. Considers both "
            "presence of golden information AND focus of extraction. Returning "
            "the entire page to capture a small section scores low. "
            "1.0 = all golden content present in a focused extraction."
        ),
    )
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Correctness of the content that IS present. 1.0 = all content is "
            "faithful to the original, 0.0 = severe distortions/hallucinations."
        ),
    )
    structure: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well document structure is preserved: heading hierarchy, lists, "
            "tables, code blocks, paragraph boundaries. 1.0 = perfect structure."
        ),
    )
    noise: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Amount of extraneous content. Use length ratio as a signal: "
            "~1x golden = 0.0, 2-5x = 0.4, 5-10x = 0.6, 10-50x = 0.8, 50x+ = 1.0. "
            "Content beyond the golden reference is noise by definition."
        ),
    )
    missing_elements: list[str] = Field(
        description=(
            "Specific pieces of content from the golden reference that are "
            "absent in the extraction. Each entry should be a brief description."
        ),
    )
    extra_elements: list[str] = Field(
        description=(
            "Extraneous content not in the golden reference: navigation, ads, "
            "footers, cookie banners, related articles, etc."
        ),
    )
    explanation: str = Field(
        description="2-3 sentence overall assessment of extraction quality.",
    )


CONTENTS_EVAL_SYSTEM_PROMPT = """\
You are an expert evaluator of web content extraction systems. You will be given:
1. **Golden content** -- a human-verified reference extraction from page screenshots
2. **Extracted content** -- output from an automated extraction pipeline

Your job is to score how well the automated extraction captures the golden content.

The golden content represents the *ideal* extraction: exactly the substantive body
content of the page, nothing more. A perfect extractor would return text that closely
matches the golden in both content and length.

## Scoring Rubric

### Completeness (0.0-1.0)
How well does the extraction *target* the golden content? This measures precision
of extraction, not just containment. Returning the entire DOM of a website to
capture a 500-character article is NOT complete -- it is a failure to extract.

Consider both: (a) is the golden information present? and (b) is the extraction
focused on it rather than burying it in unrelated content?

- 1.0: All golden content present AND extraction is focused -- output length is
  within ~2x of the golden length with no significant extraneous content
- 0.8: All or nearly all golden content present, output is somewhat longer than
  golden (2-5x) but the extra content is at least from the same page section
- 0.6: Most golden content present but extraction is unfocused -- output may be
  5-10x the golden length, or noticeable golden content is missing
- 0.4: Golden content is partially present but buried in much larger output
  (10x+ golden length), or substantial golden content is missing
- 0.2: Only fragments of golden content present, or golden content is a tiny
  fraction of a massive extraction
- 0.0: Golden content is absent or completely lost in noise

### Accuracy (0.0-1.0)
Of the content that IS present, how faithful is it?
- 1.0: Perfect -- all numbers, names, dates, code exactly match the golden reference
- 0.8: Minor issues -- small formatting differences but information is correct
- 0.5: Some errors -- a few wrong numbers, garbled text, or misattributed content
- 0.2: Significant errors -- content is distorted, wrong, or from a different section
- 0.0: Unreliable -- content contradicts or hallucinates relative to the golden version

### Structure (0.0-1.0)
How well is document structure preserved?
- 1.0: Perfect structure -- heading levels, lists, tables, code blocks all match
- 0.8: Good structure -- minor formatting issues (e.g. flat list instead of nested)
- 0.5: Partial structure -- some headings present but hierarchy wrong, tables broken
- 0.2: Poor structure -- mostly flat text, structure largely lost
- 0.0: No structure -- raw text dump with no formatting

### Noise (0.0-1.0)
How much extraneous content pollutes the extraction? (Higher = noisier = worse)

Use the ratio of extracted length to golden length as a strong signal. Content
beyond the golden reference is noise by definition -- navigation, sidebars, ads,
footers, related articles, other page sections, or duplicate/repeated content.

- 0.0: Pristine -- extraction closely matches golden length (within ~1.5x), only
  body content, no extraneous elements
- 0.2: Mostly clean -- small amount of extra content (1.5-2x golden length),
  minor nav remnants or a footer link
- 0.4: Noticeable noise -- extraction is 2-5x the golden length, some boilerplate
  or unrelated page sections mixed in
- 0.6: Significant noise -- extraction is 5-10x the golden length, substantial
  boilerplate, sidebars, or other page sections dominate
- 0.8: Very noisy -- extraction is 10-50x the golden length, mostly boilerplate
  with golden content buried within
- 1.0: Dominated by noise -- extraction is 50x+ the golden length, or golden
  content is virtually unfindable in the output

## Output
Provide scores for each dimension, list specific missing and extra elements,
and write a brief overall explanation."""


class ContentsGrader(BaseLLMGrader):
    def __init__(
        self,
        model: str = "gpt-5.4",
        temperature: float = 1.0,
        max_characters: int = 200_000,
        api_key: str | None = None,
    ):
        super().__init__(model=model, temperature=temperature, api_key=api_key)
        self.max_characters = max_characters

    async def grade(
        self,
        url: str,
        golden_content: str,
        extracted_content: str,
    ) -> GradeResult:
        golden_len = len(golden_content)
        extracted_len = len(extracted_content)

        if not extracted_content.strip():
            return GradeResult(
                scores={
                    "completeness": 0.0,
                    "accuracy": 0.0,
                    "structure": 0.0,
                    "noise": 0.0,
                    "composite_quality": 0.0,
                    **{f"det_{k}": v for k, v in _empty_det_scores(golden_len, 0).items()},
                },
                details={"explanation": "Extraction returned empty content."},
            )

        if not golden_content.strip():
            return GradeResult(
                scores={
                    "completeness": 0.0,
                    "accuracy": 0.0,
                    "structure": 0.0,
                    "noise": 0.0,
                    "composite_quality": 0.0,
                    **{f"det_{k}": v for k, v in _empty_det_scores(0, extracted_len).items()},
                },
                details={"explanation": "No golden content available for comparison."},
            )

        golden_truncated = golden_content[: self.max_characters]
        extracted_truncated = extracted_content[: self.max_characters]

        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": CONTENTS_EVAL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"## URL\n{url}\n\n## Golden Content (reference)\n{golden_truncated}",
                    },
                    {
                        "role": "user",
                        "content": f"## Extracted Content (to evaluate)\n{extracted_truncated}",
                    },
                ],
                response_format=ContentsEvalResult,
            )
            parsed = response.choices[0].message.parsed
            assert parsed is not None
        except Exception as e:
            logger.warning(f"Contents grading failed: {e}")
            return GradeResult(
                scores={
                    "completeness": 0.0,
                    "accuracy": 0.0,
                    "structure": 0.0,
                    "noise": 0.0,
                    "composite_quality": 0.0,
                    **{f"det_{k}": v for k, v in _empty_det_scores(golden_len, extracted_len).items()},
                },
                details={"explanation": f"LLM grading failed: {e}"},
            )

        structural = compute_structural_metrics(golden_content, extracted_content)
        length_ratio = extracted_len / golden_len if golden_len > 0 else 0.0
        char_delta = extracted_len - golden_len
        char_delta_pct = (char_delta / golden_len * 100.0) if golden_len > 0 else 0.0

        quality = (
            0.5 * parsed.completeness + 0.25 * parsed.accuracy + 0.25 * parsed.structure
        ) * (1.0 - parsed.noise)

        scores: dict[str, float] = {
            "completeness": parsed.completeness,
            "accuracy": parsed.accuracy,
            "structure": parsed.structure,
            "noise": parsed.noise,
            "composite_quality": round(quality, 3),
            "det_paragraph_containment": structural["paragraph_containment"],
            "det_heading_similarity": structural["heading_similarity"],
            "det_code_block_recall": structural["code_block_recall"],
            "det_table_recall": structural["table_recall"],
            "det_rouge_l": structural["rouge_l"],
            "det_golden_length": float(golden_len),
            "det_extracted_length": float(extracted_len),
            "det_length_ratio": round(length_ratio, 3),
            "det_char_delta": float(char_delta),
            "det_char_delta_pct": round(char_delta_pct, 1),
        }

        return GradeResult(
            scores=scores,
            details={
                "explanation": parsed.explanation,
                "missing_elements": parsed.missing_elements,
                "extra_elements": parsed.extra_elements,
            },
        )


def _empty_det_scores(golden_len: int, extracted_len: int) -> dict[str, float]:
    length_ratio = extracted_len / golden_len if golden_len > 0 else 0.0
    char_delta = extracted_len - golden_len
    char_delta_pct = (char_delta / golden_len * 100.0) if golden_len > 0 else 0.0
    return {
        "paragraph_containment": 0.0,
        "heading_similarity": 0.0,
        "code_block_recall": 0.0,
        "table_recall": 0.0,
        "rouge_l": 0.0,
        "golden_length": float(golden_len),
        "extracted_length": float(extracted_len),
        "length_ratio": round(length_ratio, 3),
        "char_delta": float(char_delta),
        "char_delta_pct": round(char_delta_pct, 1),
    }
