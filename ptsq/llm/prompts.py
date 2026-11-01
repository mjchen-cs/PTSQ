"""Prompt construction for semantic predicates and for symbolic decomposition."""
from __future__ import annotations

import json

SYSTEM_FILTER = (
    "You are a precise data classifier. Decide whether a database record satisfies a "
    "condition. Answer with exactly one word: Yes or No."
)

SYSTEM_FILTER_VERBAL = (
    "You are a precise data classifier. Decide whether a database record satisfies a "
    "condition. Reply ONLY with JSON: {\"answer\": \"Yes\" or \"No\", \"confidence\": "
    "an integer 0-100 giving your confidence that the answer is Yes}."
)


def render_record(row: dict, text_col: str = "text", extra_cols: list[str] | None = None,
                  max_chars: int = 2000) -> str:
    parts = []
    for c in extra_cols or []:
        if c in row and row[c] is not None:
            parts.append(f"{c}: {row[c]}")
    text = str(row.get(text_col, ""))
    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    parts.append(f"{text_col}: {text}")
    return "\n".join(parts)


def filter_messages(predicate: str, record: str, verbalized: bool = False) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_FILTER_VERBAL if verbalized else SYSTEM_FILTER},
        {"role": "user", "content": f"Condition: {predicate}\n\nRecord:\n{record}\n\n"
                                    "Does the record satisfy the condition?"},
    ]


DECOMPOSE_SYSTEM = (
    "You help a database optimizer. Given a natural-language filter condition and a table "
    "schema, propose cheap NECESSARY conditions: rules that every record satisfying the "
    "condition must also satisfy. A rule must never reject a record that satisfies the "
    "condition, so prefer broad rules (many synonyms, generous numeric bounds). "
    "You do NOT see any data."
)

DECOMPOSE_USER = """Condition: {predicate}
Schema (column: type): {schema}
Text column: {text_col}
The records are written in {lang}.

Return ONLY a JSON list of {k} candidate rules using this DSL:
  {{"type": "keyword_any", "field": "<text column>", "keywords": ["...", "..."]}}
  {{"type": "numeric", "field": "<numeric column>", "op": "<=|<|>=|>|==|!=", "value": <number>}}
  {{"type": "regex", "field": "<text column>", "pattern": "<python regex>"}}
  {{"type": "or", "args": [<rule>, ...]}}   {{"type": "and", "args": [<rule>, ...]}}
Guidelines:
- Each rule should approach the condition from a DIFFERENT angle (e.g. topic words,
  time/quantity expressions, typical complaint phrases), not repeat the same list.
- Keywords are matched case-insensitively as substrings, so short stems cover inflections
  ("wait" matches "waited", "waiting"). Use 10-40 distinct keywords per rule, all in {lang};
  do not list mechanical combinations of the same words.
- A rule must keep every record that satisfies the condition: be generous."""


def decompose_messages(predicate: str, schema: dict[str, str], text_col: str, k: int = 4,
                       lang: str = "English") -> list[dict]:
    return [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": DECOMPOSE_USER.format(
            predicate=predicate, schema=json.dumps(schema, ensure_ascii=False),
            text_col=text_col, k=k, lang=lang)},
    ]
