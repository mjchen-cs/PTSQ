"""Symbolic predicate decomposition (the neuro-symbolic part of PTSQ).

A semantic predicate ``phi`` (natural language) is decomposed *at compile time*, from the
query text and the schema only (no data), into candidate *necessary conditions* ``r`` such
that ideally ``phi(t) => r(t)``. Tuples with ``r(t) = False`` are pruned without any LLM
call. Because an LLM-proposed rule may be wrong, rules are (1) screened on a held-out
selection sample S_sel labelled by the reference tier, and (2) any residual error of the
kept rules is *accounted for* by the cascade's guarantee (pruned tuples count as rejected
tuples in the calibration bounds), so a bad rule can hurt cost but never the guarantee.
"""
from __future__ import annotations

import json
import logging
import operator
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_OPS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt,
        "==": operator.eq, "!=": operator.ne}


# --------------------------------------------------------------------------------------
# Rule DSL
# --------------------------------------------------------------------------------------
def validate_rule(rule: dict, columns: set[str]) -> bool:
    t = rule.get("type")
    if t in ("and", "or"):
        args = rule.get("args") or []
        return bool(args) and all(validate_rule(a, columns) for a in args)
    if rule.get("field") not in columns:
        return False
    if t == "keyword_any":
        kws = rule.get("keywords")
        return isinstance(kws, list) and len(kws) > 0 and all(isinstance(k, str) and k for k in kws)
    if t == "numeric":
        return rule.get("op") in _OPS and isinstance(rule.get("value"), (int, float))
    if t == "regex":
        try:
            re.compile(rule.get("pattern", ""))
            return True
        except re.error:
            return False
    return False


def eval_rule(rule: dict, df: pd.DataFrame) -> np.ndarray:
    """Vectorised evaluation; returns a boolean mask (True = may satisfy phi, keep)."""
    t = rule["type"]
    if t == "and":
        m = np.ones(len(df), dtype=bool)
        for a in rule["args"]:
            m &= eval_rule(a, df)
        return m
    if t == "or":
        m = np.zeros(len(df), dtype=bool)
        for a in rule["args"]:
            m |= eval_rule(a, df)
        return m
    col = df[rule["field"]]
    if t == "keyword_any":
        pat = "|".join(re.escape(k.lower()) for k in rule["keywords"])
        return col.astype(str).str.lower().str.contains(pat, regex=True, na=False).to_numpy()
    if t == "numeric":
        vals = pd.to_numeric(col, errors="coerce")
        # unknown values are kept (never prune on missing data)
        return (_OPS[rule["op"]](vals, rule["value"]) | vals.isna()).to_numpy()
    if t == "regex":
        return col.astype(str).str.contains(rule["pattern"], regex=True, na=False).to_numpy()
    raise ValueError(t)


def rule_to_str(rule: dict) -> str:
    t = rule["type"]
    if t in ("and", "or"):
        return "(" + f" {t.upper()} ".join(rule_to_str(a) for a in rule["args"]) + ")"
    if t == "keyword_any":
        k = rule["keywords"]
        return f"{rule['field']} CONTAINS ANY({', '.join(k[:4])}{', ...' if len(k) > 4 else ''})"
    if t == "numeric":
        return f"{rule['field']} {rule['op']} {rule['value']}"
    return f"{rule['field']} ~ /{rule['pattern']}/"


def _top_level_objects(text: str) -> list[str]:
    """Balanced {...} objects at the outermost level (survives a truncated JSON list)."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def _clean(rule: dict, max_kw: int = 60) -> dict:
    if rule.get("type") in ("and", "or"):
        return {**rule, "args": [_clean(a, max_kw) for a in rule.get("args") or [] if isinstance(a, dict)]}
    if rule.get("type") == "keyword_any" and isinstance(rule.get("keywords"), list):
        seen, kws = set(), []
        for k in rule["keywords"]:
            if isinstance(k, str) and k.strip() and k.strip().lower() not in seen:
                seen.add(k.strip().lower()); kws.append(k.strip())
        return {**rule, "keywords": kws[:max_kw]}
    return rule


def parse_rules(text: str, columns: set[str]) -> list[dict]:
    """Robustly extract a JSON list of rules from an LLM answer (dedupes keywords, and
    recovers the complete rules of a truncated list)."""
    text = text or ""
    cand = None
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            cand = json.loads(m.group(0))
        except json.JSONDecodeError:
            cand = None
    if not isinstance(cand, list):
        cand = []
        for obj in _top_level_objects(text[text.find("["):] if "[" in text else text):
            try:
                cand.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
    out = []
    for r in cand:
        if isinstance(r, dict):
            r = _clean(r)
            if validate_rule(r, columns) and r not in out:
                out.append(r)
    return out


# --------------------------------------------------------------------------------------
# Selection of rules on S_sel
# --------------------------------------------------------------------------------------
@dataclass
class RuleStats:
    rule: dict
    keep_rate: float          # fraction of tuples kept (1 - pruning power)
    pos_recall: float         # fraction of positive sample tuples kept
    n_pos: int


def select_rules(rules: list[dict], df_sel: pd.DataFrame, y_sel: np.ndarray,
                 min_recall: float = 0.99, min_prune: float = 0.05,
                 max_rules: int = 3) -> tuple[list[dict], list[RuleStats]]:
    """Greedy: keep rules whose empirical recall on positives >= ``min_recall`` and that
    prune at least ``min_prune`` of the sample; combine greedily (AND) while the combined
    recall stays >= ``min_recall``. Uses only S_sel (disjoint from the calibration sample).
    """
    y_sel = np.asarray(y_sel, dtype=bool)
    stats = []
    for r in rules:
        keep = eval_rule(r, df_sel)
        npos = int(y_sel.sum())
        rec = float(keep[y_sel].mean()) if npos else 1.0
        stats.append(RuleStats(r, float(keep.mean()), rec, npos))
    good = sorted([s for s in stats if s.pos_recall >= min_recall and 1 - s.keep_rate >= min_prune],
                  key=lambda s: s.keep_rate)
    chosen: list[dict] = []
    mask = np.ones(len(df_sel), dtype=bool)
    for s in good:
        new = mask & eval_rule(s.rule, df_sel)
        rec = float(new[y_sel].mean()) if y_sel.any() else 1.0
        if rec >= min_recall and new.mean() < mask.mean():
            chosen.append(s.rule)
            mask = new
        if len(chosen) >= max_rules:
            break
    return chosen, stats


def apply_rules(rules: list[dict], df: pd.DataFrame) -> np.ndarray:
    """Return the *pruned* mask (True = rejected symbolically)."""
    keep = np.ones(len(df), dtype=bool)
    for r in rules:
        keep &= eval_rule(r, df)
    return ~keep


async def propose_rules(backend: Any, predicate: str, schema: dict[str, str], text_col: str,
                        k: int = 4, lang: str = "English") -> list[dict]:
    """Ask a (compile-time) LLM for candidate necessary conditions.

    Only the predicate text and the schema are sent - never tuple data - so any tier may
    be used here without violating the privacy policy.
    """
    from .llm.prompts import decompose_messages
    try:
        text = await backend.complete(decompose_messages(predicate, schema, text_col, k, lang), max_tokens=2048)
    except NotImplementedError:
        return []
    rules = parse_rules(text, set(schema))
    log.info("proposed %d valid rules for %r", len(rules), predicate)
    return rules
