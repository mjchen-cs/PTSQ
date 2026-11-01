"""Tiered privacy: tuple labelling, optional on-device redaction, and the privacy shield.

The shield is the *only* component allowed to approve sending a tuple to a tier.
Every dispatch in the executor and the simulator goes through ``PrivacyShield.check``;
a request that would violate the policy is never sent (it is re-routed to the strongest
permitted tier and the attempt is counted).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .types import PrivacyLevel, TierId

# --------------------------------------------------------------------------------------
# PII detectors (regex based, run on-device). Extend as needed.
# --------------------------------------------------------------------------------------
PII_PATTERNS: dict[str, re.Pattern] = {
    # Chinese resident ID card: 17 digits + digit/X
    "cn_id": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    # Chinese mobile phone
    "cn_phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    # Generic international phone like +1 410-555-0199 / (410) 555-0199
    "phone": re.compile(r"(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]\d{3}[\s-]\d{4}"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # 13-19 digit card numbers (validated with Luhn below)
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
}


def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total, alt = 0, False
    for d in reversed(digits):
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def detect_pii(text: str) -> dict[str, list[tuple[int, int]]]:
    """Return {pii_type: [(start, end), ...]} spans found in ``text``."""
    found: dict[str, list[tuple[int, int]]] = {}
    for name, pat in PII_PATTERNS.items():
        spans = []
        for m in pat.finditer(text or ""):
            if name == "card" and not _luhn_ok(m.group(0)):
                continue
            spans.append(m.span())
        if spans:
            found[name] = spans
    return found


@dataclass
class PrivacyPolicy:
    """Maps privacy levels to the set of tiers a tuple of that level may reach.

    Default (the paper's running policy):
        HIGH   -> {DEVICE}
        MEDIUM -> {DEVICE, EDGE}
        LOW    -> {DEVICE, EDGE, CLOUD}
    """
    allowed: dict[PrivacyLevel, frozenset[TierId]] = field(default_factory=lambda: {
        PrivacyLevel.HIGH: frozenset({TierId.DEVICE}),
        PrivacyLevel.MEDIUM: frozenset({TierId.DEVICE, TierId.EDGE}),
        PrivacyLevel.LOW: frozenset({TierId.DEVICE, TierId.EDGE, TierId.CLOUD}),
    })
    # PII type -> level it induces when present in the text
    pii_levels: dict[str, PrivacyLevel] = field(default_factory=lambda: {
        "cn_id": PrivacyLevel.HIGH,
        "card": PrivacyLevel.HIGH,
        "cn_phone": PrivacyLevel.MEDIUM,
        "phone": PrivacyLevel.MEDIUM,
        "email": PrivacyLevel.MEDIUM,
        "ipv4": PrivacyLevel.MEDIUM,
    })
    # column name -> level if that column is present and non-empty (e.g. "diagnosis": HIGH)
    field_levels: dict[str, PrivacyLevel] = field(default_factory=dict)
    # If True, MEDIUM tuples whose only PII is redactable are redacted on-device and
    # promoted to LOW (they may then reach the cloud in redacted form).
    redact_promote: bool = False

    @classmethod
    def from_config(cls, cfg: dict | None) -> "PrivacyPolicy":
        pol = cls()
        if not cfg:
            return pol
        if "allowed" in cfg:
            pol.allowed = {PrivacyLevel.parse(k): frozenset(TierId[t.upper()] for t in v)
                           for k, v in cfg["allowed"].items()}
        if "pii_levels" in cfg:
            pol.pii_levels = {k: PrivacyLevel.parse(v) for k, v in cfg["pii_levels"].items()}
        if "field_levels" in cfg:
            pol.field_levels = {k: PrivacyLevel.parse(v) for k, v in cfg["field_levels"].items()}
        pol.redact_promote = bool(cfg.get("redact_promote", False))
        pol.validate()
        return pol

    def validate(self) -> None:
        for lvl in PrivacyLevel:
            tiers = self.allowed.get(lvl)
            if not tiers or TierId.DEVICE not in tiers:
                raise ValueError(f"policy must allow DEVICE for level {lvl.name}")
        # monotonicity: more sensitive levels may not reach more tiers
        for a in PrivacyLevel:
            for b in PrivacyLevel:
                if a < b and not self.allowed[b] <= self.allowed[a]:
                    raise ValueError("policy must be monotone in privacy level")

    def chain(self, level: PrivacyLevel) -> list[TierId]:
        """Ordered tiers a tuple of ``level`` may traverse; the last one is its reference."""
        return sorted(self.allowed[level])

    def reference(self, level: PrivacyLevel) -> TierId:
        return max(self.allowed[level])


def label_privacy(df: pd.DataFrame, policy: PrivacyPolicy, text_col: str = "text",
                  existing_col: str | None = "privacy") -> pd.Series:
    """Assign a privacy level to every tuple (runs on-device; no data leaves).

    Level = max(existing label if any, PII-induced level, field-induced level).
    """
    levels = []
    for _, row in df.iterrows():
        lvl = PrivacyLevel.LOW
        if existing_col and existing_col in df.columns and pd.notna(row.get(existing_col)):
            lvl = max(lvl, PrivacyLevel.parse(row[existing_col]))
        for pii in detect_pii(str(row.get(text_col, ""))):
            lvl = max(lvl, policy.pii_levels.get(pii, PrivacyLevel.LOW))
        for col, flv in policy.field_levels.items():
            if col in df.columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                lvl = max(lvl, flv)
        levels.append(int(lvl))
    return pd.Series(levels, index=df.index, name="privacy")


def redact(text: str, kinds: Iterable[str] | None = None) -> str:
    """Replace PII spans by typed placeholders, e.g. ``[EMAIL]``."""
    spans = []
    for name, ss in detect_pii(text).items():
        if kinds is None or name in kinds:
            spans += [(s, e, name) for s, e in ss]
    out, last = [], 0
    for s, e, name in sorted(spans):
        if s < last:
            continue
        out.append(text[last:s])
        out.append(f"[{name.upper()}]")
        last = e
    out.append(text[last:])
    return "".join(out)


class PrivacyViolation(RuntimeError):
    pass


class PrivacyShield:
    """Symbolic shield: approves or re-routes every (tuple, tier) dispatch.

    ``project`` maps a requested tier onto the strongest permitted tier not above it;
    ``check`` raises if a dispatch would violate the policy (used as a hard assertion
    right before any network send).
    """

    def __init__(self, policy: PrivacyPolicy):
        self.policy = policy
        self.blocked_attempts = 0
        self.sent: dict[TierId, int] = {t: 0 for t in TierId}

    def permitted(self, level: PrivacyLevel, tier: TierId) -> bool:
        return tier in self.policy.allowed[level]

    def project(self, level: PrivacyLevel, tier: TierId) -> TierId:
        if self.permitted(level, tier):
            return tier
        self.blocked_attempts += 1
        allowed = [t for t in self.policy.allowed[level] if t <= tier]
        return max(allowed) if allowed else TierId.DEVICE

    def check(self, level: PrivacyLevel, tier: TierId) -> None:
        if not self.permitted(level, tier):
            raise PrivacyViolation(f"tuple of level {level.name} may not reach {tier.name}")
        self.sent[tier] += 1
