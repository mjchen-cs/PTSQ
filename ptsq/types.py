"""Core types shared across PTSQ (Privacy-Tiered Semantic Query processing)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class PrivacyLevel(IntEnum):
    """Privacy level of a tuple. Higher = more sensitive."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    @classmethod
    def parse(cls, v: Any) -> "PrivacyLevel":
        if isinstance(v, PrivacyLevel):
            return v
        if isinstance(v, (int,)):
            return cls(int(v))
        return cls[str(v).strip().upper()]


class TierId(IntEnum):
    """Execution tiers, ordered from weakest/most private to strongest/least private."""
    DEVICE = 0   # on-device small model (e.g. Jetson Orin Nano Super)
    EDGE = 1     # private edge server (e.g. RTX 5090 workstation)
    CLOUD = 2    # public cloud LLM API (e.g. Zhipu GLM, SiliconFlow)

    @property
    def short(self) -> str:
        return "DEC"[int(self)]


@dataclass
class Request:
    """One semantic-predicate evaluation request for one tuple."""
    rid: int
    messages: list[dict]
    meta: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return sum(len(m.get("content", "").encode("utf-8")) for m in self.messages)


@dataclass
class ScoreResult:
    """Result of evaluating a semantic predicate on one tuple at one tier."""
    rid: int
    p_yes: float                 # calibrated-ish probability that predicate holds
    label: bool                  # hard decision of this model
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    money: float = 0.0           # monetary cost in the configured currency
    cached: bool = False
    raw: str | None = None


# Decision provenance codes used in plans / traces
DEC_PRUNED = "S"     # rejected by symbolic necessary condition
DEC_DEVICE = "D"
DEC_EDGE = "E"
DEC_CLOUD = "C"
DEC_SAMPLE = "Z"     # calibration-sample tuple, labelled by its reference tier
