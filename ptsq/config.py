"""Configuration loading and tier construction."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .llm.backends import Backend, build_backend
from .llm.cache import ResponseCache
from .privacy import PrivacyPolicy
from .replay import Profiles, ReplayParams
from .simulator import CostWeights, TierProfile
from .types import TierId


@dataclass
class Query:
    qid: str
    table: str                      # path to CSV / JSONL / Parquet
    predicate: str                  # natural-language semantic predicate
    sql_filter: str | None = None   # pandas .query() expression (cheap, exact predicates)
    text_col: str = "text"
    extra_cols: list[str] = field(default_factory=list)
    limit: int | None = None        # keep the first `limit` tuples after the SQL filter (tables are pre-shuffled)

    @classmethod
    def from_dict(cls, d: dict) -> "Query":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_queries(path: str | Path) -> list[Query]:
    with open(path, encoding="utf-8") as f:
        return [Query.from_dict(d) for d in json.load(f)]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tiers(cfg: dict, mock_truth: dict | None = None) -> dict[TierId, Backend]:
    cache = ResponseCache(cfg.get("cache")) if cfg.get("cache") else None
    tiers = {}
    for name, tid in (("device", TierId.DEVICE), ("edge", TierId.EDGE), ("cloud", TierId.CLOUD)):
        tc = dict(cfg["tiers"][name])
        if tc.get("kind") == "mock":
            tc.setdefault("name", name)
        tiers[tid] = build_backend(tc, cache=cache, mock_truth=mock_truth)
    return tiers


def build_policy(cfg: dict) -> PrivacyPolicy:
    return PrivacyPolicy.from_config(cfg.get("privacy"))


def build_params(cfg: dict) -> ReplayParams:
    g = cfg.get("guarantee", {})
    w = cfg.get("weights", {})
    return ReplayParams(gamma_r=g.get("gamma_r", 0.9), gamma_p=g.get("gamma_p", 0.9),
                        delta=g.get("delta", 0.05), n_cal=g.get("n_cal", 1500),
                        max_rule_subset=g.get("max_rule_subset", 2),
                        min_pos=g.get("min_pos"), n_pilot=g.get("n_pilot", 300),
                        n_cal_max=g.get("n_cal_max", 8000),
                        cert=g.get("cert", "union"), split=g.get("split", 0.5),
                        progressive=g.get("progressive", False),
                        miss_budget=g.get("miss_budget"),
                        risk=g.get("risk", False),
                        prefilter=g.get("prefilter", False),
                        test=g.get("test", "ratio"),
                        eta=g.get("eta", 0),
                        delta_certified_only=g.get("delta_certified_only", False),
                        hybrid=float(g.get("hybrid", 0.0)),
                        weights=CostWeights(**w) if w else CostWeights())


def load_profiles(path: str | Path | None, cfg: dict | None = None) -> Profiles:
    """Profiles measured by scripts/profile_tiers.py (falls back to config defaults)."""
    d = {}
    if isinstance(path, dict):
        d = path
    elif path and Path(path).exists():
        d = json.loads(Path(path).read_text())
    elif cfg and "profiles" in cfg and isinstance(cfg["profiles"], dict):
        d = cfg["profiles"]
    return Profiles(device_per_tuple_s=d.get("device_per_tuple_s", 0.05),
                    edge=TierProfile.from_dict(d.get("edge", {})),
                    cloud=TierProfile.from_dict(d.get("cloud", {"concurrency": 16, "batchable": False})))
