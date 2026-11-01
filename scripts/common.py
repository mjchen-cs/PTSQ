"""Shared helpers for experiment scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptsq.replay import MethodConfig  # noqa: E402
from ptsq.simulator import FixedRouter, HeuristicRouter  # noqa: E402


def load_traces(trace_dir: str | Path) -> dict[str, tuple[pd.DataFrame, list[dict]]]:
    out = {}
    for p in sorted(Path(trace_dir).glob("*.csv")):
        rp = p.with_suffix(".rules.json")
        rules = json.loads(rp.read_text()) if rp.exists() else []
        out[p.stem] = (pd.read_csv(p), rules)
    if not out:
        raise SystemExit(f"no traces in {trace_dir}")
    return out


def attach_features(T: dict, feat_dir: str | Path | None) -> int:
    """Attach the learned-risk features (ptsq.risk.text_features) to each trace as
    tr.attrs['X'] / tr.attrs['X_rid']. Texts come from <feat_dir>/<qid>_text.csv.gz (rid, text),
    exported on the machine that holds the data. Returns the number of traces with features."""
    if not feat_dir:
        return 0
    import numpy as np
    from ptsq.risk import text_features
    k = 0
    for qid, (tr, _) in T.items():
        p = Path(feat_dir) / f"{qid}_text.csv.gz"
        if not p.exists():
            continue
        tx = pd.read_csv(p).drop_duplicates("rid").set_index("rid")["text"]
        texts = tx.reindex(tr["rid"]).fillna("").to_numpy()
        # [s_E, log length]: no device score, which the plan may never compute (skip_device)
        num = np.c_[tr["s_E"].fillna(0.5), np.log1p([len(t) for t in texts])]
        tr.attrs["X"] = text_features(texts, num)
        tr.attrs["X_rid"] = tr["rid"].to_numpy()
        k += 1
    return k


def make_router(path: str | None):
    if path and Path(path).exists():
        from ptsq.rl.ppo import PPORouter
        return PPORouter(path)
    print("[warn] no trained router found - using FixedRouter for PTSQ", file=sys.stderr)
    return FixedRouter()


def main_methods(router) -> list[MethodConfig]:
    return [
        MethodConfig("Cloud-All", kind="cloud_all"),
        MethodConfig("Device-Only", kind="device_only"),
        MethodConfig("Static-Ref", kind="static_ref"),
        MethodConfig("Conf-Cascade", thresholds="fixed", fixed_tau=0.9, use_symbolic=False),
        MethodConfig("Cert-2Tier", use_edge=False, use_symbolic=False),
        MethodConfig("SUPG-JT", kind="supg"),
        MethodConfig("BARGAIN-JT", kind="bargain"),
        MethodConfig("BARGAIN-JT (relaxed)", kind="bargain", beta=0.01),
        MethodConfig("PTSQ", router=router),
    ]


def ablation_methods(router) -> list[MethodConfig]:
    return [
        MethodConfig("PTSQ", router=router),
        MethodConfig("w/o Symbolic", use_symbolic=False, router=router),
        MethodConfig("RL w/o selection", router=router, safe_select=False),
        MethodConfig("w/o Edge tier", use_edge=False, router=router),
        MethodConfig("w/o RL (fixed)", router=FixedRouter()),
        MethodConfig("Heuristic router", router=HeuristicRouter()),
    ]
