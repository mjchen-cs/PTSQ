#!/usr/bin/env python
"""Replay the live runs (results/live_v2.csv) under the configuration they were executed with
(propose-and-test with ratio bounds, B=80, delta split over all strata, no single-threshold
family), same query and seed, for the simulator-fidelity comparison of Fig. 10(d).

  python scripts/replay_live_config.py --live results/live_v2.csv --out results/exp_v13/live_replay.csv
"""
import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import attach_features, load_traces  # noqa: E402
from ptsq.config import build_params, build_policy, load_config, load_profiles  # noqa: E402
from ptsq.replay import MethodConfig, replay  # noqa: E402

LIVE_CONFIG = dict(test="ratio", miss_budget=80, delta_certified_only=False, hybrid=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--traces", default="results/traces_main")
    ap.add_argument("--features", default="results/feat")
    ap.add_argument("--profiles", default="results/profiles.json")
    ap.add_argument("--live", default="results/live_v2.csv")
    ap.add_argument("--out", default="results/exp_v13/live_replay.csv")
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = build_params(cfg)
    for k, v in LIVE_CONFIG.items():
        setattr(P, k, v)
    pol, prof = build_policy(cfg), load_profiles(a.profiles, cfg)
    lv = pd.read_csv(a.live)
    T = load_traces(a.traces)
    T = {q: T[q] for q in set(lv.qid)}
    attach_features(T, a.features)
    rows = []
    for _, l in lv.iterrows():
        for m in (MethodConfig("PTSQ"), MethodConfig("Static-Ref", kind="static_ref")):
            r = replay(*T[l.qid], m, copy.deepcopy(P), prof, pol, seed=int(l.seed))
            r.update(qid=l.qid)
            rows.append(r)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[live-replay] {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
