#!/usr/bin/env python
"""Train the PPO router on item pools extracted from traces (domain-randomised simulator).

  python scripts/train_router.py --traces results/traces --config configs/default.yaml --out results/router.pt

To avoid test leakage use --train-queries to train on a subset and evaluate on the rest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_traces  # noqa: E402

from ptsq.config import build_params, build_policy, load_config, load_profiles  # noqa: E402
from ptsq.replay import MethodConfig, replay  # noqa: E402
from ptsq.rl.ppo import PPOConfig, synthetic_item_pool, train_ppo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="results/traces")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--profiles", default=None)
    ap.add_argument("--train-queries", nargs="*")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-pool", action="store_true", help="dry run without traces")
    ap.add_argument("--out", default="results/router.pt")
    a = ap.parse_args()
    cfg = load_config(a.config)
    prof = load_profiles(a.profiles or cfg.get("profiles"), cfg)
    params = build_params(cfg)
    policy = build_policy(cfg)
    if a.synthetic_pool:
        pool = synthetic_item_pool(seed=a.seed)
    else:
        pool = []
        for qid, (tr, rules) in load_traces(a.traces).items():
            if a.train_queries and qid not in a.train_queries:
                continue
            for s in range(3):
                r = replay(tr, rules, MethodConfig("pool"), params, prof, policy, seed=1000 + s, return_items=True)
                pool += r["_items"]
        print(f"[router] item pool: {len(pool)} items")
    _, hist = train_ppo(pool, prof.edge, prof.cloud, params.weights,
                        PPOConfig(episodes=a.episodes, seed=a.seed), out_path=a.out)
    Path(a.out).with_suffix(".history.json").write_text(json.dumps(hist))
    print(f"[router] saved to {a.out}")


if __name__ == "__main__":
    main()
