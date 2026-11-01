#!/usr/bin/env python
"""End-to-end live execution on the real testbed (Exp-1 fidelity check, Table VI).

Run ON the Jetson (device tier local; edge = vLLM on the 5090; cloud = API):
  python scripts/run_live.py --config configs/default.yaml --queries configs/queries_example.json \
      --router results/router.pt --out results/live.csv [--dump-history data/jetson_history.jsonl]

Compares measured latency / money with what trace replay predicts for the same query, which
validates the simulator used for the large-scale experiments.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_router  # noqa: E402

from ptsq.config import build_params, build_policy, build_tiers, load_config, load_profiles, load_queries  # noqa: E402
from ptsq.executor import load_table, make_requests, prepare, run_query  # noqa: E402
from ptsq.replay import unit_costs  # noqa: E402
from ptsq.types import TierId  # noqa: E402


async def main_async(a):
    cfg = load_config(a.config)
    # compile-time rule proposal may come from the cache (same rules as the trace), but
    # tuple scoring must NOT: cached answers would make latency ~0 and money 0
    compile_tiers = build_tiers(cfg)
    tiers = compile_tiers if a.use_cache else build_tiers({**cfg, "cache": None})
    policy, params = build_policy(cfg), build_params(cfg)
    prof = load_profiles(cfg.get("profiles"), cfg)
    uc = unit_costs(prof, 300, 1200, prof.device_per_tuple_s, params.weights)
    router = make_router(a.router)
    comp = {"device": TierId.DEVICE, "edge": TierId.EDGE, "cloud": TierId.CLOUD}[cfg.get("compile_tier", "cloud")]
    rows, hist = [], []
    for q in load_queries(a.queries):
        if a.only and q.qid not in a.only:
            continue
        for s in range(a.seeds):
            r = await run_query(q, tiers, policy, params, router=router, compile_backend=compile_tiers[comp], seed=s,
                                unit_cost=uc, edge_concurrency=prof.edge.concurrency,
                                cloud_concurrency=prof.cloud.concurrency, profiles=prof,
                                feat_cache=a.features)
            sel = set(r.pop("selected"))
            r["n_selected"] = len(sel)
            r["plans"] = json.dumps(r["plans"])
            r["rules"] = json.dumps(r["rules"], ensure_ascii=False)
            r["seed"] = s
            rows.append(r)
            print(f"[live] {q.qid} seed={s}: policy={r['policy']} latency={r['latency_s']:.1f}s money={r['money']:.3f} "
                  f"edge={r['n_edge']} cloud={r['n_cloud']} pruned={r['n_pruned']}")
            if a.dump_history and s == 0:
                df = prepare(load_table(q.table), q, policy)
                for req in make_requests(df, q):
                    hist.append({"messages": req.messages, "label": req.rid in sel})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    if a.dump_history:
        with open(a.dump_history, "w", encoding="utf-8") as f:
            for h in hist:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--queries", default="configs/queries_example.json")
    ap.add_argument("--router", default="results/router.pt")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dump-history", default=None)
    ap.add_argument("--features", default="results/feat_live",
                    help="TF-IDF cache written by scripts/precompute_features.py (Jetson memory)")
    ap.add_argument("--out", default="results/live.csv")
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse cached tier answers (fast functional test; timings/money invalid)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
