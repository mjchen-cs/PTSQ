#!/usr/bin/env python
"""Collect evaluation traces: every candidate tuple scored on every tier.

  python scripts/collect_trace.py --config configs/default.yaml --queries configs/queries.json \
         --out results/traces

For each query this (1) asks the compile tier for candidate symbolic rules (query text +
schema only), (2) scores all tuples on device, edge and cloud, (3) writes
results/traces/<qid>.csv and results/traces/<qid>.rules.json.

Cost note: the cloud labels every tuple here. This is an OFFLINE EVALUATION ORACLE on public
data with synthetic PII; the online system (scripts/run_live.py) never does this. All
responses are cached, so re-running is free.

Run it on the 5090 machine with the device tier pointed at the Jetson, or run the device
tier separately on the Jetson with --tiers device and merge (see README).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.config import build_policy, build_tiers, load_config, load_queries  # noqa: E402
from ptsq.executor import collect_trace, load_table, prepare  # noqa: E402
from ptsq.symbolic import propose_rules  # noqa: E402
from ptsq.types import TierId  # noqa: E402


async def score_extra_device(spec, q, cfg, tiers, policy):
    """Score a query with an additional device model (model-size study); cached like the rest."""
    import gc
    from ptsq.executor import make_requests, score_tier
    from ptsq.llm.backends import HFLocalBackend
    tag, path = spec.split("=", 1)
    dc = cfg["tiers"]["device"]
    dev = tiers[TierId.DEVICE]
    if getattr(dev, "_model", None) is not None:   # free the base model first (8 GB shared memory)
        dev._model = None
    gc.collect()
    be = HFLocalBackend(f"device-{tag}", path, cache=dev.cache, device=dc.get("device", "cuda"),
                        dtype=dc.get("dtype", "float16"), batch_size=dc.get("batch_size", 16),
                        max_batch_tokens=dc.get("max_batch_tokens", 6144))
    df = prepare(load_table(q.table), q, policy)
    res = await score_tier(be, make_requests(df, q), None, None, TierId.DEVICE)
    be._model = None
    del be
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return res


async def main_async(a):
    cfg = load_config(a.config)
    tiers = build_tiers(cfg)
    policy = build_policy(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    compile_tier = {"device": TierId.DEVICE, "edge": TierId.EDGE, "cloud": TierId.CLOUD}[cfg.get("compile_tier", "cloud")]
    for q in load_queries(a.queries):
        if a.only and q.qid not in a.only:
            continue
        if a.prefetch:                     # fill the response cache only (no rules, no trace file)
            tids = [{"device": TierId.DEVICE, "edge": TierId.EDGE, "cloud": TierId.CLOUD}[t] for t in a.prefetch]
            await collect_trace(q, tiers, policy, [], only_tiers=tids)
            for spec in a.extra_device or []:
                await score_extra_device(spec, q, cfg, tiers, policy)
            print(f"[prefetch] {q.qid}: {', '.join(a.prefetch + (a.extra_device or []))} cached")
            continue
        rpath = out / f"{q.qid}.rules.json"
        if rpath.exists():
            rules = json.loads(rpath.read_text())
        else:
            df = prepare(load_table(q.table), q, policy)
            schema = {c: str(t) for c, t in df.dtypes.items() if c not in ("rid", "privacy")}
            rules = await propose_rules(tiers[compile_tier], q.predicate, schema, q.text_col, k=a.k_rules)
            rpath.write_text(json.dumps(rules, ensure_ascii=False, indent=1))
        tr = await collect_trace(q, tiers, policy, rules)
        for spec in a.extra_device or []:            # e.g. 0.5b=Qwen/Qwen2.5-0.5B-Instruct
            tag = spec.split("=", 1)[0]
            res = await score_extra_device(spec, q, cfg, tiers, policy)
            tr[f"s_Dm_{tag}"] = [res[int(r)].p_yes for r in tr["rid"]]
            tr[f"t_Dm_{tag}"] = [res[int(r)].latency_s for r in tr["rid"]]
        tr.to_csv(out / f"{q.qid}.csv", index=False)
        print(f"[trace] {q.qid}: {len(tr)} tuples, {len(rules)} candidate rules, "
              f"selectivity(cloud)={tr['y_C'].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--queries", default="configs/queries_example.json")
    ap.add_argument("--out", default="results/traces")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--k-rules", type=int, default=4)
    ap.add_argument("--prefetch", nargs="+", choices=["device", "edge", "cloud"],
                    help="only fill the response cache for these tiers (e.g. cloud on the 5090 while the "
                         "Jetson does device+edge); merge caches with scripts/merge_cache.py, then run normally")
    ap.add_argument("--extra-device", nargs="*", help="TAG=MODEL_PATH, extra device models to score (run on the Jetson)")
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)      # one line per request is too noisy
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
