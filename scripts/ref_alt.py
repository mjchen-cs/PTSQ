#!/usr/bin/env python
"""Reference-model robustness: label a uniform subset of every query with a SECOND strong
cloud model and compare it with the main reference (Qwen3.5-122B).

Why: PTSQ's guarantees are relative to the permitted reference. This experiment shows how
much two strong references agree, and whether the cheaper tiers look equally good against
either of them (i.e., conclusions do not hinge on one particular reference model).

  # 1) label (any machine with the API key; uses its own cache file, resumable, budget-capped)
  python scripts/ref_alt.py label --alt deepseek --n 3000 --budget 12 \
         --cache /home/cmj/ptsq_cache/cache_refalt.sqlite
  # 1b) or through the provider's batch-inference API (cheaper, finishes within the window)
  python scripts/ref_alt.py batch --alt deepseek --n 3000 \
         --cache /home/cmj/ptsq_cache/cache_refalt.sqlite          # submit + poll (re-run to resume)
  python scripts/ref_alt.py batch --alt deepseek --n 3000 --status --cache ...  # poll once
  # 2) analyse (on the machine whose cache holds the Qwen3.5 and edge answers, after merging
  #    the alt cache into it with scripts/merge_cache.py)
  python scripts/ref_alt.py analyze --alt deepseek --n 3000

The subset of each query is a uniform sample (seed 0) of its candidate tuples, identical in
both steps. Output: results/refalt/<alt>_summary.csv (+ printed table).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.config import build_backend, build_policy, build_tiers, load_config, load_queries  # noqa: E402
from ptsq.executor import load_table, make_requests, prepare, score_tier  # noqa: E402
from ptsq.llm.cache import ResponseCache  # noqa: E402
from ptsq.types import TierId  # noqa: E402

log = logging.getLogger("ref_alt")


def subset(q, policy, n: int, seed: int):
    df = prepare(load_table(q.table), q, policy)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed).sort_index()
    return df, make_requests(df, q)


def alt_backend(cfg, name, cache):
    alts = cfg.get("cloud_alternatives", {})
    if name not in alts:
        sys.exit(f"--alt {name!r} not in cloud_alternatives ({', '.join(alts)})")
    return build_backend(dict(alts[name]), cache=cache)


async def do_label(a, cfg, policy):
    cache = ResponseCache(a.cache or cfg.get("cache"))
    be = alt_backend(cfg, a.alt, cache)
    if not (be.price_in or be.price_out):
        sys.exit("set price_in/price_out of this alternative in the config (needed for --budget)")
    spent = 0.0
    for q in load_queries(a.queries):
        if a.only and q.qid not in a.only:
            continue
        _, reqs = subset(q, policy, a.n, a.seed)
        log.info("%s: %d tuples with %s", q.qid, len(reqs), be.backend_id)
        for i in range(0, len(reqs), 500):
            res = await score_tier(be, reqs[i:i + 500], None, None, TierId.CLOUD)
            spent += sum(r.money for r in res.values() if not r.cached)
            log.info("%s: %d/%d done, %.2f CNY spent in this run", q.qid, min(i + 500, len(reqs)),
                     len(reqs), spent)
            if a.budget is not None and spent >= a.budget:
                print(f"[ref_alt] budget {a.budget} CNY reached; re-run later to continue (cached)")
                return
    print(f"[ref_alt] done, {spent:.2f} CNY spent in this run")


def do_batch(a, cfg, policy):
    """Submit the subsets as batch jobs; answers land in the cache under the online keys."""
    import json
    import time
    from batch_label import DONE, BatchAPI, ingest
    cache = ResponseCache(a.cache or cfg.get("cache"))
    be = alt_backend(cfg, a.alt, cache)
    if not be.api_key:
        sys.exit(f"API key not set ({cfg['cloud_alternatives'][a.alt].get('api_key_env')})")
    api = BatchAPI(be.base_url, be.api_key)
    sdir = Path(a.out) / "batch"
    sdir.mkdir(parents=True, exist_ok=True)
    spath = sdir / f"{a.alt}_n{a.n}.json"
    state = json.loads(spath.read_text()) if spath.exists() else {"jobs": []}
    req_by_key = {}
    for q in load_queries(a.queries):
        if a.only and q.qid not in a.only:
            continue
        _, reqs = subset(q, policy, a.n, a.seed)
        req_by_key.update({cache.key(be.backend_id, {"m": r.messages}): r for r in reqs})

    def save():
        spath.write_text(json.dumps(state, indent=1))

    if not a.status:
        pending = {k for j in state["jobs"] if not j.get("ingested") for k in j["keys"]}
        todo = [k for k in req_by_key if cache.get(k) is None and k not in pending]
        print(f"[ref_alt] {len(req_by_key)} requests, {len(todo)} to submit")
        for i in range(0, len(todo), a.max_lines):
            keys = todo[i:i + a.max_lines]
            fpath = sdir / f"{a.alt}_{len(state['jobs']):03d}.jsonl"
            with open(fpath, "w", encoding="utf-8") as f:
                for k in keys:
                    f.write(json.dumps({"custom_id": k, "method": "POST", "url": "/v1/chat/completions",
                                        "body": be.request_body(req_by_key[k])}, ensure_ascii=False) + "\n")
            b = api.create(api.upload(fpath), a.window, {})
            state["jobs"].append({"batch_id": b["id"], "n": len(keys), "status": b.get("status"), "keys": keys})
            save()
            print(f"[ref_alt]   submitted {b['id']} ({len(keys)} requests)")
    while True:
        live = 0
        for j in state["jobs"]:
            if j.get("ingested"):
                continue
            b = api.get(j["batch_id"])
            j["status"] = b.get("status", j.get("status"))
            if j["status"] == "completed":
                ok = bad = 0
                if b.get("output_file_id"):
                    ok, bad = ingest(api.content(b["output_file_id"]), be, req_by_key, cache)
                if b.get("error_file_id"):
                    bad += ingest(api.content(b["error_file_id"]), be, req_by_key, cache)[1]
                j.update(ingested=True, ok=ok, bad=bad)
                print(f"[ref_alt]   {j['batch_id']}: completed, {ok} cached, {bad} failed")
            elif j["status"] in DONE:
                j["ingested"] = True
                print(f"[ref_alt]   {j['batch_id']}: {j['status']} ({b.get('errors')})")
            else:
                live += 1
                print(f"[ref_alt]   {j['batch_id']}: {j['status']} {b.get('request_counts') or ''}")
            save()
        if live == 0 or a.status:
            break
        time.sleep(a.poll)
    n_ok = sum(cache.get(k) is not None for k in req_by_key)
    print(f"[ref_alt] {n_ok}/{len(req_by_key)} answers cached")


def _pr(pred, ref):
    tp = int((pred & ref).sum())
    rec = tp / max(int(ref.sum()), 1)
    prec = tp / max(int(pred.sum()), 1)
    return rec, prec


def _kappa(x, y):
    po = float((x == y).mean())
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def do_analyze(a, cfg, policy):
    tiers = build_tiers(cfg)                       # main reference + edge, read from the cache only
    cache = tiers[TierId.CLOUD].cache
    alt = alt_backend(cfg, a.alt, cache)
    look = {"C": tiers[TierId.CLOUD], "A": alt, "E": tiers[TierId.EDGE]}
    rows = []
    for q in load_queries(a.queries):
        if a.only and q.qid not in a.only:
            continue
        df, reqs = subset(q, policy, a.n, a.seed)
        lab = {k: [] for k in look}
        keep = []
        for r in reqs:
            hits = {k: cache.get(cache.key(be.backend_id, {"m": r.messages})) for k, be in look.items()}
            if any(h is None for h in hits.values()):
                continue
            keep.append(r.rid)
            for k, h in hits.items():
                lab[k].append(bool(h["label"]))
        if not keep:
            print(f"[ref_alt] {q.qid}: no tuple has all three labels yet (merge the caches first)")
            continue
        C, A, E = (np.array(lab[k]) for k in ("C", "A", "E"))
        lv = df.set_index("rid").loc[keep, "privacy"].to_numpy()
        for name, m in (("all", np.ones(len(C), bool)), ("LOW", lv == 0)):
            if m.sum() == 0:
                continue
            c, al, e = C[m], A[m], E[m]
            rA, pA = _pr(al, c)
            rEC, pEC = _pr(e, c)
            rEA, pEA = _pr(e, al)
            rows.append(dict(qid=q.qid, stratum=name, n=int(m.sum()), sel_main=c.mean(), sel_alt=al.mean(),
                             agree=float((c == al).mean()), kappa=_kappa(c, al),
                             alt_rec=rA, alt_prec=pA,
                             edge_rec_main=rEC, edge_prec_main=pEC, edge_rec_alt=rEA, edge_prec_alt=pEA))
    if not rows:
        return
    out = pd.DataFrame(rows)
    p = Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    out.to_csv(p / f"{a.alt}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["label", "batch", "analyze"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--queries", default="configs/queries_example.json")
    ap.add_argument("--only", nargs="*", default=["yelp_delay", "yelp_hygiene", "yelp_staff_praise",
                                                   "amazon_defect"])
    ap.add_argument("--alt", default="deepseek", help="key in cloud_alternatives")
    ap.add_argument("--n", type=int, default=3000, help="tuples per query (uniform subset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None, help="stop after spending this many CNY")
    ap.add_argument("--cache", default=None, help="cache file for the labels (default: config's)")
    ap.add_argument("--out", default="results/refalt")
    ap.add_argument("--status", action="store_true", help="batch: poll once, do not submit or wait")
    ap.add_argument("--poll", type=float, default=120.0)
    ap.add_argument("--max-lines", type=int, default=5000, help="batch: requests per batch file")
    ap.add_argument("--window", default="24h", help="batch: completion window")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(a.config)
    policy = build_policy(cfg)
    if a.mode == "label":
        asyncio.run(do_label(a, cfg, policy))
    elif a.mode == "batch":
        do_batch(a, cfg, policy)
    else:
        do_analyze(a, cfg, policy)


if __name__ == "__main__":
    main()
