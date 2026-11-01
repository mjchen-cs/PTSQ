#!/usr/bin/env python
"""Compute the label-free TF-IDF features of each query's candidate tuples ON THE DEVICE, in a
separate process before run_live.py (the bigram vocabulary needs ~1 GB of peak memory, which does
not fit next to the device model on an 8 GB Jetson). run_live.py loads them from the cache.

  python scripts/precompute_features.py --config configs/live.yaml --only yelp_delay amazon_defect
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.config import build_policy, load_config, load_queries  # noqa: E402
from ptsq.executor import load_table, prepare  # noqa: E402
from ptsq.risk import query_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live.yaml")
    ap.add_argument("--queries", default="configs/queries_example.json")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="results/feat_live")
    a = ap.parse_args()
    cfg = load_config(a.config)
    policy = build_policy(cfg)
    for q in load_queries(a.queries):
        if a.only and q.qid not in a.only:
            continue
        t0 = time.perf_counter()
        df = prepare(load_table(q.table), q, policy)
        X, _ = query_features(df[q.text_col].astype(str).tolist(), cache=f"{a.out}/{q.qid}.npz")
        print(f"[feat] {q.qid}: {X.shape[0]} tuples x {X.shape[1]} terms, nnz={X.nnz}, "
              f"{time.perf_counter() - t0:.1f} s on this device")


if __name__ == "__main__":
    main()
