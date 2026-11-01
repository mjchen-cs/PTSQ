#!/usr/bin/env python
"""Export the candidate texts of each query to <out>/<qid>_text.csv.gz (columns rid, text).

The learned risk and prefilter (guarantee.risk / guarantee.prefilter) read these files during
replay (run_experiments.py --features). The candidates are the same as in collect_trace.py:
the query's table after its SQL filter and limit.

  python scripts/export_texts.py --out results/feat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.config import build_policy, load_config, load_queries  # noqa: E402
from ptsq.executor import load_table, prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--queries", default="configs/queries_example.json")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="results/feat")
    a = ap.parse_args()
    policy = build_policy(load_config(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for q in load_queries(a.queries):
        if a.only and q.qid not in a.only:
            continue
        df = prepare(load_table(q.table), q, policy)
        t = df[["rid", q.text_col]].rename(columns={q.text_col: "text"})
        t.to_csv(out / f"{q.qid}_text.csv.gz", index=False)
        print(f"[texts] {q.qid}: {len(t)} tuples -> {out / (q.qid + '_text.csv.gz')}")


if __name__ == "__main__":
    main()
