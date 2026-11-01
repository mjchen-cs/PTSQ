#!/usr/bin/env python
"""Write SYNTHETIC traces for a dry run of the whole experiment pipeline.

These numbers test the code paths only. Never report them as results.
  python scripts/make_mock_traces.py --out results_mock/traces
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.mock import synthetic_trace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_mock/traces")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for qid, sel, qd in (("mock_q1", 0.10, 1.0), ("mock_q2", 0.25, 1.4), ("mock_q3", 0.40, 0.8)):
        tr, rules = synthetic_trace(a.n, sel, q_device=qd, seed=hash(qid) % 1000)
        # pretend federated rounds improve the device model (for the federated suite dry run)
        for k, q in ((5, qd + 0.3), (10, qd + 0.5), (20, qd + 0.6)):
            tr[f"s_D_r{k}"] = synthetic_trace(a.n, sel, q_device=q, seed=hash(qid) % 1000)[0]["s_D"]
        for tag, dq, ts in (("0.5b", -0.4, 0.4), ("3b", 0.4, 2.0)):
            tr[f"s_Dm_{tag}"] = synthetic_trace(a.n, sel, q_device=qd + dq, seed=hash(qid) % 1000)[0]["s_D"]
            tr[f"t_Dm_{tag}"] = tr["t_D"] * ts
        tr.to_csv(out / f"{qid}.csv", index=False)
        (out / f"{qid}.rules.json").write_text(json.dumps(rules))
        print("wrote", qid)


if __name__ == "__main__":
    main()
