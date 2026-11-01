#!/usr/bin/env python
"""Federated LoRA experiment (Exp-5). Run on the RTX 5090 machine.

  python scripts/run_federated.py --config configs/default.yaml --queries configs/queries_example.json \
      --traces results/traces --train-queries yelp_delivery yelp_hygiene amazon_defect \
      --clients 20 --alpha 0.3 --rounds 20 --out results/traces_fed [--jetson http://<ip>:8765]

Training data of a client = (prompt, permitted-reference label) pairs of tuples it already
queried (the "history" queries). After every round the global adapter scores the tuples of
all traces; columns s_D_r<k> are added so that run_experiments.py --suite federated can
replay PTSQ with the round-k device model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_traces  # noqa: E402

from ptsq.config import build_policy, load_config, load_queries  # noqa: E402
from ptsq.executor import load_table, make_requests, prepare  # noqa: E402
from ptsq.federated.fedlora import FedConfig, dirichlet_split, run_federated  # noqa: E402
from ptsq.replay import reference_labels  # noqa: E402
from ptsq.types import TierId  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--queries", default="configs/queries_example.json")
    ap.add_argument("--traces", default="results/traces")
    ap.add_argument("--train-queries", nargs="+", required=True)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet heterogeneity")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-eval", type=int, default=20000)
    ap.add_argument("--noise", type=float, default=0.0, help="DP noise multiplier")
    ap.add_argument("--topk", type=float, default=0.1)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--jetson", default=None, help="URL of scripts/fed_device_worker.py on the Jetson")
    ap.add_argument("--include-high", action="store_true",
                    help="also train on HIGH tuples (requires an explicit DP policy exception)")
    ap.add_argument("--out", default="results/traces_fed")
    a = ap.parse_args()

    cfg = load_config(a.config)
    policy = build_policy(cfg)
    dev_cfg = cfg["tiers"]["device"]
    queries = {q.qid: q for q in load_queries(a.queries)}
    traces = load_traces(a.traces)

    # ---- build (messages, reference label) pairs per query --------------------------------
    msgs, groups, labels = [], [], []
    eval_sets = {}
    for qid, (tr, rules) in traces.items():
        q = queries[qid]
        df = prepare(load_table(q.table), q, policy)
        reqs = {r.rid: r for r in make_requests(df, q)}
        ref = reference_labels(tr["privacy"].to_numpy(),
                               {TierId.DEVICE: tr["y_D"].to_numpy(bool), TierId.EDGE: tr["y_E"].to_numpy(bool),
                                TierId.CLOUD: tr["y_C"].to_numpy(bool)}, policy)
        if qid in a.train_queries:
            for r, y, lv in zip(tr["rid"], ref, tr["privacy"]):
                if lv == 2 and not a.include_high:
                    continue          # HIGH tuples may not influence anything that leaves the device
                msgs.append(reqs[int(r)].messages); labels.append(bool(y)); groups.append(f"{qid}:{int(y)}")
        eval_sets[qid] = [reqs[int(r)] for r in tr["rid"][: a.max_eval]]
    parts = dirichlet_split(np.array(groups), a.clients, a.alpha)
    client_data = [[(msgs[i], labels[i]) for i in p] for p in parts]
    print(f"[fed] {len(msgs)} training pairs over {a.clients} clients; sizes={[len(c) for c in client_data]}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scored = {qid: {} for qid in traces}

    def eval_fn(model, tok, rnd):
        if rnd % a.eval_every and rnd != a.rounds:
            return {}
        import torch
        from ptsq.llm.backends import yes_no_token_ids
        yes, no = yes_no_token_ids(tok)
        for qid, rs in eval_sets.items():
            ps = []
            for i in range(0, len(rs), 32):
                chunk = rs[i:i + 32]
                prompts = [tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True) for r in chunk]
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
                lg = model(**enc).logits[:, -1, :].float()
                ps += torch.sigmoid(torch.logsumexp(lg[:, yes], -1) - torch.logsumexp(lg[:, no], -1)).tolist()
            scored[qid][rnd] = ps
        return {"evaluated": True}

    remote = []
    if a.jetson:
        from ptsq.federated.transport import RemoteClient
        remote = [RemoteClient(a.jetson)]
    fc = FedConfig(rounds=a.rounds, noise_multiplier=a.noise, topk_frac=a.topk, quant_bits=a.bits)
    hist = run_federated(dev_cfg["model_path"], client_data, fc, device="cuda", eval_fn=eval_fn,
                         save_dir=str(out / "adapters"), remote_clients=remote)
    (out / "fed_history.json").write_text(json.dumps(hist, indent=1, default=str))
    for qid, (tr, rules) in traces.items():
        tr = tr.copy()
        for rnd, ps in scored[qid].items():
            col = np.full(len(tr), np.nan)
            col[: len(ps)] = ps
            tr[f"s_D_r{rnd}"] = col
        tr = tr.iloc[: a.max_eval]
        tr.to_csv(out / f"{qid}.csv", index=False)
        (out / f"{qid}.rules.json").write_text(json.dumps(rules))
    print(f"[fed] wrote traces with per-round device scores to {out}")


if __name__ == "__main__":
    main()
