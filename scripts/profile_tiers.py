#!/usr/bin/env python
"""Measure tier performance profiles used by the simulator (results/profiles.json).

  # on the 5090 machine (edge + cloud), device profile measured on the Jetson separately:
  python scripts/profile_tiers.py --config configs/default.yaml --tiers edge cloud --out results/profiles.json
  # on the Jetson:
  python scripts/profile_tiers.py --config configs/default.yaml --tiers device --out results/profile_device.json

Fits latency = fixed_s + per_token_s * tokens by least squares over prompts of increasing
length (batches of 32 for the edge, single requests for the cloud). Bandwidth/RTT must be
measured with iperf3/ping from the Jetson (enter them with --edge-bw/--cloud-bw/--rtt-*).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.config import build_tiers, load_config  # noqa: E402
from ptsq.llm.prompts import filter_messages  # noqa: E402
from ptsq.types import Request, TierId  # noqa: E402

LOREM = ("The food was fine but the delivery took more than an hour and the driver was rude. "
         "We will probably order again because the prices are reasonable. ")


def reqs(n: int, reps: int, start: int) -> list[Request]:
    text = LOREM * reps
    return [Request(rid=start + i, messages=filter_messages(f"Complains about delivery (probe {start + i})",
                                                              f"text: {text}")) for i in range(n)]


async def profile(tier, batch: int, lengths=(1, 3, 6, 12), reps: int = 3):
    """Warm up once, then time `reps` batches per prompt length. The fit uses the median
    per length, so a single slow/retried request (cloud timeouts, first-call CUDA init)
    does not distort it. Raw samples are returned for inspection."""
    rid = 10_000_000
    await tier.score(reqs(batch, 1, rid)); rid += batch          # warm-up (not timed)
    xs, ys, raw = [], [], []
    for L in lengths:
        tl, dl = [], []
        for _ in range(reps):
            rs = reqs(batch, L, rid)
            rid += batch
            t0 = time.perf_counter()
            res = await tier.score(rs)
            dt = time.perf_counter() - t0
            tok = sum(r.prompt_tokens for r in res) or sum(len(r.messages[1]["content"]) // 4 for r in rs)
            tl.append(tok); dl.append(dt)
            raw.append({"reps": L, "tokens": tok, "seconds": round(dt, 4)})
        xs.append(float(np.median(tl))); ys.append(float(np.median(dl)))
    A = np.vstack([np.ones(len(xs)), xs]).T
    fixed, per_tok = np.linalg.lstsq(A, np.array(ys), rcond=None)[0]
    for r in raw:
        print(f"  [{tier.name}] batch={batch} tokens={r['tokens']:>6} t={r['seconds']:.3f}s", flush=True)
    return max(float(fixed), 0.0), max(float(per_tok), 1e-7), raw


async def main_async(a):
    cfg = load_config(a.config)
    cfg["cache"] = None                      # never use cached timings
    tiers = build_tiers(cfg)
    prof = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    if "device" in a.tiers:
        fixed, per_tok, raw = await profile(tiers[TierId.DEVICE], 16)
        prof.setdefault("raw", {})["device"] = raw
        prof["device_per_tuple_s"] = (fixed + per_tok * 300 * 16) / 16
        prof["device_fit"] = {"fixed_s": fixed, "per_token_s": per_tok, "batch": 16}
    if "edge" in a.tiers:
        fixed, per_tok, raw = await profile(tiers[TierId.EDGE], 32)
        prof.setdefault("raw", {})["edge"] = raw
        # timings are end-to-end from the Jetson and already contain the network RTT; the
        # simulator adds rtt_s + transfer on top of fixed_s, so store the service part only
        prof["edge"] = {"fixed_s": max(fixed - a.rtt_edge, 0.0), "fixed_measured_s": fixed, "per_token_s": per_tok, "concurrency": a.edge_conc, "batchable": True,
                        "rtt_s": a.rtt_edge, "bw_mbps": a.edge_bw}
    if "cloud" in a.tiers:
        fixed, per_tok, raw = await profile(tiers[TierId.CLOUD], 1)
        prof.setdefault("raw", {})["cloud"] = raw
        c = cfg["tiers"]["cloud"]
        prof["cloud"] = {"fixed_s": max(fixed - a.rtt_cloud, 0.0), "fixed_measured_s": fixed, "per_token_s": per_tok, "concurrency": a.cloud_conc, "batchable": False,
                         "rtt_s": a.rtt_cloud, "bw_mbps": a.cloud_bw,
                         "price_in": c.get("price_in", 0.0), "price_out": c.get("price_out", 0.0),
                         "tpm": float(c.get("tpm_limit") or 0.0)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(prof, indent=1))
    print(json.dumps({k: v for k, v in prof.items() if k != "raw"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tiers", nargs="+", default=["device", "edge", "cloud"])
    ap.add_argument("--out", default="results/profiles.json")
    ap.add_argument("--edge-bw", type=float, default=100.0, help="Jetson->5090 uplink Mbps (iperf3)")
    ap.add_argument("--cloud-bw", type=float, default=20.0, help="Jetson->Internet uplink Mbps")
    ap.add_argument("--rtt-edge", type=float, default=0.002)
    ap.add_argument("--rtt-cloud", type=float, default=0.04)
    ap.add_argument("--edge-conc", type=int, default=1,
                    help="parallel batches on the edge GPU (1: vLLM already batches inside a dispatch)")
    ap.add_argument("--cloud-conc", type=int, default=16)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
