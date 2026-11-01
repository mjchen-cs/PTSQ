#!/usr/bin/env python
"""Diagnose cloud ReadTimeouts during live runs on the Jetson.

Measures (1) cloud request latency alone, (2) cloud request latency while the device model
scores tuples in a worker thread, and (3) the asyncio event-loop lag in both phases. If the
loop lag is large while the device scores, the device thread starves the event loop and the
cloud responses are read too late (ReadTimeout), i.e. the timeouts are caused locally.

  python scripts/diag_loop.py --config configs/live.yaml          # ~3-4 minutes, ~0.05 CNY
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.config import build_policy, build_tiers, load_config, load_queries  # noqa: E402
from ptsq.executor import load_table, make_requests, prepare  # noqa: E402
from ptsq.types import TierId  # noqa: E402


async def lag_probe(stop: asyncio.Event, out: list):
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(0.1)
        out.append(time.perf_counter() - t - 0.1)


async def cloud_probe(be, reqs, out: list):
    for k in range(0, len(reqs), 8):            # 8 concurrent requests at a time
        t = time.perf_counter()
        await be.score(reqs[k:k + 8])
        out.append((time.perf_counter() - t))


def summary(name, lag, cl):
    print(f"[{name}] loop lag: median {statistics.median(lag) * 1e3:.0f} ms, max {max(lag):.1f} s; "
          f"cloud round of 8 requests: median {statistics.median(cl):.1f} s, max {max(cl):.1f} s")


async def main_async(a):
    cfg = load_config(a.config)
    tiers = build_tiers({**cfg, "cache": None})
    q = [q for q in load_queries(a.queries) if q.qid == "yelp_delay"][0]
    df = prepare(load_table(q.table), q, build_policy(cfg))
    reqs = make_requests(df, q)
    cloud = tiers[TierId.CLOUD]
    dev = tiers[TierId.DEVICE]
    await dev.score(reqs[:4])                    # load the device model first

    lag, cl = [], []
    stop = asyncio.Event()
    probe = asyncio.create_task(lag_probe(stop, lag))
    await cloud_probe(cloud, reqs[100:100 + a.n_cloud], cl)
    stop.set(); await probe
    summary("cloud alone", lag, cl)

    lag2, cl2 = [], []
    stop = asyncio.Event()
    probe = asyncio.create_task(lag_probe(stop, lag2))
    dev_task = asyncio.create_task(dev.score(reqs[1000:1000 + a.n_dev]))
    await cloud_probe(cloud, reqs[200:200 + a.n_cloud], cl2)
    stop.set(); await probe
    await dev_task
    summary("cloud + device scoring", lag2, cl2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live.yaml")
    ap.add_argument("--queries", default="configs/queries_example.json")
    ap.add_argument("--n-cloud", type=int, default=48)
    ap.add_argument("--n-dev", type=int, default=600)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
