#!/usr/bin/env python
"""Offline ORACLE labels through the provider's batch-inference API (OpenAI-compatible
/files + /batches, e.g. SiliconFlow "批量推理").

The cloud labels used for evaluation (collect_trace) do not need low latency, so they are
submitted as batch jobs instead of being paced through the online API's TPM limit. Results
are written into the normal response cache under exactly the keys the online backend uses,
so a subsequent `collect_trace.py` run finds every cloud answer in the cache and makes no
online cloud calls. Online experiments (run_live) are NOT affected and keep using the
real-time API.

  # try the API with 20 requests first
  python scripts/batch_label.py --only yelp_delay --limit 20
  # all tuples of the selected queries (submit, then poll until done; re-run to resume)
  python scripts/batch_label.py --only yelp_delay
  python scripts/batch_label.py --only yelp_delay --status      # poll once, ingest finished jobs

Job state is kept in results/batch/<qid>.json, so the script can be stopped and re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.config import build_policy, build_tiers, load_config, load_queries  # noqa: E402
from ptsq.executor import load_table, make_requests, prepare  # noqa: E402
from ptsq.types import TierId  # noqa: E402

DONE = {"completed", "failed", "expired", "cancelled", "canceled"}


class BatchAPI:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120):
        import httpx
        self.base = base_url.rstrip("/")
        # force IPv4: on hosts with a broken IPv6 route the sync client's connect() hangs
        # until the timeout on every AAAA address before trying IPv4
        self.c = httpx.Client(timeout=httpx.Timeout(timeout, connect=15.0),
                              transport=httpx.HTTPTransport(local_address="0.0.0.0", retries=2),
                              headers={"Authorization": f"Bearer {api_key}"})

    def _ok(self, r):
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} {r.request.method} {r.request.url}: {r.text[:500]}")
        return r

    def upload(self, path: Path) -> str:
        with open(path, "rb") as f:
            r = self._ok(self.c.post(f"{self.base}/files", data={"purpose": "batch"},
                                     files={"file": (path.name, f, "application/jsonl")}))
        d = r.json()
        return d.get("id") or d["data"]["id"]

    def create(self, file_id: str, window: str, extra: dict) -> dict:
        body = {"input_file_id": file_id, "endpoint": "/v1/chat/completions",
                "completion_window": window, **extra}
        return self._ok(self.c.post(f"{self.base}/batches", json=body)).json()

    def get(self, batch_id: str) -> dict:
        return self._ok(self.c.get(f"{self.base}/batches/{batch_id}")).json()

    def content(self, file_id: str) -> str:
        if file_id.startswith(("http://", "https://")):
            # some providers (e.g. SiliconFlow) return a pre-signed download URL instead of a
            # file id; fetch it directly and WITHOUT our API key (it must not go to a third host)
            import httpx
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0),
                              transport=httpx.HTTPTransport(local_address="0.0.0.0", retries=2)) as c:
                return self._ok(c.get(file_id, follow_redirects=True)).text
        return self._ok(self.c.get(f"{self.base}/files/{file_id}/content")).text


def ingest(text: str, be, req_by_key: dict, cache) -> tuple[int, int]:
    ok = bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        req = req_by_key.get(rec.get("custom_id"))
        resp = rec.get("response") or {}
        body = resp.get("body")
        if req is None or rec.get("error") or resp.get("status_code", 200) >= 400 or not body:
            bad += 1
            continue
        try:
            res = be.parse_response(req, body, 0.0)       # batch: latency is not meaningful
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        d = dict(res.__dict__)
        d.pop("rid", None)
        d["cached"] = False
        cache.put(rec["custom_id"], d)
        ok += 1
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--queries", default="configs/queries_example.json")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="only the first N uncached tuples (API test)")
    ap.add_argument("--max-lines", type=int, default=None, help="requests per batch file")
    ap.add_argument("--status", action="store_true", help="poll once and ingest; do not submit or wait")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--state-dir", default="results/batch")
    a = ap.parse_args()

    cfg = load_config(a.config)
    bcfg = dict(cfg["tiers"]["cloud"].get("batch", {}))
    max_lines = a.max_lines or int(bcfg.get("max_lines", 5000))
    window = str(bcfg.get("completion_window", "24h"))
    extra = dict(bcfg.get("extra", {}))
    tiers = build_tiers(cfg)
    be = tiers[TierId.CLOUD]
    cache = be.cache
    if cache is None or not be.api_key:
        sys.exit("need a response cache in the config and the cloud API key in the environment")
    api = BatchAPI(be.base_url, be.api_key)
    policy = build_policy(cfg)
    sdir = Path(a.state_dir)
    sdir.mkdir(parents=True, exist_ok=True)

    for q in load_queries(a.queries):
        if a.only and q.qid not in a.only:
            continue
        df = prepare(load_table(q.table), q, policy)
        reqs = make_requests(df, q)
        req_by_key = {cache.key(be.backend_id, {"m": r.messages}): r for r in reqs}
        spath = sdir / f"{q.qid}.json"
        state = json.loads(spath.read_text()) if spath.exists() else {"jobs": []}

        def save():
            spath.write_text(json.dumps(state, indent=1))

        # ---- submit everything that is neither cached nor already in a live job -----------
        if not a.status:
            pending = {k for j in state["jobs"] if j.get("status") not in DONE for k in j["keys"]}
            todo = [k for k in req_by_key if cache.get(k) is None and k not in pending]
            if a.limit is not None:
                todo = todo[:a.limit]
            print(f"[batch] {q.qid}: {len(reqs)} tuples, {len(todo)} to submit")
            for i in range(0, len(todo), max_lines):
                keys = todo[i:i + max_lines]
                fpath = sdir / f"{q.qid}_{len(state['jobs']):03d}.jsonl"
                with open(fpath, "w", encoding="utf-8") as f:
                    for k in keys:
                        f.write(json.dumps({"custom_id": k, "method": "POST", "url": "/v1/chat/completions",
                                            "body": be.request_body(req_by_key[k])}, ensure_ascii=False) + "\n")
                fid = api.upload(fpath)
                b = api.create(fid, window, extra)
                state["jobs"].append({"batch_id": b["id"], "file": fpath.name, "n": len(keys),
                                      "status": b.get("status", "submitted"), "keys": keys})
                save()
                print(f"[batch]   submitted {b['id']} ({len(keys)} requests)")

        # ---- poll / ingest -----------------------------------------------------------------
        while True:
            live = 0
            for j in state["jobs"]:
                if j.get("ingested"):
                    continue
                b = api.get(j["batch_id"])
                j["status"] = b.get("status", j.get("status"))
                rc = b.get("request_counts") or {}
                if j["status"] == "completed":
                    ok = bad = 0
                    if b.get("output_file_id"):
                        ok, bad = ingest(api.content(b["output_file_id"]), be, req_by_key, cache)
                    if b.get("error_file_id"):
                        _, e = ingest(api.content(b["error_file_id"]), be, req_by_key, cache)
                        bad += e
                    j.update(ingested=True, ok=ok, bad=bad)
                    print(f"[batch]   {j['batch_id']}: completed, {ok} cached, {bad} failed")
                elif j["status"] in DONE:
                    j["ingested"] = True
                    print(f"[batch]   {j['batch_id']}: {j['status']} ({b.get('errors')})")
                else:
                    live += 1
                    print(f"[batch]   {j['batch_id']}: {j['status']} {rc}")
                save()
            if live == 0 or a.status:
                break
            time.sleep(a.poll)
        n_cached = sum(cache.get(k) is not None for k in req_by_key)
        print(f"[batch] {q.qid}: {n_cached}/{len(reqs)} cloud labels now cached"
              + ("" if n_cached == len(reqs) else " (collect_trace will fetch the rest online)"))


if __name__ == "__main__":
    main()
