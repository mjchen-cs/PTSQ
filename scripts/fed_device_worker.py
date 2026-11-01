#!/usr/bin/env python
"""Federated worker that runs ON the Jetson Orin Nano Super.

  python scripts/fed_device_worker.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/jetson_history.jsonl --port 8765

data: JSONL with {"messages": [...], "label": true/false} built from the Jetson's own past
queries (scripts/run_live.py --dump-history writes it). The data never leaves the device.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.federated.transport import serve_device_worker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    data = [(d["messages"], bool(d["label"])) for d in map(json.loads, open(a.data, encoding="utf-8"))]
    serve_device_worker(a.model, data, a.port)


if __name__ == "__main__":
    main()
