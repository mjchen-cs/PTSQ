"""Minimal HTTP transport so that a real Jetson can join federated rounds.

On the Jetson:   python scripts/fed_device_worker.py --model <path> --data <local.jsonl> --port 8765
On the 5090:     RemoteClient("http://<jetson-ip>:8765") passed to run_federated(...)

Only the global adapter goes down and only the compressed, local adapter update comes up;
the Jetson's tuples never leave it.
"""
from __future__ import annotations

import io
import pickle
import time
import urllib.request
from dataclasses import asdict

from .fedlora import FedConfig, decompress


def _dumps(obj) -> bytes:
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _loads(b: bytes):
    import torch
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


class RemoteClient:
    def __init__(self, url: str, timeout_s: float = 3600):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.last_round_s = None

    def train_round(self, global_state: dict, cfg: FedConfig):
        body = _dumps({"state": global_state, "cfg": asdict(cfg)})
        req = urllib.request.Request(f"{self.url}/train", data=body, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            resp = _loads(r.read())
        self.last_round_s = time.perf_counter() - t0
        return decompress(resp["payload"]), int(resp["nbytes"])


def serve_device_worker(model_path: str, data: list, port: int = 8765, device: str = "cuda"):
    """Blocking HTTP server run on the Jetson."""
    import random
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from .fedlora import FedConfig, build_lora_model, compress, local_train

    state = {"model": None, "tok": None}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/train":
                self.send_response(404); self.end_headers(); return
            n = int(self.headers.get("Content-Length", 0))
            msg = _loads(self.rfile.read(n))
            cfg = FedConfig(**{k: (tuple(v) if k == "target_modules" else v) for k, v in msg["cfg"].items()})
            if state["model"] is None:
                state["model"], state["tok"] = build_lora_model(model_path, cfg, device, dtype="float16")
            delta, loss = local_train(state["model"], state["tok"], data, msg["state"], cfg, device,
                                      random.Random(cfg.seed))
            payload, nb = compress(delta, cfg.topk_frac, cfg.quant_bits)
            out = _dumps({"payload": payload, "nbytes": nb, "loss": loss})
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    print(f"[fed-worker] listening on :{port}")
    HTTPServer(("0.0.0.0", port), H).serve_forever()


__all__ = ["RemoteClient", "serve_device_worker", "pickle"]
