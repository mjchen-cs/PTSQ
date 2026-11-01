"""LLM backends that score a semantic predicate on a batch of tuples.

* ``OpenAICompatBackend`` - any OpenAI-compatible chat endpoint: vLLM / llama.cpp server
  (edge, device) and commercial APIs such as Zhipu (GLM) and SiliconFlow (cloud).
* ``HFLocalBackend``      - a local HuggingFace model (+ optional LoRA adapter); reads the
  Yes/No next-token logits directly. Recommended for the device tier because the federated
  LoRA adapters can be hot-swapped.
* ``MockBackend``         - synthetic, deterministic scores for tests and dry runs. Results
  produced with it are NOT experimental results.

All backends return ``ScoreResult`` with ``p_yes`` in [0, 1].
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Sequence

from ..types import Request, ScoreResult
from .cache import ResponseCache

log = logging.getLogger(__name__)

_YES = re.compile(r"^\W*(yes|是|true)\b", re.I)
_NO = re.compile(r"^\W*(no|否|不是|false)\b", re.I)


def parse_yes_no(text: str) -> bool | None:
    t = (text or "").strip()
    if _YES.match(t):
        return True
    if _NO.match(t):
        return False
    low = t.lower()
    if "yes" in low and "no" not in low:
        return True
    if "no" in low and "yes" not in low:
        return False
    return None


class Backend(ABC):
    """Abstract scorer. Subclasses implement ``_score_batch``."""

    def __init__(self, name: str, cache: ResponseCache | None = None,
                 price_in: float = 0.0, price_out: float = 0.0):
        self.name = name
        self.cache = cache
        # price per 1M tokens (currency of your choice, e.g. CNY)
        self.price_in = price_in
        self.price_out = price_out

    @property
    def backend_id(self) -> str:
        return self.name

    def money(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_in + completion_tokens * self.price_out) / 1e6

    async def score(self, reqs: Sequence[Request]) -> list[ScoreResult]:
        out: list[ScoreResult | None] = [None] * len(reqs)
        todo_idx, todo = [], []
        for i, r in enumerate(reqs):
            if self.cache is not None:
                hit = self.cache.get(self.cache.key(self.backend_id, {"m": r.messages}))
                if hit is not None:
                    res = ScoreResult(**{**hit, "rid": r.rid})
                    res.cached = True
                    out[i] = res
                    continue
            todo_idx.append(i)
            todo.append(r)
        if todo:
            fresh = await self._score_batch(todo)
            for i, r, res in zip(todo_idx, todo, fresh):
                out[i] = res
                if self.cache is not None:
                    d = dict(res.__dict__)
                    d.pop("rid", None)
                    d["cached"] = False
                    self.cache.put(self.cache.key(self.backend_id, {"m": r.messages}), d)
        return [o for o in out if o is not None]

    @abstractmethod
    async def _score_batch(self, reqs: Sequence[Request]) -> list[ScoreResult]:
        ...

    async def complete(self, messages: list[dict], max_tokens: int = 1024) -> str:
        """Free-form completion (used only at query-compile time)."""
        raise NotImplementedError


# ======================================================================================
# OpenAI-compatible HTTP backend
# ======================================================================================
class OpenAICompatBackend(Backend):
    """score_mode:
        "logprobs"   - max_tokens=1 with top_logprobs; p_yes = P(yes)/(P(yes)+P(no)).
        "verbalized" - model returns JSON {answer, confidence}; p_yes = confidence/100.
        "label"      - hard label only; p_yes in {0, 1} (enough for a reference tier).
    If "logprobs" is requested but the endpoint returns none, the backend falls back to
    "label" for that request and logs a warning once.
    """

    def __init__(self, name: str, base_url: str, model: str, api_key_env: str | None = None,
                 score_mode: str = "logprobs", max_concurrency: int = 8, timeout_s: float = 60,
                 max_retries: int = 5, extra_body: dict | None = None,
                 tpm_limit: float | None = None, rpm_limit: float | None = None,
                 batch: dict | None = None, **kw):
        super().__init__(name, **kw)
        # provider rate limits (e.g. SiliconFlow L0: 1000 RPM / 20000 TPM); requests are
        # paced client-side so the run does not degrade into 429 retries
        self._limiter = _RateLimiter(tpm_limit, rpm_limit) if (tpm_limit or rpm_limit) else None
        self.batch_cfg = batch or {}             # used only by scripts/batch_label.py
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.score_mode = score_mode
        self.sem = asyncio.Semaphore(max_concurrency)
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self._warned = False
        self._client = None

    @property
    def backend_id(self) -> str:
        return f"{self.name}|{self.model}|{self.score_mode}"

    def _http(self):
        if self._client is None:
            import httpx
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            # IPv4 only (broken IPv6 routes cause sporadic ConnectErrors on WSL/Jetson) and a
            # pool sized to the concurrency so a stalled network cannot exhaust it
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s, connect=20.0, pool=120.0), headers=headers,
                limits=httpx.Limits(max_connections=max(64, 2 * self.max_concurrency), max_keepalive_connections=max(32, self.max_concurrency)),
                transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=2))
        return self._client

    async def _post(self, body: dict) -> dict:
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                async with self.sem:
                    r = await self._http().post(f"{self.base_url}/chat/completions", json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                if attempt == self.max_retries:
                    raise
                log.warning("%s request failed (%r); retry in %.1fs", self.name, e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def request_body(self, r: Request) -> dict:
        """Chat-completions body for scoring one request (shared with the batch API path)."""
        from .prompts import SYSTEM_FILTER, SYSTEM_FILTER_VERBAL
        msgs = [dict(m) for m in r.messages]
        if self.score_mode == "verbalized" and msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = SYSTEM_FILTER_VERBAL
        elif msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = SYSTEM_FILTER
        body = {"model": self.model, "messages": msgs, "temperature": 0, **self.extra_body}
        if self.score_mode == "logprobs":
            body.update({"max_tokens": 1, "logprobs": True, "top_logprobs": 10})
        elif self.score_mode == "verbalized":
            body.update({"max_tokens": 40})
        else:
            body.update({"max_tokens": 3})
        return body

    def parse_response(self, r: Request, data: dict, lat: float) -> ScoreResult:
        choice = data["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        p_yes = None
        if self.score_mode == "logprobs":
            p_yes = _p_yes_from_logprobs(choice.get("logprobs"))
            if p_yes is None and not self._warned:
                log.warning("%s: no logprobs returned; falling back to hard labels", self.name)
                self._warned = True
        elif self.score_mode == "verbalized":
            p_yes = _p_yes_from_verbal(text)
        lab = parse_yes_no(text)
        if p_yes is None:
            p_yes = 1.0 if lab else 0.0
        if lab is None:
            lab = p_yes >= 0.5
        return ScoreResult(rid=r.rid, p_yes=float(p_yes), label=bool(lab), prompt_tokens=pt,
                           completion_tokens=ct, latency_s=lat, money=self.money(pt, ct), raw=text)

    async def _one(self, r: Request) -> ScoreResult:
        body = self.request_body(r)
        ev = None
        if self._limiter is not None:
            est = sum(len(m.get("content") or "") for m in body["messages"]) / 3.5 + body.get("max_tokens", 1) + 8
            ev = await self._limiter.acquire(est)
        t0 = time.perf_counter()
        data = await self._post(body)
        if ev is not None:
            self._limiter.settle(ev, est, (data.get("usage") or {}).get("total_tokens"))
        return self.parse_response(r, data, time.perf_counter() - t0)

    async def _score_batch(self, reqs: Sequence[Request]) -> list[ScoreResult]:
        return list(await asyncio.gather(*[self._one(r) for r in reqs]))

    async def complete(self, messages: list[dict], max_tokens: int = 1024) -> str:
        key = None
        if self.cache is not None:
            key = self.cache.key(self.backend_id + "|complete", {"m": messages})
            hit = self.cache.get(key)
            if hit is not None:
                return hit["text"]
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "max_tokens": max_tokens, **self.extra_body}
        data = await self._post(body)
        text = data["choices"][0]["message"]["content"]
        if key is not None:
            self.cache.put(key, {"text": text})
        return text

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()


class _RateLimiter:
    """Sliding 60-s window over tokens and requests (client-side pacing).

    Token counts are estimated from characters before the call; ``settle`` replaces the
    estimate with the provider's reported usage afterwards and adapts the chars->tokens
    ratio, so the pace converges to the real TPM limit instead of a pessimistic guess."""

    def __init__(self, tpm: float | None, rpm: float | None):
        self.tpm, self.rpm = tpm, rpm
        self.events: list[list[float]] = []               # [time, tokens]
        self.lock = asyncio.Lock()
        self.ratio = 1.0                                   # actual / estimated tokens (EMA)

    async def acquire(self, tokens: float) -> list[float]:
        tokens = tokens * self.ratio
        async with self.lock:
            while True:
                now = time.monotonic()
                self.events = [e for e in self.events if now - e[0] < 60.0]
                used = sum(e[1] for e in self.events)
                ok_t = self.tpm is None or used + tokens <= self.tpm or not self.events
                ok_r = self.rpm is None or len(self.events) + 1 <= self.rpm
                if ok_t and ok_r:
                    ev = [now, tokens]
                    self.events.append(ev)
                    return ev
                await asyncio.sleep(max(0.05, min(60.0 - (now - self.events[0][0]), 2.0)))

    def settle(self, ev: list[float], est_raw: float, actual: int | None):
        if actual and actual > 0 and est_raw > 0:
            ev[1] = float(actual)
            self.ratio = 0.9 * self.ratio + 0.1 * (actual / est_raw)


def _p_yes_from_logprobs(lp: dict | None) -> float | None:
    if not lp:
        return None
    content = lp.get("content") or []
    if not content:
        return None
    tops = content[0].get("top_logprobs") or []
    py = pn = 0.0
    for t in tops:
        tok = (t.get("token") or "").strip().lower()
        p = math.exp(t.get("logprob", -1e9))
        if tok.startswith("yes") or tok == "是":
            py += p
        elif tok.startswith("no") or tok in ("否", "不"):
            pn += p
    if py + pn <= 0:
        return None
    return py / (py + pn)


def _p_yes_from_verbal(text: str) -> float | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        conf = float(d.get("confidence"))
        return min(max(conf / 100.0, 0.0), 1.0)
    except Exception:  # noqa: BLE001
        return None


# ======================================================================================
# Local HuggingFace backend (device tier, supports LoRA adapters)
# ======================================================================================
class HFLocalBackend(Backend):
    def __init__(self, name: str, model_path: str, lora_path: str | None = None,
                 device: str = "cuda", dtype: str = "float16", batch_size: int = 8,
                 max_length: int = 1024, max_batch_tokens: int = 6144, **kw):
        super().__init__(name, **kw)
        self.model_path, self.lora_path = model_path, lora_path
        self.device, self.dtype = device, dtype
        self.batch_size, self.max_length = batch_size, max_length
        self.max_batch_tokens = max_batch_tokens
        self._model = self._tok = None
        self._yes_ids = self._no_ids = None

    @property
    def backend_id(self) -> str:
        return f"{self.name}|{self.model_path}|{self.lora_path}"

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._tok.padding_side = "left"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        # load straight onto the target device: on the Jetson CPU and GPU share 8 GB, so a
        # CPU copy followed by .to(cuda) doubles the peak and gets the process OOM-killed
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=getattr(torch, self.dtype), low_cpu_mem_usage=True,
            device_map={"": self.device})
        if self.lora_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.lora_path)
        model.eval()
        self._model = model
        self._yes_ids, self._no_ids = yes_no_token_ids(self._tok)

    def set_adapter(self, lora_path: str | None):
        """Swap the LoRA adapter (after a federated round)."""
        self.lora_path = lora_path
        self._model = None
        self.load()

    def _last_logits(self, enc):
        """Logits of the last position only. model(**enc).logits would materialise
        batch x seq x vocab (152k for Qwen) logits - several GB on the Jetson."""
        m = self._model
        base = m.get_base_model() if hasattr(m, "get_base_model") else m
        if hasattr(base, "model") and hasattr(base, "lm_head"):
            h = base.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            return base.lm_head(h[:, -1, :])
        return m(**enc).logits[:, -1, :]

    def _prompts(self, reqs):
        return [self._tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True)
                for r in reqs]

    def _batches(self, prompts: list[str]) -> list[list[int]]:
        """Length-sorted batches under a padded-token budget. Real reviews vary a lot in
        length; fixed batches of 16 padded to the longest one (up to max_length) exhaust the
        Jetson's shared 8 GB, so cap batch_size x longest_prompt at max_batch_tokens."""
        lens = [min(len(x), self.max_length) for x in
                self._tok(prompts, add_special_tokens=False)["input_ids"]]
        order = sorted(range(len(prompts)), key=lambda i: lens[i])
        out, cur, longest = [], [], 0
        for i in order:
            L = max(longest, lens[i])
            if cur and (len(cur) + 1 > self.batch_size or (len(cur) + 1) * L > self.max_batch_tokens):
                out.append(cur); cur, L = [], lens[i]
            cur.append(i); longest = L
        if cur:
            out.append(cur)
        return out

    def _score_sync(self, reqs: Sequence[Request]) -> list[ScoreResult]:
        import torch
        self.load()
        prompts = self._prompts(reqs)
        res: list[ScoreResult | None] = [None] * len(reqs)
        for idx in self._batches(prompts):
            enc = self._tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                            truncation=True, max_length=self.max_length).to(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self._last_logits(enc).float()
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) / len(idx)
            ly = torch.logsumexp(logits[:, self._yes_ids], dim=-1)
            ln = torch.logsumexp(logits[:, self._no_ids], dim=-1)
            p = torch.sigmoid(ly - ln).tolist()
            ntok = enc["attention_mask"].sum(dim=1).tolist()
            for i, pi, nt in zip(idx, p, ntok):
                res[i] = ScoreResult(rid=reqs[i].rid, p_yes=float(pi), label=pi >= 0.5,
                                     prompt_tokens=int(nt), completion_tokens=1, latency_s=lat)
            del enc, logits
        if self.device.startswith("cuda"):
            # varying sequence lengths fragment the caching allocator; on the Jetson the
            # cached blocks come out of the shared 8 GB, so hand them back after each call
            torch.cuda.empty_cache()
        return res  # type: ignore[return-value]

    async def _score_batch(self, reqs: Sequence[Request]) -> list[ScoreResult]:
        return await asyncio.to_thread(self._score_sync, reqs)


def yes_no_token_ids(tok) -> tuple[list[int], list[int]]:
    def ids(words):
        s = set()
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if t:
                s.add(t[0])
        return sorted(s)
    return ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])


# ======================================================================================
# Mock backend (tests / dry runs only)
# ======================================================================================
class MockBackend(Backend):
    """Deterministic synthetic scorer.

    ``truth`` maps rid -> bool (the synthetic ground truth). ``quality`` controls the
    separation of the score distribution (higher = more accurate); ``flip`` is the rate
    of confidently wrong answers. A quality of ``None`` returns the truth exactly
    (acts as a perfect reference).
    """

    def __init__(self, name: str, truth: dict[int, bool], quality: float | None = 2.0,
                 flip: float = 0.0, seed: int = 0, base_latency_s: float = 0.02,
                 per_token_s: float = 2e-5, tokens_per_char: float = 0.3, **kw):
        super().__init__(name, **kw)
        self.truth, self.quality, self.flip, self.seed = truth, quality, flip, seed
        self.base_latency_s, self.per_token_s = base_latency_s, per_token_s
        self.tokens_per_char = tokens_per_char

    def _u(self, rid: int, salt: str) -> float:
        h = hashlib.sha256(f"{self.seed}|{self.name}|{rid}|{salt}".encode()).digest()
        return (int.from_bytes(h[:8], "big") + 0.5) / 2**64

    def score_one(self, rid: int, nbytes: int = 400) -> ScoreResult:
        y = bool(self.truth.get(rid, False))
        if self.quality is None:
            p = 1.0 if y else 0.0
        else:
            u1, u2 = self._u(rid, "a"), self._u(rid, "b")
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            sign = 1.0 if y else -1.0
            if self._u(rid, "flip") < self.flip:
                sign = -sign
            p = 1 / (1 + math.exp(-(self.quality * sign + z)))
        pt = int(nbytes * self.tokens_per_char)
        return ScoreResult(rid=rid, p_yes=p, label=p >= 0.5, prompt_tokens=pt, completion_tokens=1,
                           latency_s=self.base_latency_s + self.per_token_s * pt,
                           money=self.money(pt, 1))

    async def _score_batch(self, reqs: Sequence[Request]) -> list[ScoreResult]:
        return [self.score_one(r.rid, r.nbytes) for r in reqs]

    async def complete(self, messages: list[dict], max_tokens: int = 1024) -> str:
        return json.dumps(self.meta_rules) if hasattr(self, "meta_rules") else "[]"


def build_backend(cfg: dict, cache: ResponseCache | None = None, mock_truth: dict | None = None) -> Backend:
    """Factory from a config dict (see configs/default.yaml)."""
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    common = {"cache": cache, "price_in": cfg.pop("price_in", 0.0), "price_out": cfg.pop("price_out", 0.0)}
    if kind == "openai":
        return OpenAICompatBackend(**cfg, **common)
    if kind == "hf":
        return HFLocalBackend(**cfg, **common)
    if kind == "mock":
        return MockBackend(truth=mock_truth or {}, **cfg, **{**common, "cache": None})
    raise ValueError(f"unknown backend kind {kind}")
