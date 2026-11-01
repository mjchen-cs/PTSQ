"""Discrete-event simulator of PTSQ's streaming execution phase.

Used for RL training and for trace replay. Tier performance is described by
``TierProfile`` objects *measured* on the real hardware with ``scripts/profile_tiers.py``
(Jetson Orin Nano Super, RTX 5090 + vLLM, cloud APIs).

After calibration every remaining tuple enters the device queue. Three resources work in
parallel: the device (one batched worker), the edge server (``concurrency`` batched
dispatches) and the cloud (``concurrency`` single requests). For each tuple, the outcome of
device scoring under the certified plan is precomputed (``dev_outcome``):

    "acc" / "rej"  decided on the device
    "E_final"      escalate to E, which is the reference (MEDIUM stratum)
    "E_mid"        escalate to E; if ``up_after_e`` E's score is uncertain -> C
    "C_final"      escalate straight to C (reference)

Router actions are *monotone escalations* (Theorem 2): they can only send a tuple to its
permitted reference earlier than the plan would, never to a weaker tier:
    beta - fraction of the next device batch's LOW/MEDIUM tuples that BYPASS the device
           and go straight to their reference (C for LOW, E for MEDIUM);
    rho  - fraction of the E_mid tuples of the next edge batch that skip E and go to C;
    b_E  - edge batch size.
Because the reference decides exactly, no action can lower recall or precision w.r.t.
the reference; the privacy shield asserts every dispatch.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .privacy import PrivacyShield
from .types import PrivacyLevel, TierId

BYPASS_RATIOS = (0.0, 0.25, 0.5, 1.0)
SKIP_RATIOS = (0.0, 0.5, 1.0)
EDGE_BATCHES = (32, 128)
ACTIONS = [(be, rho, b) for be in BYPASS_RATIOS for rho in SKIP_RATIOS for b in EDGE_BATCHES]
STATE_DIM = 12
DEVICE_BATCH = 16


@dataclass
class TierProfile:
    rtt_s: float = 0.0              # round-trip time from the device
    bw_mbps: float = 1e9            # uplink bandwidth from the device
    fixed_s: float = 0.05           # per-dispatch fixed latency (prefill launch, HTTP)
    per_token_s: float = 1e-4       # marginal latency per prompt token
    concurrency: int = 1            # concurrent dispatches
    batchable: bool = True          # True: a dispatch carries a batch (vLLM); False: 1 request
    price_in: float = 0.0           # per 1M prompt tokens
    price_out: float = 0.0          # per 1M completion tokens
    tpm: float = 0.0                # provider rate limit, tokens per minute (0 = unlimited)

    def rate_s(self, tokens: float) -> float:
        """Minimum spacing between requests imposed by the provider's TPM limit."""
        return tokens * 60.0 / self.tpm if self.tpm and self.tpm > 0 else 0.0

    def net_s(self, nbytes: int) -> float:
        return self.rtt_s + nbytes * 8 / (self.bw_mbps * 1e6)

    def service_s(self, tokens: int, n_items: int = 1) -> float:
        return self.fixed_s + self.per_token_s * tokens

    def money(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_in + completion_tokens * self.price_out) / 1e6

    @classmethod
    def from_dict(cls, d: dict) -> "TierProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Item:
    rid: int
    level: PrivacyLevel
    dev_outcome: str            # acc | rej | E_final | E_mid | C_final
    tokens: int
    nbytes: int
    t_dev: float = 0.05         # measured device latency for this tuple (amortised in batch)
    up_after_e: bool = False
    reference: TierId = TierId.DEVICE
    start_remote: bool = False   # plan skips the device: start at the edge/cloud queue per dev_outcome


@dataclass
class SimResult:
    makespan_s: float = 0.0
    money: float = 0.0
    bytes_up: dict = field(default_factory=lambda: {TierId.EDGE: 0, TierId.CLOUD: 0})
    n_tier: dict = field(default_factory=lambda: {TierId.DEVICE: 0, TierId.EDGE: 0, TierId.CLOUD: 0})
    bypassed: int = 0
    skipped: int = 0
    decided_by: dict = field(default_factory=dict)   # rid -> TierId that made the final decision


@dataclass
class CostWeights:
    latency: float = 1.0      # per second
    money: float = 1.0        # per currency unit
    bytes: float = 0.0        # per MB uplink


class RoutingSim:
    """Event-driven simulation of one query's streaming execution phase."""

    def __init__(self, items: list[Item], prof_e: TierProfile, prof_c: TierProfile,
                 shield: PrivacyShield | None = None, weights: CostWeights | None = None,
                 money_budget: float | None = None, device_batch: int = DEVICE_BATCH):
        self.items = items
        self.pe, self.pc = prof_e, prof_c
        self.shield = shield
        self.w = weights or CostWeights()
        self.budget = money_budget
        self.device_batch = device_batch
        self.reset()

    # ----------------------------------------------------------------------------------
    def reset(self):
        self.t = 0.0
        self.qd = deque(it for it in self.items if not it.start_remote)
        self.qe: deque[Item] = deque()
        self.qc: deque[Item] = deque(it for it in self.items if it.start_remote and it.dev_outcome == "C_final")
        self.n_mid_qe = 0
        self.n_byp_qd = sum(1 for it in self.qd if it.reference != TierId.DEVICE)
        for it in self.items:
            if it.start_remote and it.dev_outcome in ("E_mid", "E_final"):
                self._push_e(it)
        self.n0 = max(len(self.items), 1)
        self.busy_d = self.busy_e = self.busy_c = 0
        self.c_free_at = 0.0            # token bucket of the cloud TPM limit
        self.events: list = []
        self._seq = 0
        self.res = SimResult()
        self.ewma_d = float(np.mean([it.t_dev for it in self.items])) if self.items else 0.05
        self.ewma_e = (self.pe.service_s(300 * 32) + self.pe.net_s(1500 * 32)) / 32
        self.ewma_c = self.pc.service_s(300) + self.pc.net_s(1500)
        self._last_cost = 0.0
        self._action = ACTIONS.index((0.0, 0.0, 32))
        return self.state()

    def _push_e(self, it: Item):
        self.qe.append(it)
        if it.dev_outcome == "E_mid":
            self.n_mid_qe += 1

    def _push(self, t, kind, payload):
        heapq.heappush(self.events, (t, self._seq, kind, payload))
        self._seq += 1

    # ---------------------------------------------------------------- dispatchers
    def _to_reference(self, it: Item):
        if it.reference == TierId.CLOUD:
            self.qc.append(it)
        elif it.reference == TierId.EDGE:
            it.dev_outcome = "E_final"
            self._push_e(it)
        else:
            raise AssertionError("HIGH tuples cannot bypass the device")

    def _dispatch_d(self):
        if self.busy_d or not self.qd:
            return
        beta, _, _ = ACTIONS[self._action]
        batch = []
        while self.qd and len(batch) < self.device_batch:
            it = self.qd.popleft()
            bypassable = it.reference != TierId.DEVICE
            if bypassable:
                self.n_byp_qd -= 1
            if bypassable and beta > 0 and (beta >= 1 or self._bypass_coin(it, beta)):
                self.res.bypassed += 1
                self._to_reference(it)
                continue
            batch.append(it)
        if batch:
            dur = sum(it.t_dev for it in batch)
            self.busy_d = 1
            self.res.n_tier[TierId.DEVICE] += len(batch)
            self._push(self.t + dur, "d_done", (batch, dur))

    @staticmethod
    def _bypass_coin(it: Item, beta: float) -> bool:
        # deterministic pseudo-random coin per tuple (reproducible)
        return ((it.rid * 2654435761) % 1000) / 1000.0 < beta

    def _dispatch_e(self):
        _, rho, b = ACTIONS[self._action]
        while self.busy_e < self.pe.concurrency and self.qe:
            batch = [self.qe.popleft() for _ in range(min(b, len(self.qe)))]
            self.n_mid_qe -= sum(1 for it in batch if it.dev_outcome == "E_mid")
            to_e = []
            for it in batch:
                if it.dev_outcome == "E_mid" and it.level == PrivacyLevel.LOW and rho > 0 and \
                        (rho >= 1 or self._bypass_coin(it, rho)):
                    self.res.skipped += 1
                    self.qc.append(it)
                else:
                    to_e.append(it)
            if not to_e:
                continue
            if self.shield is not None:
                for it in to_e:
                    self.shield.check(it.level, TierId.EDGE)
            nbytes = sum(it.nbytes for it in to_e)
            toks = sum(it.tokens for it in to_e)
            if self.pe.batchable:
                dur = self.pe.net_s(nbytes) + self.pe.service_s(toks, len(to_e))
            else:
                dur = sum(self.pe.net_s(it.nbytes) + self.pe.service_s(it.tokens) for it in to_e)
            self.res.bytes_up[TierId.EDGE] += nbytes
            self.res.n_tier[TierId.EDGE] += len(to_e)
            for it in to_e:
                self.res.decided_by[it.rid] = TierId.EDGE
            self.busy_e += 1
            self._push(self.t + dur, "e_done", (to_e, dur))

    def _dispatch_c(self):
        while self.busy_c < self.pc.concurrency and self.qc:
            # The money budget is only a state feature: escalations to the reference are
            # never dropped, otherwise the accuracy guarantee would be void.
            it = self.qc.popleft()
            if self.shield is not None:
                self.shield.check(it.level, TierId.CLOUD)
            dur = self.pc.net_s(it.nbytes) + self.pc.service_s(it.tokens)
            start = self.t
            if self.pc.tpm:                      # provider rate limit delays the dispatch
                start = max(self.t, self.c_free_at)
                self.c_free_at = start + self.pc.rate_s(it.tokens + 1)
            dur += start - self.t
            self.res.money += self.pc.money(it.tokens, 1)
            self.res.bytes_up[TierId.CLOUD] += it.nbytes
            self.res.n_tier[TierId.CLOUD] += 1
            self.res.decided_by[it.rid] = TierId.CLOUD
            self.busy_c += 1
            self._push(self.t + dur, "c_done", dur)

    def _dispatch_all(self):
        self._dispatch_d()
        self._dispatch_e()
        self._dispatch_c()

    # ---------------------------------------------------------------- RL interface
    def _cost(self) -> float:
        return (self.w.latency * self.t + self.w.money * self.res.money
                + self.w.bytes * sum(self.res.bytes_up.values()) / 1e6)

    def state(self) -> np.ndarray:
        n_mid, n_byp = self.n_mid_qe, self.n_byp_qd
        return np.array([
            len(self.qd) / self.n0,
            len(self.qe) / self.n0,
            len(self.qc) / self.n0,
            n_mid / max(len(self.qe), 1),
            n_byp / max(len(self.qd), 1),
            self.busy_e / max(self.pe.concurrency, 1),
            self.busy_c / max(self.pc.concurrency, 1),
            min(self.ewma_d, 2.0) / 2.0,
            min(self.ewma_e, 2.0) / 2.0,
            min(self.ewma_c, 10.0) / 10.0,
            (self.res.money / self.budget) if self.budget else 0.0,
            min(self.t, 1800.0) / 1800.0,
        ], dtype=np.float32)

    def needs_decision(self) -> bool:
        return (bool(self.qd) and not self.busy_d) or (bool(self.qe) and self.busy_e < self.pe.concurrency)

    def done(self) -> bool:
        return not (self.qd or self.qe or self.qc or self.busy_d or self.busy_e or self.busy_c)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self._action = action
        self._dispatch_all()
        self._advance()
        cost = self._cost()
        r = -(cost - self._last_cost)
        self._last_cost = cost
        return self.state(), r, self.done()

    def _advance(self):
        """Process events until a router decision is needed or the work is done."""
        self._dispatch_c()
        while not self.done() and not self.needs_decision():
            if not self.events:
                break
            t, _, kind, payload = heapq.heappop(self.events)
            self.t = t
            if kind == "d_done":
                items, dur = payload
                self.busy_d = 0
                self.ewma_d = 0.8 * self.ewma_d + 0.2 * dur / max(len(items), 1)
                for it in items:
                    o = it.dev_outcome
                    if o in ("acc", "rej"):
                        self.res.decided_by[it.rid] = TierId.DEVICE
                    elif o in ("E_final", "E_mid"):
                        self._push_e(it)
                    else:
                        self.qc.append(it)
            elif kind == "e_done":
                items, dur = payload
                self.busy_e -= 1
                self.ewma_e = 0.8 * self.ewma_e + 0.2 * dur / max(len(items), 1)
                for it in items:
                    if it.dev_outcome == "E_mid" and it.up_after_e:
                        self.qc.append(it)
            else:
                self.busy_c -= 1
                self.ewma_c = 0.8 * self.ewma_c + 0.2 * payload
            self._dispatch_c()

    def run(self, router) -> SimResult:
        self.reset()
        self._advance()
        guard = 0
        while not self.done():
            if self.needs_decision():
                self.step(router.act(self.state()))
            elif self.events:
                self._advance()
            else:
                break
            guard += 1
            if guard > 10_000_000:
                raise RuntimeError("simulation did not terminate")
        self.res.makespan_s = self.t
        return self.res


class FixedRouter:
    """Non-learning router (ablation 'w/o RL'): no bypass, no skip, fixed edge batch."""

    def __init__(self, beta: float = 0.0, rho: float = 0.0, batch: int = 32):
        self.a = ACTIONS.index((beta, rho, batch))

    def act(self, state) -> int:
        return self.a


class HeuristicRouter:
    """Hand-written rule: bypass the device when its backlog dominates, skip E when the
    edge server is saturated and the cloud is idle."""

    def act(self, s) -> int:
        dev_backlog, edge_backlog, c_busy, e_busy = s[0], s[1], s[6], s[5]
        beta = 0.5 if dev_backlog > 0.3 and c_busy < 0.5 else 0.0
        rho = 0.5 if (e_busy >= 1.0 and edge_backlog > 0.1 and c_busy < 0.5) else 0.0
        return ACTIONS.index((beta, rho, 128 if edge_backlog > 0.1 else 32))
