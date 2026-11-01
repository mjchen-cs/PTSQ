"""PPO router for the routing phase (guarantee-preserving by construction).

The action space only contains batch sizes and *skip-to-reference* ratios (see
``simulator.ACTIONS``), so any policy - including a badly trained one - preserves the
accuracy guarantee (Theorem 2). The policy is a small MLP so it runs on the Jetson CPU.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..simulator import ACTIONS, STATE_DIM, CostWeights, Item, RoutingSim, TierProfile
from ..types import PrivacyLevel


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class PolicyNet:
    """Actor-critic MLP (created lazily so that importing PTSQ does not require torch)."""

    def __init__(self, hidden: int = 64):
        torch, nn = _torch()

        class AC(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.Tanh(),
                                          nn.Linear(hidden, hidden), nn.Tanh())
                self.pi = nn.Linear(hidden, len(ACTIONS))
                self.v = nn.Linear(hidden, 1)

            def forward(self, x):
                h = self.body(x)
                return self.pi(h), self.v(h).squeeze(-1)

        self.net = AC()


class PPORouter:
    """Router wrapper used by the simulator and the live executor."""

    def __init__(self, path: str | None = None, greedy: bool = True):
        torch, _ = _torch()
        self.torch = torch
        self.pn = PolicyNet()
        if path:
            self.pn.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.pn.net.eval()
        self.greedy = greedy

    def act(self, state) -> int:
        with self.torch.no_grad():
            logits, _ = self.pn.net(self.torch.as_tensor(state, dtype=self.torch.float32)[None])
        if self.greedy:
            return int(logits.argmax(-1).item())
        return int(self.torch.distributions.Categorical(logits=logits).sample().item())


@dataclass
class PPOConfig:
    episodes: int = 2000
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    batch_episodes: int = 16
    entropy: float = 0.01
    reward_scale: float = 1.0
    seed: int = 0


def random_env(item_pool: list[Item], base_e: TierProfile, base_c: TierProfile,
               weights: CostWeights, rng: random.Random, n_range=(500, 6000)) -> RoutingSim:
    """Domain randomisation: resample items, device speed, bandwidth, concurrency and
    service speed so that the policy generalises across queries and network conditions."""
    n = rng.randint(*n_range)
    dev_speed = rng.uniform(0.3, 3.0)
    items = []
    for i in range(n):
        it = rng.choice(item_pool)
        items.append(Item(rid=i, level=it.level, dev_outcome=it.dev_outcome, tokens=it.tokens,
                          nbytes=it.nbytes, t_dev=it.t_dev * dev_speed, up_after_e=it.up_after_e,
                          reference=it.reference, start_remote=it.start_remote))

    def jitter(p: TierProfile) -> TierProfile:
        return TierProfile(rtt_s=p.rtt_s * rng.uniform(0.5, 3), bw_mbps=p.bw_mbps * rng.uniform(0.05, 1.5),
                           fixed_s=p.fixed_s * rng.uniform(0.5, 2), per_token_s=p.per_token_s * rng.uniform(0.5, 3),
                           concurrency=max(1, int(round(p.concurrency * rng.uniform(0.25, 2)))),
                           batchable=p.batchable, price_in=p.price_in * rng.uniform(0.5, 2),
                           price_out=p.price_out * rng.uniform(0.5, 2), tpm=jitter_tpm(p.tpm))

    def jitter_tpm(tpm: float) -> float:
        # the provider's tokens-per-minute limit decides whether the cloud or the device is the
        # bottleneck, so it must be part of the randomisation (dropping it trains the router
        # on an unlimited cloud, where skipping the edge looks free). 25% of the episodes are
        # unlimited, the rest scale the measured limit.
        if not tpm or rng.random() < 0.25:
            return 0.0
        return tpm * rng.uniform(0.5, 3.0)
    return RoutingSim(items, jitter(base_e), jitter(base_c), weights=weights)


def synthetic_item_pool(n: int = 4000, seed: int = 0) -> list[Item]:
    """Fallback pool when no trace is available (dry runs only)."""
    from ..types import TierId
    rng = random.Random(seed)
    pool = []
    for i in range(n):
        lv = rng.choices([PrivacyLevel.LOW, PrivacyLevel.MEDIUM, PrivacyLevel.HIGH], [0.6, 0.3, 0.1])[0]
        ref = {PrivacyLevel.LOW: TierId.CLOUD, PrivacyLevel.MEDIUM: TierId.EDGE, PrivacyLevel.HIGH: TierId.DEVICE}[lv]
        if lv == PrivacyLevel.HIGH:
            o = rng.choice(["acc", "rej", "rej"])
        else:
            o = rng.choices(["acc", "rej", "E_mid" if lv == PrivacyLevel.LOW else "E_final"], [0.15, 0.45, 0.4])[0]
        tok = rng.randint(80, 900)
        pool.append(Item(rid=i, level=lv, dev_outcome=o, tokens=tok, nbytes=int(tok * 3.3),
                         t_dev=0.02 + 5e-5 * tok, up_after_e=rng.random() < 0.3, reference=ref))
    return pool


def train_ppo(item_pool: list[Item], base_e: TierProfile, base_c: TierProfile,
              weights: CostWeights, cfg: PPOConfig = PPOConfig(), out_path: str | None = None,
              log_every: int = 50):
    torch, _ = _torch()
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    pn = PolicyNet()
    opt = torch.optim.Adam(pn.net.parameters(), lr=cfg.lr)
    history = []
    ep = 0
    while ep < cfg.episodes:
        S, A, LP, R, V, D = [], [], [], [], [], []
        ep_costs = []
        for _ in range(cfg.batch_episodes):
            env = random_env(item_pool, base_e, base_c, weights, rng)
            s = env.reset()
            env._advance()
            s = env.state()
            total = 0.0
            while not env.done():
                if not env.needs_decision():
                    if not env.events:
                        break
                    env._advance()
                    continue
                st = torch.as_tensor(s, dtype=torch.float32)
                with torch.no_grad():
                    logits, v = pn.net(st[None])
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                s2, r, done = env.step(int(a.item()))
                S.append(st); A.append(a.squeeze(0)); LP.append(dist.log_prob(a).squeeze(0))
                R.append(r * cfg.reward_scale / max(env.n0 / 1000.0, 0.1)); V.append(v.squeeze(0)); D.append(done)
                total += r
                s = s2
            if D:
                D[-1] = True
            ep_costs.append(-total)
            ep += 1
        if not S:
            continue
        # GAE
        R_t = torch.tensor(R, dtype=torch.float32)
        V_t = torch.stack(V)
        D_t = torch.tensor(D, dtype=torch.float32)
        adv = torch.zeros_like(R_t)
        last = 0.0
        for t in reversed(range(len(R))):
            nv = 0.0 if D[t] or t == len(R) - 1 else V_t[t + 1]
            delta = R_t[t] + cfg.gamma * nv * (1 - D_t[t]) - V_t[t]
            last = delta + cfg.gamma * cfg.lam * (1 - D_t[t]) * last
            adv[t] = last
        ret = adv + V_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        S_t, A_t, LP_t = torch.stack(S), torch.stack(A), torch.stack(LP)
        for _ in range(cfg.epochs):
            logits, v = pn.net(S_t)
            dist = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(A_t) - LP_t)
            l1, l2 = ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
            loss = -torch.min(l1, l2).mean() + 0.5 * ((v - ret) ** 2).mean() - cfg.entropy * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pn.net.parameters(), 0.5)
            opt.step()
        history.append(float(np.mean(ep_costs)))
        if log_every and len(history) % max(1, log_every // cfg.batch_episodes) == 0:
            print(f"[ppo] episodes={ep} mean_cost={history[-1]:.3f}")
    if out_path:
        torch.save(pn.net.state_dict(), out_path)
    return pn, history
