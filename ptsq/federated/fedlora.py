"""Federated LoRA fine-tuning of the device-tier model (DP-FedAvg + update compression).

Each device holds tuples it has queried before together with the labels returned by its
permitted reference tier (free supervision produced by normal query processing). Clients
train a LoRA adapter so that the device model's Yes/No logits match those labels; only
(compressed, clipped) adapter *updates* leave the device. The server (the private edge
server) aggregates them with DP-FedAvg. A better-calibrated device model shrinks the
uncertain band of the cascade, so fewer tuples are escalated.

Heavy dependencies (torch, transformers, peft) are imported lazily.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class FedConfig:
    rounds: int = 20
    clients_per_round: int = 8
    local_steps: int = 30
    batch_size: int = 8
    lr: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")
    # compression of the uplink update
    topk_frac: float = 0.1          # keep this fraction of entries per tensor (1.0 = dense)
    quant_bits: int = 8             # 8 = int8 quantisation, 32 = none
    # DP-FedAvg (server-side Gaussian mechanism on clipped client updates)
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0   # sigma; 0 disables DP
    seed: int = 0


# --------------------------------------------------------------------------------------
# Non-IID partitioning
# --------------------------------------------------------------------------------------
def dirichlet_split(groups: np.ndarray, n_clients: int, alpha: float, seed: int = 0) -> list[np.ndarray]:
    """Split indices across clients with Dirichlet(alpha) proportions per group
    (smaller alpha = more heterogeneous)."""
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in range(n_clients)]
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        rng.shuffle(idx)
        p = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for c, chunk in enumerate(np.split(idx, cuts)):
            parts[c].extend(chunk.tolist())
    return [np.array(sorted(p), dtype=int) for p in parts]


# --------------------------------------------------------------------------------------
# Update compression
# --------------------------------------------------------------------------------------
def compress(delta: dict, topk_frac: float, bits: int) -> tuple[dict, int]:
    """Top-k sparsify + uniform quantise each tensor. Returns (payload, uplink bytes)."""
    import torch
    payload, nbytes = {}, 0
    for k, t in delta.items():
        flat = t.flatten().float()
        n = flat.numel()
        kk = n if topk_frac >= 1 else max(1, int(math.ceil(topk_frac * n)))
        if kk < n:
            idx = flat.abs().topk(kk).indices
            vals = flat[idx]
        else:
            idx, vals = None, flat
        if bits < 32:
            scale = vals.abs().max().clamp_min(1e-12) / (2 ** (bits - 1) - 1)
            q = torch.round(vals / scale).to(torch.int8 if bits <= 8 else torch.int16)
            payload[k] = (tuple(t.shape), idx, q, scale)
            nbytes += q.numel() * (bits // 8) + 4
        else:
            payload[k] = (tuple(t.shape), idx, vals, None)
            nbytes += vals.numel() * 4
        if idx is not None:
            nbytes += idx.numel() * 4
    return payload, nbytes


def decompress(payload: dict) -> dict:
    import torch
    out = {}
    for k, (shape, idx, vals, scale) in payload.items():
        v = vals.float() * scale if scale is not None else vals.float()
        n = int(np.prod(shape))
        flat = torch.zeros(n)
        if idx is None:
            flat = v
        else:
            flat[idx] = v
        out[k] = flat.view(shape)
    return out


# --------------------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------------------
def build_lora_model(model_path: str, cfg: FedConfig, device: str = "cuda", dtype: str = "bfloat16"):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, dtype)).to(device)
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.target_modules), task_type="CAUSAL_LM")
    return get_peft_model(base, lcfg), tok


def lora_state(model) -> dict:
    from peft import get_peft_model_state_dict
    return {k: v.detach().float().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def load_lora_state(model, state: dict):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state)


def yes_no_loss(model, tok, messages_batch, labels, device):
    import torch
    from ..llm.backends import yes_no_token_ids
    yes_ids, no_ids = yes_no_token_ids(tok)
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    logits = model(**enc).logits[:, -1, :].float()
    ly = torch.logsumexp(logits[:, yes_ids], -1)
    ln = torch.logsumexp(logits[:, no_ids], -1)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    return torch.nn.functional.binary_cross_entropy_with_logits(ly - ln, y)


def local_train(model, tok, data: list[tuple[list[dict], bool]], global_state: dict,
                cfg: FedConfig, device: str, rng: random.Random) -> tuple[dict, float]:
    """Run local steps from the global adapter; return (delta, mean loss)."""
    import torch
    load_lora_state(model, global_state)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    losses = []
    for _ in range(cfg.local_steps):
        batch = [rng.choice(data) for _ in range(cfg.batch_size)]
        loss = yes_no_loss(model, tok, [b[0] for b in batch], [float(b[1]) for b in batch], device)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
    new = lora_state(model)
    delta = {k: new[k] - global_state[k] for k in new}
    return delta, float(np.mean(losses)) if losses else 0.0


def clip_delta(delta: dict, clip: float) -> dict:
    import torch
    norm = torch.sqrt(sum((v.float() ** 2).sum() for v in delta.values()))
    f = min(1.0, clip / (float(norm) + 1e-12))
    return {k: v * f for k, v in delta.items()}


def aggregate(global_state: dict, deltas: list[dict], cfg: FedConfig, rng_t=None) -> dict:
    """DP-FedAvg: average clipped deltas, add N(0, (sigma*C/m)^2) noise."""
    import torch
    m = len(deltas)
    avg = {k: sum(d[k] for d in deltas) / m for k in global_state}
    if cfg.noise_multiplier > 0:
        g = rng_t or torch.Generator().manual_seed(cfg.seed)
        std = cfg.noise_multiplier * cfg.clip_norm / m
        avg = {k: v + torch.randn(v.shape, generator=g) * std for k, v in avg.items()}
    return {k: global_state[k] + avg[k] for k in global_state}


def run_federated(model_path: str, client_data: list[list[tuple[list[dict], bool]]], cfg: FedConfig,
                  device: str = "cuda", eval_fn=None, save_dir: str | None = None,
                  remote_clients: list | None = None) -> list[dict]:
    """Simulate federated training on one GPU (clients run sequentially).

    ``remote_clients`` may contain ``transport.RemoteClient`` objects (e.g. the Jetson) that
    train on their own data and return compressed updates over HTTP.
    ``eval_fn(model, tok, round)`` returns a metrics dict logged per round.
    """
    import os
    import torch
    rng = random.Random(cfg.seed)
    model, tok = build_lora_model(model_path, cfg, device)
    gstate = lora_state(model)
    history = []
    n_local = len(client_data)
    remote_clients = remote_clients or []
    for rnd in range(1, cfg.rounds + 1):
        total = n_local + len(remote_clients)
        chosen = rng.sample(range(total), min(cfg.clients_per_round, total))
        deltas, up_bytes, losses = [], 0, []
        for c in chosen:
            if c < n_local:
                if not client_data[c]:
                    continue
                d, l = local_train(model, tok, client_data[c], gstate, cfg, device, rng)
                losses.append(l)
                payload, nb = compress(d, cfg.topk_frac, cfg.quant_bits)
                d = decompress(payload)
            else:
                d, nb = remote_clients[c - n_local].train_round(gstate, cfg)
            up_bytes += nb
            deltas.append(clip_delta(d, cfg.clip_norm))
        if not deltas:
            continue
        gstate = aggregate(gstate, deltas, cfg)
        load_lora_state(model, gstate)
        rec = {"round": rnd, "clients": len(deltas), "uplink_bytes": up_bytes,
               "train_loss": float(np.mean(losses)) if losses else None}
        if eval_fn is not None:
            model.eval()
            with torch.no_grad():
                rec.update(eval_fn(model, tok, rnd))
        if save_dir:
            path = os.path.join(save_dir, f"round_{rnd:03d}")
            model.save_pretrained(path)
            rec["adapter"] = path
        history.append(rec)
        print(f"[fed] round {rnd}: {rec}")
    return history
