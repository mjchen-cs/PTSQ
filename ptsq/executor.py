"""Live executor: runs one semantic query end-to-end on the real three tiers.

Phases (Algorithm 1 in the paper):
  A. on-device: SQL pre-filter, privacy labelling (+ optional redaction), symbolic rules
  B. per-stratum calibration sample -> certified rule subset + thresholds (cascade.py)
  C. streaming execution with the RL router; every dispatch is checked by the privacy shield
It also provides ``collect_trace`` which records all tiers' outputs for offline replay.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cascade import StratumInput, StratumPlan, calibrate_stratum, threshold_certify, split_delta, choose_n_cal, cascade_can_pay_off, reference_is_bottleneck
from .config import Query
from .llm.backends import Backend
from .llm.prompts import filter_messages, render_record
from .privacy import PrivacyPolicy, PrivacyShield, label_privacy, redact
from .replay import Profiles, ReplayParams, calibration_cannot_pay_off, unit_costs
from .simulator import ACTIONS, DEVICE_BATCH, FixedRouter
from .symbolic import eval_rule, propose_rules
from .types import PrivacyLevel, Request, ScoreResult, TierId

log = logging.getLogger(__name__)


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        df = pd.read_csv(p)
    elif p.suffix in (".jsonl", ".json"):
        df = pd.read_json(p, lines=p.suffix == ".jsonl")
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(p.suffix)
    if "rid" not in df.columns:
        df = df.reset_index(drop=True)
        df["rid"] = np.arange(len(df))
    return df


def make_requests(df: pd.DataFrame, q: Query) -> list[Request]:
    reqs = []
    for row in df.to_dict("records"):
        rec = render_record(row, q.text_col, q.extra_cols)
        reqs.append(Request(rid=int(row["rid"]), messages=filter_messages(q.predicate, rec)))
    return reqs


def prepare(df: pd.DataFrame, q: Query, policy: PrivacyPolicy) -> pd.DataFrame:
    """Phase A (on-device): SQL filter, privacy labels, optional redaction-promotion."""
    if q.sql_filter:
        df = df.query(q.sql_filter, engine="python")
    if q.limit:
        df = df.iloc[:int(q.limit)]
    df = df.copy()
    df["privacy"] = label_privacy(df, policy, q.text_col).to_numpy()
    if policy.redact_promote:
        m = df["privacy"] == int(PrivacyLevel.MEDIUM)
        df.loc[m, q.text_col] = df.loc[m, q.text_col].astype(str).map(redact)
        df.loc[m, "privacy"] = int(PrivacyLevel.LOW)
    return df.reset_index(drop=True)


async def score_tier(backend: Backend, reqs: list[Request], shield: PrivacyShield | None,
                     levels: dict[int, int] | None, tier: TierId, chunk: int = 64) -> dict[int, ScoreResult]:
    out = {}
    # keep several waves of requests in flight per chunk: a chunk waits for its slowest
    # request, so chunk == concurrency lets one slow call idle all other slots
    chunk = max(chunk, 8 * int(getattr(backend, "max_concurrency", 0) or 0))
    t0, last = time.perf_counter(), 0
    for i in range(0, len(reqs), chunk):
        part = reqs[i:i + chunk]
        if len(reqs) >= 2000 and i - last >= 2000:      # progress for long offline runs
            last = i
            el = time.perf_counter() - t0
            log.info("%s: %d/%d tuples, %.0f s elapsed, ETA %.1f h", getattr(backend, "name", tier.name),
                     i, len(reqs), el, el / max(i, 1) * (len(reqs) - i) / 3600)
        if shield is not None and levels is not None:
            for r in part:
                shield.check(PrivacyLevel(levels[r.rid]), tier)
        for res in await backend.score(part):
            out[res.rid] = res
    return out


async def run_query(q: Query, tiers: dict[TierId, Backend], policy: PrivacyPolicy, params: ReplayParams,
                    router=None, compile_backend: Backend | None = None, seed: int = 0,
                    use_symbolic: bool = True, unit_cost: dict | None = None,
                    edge_concurrency: int = 2, cloud_concurrency: int = 16,
                    profiles=None, safe_select: bool = True, feat_cache: str | None = None) -> dict:
    """Execute one query live. Mirrors ``replay.replay`` step by step."""
    rng = np.random.default_rng(seed)
    shield = PrivacyShield(policy)
    T = {"start": time.perf_counter()}
    df = prepare(load_table(q.table), q, policy)
    N = len(df)
    lv = df["privacy"].to_numpy().astype(int)
    rid = df["rid"].to_numpy().astype(int)
    levels_map = dict(zip(rid, lv))
    reqs = make_requests(df, q)
    req_by_rid = {r.rid: r for r in reqs}
    levels = [PrivacyLevel(l) for l in np.unique(lv)]
    chains = {l: policy.chain(l) for l in levels}
    spent = {"money": 0.0, "bytes_E": 0, "bytes_C": 0, "n_E": 0, "n_C": 0, "n_D": 0}
    # label-free text features for the learned risk / pre-filter (computed on the device)
    Xt = loglen = None
    if (params.risk or params.prefilter) and params.cert == "split":
        from .risk import assemble, query_features
        Xt, loglen = query_features(df[q.text_col].astype(str).tolist(),
                                    cache=f"{feat_cache}/{q.qid}.npz" if feat_cache else None)

    # ---- A. symbolic candidate rules (compile time: query text + schema only) ----------
    rules, rule_keep = [], None
    if use_symbolic and compile_backend is not None:
        schema = {c: str(t) for c, t in df.dtypes.items() if c not in ("rid", "privacy")}
        rules = await propose_rules(compile_backend, q.predicate, schema, q.text_col)
        if rules:
            rule_keep = np.stack([eval_rule(r, df) for r in rules], 1)   # local, cheap
    T["compile"] = time.perf_counter()

    # ---- B. calibration: device scores + reference labels (+ E scores) of S_cal ----------
    deltas = split_delta(params.delta, levels, [l for l in levels if len(policy.chain(l)) > 1]
                         if params.delta_certified_only else None)
    uc = unit_cost or unit_costs(Profiles(), 300, 1500, 0.05, params.weights)
    s_dev = np.full(N, np.nan)
    pred: dict[int, bool] = {}
    plans, pruned = {}, np.zeros(N, dtype=bool)
    sample_scores: dict = {}                # stratum -> (sample indices, s_dev, s_mid) for policy selection

    skip_all = calibration_cannot_pay_off(levels, chains, uc)

    async def calib(l):
        idx = np.flatnonzero(lv == int(l))
        chain = chains[l]
        if skip_all:                             # no bottleneck reference anywhere: as Static-Ref
            plans[l] = (StratumPlan(level=l, chain=chain, lo_d=0.5 - 1e-12, hi_d=0.5) if len(chain) == 1
                        else StratumPlan(level=l, chain=chain, direct=True))
            return
        n_cal = params.n_cal
        pilot_pos_keep = None
        if params.target_pos():                  # independent pilot -> calibration size
            ploc = rng.integers(0, len(idx), size=min(params.n_pilot, len(idx)))
            pu = np.unique(idx[ploc])
            ptier = chain[-1]
            pout = await score_tier(tiers[ptier], [reqs[i] for i in pu],
                                    shield if ptier != TierId.DEVICE else None, levels_map, ptier)
            if ptier == TierId.DEVICE:
                plab = {k: v.p_yes >= 0.5 for k, v in pout.items()}
                spent["n_D"] += len(pu)
            else:
                _account(spent, ptier, [reqs[i] for i in pu], pout)
                plab = {k: v.label for k, v in pout.items()}
            pred.update(plab)
            n_cal = choose_n_cal(np.array([plab[int(rid[i])] for i in idx[ploc]]), params.n_cal,
                                 params.target_pos(), params.n_cal_max)
            if rule_keep is not None:
                pp = [i for i in idx[ploc] if plab[int(rid[i])]]
                pilot_pos_keep = rule_keep[pp] if pp else None
        use_pf = params.prefilter and params.cert == "split" and reference_is_bottleneck(chain, uc)
        if len(chain) > 1 and not use_pf and not cascade_can_pay_off(
                chain, uc, rule_keep[idx] if rule_keep is not None else None, params.max_rule_subset,
                min(n_cal, 10 * len(idx)), len(idx), pilot_pos_keep, params.gamma_r):
            plans[l] = StratumPlan(level=l, chain=chain, direct=True)
            return
        loc = rng.integers(0, len(idx), size=min(n_cal, 10 * len(idx)))
        samp = idx[loc]
        uniq = np.unique(samp)
        rs = [reqs[i] for i in uniq]
        jobs = [score_tier(tiers[TierId.DEVICE], rs, None, None, TierId.DEVICE)]
        if chain[-1] != TierId.DEVICE:
            jobs.append(score_tier(tiers[chain[-1]], rs, shield, levels_map, chain[-1]))
        if len(chain) == 3:
            jobs.append(score_tier(tiers[chain[1]], rs, shield, levels_map, chain[1]))
        outs = await asyncio.gather(*jobs)
        spent["n_D"] += len(rs)
        dmap = {k: v.p_yes for k, v in outs[0].items()}
        for i in uniq:
            s_dev[i] = dmap[int(rid[i])]
        if chain[-1] == TierId.DEVICE:
            lab = {k: v >= 0.5 for k, v in dmap.items()}
        else:
            _account(spent, chain[-1], rs, outs[1])
            lab = {k: v.label for k, v in outs[1].items()}
        s_mid = None
        if len(chain) == 3:
            _account(spent, chain[1], rs, outs[2])
            mmap = {k: v.p_yes for k, v in outs[2].items()}
            s_mid = np.array([mmap[int(rid[i])] for i in samp])
        pred.update(lab)
        feat_s = feat_p = None
        if Xt is not None and len(chain) >= 2:
            # s_E is known on the sample; for the population it is a placeholder: only the
            # pre-filter (which drops s_E) uses population features, the risk is scored lazily
            feat_s = assemble(Xt, loglen, s_mid if s_mid is not None else np.full(len(samp), 0.5), samp)
            feat_p = assemble(Xt, loglen, np.full(len(idx), 0.5), idx)
        inp = StratumInput(level=l, chain=chain, N=len(idx), sample_idx=loc,
                           y_sample=np.array([lab[int(rid[i])] for i in samp]),
                           s_dev_sample=s_dev[samp], s_mid_sample=s_mid,
                           rule_keep_sample=rule_keep[samp] if rule_keep is not None else None,
                           rule_keep_pop=rule_keep[idx] if rule_keep is not None else None,
                           feat_sample=feat_s, feat_pop=feat_p)
        hyb = getattr(params, "hybrid", 0.0) > 0 and len(chain) > 1
        plan = calibrate_stratum(inp, params.gamma_r, params.gamma_p,
                                 deltas[l] * (1 - params.hybrid) if hyb else deltas[l], uc,
                                 max_rule_subset=params.max_rule_subset, cert=params.cert, split=params.split,
                                 bonferroni=params.bonferroni, prefilter=use_pf, test=params.test, eta=params.eta)
        if hyb:   # single-threshold family on the whole sample at hybrid * delta (Sec. IV)
            prox = s_mid if chain[-2] == TierId.EDGE else s_dev[samp]
            tp = threshold_certify(l, chain, len(idx), inp.y_sample, prox, params.gamma_r,
                                   deltas[l] * params.hybrid, uc)
            cur = plan.est_cost if (plan.feasible or plan.direct) else float(len(idx) * uc[chain[-1]])
            if tp is not None and tp.est_cost < min(cur, len(idx) * uc[chain[-1]]):
                plan = tp
        plans[l] = plan
        risk_s = plan.risk_model(feat_s) if plan.risk_model is not None else None
        sample_scores[l] = (samp, s_dev[samp].copy(), s_mid, risk_s)
        pm = plan.pruned_mask(rule_keep[idx] if rule_keep is not None else None)
        if pm is not None:
            pruned[idx] = pm

    for l in levels:          # strata share the device, so calibrate them one after another
        await calib(l)
    T["calib"] = time.perf_counter()

    # ---- C. streaming execution with the router ---------------------------------------
    items = []
    for l in levels:
        plan, chain = plans[l], chains[l]
        direct = (plan.direct or not plan.feasible) and len(chain) > 1
        skipdev = plan.skip_device and len(chain) > 1 and not direct
        for i in np.flatnonzero(lv == int(l)):
            r = int(rid[i])
            if r in pred or pruned[i]:
                pred.setdefault(r, False)
                continue
            items.append({"rid": r, "i": int(i), "level": l, "plan": plan, "chain": chain, "direct": direct,
                          "skipdev": skipdev})
    # safe policy selection (same procedure as replay.replay): simulate the learned router and
    # the fixed plan on a bootstrap of the labelled sample and keep the cheaper policy
    chosen = router or FixedRouter()
    policy_name = "fixed" if router is None else "router"
    if router is not None and safe_select and profiles is not None and items:
        from .replay import plan_outcome, select_policy
        from .simulator import Item
        preview = []
        for l in levels:
            n_l = sum(1 for it in items if it["level"] == l)
            if not n_l or l not in sample_scores:
                continue
            samp, sd, sm, rk = sample_scores[l]
            keep = [k for k in range(len(samp)) if not pruned[samp[k]]]
            if not keep:
                continue
            for k in rng.choice(keep, size=n_l, replace=True):
                rq = reqs[samp[k]]
                o, mid, start_remote = plan_outcome(plans[l], chains[l], sd[k], None if sm is None else sm[k],
                                                    None if rk is None else rk[k])
                preview.append(Item(rid=len(preview), level=l, dev_outcome=o, tokens=max(1, rq.nbytes // 4),
                                    nbytes=rq.nbytes, t_dev=profiles.device_per_tuple_s,
                                    up_after_e=(mid == "up"), reference=chains[l][-1], start_remote=start_remote))
        if preview and select_policy(preview, router, profiles, params.weights) == "fixed":
            chosen, policy_name = FixedRouter(), "fixed"
    def risk_of(it, s_e):                  # learned risk of one tuple, once its edge score is known
        m = it["plan"].risk_model
        return None if m is None else float(m(assemble(Xt, loglen, [s_e], [it["i"]]))[0])

    await _stream_live(items, tiers, req_by_rid, shield, chosen, pred, spent,
                       edge_concurrency, cloud_concurrency, risk_of)
    T["end"] = time.perf_counter()

    return {"qid": q.qid, "N": N, "selected": sorted(r for r, v in pred.items() if v),
            "latency_s": T["end"] - T["start"], "t_compile": T["compile"] - T["start"],
            "t_calib": T["calib"] - T["compile"], "t_stream": T["end"] - T["calib"],
            "money": spent["money"], "mb_up": (spent["bytes_E"] + spent["bytes_C"]) / 1e6,
            "n_edge": spent["n_E"], "n_cloud": spent["n_C"], "n_device": spent["n_D"],
            "rules": [rules[j] if j < len(rules) else f"learned_prefilter_{j - len(rules)}"
                      for p in plans.values() for j in p.rules], "n_pruned": int(pruned.sum()),
            "risk_used": sum(1 for p in plans.values() if p.risk_model is not None),
            "shield_blocked": shield.blocked_attempts, "policy": policy_name,
            "plans": {l.name: {"rules": list(p.rules), "lo_d": p.lo_d, "hi_d": p.hi_d, "lo_m": p.lo_m,
                               "hi_m": p.hi_m, "rho_rej": p.rho_rej, "rho_acc": p.rho_acc, "feasible": p.feasible, **p.bounds} for l, p in plans.items()}}


def _account(spent, tier, reqs, res):
    if tier == TierId.EDGE:
        spent["n_E"] += len(reqs); spent["bytes_E"] += sum(r.nbytes for r in reqs)
    elif tier == TierId.CLOUD:
        spent["n_C"] += len(reqs); spent["bytes_C"] += sum(r.nbytes for r in reqs)
    spent["money"] += sum(v.money for v in res.values() if not v.cached)


async def _stream_live(items, tiers, req_by_rid, shield, router, pred, spent, c_edge, c_cloud, risk_of=None):
    """Live counterpart of simulator.RoutingSim: same actions, same state features."""
    def first_remote(it):
        plan, chain = it["plan"], it["chain"]
        trivial_mid = not (math.isfinite(plan.lo_m) or math.isfinite(plan.hi_m))
        if not it["direct"] and len(chain) == 3 and not trivial_mid:
            return "E_mid"
        return "E_final" if chain[-1] == TierId.EDGE else "C_final"

    qd = [it for it in items if not (it["direct"] or it["skipdev"])]
    remote = [(it, first_remote(it)) for it in items if it["direct"] or it["skipdev"]]
    qe = [dict(it, kind=k) for it, k in remote if k != "C_final"]
    qc = [it for it, k in remote if k == "C_final"]
    n0 = max(len(items), 1)
    t0 = time.perf_counter()
    st = {"busy_e": 0, "busy_c": 0, "busy_d": 0, "ewma_d": 0.05, "ewma_e": 0.05, "ewma_c": 1.0,
          "action": ACTIONS.index((0.0, 0.0, 32)), "dev_done": False}

    def state():
        n_mid = sum(1 for x in qe if x.get("kind") == "E_mid")
        n_byp = sum(1 for x in qd if x["chain"][-1] != TierId.DEVICE)
        return np.array([len(qd) / n0, len(qe) / n0, len(qc) / n0, n_mid / max(len(qe), 1),
                         n_byp / max(len(qd), 1), st["busy_e"] / c_edge, st["busy_c"] / c_cloud,
                         min(st["ewma_d"], 2) / 2, min(st["ewma_e"], 2) / 2, min(st["ewma_c"], 10) / 10,
                         0.0, min(time.perf_counter() - t0, 1800) / 1800], dtype=np.float32)

    def to_reference(it):
        if it["chain"][-1] == TierId.CLOUD:
            qc.append(it)
        else:
            qe.append(dict(it, kind="E_final"))

    async def device_loop():
        while qd:
            st["action"] = router.act(state())
            beta = ACTIONS[st["action"]][0]
            batch = []
            while qd and len(batch) < DEVICE_BATCH:
                it = qd.pop(0)
                if it["chain"][-1] != TierId.DEVICE and beta > 0 and \
                        (beta >= 1 or ((it["rid"] * 2654435761) % 1000) / 1000.0 < beta):
                    to_reference(it)
                    continue
                batch.append(it)
            if not batch:
                continue
            t1 = time.perf_counter()
            res = await tiers[TierId.DEVICE].score([req_by_rid[x["rid"]] for x in batch])
            st["ewma_d"] = 0.8 * st["ewma_d"] + 0.2 * (time.perf_counter() - t1) / len(batch)
            spent["n_D"] += len(batch)
            rmap = {r.rid: r.p_yes for r in res}
            for it in batch:
                plan, chain = it["plan"], it["chain"]
                d = plan.route_device(rmap[it["rid"]])
                trivial_mid = not (math.isfinite(plan.lo_m) or math.isfinite(plan.hi_m))
                if d in ("acc", "rej"):
                    pred[it["rid"]] = d == "acc"
                elif len(chain) == 3 and not trivial_mid:
                    qe.append(dict(it, kind="E_mid"))
                elif chain[-1] == TierId.EDGE:
                    qe.append(dict(it, kind="E_final"))
                else:
                    qc.append(it)
        st["dev_done"] = True

    async def edge_worker():
        while True:
            if not qe:
                if st["dev_done"] and st["busy_d"] == 0:
                    return
                await asyncio.sleep(0.005)
                continue
            st["action"] = router.act(state())
            _, rho, b = ACTIONS[st["action"]]
            batch = qe[:b]
            del qe[:b]
            to_e = []
            for x in batch:
                if x["kind"] == "E_mid" and x["level"] == PrivacyLevel.LOW and rho > 0 and \
                        (rho >= 1 or ((x["rid"] * 2654435761) % 1000) / 1000.0 < rho):
                    qc.append(x)
                else:
                    to_e.append(x)
            if not to_e:
                continue
            for x in to_e:
                shield.check(PrivacyLevel(x["level"]), TierId.EDGE)
            rs = [req_by_rid[x["rid"]] for x in to_e]
            st["busy_e"] += 1
            t1 = time.perf_counter()
            res = await tiers[TierId.EDGE].score(rs)
            st["busy_e"] -= 1
            st["ewma_e"] = 0.8 * st["ewma_e"] + 0.2 * (time.perf_counter() - t1) / len(rs)
            _account(spent, TierId.EDGE, rs, {r.rid: r for r in res})
            rmap = {r.rid: r for r in res}
            for x in to_e:
                r = rmap[x["rid"]]
                if x["kind"] == "E_final":
                    pred[x["rid"]] = r.label
                else:
                    d = x["plan"].route_mid(r.p_yes, risk_of(x, r.p_yes) if risk_of else None)
                    if d == "up":
                        qc.append(x)
                    else:
                        pred[x["rid"]] = d == "acc"

    async def cloud_worker():
        while True:
            if not qc:
                if st["dev_done"] and not qe and st["busy_e"] == 0:
                    return
                await asyncio.sleep(0.005)
                continue
            x = qc.pop(0)
            shield.check(PrivacyLevel(x["level"]), TierId.CLOUD)
            st["busy_c"] += 1
            t1 = time.perf_counter()
            res = (await tiers[TierId.CLOUD].score([req_by_rid[x["rid"]]]))[0]
            st["busy_c"] -= 1
            st["ewma_c"] = 0.8 * st["ewma_c"] + 0.2 * (time.perf_counter() - t1)
            _account(spent, TierId.CLOUD, [req_by_rid[x["rid"]]], {x["rid"]: res})
            pred[x["rid"]] = res.label

    await asyncio.gather(device_loop(), *[edge_worker() for _ in range(c_edge)],
                         *[cloud_worker() for _ in range(c_cloud)])


async def collect_trace(q: Query, tiers: dict[TierId, Backend], policy: PrivacyPolicy,
                        rules: list[dict], cloud_for_all: bool = True,
                        only_tiers: list[TierId] | None = None) -> pd.DataFrame | None:
    """Score every candidate tuple on every tier (offline evaluation oracle).

    Only use on public data with synthetic privacy labels. ``rules`` are the candidate
    symbolic rules proposed for the query; their keep-masks are stored as columns.
    """
    df = prepare(load_table(q.table), q, policy)
    reqs = make_requests(df, q)
    res = {}
    for tid in (TierId.DEVICE, TierId.EDGE, TierId.CLOUD):
        if tid == TierId.CLOUD and not cloud_for_all:
            continue
        if only_tiers is not None and tid not in only_tiers:
            continue
        t0 = time.perf_counter()
        res[tid] = await score_tier(tiers[tid], reqs, None, None, tid)
        log.info("tier %s scored %d tuples in %.1fs", tid.name, len(reqs), time.perf_counter() - t0)
    if only_tiers is not None:          # prefetch mode: answers are now in the cache
        return None
    rows = []
    for r, (_, row) in zip(reqs, df.iterrows()):
        d, e, c = res[TierId.DEVICE][r.rid], res[TierId.EDGE][r.rid], res[TierId.CLOUD][r.rid]
        rows.append({"rid": r.rid, "privacy": int(row["privacy"]), "tokens": d.prompt_tokens or e.prompt_tokens,
                     "nbytes": r.nbytes, "s_D": d.p_yes, "y_D": d.p_yes >= 0.5, "t_D": d.latency_s,
                     "s_E": e.p_yes, "y_E": e.label, "t_E": e.latency_s, "y_C": c.label, "t_C": c.latency_s,
                     "money_C": c.money})
    tr = pd.DataFrame(rows)
    for i, rule in enumerate(rules):
        tr[f"rule_{i}"] = eval_rule(rule, df)
    return tr
