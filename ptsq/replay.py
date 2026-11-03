"""Trace replay: evaluate PTSQ and baselines offline on a collected trace.

A *trace* (see ``scripts/collect_trace.py``) holds, for every tuple of a query's candidate
set, the device / edge scores, the cloud label, token counts, the measured device latency
and the keep-masks of all candidate symbolic rules. Collecting cloud labels for all tuples
is an *offline evaluation oracle only*: datasets are public with synthetic privacy labels.
The online system never sends a tuple to a tier its policy forbids (enforced by the
shield); the cloud-all baseline is reported as privacy-violating.

Replay lets us repeat every query hundreds of times with fresh calibration samples, which is
what validates the probabilistic guarantee empirically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cascade import StratumInput, StratumPlan, calibrate_stratum, threshold_certify, split_delta, choose_n_cal, cascade_can_pay_off, reference_is_bottleneck
from .privacy import PrivacyPolicy, PrivacyShield
from .simulator import CostWeights, FixedRouter, Item, RoutingSim, TierProfile
from .types import PrivacyLevel, TierId


@dataclass
class MethodConfig:
    name: str
    kind: str = "cascade"            # cascade | cloud_all | device_only | static_ref | supg | bargain
    use_symbolic: bool = True
    use_edge: bool = True
    thresholds: str = "certified"    # certified | fixed  (fixed = confidence cascade, no guarantee)
    fixed_tau: float = 0.9
    router: object = None            # object with .act(state) -> action; None = FixedRouter()
    safe_select: bool = True         # simulate router vs. fixed policy on the labelled sample first
    dev_col: str = "s_D"             # which device-score column to use (federated rounds, model size)
    tdev_col: str | None = None      # matching device-latency column (default: t_D)
    beta: float = 0.0                # bargain: minimum positive density (0 = exact guarantee; > 0 relaxed)


@dataclass
class ReplayParams:
    gamma_r: float = 0.9
    gamma_p: float | None = 0.9
    delta: float = 0.05
    n_cal: int = 1500                # minimum calibration size per stratum
    max_rule_subset: int = 2
    min_pos: int | None = None       # adaptive n: target #reference positives (None = fixed n_cal)
    n_pilot: int = 300               # pilot draws used only to choose n
    n_cal_max: int = 8000
    cert: str = "union"              # "union" (union bound over the grid) | "split" (propose + fixed-sequence test)
    split: float = 0.5               # fraction of the calibration draws used to propose (cert="split")
    bonferroni: int = 0              # cert="split": 0 = fixed-sequence testing; L = Bonferroni over L frontier points
    test: str = "ratio"              # cert="split": "ratio" (population-fraction bounds) | "cond" (recall/precision tested directly)
    eta: int = 0                     # cert="split": fixed-sequence tolerance (continue past eta failed tests; level delta/(eta+1))
    delta_certified_only: bool = False   # split delta only among strata that need certification (chain length > 1)
    prefilter: bool = False          # learned text pre-filter as candidate pruning rules (cert="split", features)
    risk: bool = False               # learned disagreement risk for the middle tier (needs cert="split" and features)
    miss_budget: float | None = None  # adaptive n: target (1-gamma_r) * #positives in S (overrides min_pos)
    hybrid: float = 0.0              # > 0: also certify the single-threshold family on the whole sample at this share of delta_s
    progressive: bool = False        # grow the sample in rounds n_cal, 2 n_cal, ... n_cal_max (delta split over rounds)
    weights: CostWeights = field(default_factory=CostWeights)

    def target_pos(self) -> int | None:
        """Reference positives the calibration sample should contain. With a miss budget B the
        sample is sized so that about B sampled misses are allowed at the recall target, i.e.
        k+ = B / (1 - gamma_r): stricter targets need proportionally more positives."""
        if self.miss_budget:
            return int(np.ceil(self.miss_budget / max(1.0 - self.gamma_r, 1e-3)))
        return self.min_pos


@dataclass
class Profiles:
    device_per_tuple_s: float = 0.05     # used when the trace lacks measured t_D
    edge: TierProfile = field(default_factory=TierProfile)
    cloud: TierProfile = field(default_factory=lambda: TierProfile(concurrency=16, batchable=False))


def bulk_time(p: TierProfile, n: int, tok: float, nbytes: float, batch: int = 128) -> float:
    """Makespan of n independent requests on a tier (used for calibration / baselines)."""
    if n <= 0:
        return 0.0
    if p.batchable:
        nb = math.ceil(n / batch)
        per = p.net_s(int(nbytes * batch)) + p.service_s(int(tok * batch), batch)
        return math.ceil(nb / max(p.concurrency, 1)) * per
    per = p.net_s(int(nbytes)) + p.service_s(int(tok))
    return max(math.ceil(n / max(p.concurrency, 1)) * per, n * p.rate_s(tok + 1) + per)


def unit_costs(prof: Profiles, tok: float, nbytes: float, t_dev: float, w: CostWeights) -> dict[TierId, float]:
    """Marginal cost of processing one tuple at each tier (throughput-normalised latency)."""
    e, c = prof.edge, prof.cloud
    lat_e = (e.net_s(int(nbytes * 128)) + e.service_s(int(tok * 128), 128)) / 128 / max(e.concurrency, 1)
    lat_c = max((c.net_s(int(nbytes)) + c.service_s(int(tok))) / max(c.concurrency, 1), c.rate_s(tok + 1))
    return {TierId.DEVICE: w.latency * t_dev,
            TierId.EDGE: w.latency * lat_e + w.bytes * nbytes / 1e6,
            TierId.CLOUD: w.latency * lat_c + w.money * c.money(int(tok), 1) + w.bytes * nbytes / 1e6}


def chain_for(policy: PrivacyPolicy, lv: PrivacyLevel, use_edge: bool) -> list[TierId]:
    ch = policy.chain(lv)
    if not use_edge:
        ch = [t for t in ch if t != TierId.EDGE]
    return ch


def reference_labels(lv_arr: np.ndarray, y: dict, policy: PrivacyPolicy, use_edge: bool = True) -> np.ndarray:
    ref = np.zeros(len(lv_arr), dtype=bool)
    for lv in PrivacyLevel:
        m = lv_arr == int(lv)
        ref[m] = y[chain_for(policy, lv, use_edge)[-1]][m]
    return ref


def prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    tp = float((pred & truth).sum())
    rec = tp / truth.sum() if truth.sum() else 1.0
    prec = tp / pred.sum() if pred.sum() else 1.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return rec, prec, f1


def plan_outcome(plan: StratumPlan, chain, s_dev: float, s_mid: float | None, risk: float | None = None):
    """Outcome of one tuple under a stratum plan: (outcome, mid-tier decision, starts remote)."""
    trivial_mid = not (np.isfinite(plan.lo_m) or np.isfinite(plan.hi_m))
    direct = (plan.direct or not plan.feasible) and len(chain) > 1
    remote_start = direct or (plan.skip_device and len(chain) > 1)
    d = "up" if remote_start else plan.route_device(s_dev)
    if direct:
        o = "E_final" if chain[-1] == TierId.EDGE else "C_final"
    elif d in ("acc", "rej"):
        o = d
    elif len(chain) == 3 and not trivial_mid:
        o = "E_mid"
    elif chain[-1] == TierId.EDGE:
        o = "E_final"
    else:
        o = "C_final"
    mid = plan.route_mid(s_mid, risk) if o == "E_mid" else None
    return o, mid, remote_start


def calibration_cannot_pay_off(levels, chains, uc) -> bool:
    """Decision made before any label is bought. If in no stratum the permitted reference costs
    more per tuple than a cheaper permitted tier (e.g. the cloud has no rate limit), a cascade can
    save only through symbolic pruning, while its calibration delays the whole query and adds
    device and edge work; every stratum then goes straight to its reference (as Static-Ref).
    This decision affects cost only, never the guarantee."""
    certifiable = [lv for lv in levels if len(chains[lv]) > 1]
    return bool(certifiable) and not any(reference_is_bottleneck(chains[lv], uc) for lv in certifiable)


def select_policy(preview: list[Item], router, prof: "Profiles", w: CostWeights) -> str:
    """Safe policy selection: 'fixed' unless the router is cheaper in simulation on ``preview``."""
    def sim_cost(r):
        rs = RoutingSim([Item(**{**it.__dict__}) for it in preview], prof.edge, prof.cloud,
                        shield=None, weights=w).run(r)
        return w.latency * rs.makespan_s + w.money * rs.money + w.bytes * sum(rs.bytes_up.values()) / 1e6
    return "fixed" if sim_cost(FixedRouter()) <= sim_cost(router) else "router"


def replay(tr: pd.DataFrame, rules: list[dict], method: MethodConfig, params: ReplayParams,
           prof: Profiles, policy: PrivacyPolicy | None = None, seed: int = 0,
           return_items: bool = False) -> dict:
    """Replay one query under one method. Returns a metrics dict."""
    policy = policy or PrivacyPolicy()
    rng = np.random.default_rng(seed)
    N = len(tr)
    lv_arr = tr["privacy"].to_numpy().astype(int)
    tok = tr["tokens"].to_numpy(dtype=float)
    nby = tr["nbytes"].to_numpy(dtype=float)
    s_dev = tr[method.dev_col].to_numpy(dtype=float)
    s_mid = tr["s_E"].to_numpy(dtype=float)
    y_D = s_dev >= 0.5
    y_E = tr["y_E"].to_numpy(dtype=bool)
    y_C = tr["y_C"].to_numpy(dtype=bool)
    ylab = {TierId.DEVICE: y_D, TierId.EDGE: y_E, TierId.CLOUD: y_C}
    tcol = method.tdev_col or "t_D"
    t_dev = tr[tcol].to_numpy(dtype=float) if tcol in tr else np.full(N, prof.device_per_tuple_s)
    ref_full = reference_labels(lv_arr, ylab, policy, True)       # full system's permitted reference
    tok_avg, nb_avg, tdev_avg = float(tok.mean()), float(nby.mean()), float(t_dev.mean())
    w = params.weights
    shield = PrivacyShield(policy)
    out = {"method": method.name, "seed": seed, "N": N}

    def finish(pred, t_total, money, bytes_e, bytes_c, n_e, n_c, n_dev, violations, extra=None):
        r_ref, p_ref, f_ref = prf(pred, ref_full)
        r_abs, p_abs, f_abs = prf(pred, y_C)
        out.update(dict(latency_s=t_total, money=money, mb_up=(bytes_e + bytes_c) / 1e6,
                        mb_up_cloud=bytes_c / 1e6, n_edge=n_e, n_cloud=n_c, n_device=n_dev,
                        recall=r_ref, precision=p_ref, f1=f_ref, recall_abs=r_abs,
                        precision_abs=p_abs, f1_abs=f_abs, privacy_violations=violations,
                        cost=w.latency * t_total + w.money * money + w.bytes * (bytes_e + bytes_c) / 1e6))
        if extra:
            out.update(extra)
        return out

    # ------------------------------------------------------------------ simple baselines
    if method.kind == "cloud_all":
        viol = int((lv_arr != int(PrivacyLevel.LOW)).sum())
        t = bulk_time(prof.cloud, N, tok_avg, nb_avg)
        return finish(y_C.copy(), t, prof.cloud.money(int(tok.sum()), N), 0, nby.sum(), 0, N, 0, viol)
    if method.kind == "device_only":
        return finish(y_D.copy(), float(t_dev.sum()), 0.0, 0, 0, 0, 0, N, 0)
    if method.kind == "static_ref":
        refs = np.array([int(policy.reference(PrivacyLevel(l))) for l in lv_arr])
        mD, mE, mC = refs == 0, refs == 1, refs == 2
        t = max(float(t_dev[mD].sum()), bulk_time(prof.edge, int(mE.sum()), tok_avg, nb_avg),
                bulk_time(prof.cloud, int(mC.sum()), tok_avg, nb_avg))   # tiers run in parallel
        return finish(ref_full.copy(), t, prof.cloud.money(int(tok[mC].sum()), int(mC.sum())), nby[mE].sum(),
                      nby[mC].sum(), int(mE.sum()), int(mC.sum()), int(mD.sum()), 0)

    if method.kind in ("supg", "bargain"):
        return _supg_jt(tr, lv_arr, ylab, s_dev, s_mid, t_dev, tok, nby, tok_avg, nb_avg, policy, params, prof,
                        rng, ref_full, finish, selector=method.kind, beta=method.beta)

    # ------------------------------------------------------------------ cascade methods
    levels = [PrivacyLevel(l) for l in sorted(set(lv_arr.tolist()))]
    chains = {lv: chain_for(policy, lv, method.use_edge) for lv in levels}
    ref_lab = reference_labels(lv_arr, ylab, policy, method.use_edge)
    rule_keep = (tr[[f"rule_{i}" for i in range(len(rules))]].to_numpy(dtype=bool)
                 if (method.use_symbolic and rules) else None)

    Xr = None                              # per-tuple features for the learned risk (rows aligned with tr)
    if (params.risk or params.prefilter) and tr.attrs.get("X") is not None:
        xr = tr.attrs["X_rid"]
        if len(xr) == len(tr) and np.array_equal(xr, tr["rid"].to_numpy()):
            Xr = tr.attrs["X"]                        # same row order: no copy
        else:
            pos = pd.Series(np.arange(len(xr)), index=xr)
            Xr = tr.attrs["X"][pos.reindex(tr["rid"].to_numpy()).to_numpy()]
    risk_all = np.full(N, np.nan)
    labelled = np.zeros(N, dtype=bool)     # tuples whose reference label we paid for
    dev_scored = np.zeros(N, dtype=bool)   # tuples scored on the device during calibration
    deltas = split_delta(params.delta, levels, [lv for lv in levels if len(chains[lv]) > 1]
                         if params.delta_certified_only else None)
    uc = unit_costs(prof, tok_avg, nb_avg, tdev_avg, w)
    plans: dict[PrivacyLevel, StratumPlan] = {}
    pruned = np.zeros(N, dtype=bool)

    def progressive_calibration(idx, chain, lv):
        """Grow the calibration sample in rounds n_cal * 2^r (up to n_cal_max). Each round is
        certified at delta_s / #rounds, so every round's plan is valid simultaneously and we may
        stop after any round. We stop when the best certified plan is within the cost of the
        next round's labels of the best plan the sample suggests is achievable at all."""
        sched = []
        n_r = params.n_cal
        while n_r < params.n_cal_max:
            sched.append(n_r)
            n_r *= 2
        sched.append(params.n_cal_max)
        sched = [min(x, 10 * len(idx)) for x in sched]
        sched = sorted(set(sched))
        loc_all = rng.integers(0, len(idx), size=sched[-1])
        d_r = deltas[lv] / len(sched)
        u_lab = uc[chain[-1]] + (uc[chain[1]] if len(chain) == 3 else 0.0)   # labels + middle scores
        best, used = None, sched[0]
        for r, n_r in enumerate(sched):
            loc = loc_all[:n_r]
            samp = idx[loc]
            inp = StratumInput(level=lv, chain=chain, N=len(idx), sample_idx=loc, y_sample=ref_lab[samp],
                               s_dev_sample=s_dev[samp], s_mid_sample=s_mid[samp] if len(chain) == 3 else None,
                               rule_keep_sample=rule_keep[samp] if rule_keep is not None else None,
                               rule_keep_pop=rule_keep[idx] if rule_keep is not None else None)
            p_r = calibrate_stratum(inp, params.gamma_r, params.gamma_p, d_r, uc,
                                    max_rule_subset=params.max_rule_subset, cert=params.cert, split=params.split,
                                 bonferroni=params.bonferroni, prefilter=params.prefilter, test=params.test, eta=params.eta)
            used = n_r
            if best is None or p_r.est_cost < best.est_cost:
                best = p_r
            if r + 1 == len(sched):
                break
            opt = p_r.bounds.get("opt_cost", 0.0)
            if best.est_cost - min(opt, best.est_cost) <= (sched[r + 1] - n_r) * u_lab:
                break                                     # more labels cannot pay for themselves
        return best, loc_all[:used]

    # 0) can calibration pay off at all? (cost only; the guarantee is unaffected) ---------------
    skip_all = method.thresholds != "fixed" and calibration_cannot_pay_off(levels, chains, uc)

    # 1) calibration per stratum ------------------------------------------------------------
    for lv in levels:
        idx = np.flatnonzero(lv_arr == int(lv))
        chain = chains[lv]
        if skip_all and len(chain) > 1:
            plans[lv] = StratumPlan(level=lv, chain=chain, direct=True)
            continue
        if skip_all and len(chain) == 1:
            # the device is the stratum's reference: its answers are exact, nothing to calibrate
            plans[lv] = StratumPlan(level=lv, chain=chain, lo_d=0.5 - 1e-12, hi_d=0.5)
            continue
        if method.thresholds == "fixed":
            tau = method.fixed_tau
            plan = StratumPlan(level=lv, chain=chain, lo_d=1 - tau, hi_d=tau, lo_m=1 - tau, hi_m=tau)
            if len(chain) == 1:
                plan.lo_d, plan.hi_d = 0.5 - 1e-12, 0.5
            plans[lv] = plan
            continue
        if params.progressive and len(chain) > 1:
            plan, used = progressive_calibration(idx, chain, lv)
            plans[lv] = plan
            labelled[idx[used]] = True
            dev_scored[idx[used]] = True
            pm = plan.pruned_mask(rule_keep[idx] if rule_keep is not None else None)
            if pm is not None:
                pruned[idx] = pm
            continue
        n_cal = params.n_cal
        pilot_pos_keep = None
        if params.target_pos():
            ploc = rng.integers(0, len(idx), size=min(params.n_pilot, len(idx)))
            labelled[idx[ploc]] = True                  # pilot labels are paid for (and exact)
            n_cal = choose_n_cal(ref_lab[idx[ploc]], params.n_cal, params.target_pos(), params.n_cal_max)
            if rule_keep is not None:
                pp = idx[ploc][ref_lab[idx[ploc]]]
                pilot_pos_keep = rule_keep[pp]
        use_pf = params.prefilter and params.cert == "split" and reference_is_bottleneck(chain, uc)
        if len(chain) > 1 and not use_pf and not cascade_can_pay_off(
                chain, uc, rule_keep[idx] if rule_keep is not None else None, params.max_rule_subset,
                min(n_cal, 10 * len(idx)), len(idx), pilot_pos_keep, params.gamma_r):
            plans[lv] = StratumPlan(level=lv, chain=chain, direct=True)
            continue
        n_cal = min(n_cal, 10 * len(idx))
        loc = rng.integers(0, len(idx), size=n_cal)
        samp = idx[loc]
        labelled[samp] = True
        dev_scored[samp] = True
        use_x = Xr is not None and len(chain) >= 2
        inp = StratumInput(level=lv, chain=chain, N=len(idx), sample_idx=loc, y_sample=ref_lab[samp],
                           s_dev_sample=s_dev[samp], s_mid_sample=s_mid[samp] if len(chain) == 3 else None,
                           rule_keep_sample=rule_keep[samp] if rule_keep is not None else None,
                           rule_keep_pop=rule_keep[idx] if rule_keep is not None else None,
                           feat_sample=Xr[samp] if use_x else None, feat_pop=Xr[idx] if use_x else None)
        hyb = params.hybrid > 0 and len(chain) > 1
        d_c = deltas[lv] * (1 - params.hybrid) if hyb else deltas[lv]
        plan = calibrate_stratum(inp, params.gamma_r, params.gamma_p, d_c, uc,
                                 max_rule_subset=params.max_rule_subset, cert=params.cert, split=params.split,
                                 bonferroni=params.bonferroni, prefilter=use_pf, test=params.test, eta=params.eta)
        if hyb:
            # second certified family on the same sample at w * delta_s (the cascade at (1 - w) delta_s):
            # both certificates hold simultaneously w.p. >= 1 - delta_s, so the cheaper plan keeps Theorem 1
            prox = s_mid[samp] if chain[-2] == TierId.EDGE else s_dev[samp]
            tp = threshold_certify(lv, chain, len(idx), ref_lab[samp], prox, params.gamma_r, deltas[lv] * params.hybrid, uc)
            cur = plan.est_cost if (plan.feasible or plan.direct) else float(len(idx) * uc[chain[-1]])
            if tp is not None and tp.est_cost < min(cur, len(idx) * uc[chain[-1]]):
                plan = tp
        plans[lv] = plan
        if plan.risk_pop is not None:
            risk_all[idx] = plan.risk_pop
        pm = plan.pruned_mask(rule_keep[idx] if rule_keep is not None else None)
        if pm is not None:
            pruned[idx] = pm

    # cost of labelling S_cal with each stratum's reference (+ the middle tier)
    calib_money = calib_bytes_e = calib_bytes_c = 0.0
    calib_n_e = calib_n_c = 0
    lab_idx = np.flatnonzero(labelled)
    for lv in levels:
        li = lab_idx[lv_arr[lab_idx] == int(lv)]
        chain = chains[lv]
        for tier in chain[1:]:                 # calibration labelling also passes the shield
            for _ in range(len(li)):
                shield.check(lv, tier)
        if chain[-1] == TierId.CLOUD:
            calib_n_c += len(li); calib_bytes_c += nby[li].sum()
            calib_money += prof.cloud.money(int(tok[li].sum()), len(li))
        if TierId.EDGE in chain:
            calib_n_e += len(li); calib_bytes_e += nby[li].sum()
    # device scoring of S_cal runs in parallel with reference labelling
    t_calib = max(float(t_dev[dev_scored].sum()), bulk_time(prof.cloud, calib_n_c, tok_avg, nb_avg),
                  bulk_time(prof.edge, calib_n_e, tok_avg, nb_avg))

    # 2) streaming execution of all remaining tuples -------------------------------------------
    pred = np.zeros(N, dtype=bool)
    items: list[Item] = []
    outcome = {}
    def route(i, lv):
        """Plan outcome of tuple i of stratum lv (uses only scores the plan would compute)."""
        chain = chains[lv]
        o, mid, remote_start = plan_outcome(plans[lv], chain, s_dev[i], s_mid[i],
                                            risk_all[i] if plans[lv].risk_pop is not None else None)
        it = Item(rid=int(i), level=lv, dev_outcome=o, tokens=int(tok[i]), nbytes=int(nby[i]),
                  t_dev=float(t_dev[i]), up_after_e=(mid == "up"), reference=chain[-1],
                  start_remote=remote_start)
        return o, mid, it

    for lv in levels:
        for i in np.flatnonzero(lv_arr == int(lv)):
            if labelled[i]:
                pred[i] = ref_lab[i]
                continue
            if pruned[i]:
                continue                                   # rejected symbolically, free
            o, mid, it = route(i, lv)
            outcome[i] = (o, mid)
            items.append(it)
    if return_items:          # item pool for RL training (scripts/train_router.py)
        out["_items"] = [Item(**{**it.__dict__}) for it in items]

    # Safe policy selection: before streaming, simulate the learned router and the fixed policy
    # on a bootstrap of the LABELLED sample (whose plan outcomes are known after calibration,
    # so no information about the streaming tuples is used) and keep the cheaper one. The
    # guarantee is unaffected (Theorem 3 holds for any router).
    router = method.router or FixedRouter()
    chose = "fixed" if method.router is None else "router"
    if method.router is not None and method.safe_select and items:
        n_by_lv = {lv: sum(1 for it in items if it.level == lv) for lv in levels}
        preview = []
        for lv in levels:
            pool = [i for i in np.flatnonzero(labelled & (lv_arr == int(lv))) if not pruned[i]]
            if not pool or not n_by_lv[lv]:
                continue
            for k, i in enumerate(rng.choice(pool, size=n_by_lv[lv], replace=True)):
                it = route(int(i), lv)[2]
                it.rid = len(preview)
                preview.append(it)

        if preview and select_policy(preview, method.router, prof, w) == "fixed":
            router, chose = FixedRouter(), "fixed"
    sim = RoutingSim(items, prof.edge, prof.cloud, shield=shield, weights=w)
    res = sim.run(router)
    for i, (o, mid) in outcome.items():
        by = res.decided_by.get(i)
        if by == TierId.CLOUD:
            pred[i] = y_C[i]
        elif by == TierId.EDGE and o == "E_mid" and mid != "up":
            pred[i] = mid == "acc"
        elif by == TierId.EDGE:
            pred[i] = y_E[i]                               # E is the reference (E_final / bypass)
        else:
            pred[i] = o == "acc"

    t_total = t_calib + res.makespan_s
    money = calib_money + res.money
    extra = {"t_calib": t_calib, "t_stream": res.makespan_s, "n_pruned": int((pruned & ~labelled).sum()),
             "rules": sum(len(p.rules) for p in plans.values()), "n_labelled": int(labelled.sum()),
             "bypassed": res.bypassed, "skipped": res.skipped,
             "infeasible_strata": sum(1 for p in plans.values() if not p.feasible),
             "policy": chose,
             "risk_used": sum(1 for p in plans.values() if p.risk_pop is not None)}
    # per-stratum recall/precision w.r.t. the method's own permitted reference (Theorem 1)
    for lv in levels:
        m = lv_arr == int(lv)
        r_s, p_s, _ = prf(pred[m], ref_lab[m])
        extra[f"recall_{lv.name}"] = r_s
        extra[f"rules_{lv.name}"] = ";".join(map(str, plans[lv].rules))
        extra[f"precision_{lv.name}"] = p_s
    return finish(pred, t_total, money, calib_bytes_e + res.bytes_up[TierId.EDGE],
                  calib_bytes_c + res.bytes_up[TierId.CLOUD], calib_n_e + res.n_tier[TierId.EDGE],
                  calib_n_c + res.n_tier[TierId.CLOUD], int(dev_scored.sum() + res.n_tier[TierId.DEVICE]),
                  0, extra)



def _supg_jt(tr, lv_arr, ylab, s_dev, s_mid, t_dev, tok, nby, tok_avg, nb_avg, policy, params, prof, rng,
             ref_full, finish, selector: str = "supg", beta: float = 0.0):
    """SUPG baseline [Kang et al.], applied per privacy stratum (our adaptation).

    Per stratum, the proxy is the strongest non-reference permitted tier (edge for LOW, device for
    MEDIUM) and the oracle is the stratum's reference; HIGH tuples are answered by the device, their
    reference. SUPG's joint-target (JT) query first returns a set with recall >= gamma_r w.p.
    1 - delta_s (importance sampling with weights ~ sqrt(proxy score), defensive mixing 10%, normal
    confidence bounds), then labels every returned tuple with the oracle, so precision is 1. The
    oracle budget is the same pilot-based sample size PTSQ uses. Proxy scoring of the whole stratum
    is charged.

    selector="bargain" replaces SUPG's threshold selection by BARGAIN's recall-target procedure
    [Zeighami et al., SIGMOD 2025] (ptsq/bargain.py): betting tests on uniform samples (R-U, beta=0,
    exact) or with the positive-density pre-filter (R-A, beta>0, relaxed guarantee); everything else
    (proxy, oracle, budget, JT verification) is identical."""
    from scipy.stats import norm
    from .cascade import choose_n_cal, split_delta
    levels = [PrivacyLevel(l) for l in sorted(set(lv_arr.tolist()))]
    deltas = split_delta(params.delta, levels, [lv for lv in levels if len(policy.chain(lv)) > 1]
                         if params.delta_certified_only else None)
    pred = np.zeros(len(lv_arr), dtype=bool)
    dev_time, n_e, n_c, b_e, b_c, money = 0.0, 0, 0, 0.0, 0.0, 0.0
    edge_low, cloud_low, n_dev, n_lab = 0, 0, 0, 0
    extra = {}
    for lv in levels:
        idx = np.flatnonzero(lv_arr == int(lv))
        chain = policy.chain(lv)
        ref = chain[-1]
        y = ylab[ref][idx]
        if len(chain) == 1:                          # device is the reference
            pred[idx] = y
            dev_time += float(t_dev[idx].sum()); n_dev += len(idx)
            extra[f"recall_{lv.name}"], extra[f"precision_{lv.name}"], _ = prf(pred[idx], y)
            continue
        if TierId.EDGE in chain[:-1]:
            A = s_mid[idx]; n_e += len(idx); b_e += nby[idx].sum()
            if ref == TierId.CLOUD:
                edge_low += len(idx)
        else:
            A = s_dev[idx]; dev_time += float(t_dev[idx].sum()); n_dev += len(idx)
        A = np.nan_to_num(A, nan=0.5)
        Ns = len(idx)
        # oracle budget: the same independent pilot + adaptive size rule as PTSQ
        pilot = rng.integers(0, Ns, size=min(params.n_pilot, Ns))
        budget = choose_n_cal(y[pilot], params.n_cal, params.target_pos(), params.n_cal_max)
        budget = min(budget, 10 * Ns)
        if selector == "bargain":
            from .bargain import bargain_rt
            tau, S = bargain_rt(A, y, budget, params.gamma_r, deltas[lv], rng, beta=beta)
            O = y[S].astype(float)
        else:
            tau, S, O = _supg_select(A, y, budget, params.gamma_r, deltas[lv], rng)
        ret = A >= tau
        ret[S[O > 0]] = True
        labelled = np.zeros(Ns, dtype=bool)
        labelled[pilot] = True
        labelled[S] = True
        oracle = labelled | ret                        # JT: verify every returned tuple with the oracle
        pred[idx] = ret & y
        n_lab += int(labelled.sum())
        extra[f"recall_{lv.name}"], extra[f"precision_{lv.name}"], _ = prf(pred[idx], y)
        k = int(oracle.sum())
        if ref == TierId.CLOUD:
            n_c += k; b_c += nby[idx][oracle].sum(); cloud_low += k
            money += prof.cloud.money(int(tok[idx][oracle].sum()), k)
        else:
            n_e += k; b_e += nby[idx][oracle].sum()
    t = max(dev_time, bulk_time(prof.edge, n_e, tok_avg, nb_avg),
            bulk_time(prof.edge, edge_low, tok_avg, nb_avg) + bulk_time(prof.cloud, cloud_low, tok_avg, nb_avg))
    extra["n_labelled"] = n_lab
    extra["infeasible_strata"] = 0
    return finish(pred, t, money, b_e, b_c, n_e, n_c, n_dev, 0, extra)


def _supg_select(A, y, budget, gamma_r, delta_s, rng):
    """SUPG's importance-sampled recall-target threshold (normal confidence bounds)."""
    from scipy.stats import norm
    Ns = len(A)
    if True:
        wgt = 0.9 * np.sqrt(np.clip(A, 1e-6, 1)) / np.sqrt(np.clip(A, 1e-6, 1)).sum() + 0.1 / Ns
        wgt = wgt / wgt.sum()
        S = rng.choice(Ns, size=budget, replace=True, p=wgt)
        m = (1.0 / Ns) / wgt[S]
        O = y[S].astype(float)
        z = norm.ppf(1 - delta_s / 2)
        cands = np.unique(A[S][O > 0])[::-1]           # thresholds from sampled positives, high -> low
        tau = -np.inf
        for t in cands:
            above = A[S] >= t
            z1, z2 = O * m * above, O * m * (~above)
            lb1 = z1.mean() - z * z1.std(ddof=1) / np.sqrt(budget)
            ub2 = z2.mean() + z * z2.std(ddof=1) / np.sqrt(budget)
            if lb1 > 0 and lb1 / (lb1 + max(ub2, 0)) >= gamma_r:
                tau = t
                break
    return tau, S, O

__all__ = ["MethodConfig", "ReplayParams", "Profiles", "replay", "plan_outcome", "select_policy", "reference_labels", "unit_costs", "bulk_time"]
