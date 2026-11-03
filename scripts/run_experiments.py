#!/usr/bin/env python
"""Run the paper's experiments by trace replay.

  python scripts/run_experiments.py --suite all --traces results/traces --out results/exp

Suites (paper figure/table in brackets):
  main         end-to-end comparison with baselines               [Table III, Table IV, Fig. 2]
  guarantee    empirical validity of Theorem 1 over many seeds    [Fig. 3]
  ablation     contribution of each component                     [Table V]
  bad_rules    robustness to an injected leaky rule (Prop. 1)     [Table VII]
  sensitivity  gamma, privacy mix, bandwidth, calibration size    [Fig. 4]
  federated    effect of federated LoRA rounds (needs s_D_r* cols)[Fig. 5]
  device_model device model size (needs s_Dm_* cols)              [Fig. 6]
  overhead     optimiser / shield / router overheads              [Table VIII]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ablation_methods, load_traces, main_methods, make_router  # noqa: E402

from ptsq.cascade import StratumInput, calibrate_stratum  # noqa: E402
from ptsq.config import build_params, build_policy, load_config, load_profiles  # noqa: E402
from ptsq.privacy import PrivacyShield  # noqa: E402
from ptsq.replay import MethodConfig, replay  # noqa: E402
from ptsq.simulator import TierProfile  # noqa: E402
from ptsq.types import PrivacyLevel, TierId  # noqa: E402


JOBS = 1
_G = {}


def _one(task):
    qid, mi, s, extra, tag = task
    tr, rules = _G["T"][qid]
    r = replay(tr, rules, _G["M"][mi], _G["P"], _G["prof"], _G["pol"], seed=s)
    r.update(qid=qid, **extra)
    if tag:
        r["setting"] = tag
    return r


def run_grid(traces, methods, params, prof, policy, seeds, tag=None, **extra_cols):
    """Replay every (query, method, seed); parallel over processes when --jobs > 1."""
    _G.update(T=traces, M=methods, P=params, prof=prof, pol=policy)
    tasks = [(q, mi, s, extra_cols, tag) for q in traces for mi in range(len(methods)) for s in seeds]
    if JOBS > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(JOBS) as pool:
            return pool.map(_one, tasks, chunksize=max(1, len(tasks) // (JOBS * 4)))
    return [_one(t) for t in tasks]


def remix_privacy(tr: pd.DataFrame, high: float, medium: float) -> pd.DataFrame:
    tr = tr.copy()
    u = np.array([int(hashlib.md5(str(r).encode()).hexdigest()[:8], 16) / 16**8 for r in tr["rid"]])
    tr["privacy"] = np.where(u < high, 2, np.where(u < high + medium, 1, 0))
    return tr


def scale_bw(prof, f):
    p = copy.deepcopy(prof)
    p.edge.bw_mbps *= f
    p.cloud.bw_mbps *= f
    return p


def suite_main(T, P, prof, pol, router, seeds):
    return pd.DataFrame(run_grid(T, main_methods(router), P, prof, pol, seeds))


def suite_guarantee(T, P, prof, pol, router, n_seeds):
    rows = []
    for g in (0.75, 0.8, 0.9, 0.95):
        p = copy.deepcopy(P)
        p.gamma_r = p.gamma_p = g
        methods = [MethodConfig("PTSQ", router=router), MethodConfig("Conf-Cascade", thresholds="fixed",
                                                                     fixed_tau=g, use_symbolic=False),
                   MethodConfig("SUPG-JT", kind="supg"), MethodConfig("BARGAIN-JT", kind="bargain"),
                   MethodConfig("BARGAIN-JT (relaxed)", kind="bargain", beta=0.01)]
        rows += run_grid(T, methods, p, prof, pol, range(n_seeds), gamma=g)
    return pd.DataFrame(rows)


def suite_baselines(T, P, prof, pol, router, n_seeds):
    """Single-proxy baselines with guarantees, per stratum: SUPG-JT (asymptotic), BARGAIN-JT (exact,
    beta=0) and BARGAIN-JT with the relaxed positive-density guarantee (beta=0.01), at four targets
    and n_seeds seeds (cheap: no cascade calibration). Seeds 0-9 at gamma=0.8 match the main suite."""
    rows = []
    methods = [MethodConfig("SUPG-JT", kind="supg"), MethodConfig("BARGAIN-JT", kind="bargain"),
               MethodConfig("BARGAIN-JT (relaxed)", kind="bargain", beta=0.01)]
    for g in (0.75, 0.8, 0.9, 0.95):
        p = copy.deepcopy(P)
        p.gamma_r = p.gamma_p = g
        rows += run_grid(T, methods, p, prof, pol, range(n_seeds), gamma=g)
    return pd.DataFrame(rows)


def suite_ablation(T, P, prof, pol, router, seeds):
    return pd.DataFrame(run_grid(T, ablation_methods(router), P, prof, pol, seeds))


def suite_sensitivity(T, P, prof, pol, router, seeds):
    rows = []
    ptsq = [MethodConfig("PTSQ", router=router), MethodConfig("Static-Ref", kind="static_ref")]
    for g in (0.75, 0.8, 0.85, 0.9, 0.95):
        p = copy.deepcopy(P); p.gamma_r = p.gamma_p = g
        rows += run_grid(T, ptsq, p, prof, pol, seeds, tag="gamma", x=g)
    for h in (0.0, 0.1, 0.2, 0.4):
        T2 = {q: (remix_privacy(tr, h, 0.3), r) for q, (tr, r) in T.items()}
        rows += run_grid(T2, ptsq, P, prof, pol, seeds, tag="high_frac", x=h)
    for f in (0.1, 0.3, 1.0, 3.0):
        rows += run_grid(T, ptsq, P, scale_bw(prof, f), pol, seeds, tag="bandwidth", x=f)
    # certification procedure at three targets: x=0 union bound over the grid with k+=800 (first
    # version); x=0.25 propose-and-test with population-fraction ("ratio") bounds and B=80;
    # x=0.5 propose-and-test with the conditional recall/precision test (configured B, delta only
    # over certified strata); x=0.75 + learned risk; x=0.875 + learned prefilter; x=1 the configured
    # procedure (+ the single-threshold family)
    for g in (0.75, 0.8, 0.9, 0.95):
        for x, kw in ((0, dict(cert="union", miss_budget=None, min_pos=800, n_cal=1500, risk=False, prefilter=False,
                               test="ratio", delta_certified_only=False, hybrid=0.0)),
                      (0.25, dict(risk=False, prefilter=False, test="ratio", miss_budget=80, delta_certified_only=False,
                                  hybrid=0.0)),
                      (0.5, dict(risk=False, prefilter=False, hybrid=0.0)), (0.75, dict(prefilter=False, hybrid=0.0)),
                      (0.875, dict(hybrid=0.0)), (1, {})):
            p = copy.deepcopy(P); p.gamma_r = p.gamma_p = g
            for k_, v_ in kw.items():
                setattr(p, k_, v_)
            rows += run_grid(T, ptsq, p, prof, pol, seeds, tag=f"cert{g}", x=x)
    # provider rate limit (tokens per minute): measured L0 account, the provider's higher
    # account levels, and no limit (x = 0)
    for tpm in (19000.0, 28500.0, 38000.0, 950000.0, 0.0):
        pr = copy.deepcopy(prof); pr.cloud.tpm = tpm
        rows += run_grid(T, ptsq, P, pr, pol, seeds, tag="tpm", x=tpm)
    for n in (500, 1000, 1500, 2500, 4000):
        p = copy.deepcopy(P); p.n_cal = n; p.min_pos = None; p.miss_budget = None   # fixed-n sweep
        rows += run_grid(T, ptsq, p, prof, pol, seeds, tag="n_cal", x=n)
    # data size: random subsets (without replacement) of each measured trace. Calibration
    # cost does not grow with N, so the saving should improve roughly like a + b/N.
    for n_sub in (2500, 5000, 10000, 20000, None):
        T2 = {}
        for q, (tr, r) in T.items():
            k = len(tr) if n_sub is None or n_sub >= len(tr) else n_sub
            T2[q] = (tr.sample(n=k, random_state=0).reset_index(drop=True) if k < len(tr) else tr, r)
        rows += run_grid(T2, ptsq, P, prof, pol, seeds, tag="N",
                         x=n_sub if n_sub is not None else -1)
    df = pd.DataFrame(rows)
    df.loc[(df.setting == "N") & (df.x == -1), "x"] = df.loc[(df.setting == "N") & (df.x == -1), "N"]
    return df


def suite_overhead(T, P, prof, pol, router, seeds):
    rows = []
    rng = np.random.default_rng(0)
    for n in (500, 1500, 4000):
        y = rng.random(n) < 0.2
        inp = StratumInput(level=PrivacyLevel.LOW, chain=[TierId.DEVICE, TierId.EDGE, TierId.CLOUD], N=50_000,
                           sample_idx=np.arange(n), y_sample=y, s_dev_sample=rng.random(n),
                           s_mid_sample=rng.random(n), rule_keep_sample=rng.random((n, 4)) < 0.8,
                           rule_keep_pop=rng.random((1000, 4)) < 0.8)
        uc = {TierId.DEVICE: 0.05, TierId.EDGE: 0.01, TierId.CLOUD: 0.2}
        t0 = time.perf_counter()
        plan = calibrate_stratum(inp, P.gamma_r, P.gamma_p, P.delta / 3, uc)
        rows.append({"component": "calibration (per stratum)", "n_cal": n, "configs": plan.n_configs,
                     "ms": (time.perf_counter() - t0) * 1e3})
    sh = PrivacyShield(pol)
    t0 = time.perf_counter()
    for _ in range(100_000):
        sh.permitted(PrivacyLevel.LOW, TierId.CLOUD)
    rows.append({"component": "privacy shield check", "ms": (time.perf_counter() - t0) / 100_000 * 1e3})
    if hasattr(router, "act"):
        s = np.zeros(12, dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(2000):
            router.act(s)
        rows.append({"component": "router decision", "ms": (time.perf_counter() - t0) / 2000 * 1e3})
    return pd.DataFrame(rows)


def suite_federated(T, P, prof, pol, router, seeds):
    rows = []
    for qid, (tr, rules) in T.items():
        cols = sorted([c for c in tr.columns if c.startswith("s_D_r")], key=lambda c: int(c[5:]))
        if not cols:
            continue
        for c in ["s_D"] + cols:
            rnd = 0 if c == "s_D" else int(c[5:])
            for s in seeds:
                r = replay(tr, rules, MethodConfig("PTSQ", router=router, dev_col=c), P, prof, pol, seed=s)
                y = tr["y_C"].to_numpy(bool)
                sc = tr[c].to_numpy(float)
                r.update(qid=qid, round=rnd, dev_auc=_auc(sc, y), dev_acc=float(((sc >= 0.5) == y).mean()))
                rows.append(r)
    return pd.DataFrame(rows)


def suite_device_model(T, P, prof, pol, router, seeds):
    """Device model size (columns s_Dm_<tag>, t_Dm_<tag> written by collect_trace --extra-device)."""
    rows = []
    for qid, (tr, rules) in T.items():
        extra = [c[5:] for c in tr.columns if c.startswith("s_Dm_")]
        if not extra:                  # only queries scored with additional device models
            continue
        tags = ["base"] + extra
        for tag in tags:
            sc, tc = ("s_D", "t_D") if tag == "base" else (f"s_Dm_{tag}", f"t_Dm_{tag}")
            for s in seeds:
                r = replay(tr, rules, MethodConfig("PTSQ", router=router, dev_col=sc, tdev_col=tc), P, prof, pol, seed=s)
                y = tr["y_C"].to_numpy(bool)
                r.update(qid=qid, device_model=tag, dev_auc=_auc(tr[sc].to_numpy(float), y))
                rows.append(r)
    return pd.DataFrame(rows)


def suite_bad_rules(T, P, prof, pol, router, seeds):
    """Robustness to wrong LLM rules (Prop. 1): inject an attractive but leaky rule that keeps only
    30% of negatives and drops a fraction `leak` of the (cloud) positives."""
    rows = []
    for qid, (tr, rules) in T.items():
        y = tr["y_C"].to_numpy(bool)
        rng = np.random.default_rng(123)
        u1, u2 = rng.random(len(tr)), rng.random(len(tr))
        for leak in (0.0, 0.05, 0.2, 0.5):
            tr2 = tr.copy()
            j = len(rules)
            tr2[f"rule_{j}"] = (~y & (u1 < 0.3)) | (y & (u2 >= leak))
            rules2 = rules + [{"type": "keyword_any", "field": "text", "keywords": [f"injected-leak-{leak}"]}]
            for s in seeds:
                r = replay(tr2, rules2, MethodConfig("PTSQ", router=router), P, prof, pol, seed=s)
                used = any(str(j) in str(r.get(f"rules_{lv}", "")).split(";") for lv in ("LOW", "MEDIUM", "HIGH"))
                r.update(qid=qid, leak=leak, injected_used=used,
                         min_stratum_recall=min(r.get(f"recall_{lv}", 1.0) for lv in ("LOW", "MEDIUM", "HIGH")))
                rows.append(r)
    return pd.DataFrame(rows)


def _auc(s, y):
    from scipy.stats import rankdata
    if y.all() or not y.any():
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


SUITES = {"main": suite_main, "ablation": suite_ablation, "sensitivity": suite_sensitivity,
          "overhead": suite_overhead, "federated": suite_federated, "device_model": suite_device_model,
          "bad_rules": suite_bad_rules}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", default=["all"])
    ap.add_argument("--traces", default="results/traces")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--profiles", default=None)
    ap.add_argument("--router", default="results/router.pt")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--guarantee-seeds", type=int, default=200)
    ap.add_argument("--out", default="results/exp")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--features", default="results/feat",
                    help="dir with <qid>_text.csv.gz for the learned risk (guarantee.risk: true)")
    a = ap.parse_args()
    global JOBS
    JOBS = a.jobs
    cfg = load_config(a.config)
    P, pol = build_params(cfg), build_policy(cfg)
    prof = load_profiles(a.profiles or cfg.get("profiles"), cfg)
    T = load_traces(a.traces)
    if P.risk or P.prefilter:
        from common import attach_features
        k = attach_features(T, a.features)
        print(f"[exp] learned risk: features for {k}/{len(T)} queries from {a.features}")
    router = make_router(a.router)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    suites = list(SUITES) + ["guarantee"] if "all" in a.suite else a.suite
    for s in suites:
        t0 = time.perf_counter()
        if s == "guarantee":
            df = suite_guarantee(T, P, prof, pol, router, a.guarantee_seeds)
        elif s == "baselines":
            df = suite_baselines(T, P, prof, pol, router, a.guarantee_seeds)
        else:
            df = SUITES[s](T, P, prof, pol, router, range(a.seeds))
        df.to_csv(out / f"{s}.csv", index=False)
        print(f"[exp] {s}: {len(df)} rows in {time.perf_counter() - t0:.1f}s -> {out / (s + '.csv')}")


if __name__ == "__main__":
    main()
