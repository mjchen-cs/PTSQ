"""Unit and property tests (synthetic data only)."""
import asyncio
import math

import numpy as np
import pandas as pd
import pytest

from ptsq.bounds import cp_lower, cp_upper
from ptsq.cascade import StratumInput, calibrate_stratum
from ptsq.mock import mock_profiles, synthetic_trace
from ptsq.privacy import PrivacyPolicy, PrivacyShield, PrivacyViolation, detect_pii, label_privacy, redact
from ptsq.replay import MethodConfig, ReplayParams, replay
from ptsq.simulator import ACTIONS, FixedRouter, HeuristicRouter, Item, RoutingSim, TierProfile
from ptsq.symbolic import eval_rule, parse_rules
from ptsq.types import PrivacyLevel, TierId


class RandomRouter:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, s):
        return int(self.rng.integers(len(ACTIONS)))


# ------------------------------------------------------------------------------- bounds
def test_cp_bounds_bracket_estimate():
    for k, n in [(0, 100), (5, 100), (50, 100), (100, 100)]:
        lo, hi = cp_lower(k, n, 0.05), cp_upper(k, n, 0.05)
        assert 0 <= lo <= k / n <= hi <= 1


def test_cp_upper_coverage():
    rng = np.random.default_rng(0)
    p, n, a = 0.07, 300, 0.05
    ks = rng.binomial(n, p, size=4000)
    cover = (cp_upper(ks, n, a) >= p).mean()
    assert cover >= 1 - a - 0.01


# ------------------------------------------------------------------------------ privacy
def test_pii_detection_and_levels():
    df = pd.DataFrame({"text": ["hello", "call me 13812345678", "id 11010119900307663X", "a@b.com"]})
    lv = label_privacy(df, PrivacyPolicy(), existing_col=None).tolist()
    assert lv == [0, 1, 2, 1]
    assert "cn_phone" in detect_pii("13812345678")
    assert redact("mail a@b.com now") == "mail [EMAIL] now"


def test_policy_and_shield():
    pol = PrivacyPolicy()
    assert pol.chain(PrivacyLevel.LOW) == [TierId.DEVICE, TierId.EDGE, TierId.CLOUD]
    assert pol.reference(PrivacyLevel.MEDIUM) == TierId.EDGE
    sh = PrivacyShield(pol)
    sh.check(PrivacyLevel.LOW, TierId.CLOUD)
    with pytest.raises(PrivacyViolation):
        sh.check(PrivacyLevel.HIGH, TierId.EDGE)
    assert sh.project(PrivacyLevel.MEDIUM, TierId.CLOUD) == TierId.EDGE
    with pytest.raises(ValueError):
        PrivacyPolicy.from_config({"allowed": {"HIGH": ["CLOUD"], "MEDIUM": ["DEVICE"], "LOW": ["DEVICE"]}})


# ----------------------------------------------------------------------------- symbolic
def test_rules_parse_and_eval():
    txt = 'blah [{"type":"keyword_any","field":"text","keywords":["late","delay"]},' \
          '{"type":"numeric","field":"stars","op":"<=","value":3},{"type":"numeric","field":"nope","op":"<","value":1}]'
    rules = parse_rules(txt, {"text", "stars"})
    assert len(rules) == 2
    df = pd.DataFrame({"text": ["very LATE", "fine"], "stars": [2, 5]})
    assert eval_rule(rules[0], df).tolist() == [True, False]
    assert eval_rule(rules[1], df).tolist() == [True, False]


# ------------------------------------------------------------------------------ cascade
def _stratum(n_pop, sel, q_dev, rng, chain_len=3, n=1500):
    y_pop = rng.random(n_pop) < sel
    sd = 1 / (1 + np.exp(-(q_dev * (2 * y_pop - 1) + rng.normal(size=n_pop))))
    sm = 1 / (1 + np.exp(-(2.5 * (2 * y_pop - 1) + rng.normal(size=n_pop))))
    loc = rng.integers(0, n_pop, size=n)
    chain = [TierId.DEVICE, TierId.EDGE, TierId.CLOUD][:chain_len]
    keep = np.stack([y_pop | (rng.random(n_pop) < 0.4), (y_pop & (rng.random(n_pop) > 0.1)) | (rng.random(n_pop) < 0.3)], 1)
    inp = StratumInput(level=PrivacyLevel.LOW, chain=chain, N=n_pop, sample_idx=loc, y_sample=y_pop[loc],
                       s_dev_sample=sd[loc], s_mid_sample=sm[loc] if chain_len == 3 else None,
                       rule_keep_sample=keep[loc], rule_keep_pop=keep)
    return inp, y_pop, sd, sm, keep


def _realised(plan, y, sd, sm, keep):
    pr = plan.pruned_mask(keep)
    pr = np.zeros(len(y), bool) if pr is None else pr
    pred = np.zeros(len(y), bool)
    for i in range(len(y)):
        if pr[i]:
            continue
        d = plan.route_device(sd[i])
        if d != "up":
            pred[i] = d == "acc"
        elif len(plan.chain) == 3:
            m = plan.route_mid(sm[i])
            pred[i] = y[i] if m == "up" else m == "acc"
        else:
            pred[i] = y[i]
    tp = (pred & y).sum()
    return tp / max(y.sum(), 1), tp / max(pred.sum(), 1)


@pytest.mark.parametrize("chain_len,cert,test", [(2, "union", "ratio"), (3, "union", "ratio"), (3, "split", "ratio"),
                                                (2, "split", "cond"), (3, "split", "cond")])
def test_guarantee_holds_empirically(chain_len, cert, test):
    """Theorem 1: P[recall < gamma or precision < gamma] <= delta (checked over 60 draws)."""
    rng = np.random.default_rng(42)
    uc = {TierId.DEVICE: 0.01, TierId.EDGE: 0.02, TierId.CLOUD: 1.0}
    fails, n_feas = 0, 0
    for t in range(60):
        inp, y, sd, sm, keep = _stratum(20000, 0.25, 1.0, rng, chain_len)
        plan = calibrate_stratum(inp, 0.9, 0.9, 0.05, uc, cert=cert, test=test)
        rec, prec = _realised(plan, y, sd, sm, keep)
        fails += (rec < 0.9) or (prec < 0.9)
        n_feas += plan.feasible
    assert n_feas > 30            # the optimiser does find cheap certified plans
    assert fails / 60 <= 0.05 + 0.05


def test_infeasible_falls_back_to_reference():
    rng = np.random.default_rng(1)
    inp, y, sd, sm, keep = _stratum(5000, 0.01, 0.2, rng, 3, n=200)   # rare positives, tiny sample
    plan = calibrate_stratum(inp, 0.99, 0.99, 0.01, {TierId.DEVICE: 0, TierId.EDGE: 1, TierId.CLOUD: 1})
    assert not plan.feasible and plan.rules == ()
    assert all(plan.route_device(s) == "up" for s in (0.0, 0.5, 1.0))


# ---------------------------------------------------------------------------- simulator
def _items(n=600, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lv = PrivacyLevel(int(rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1])))
        ref = {0: TierId.CLOUD, 1: TierId.EDGE, 2: TierId.DEVICE}[int(lv)]
        o = "acc" if lv == 2 else ("E_mid" if lv == 0 else "E_final")
        out.append(Item(rid=i, level=lv, dev_outcome=o, tokens=300, nbytes=1000, t_dev=0.01,
                        up_after_e=bool(rng.random() < 0.3), reference=ref))
    return out


@pytest.mark.parametrize("router", [FixedRouter(), HeuristicRouter(), RandomRouter(0), FixedRouter(1.0, 1.0, 128)])
def test_sim_terminates_and_respects_privacy(router):
    pe, pc = TierProfile(fixed_s=0.05, per_token_s=1e-5, concurrency=2), TierProfile(fixed_s=0.3, concurrency=8, batchable=False)
    shield = PrivacyShield(PrivacyPolicy())
    sim = RoutingSim(_items(), pe, pc, shield=shield)
    res = sim.run(router)       # shield.check raises on any violation
    assert sim.done() and res.makespan_s > 0
    hi = {it.rid for it in _items() if it.level == PrivacyLevel.HIGH}
    assert not any(r in hi for r in res.decided_by if res.decided_by[r] != TierId.DEVICE)


def test_router_never_changes_correctness():
    """Theorem 2: any router (even random) yields recall/precision >= the fixed router's."""
    tr, rules = synthetic_trace(6000, 0.2, seed=5)
    prof, P = mock_profiles(), ReplayParams(n_cal=800)
    base = replay(tr, rules, MethodConfig("fixed"), P, prof, seed=1)
    for r in (RandomRouter(1), FixedRouter(1.0, 1.0, 128), HeuristicRouter()):
        o = replay(tr, rules, MethodConfig("r", router=r), P, prof, seed=1)
        assert o["recall"] >= base["recall"] - 1e-12 and o["precision"] >= base["precision"] - 1e-12
        assert o["privacy_violations"] == 0


def test_replay_baselines():
    tr, rules = synthetic_trace(3000, 0.2, seed=2)
    prof, P = mock_profiles(), ReplayParams(n_cal=500)
    ca = replay(tr, rules, MethodConfig("c", kind="cloud_all"), P, prof)
    assert ca["privacy_violations"] == int((tr.privacy > 0).sum())
    sr = replay(tr, rules, MethodConfig("s", kind="static_ref"), P, prof)
    assert sr["recall"] == 1.0 and sr["precision"] == 1.0 and sr["privacy_violations"] == 0


# --------------------------------------------------------------------- live executor (mock)
def test_live_executor_with_mock_backends(tmp_path):
    from ptsq.config import Query
    from ptsq.executor import run_query
    from ptsq.llm.backends import MockBackend
    rng = np.random.default_rng(0)
    n = 1200
    texts = [("late delivery " if rng.random() < 0.3 else "nice food ") + str(i) for i in range(n)]
    texts = [t + (" call 13812345678" if i % 7 == 0 else "") for i, t in enumerate(texts)]
    df = pd.DataFrame({"rid": np.arange(n), "text": texts})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    truth = {i: "late" in t for i, t in enumerate(texts)}
    tiers = {TierId.DEVICE: MockBackend("d", truth, quality=1.5), TierId.EDGE: MockBackend("e", truth, quality=3.0),
             TierId.CLOUD: MockBackend("c", truth, quality=None, price_in=4, price_out=12)}
    q = Query(qid="t", table=str(p), predicate="complains about late delivery")
    from ptsq.replay import Profiles
    out = asyncio.run(run_query(q, tiers, PrivacyPolicy(), ReplayParams(n_cal=300), router=HeuristicRouter(),
                                profiles=Profiles()))
    assert out["policy"] in ("fixed", "router")
    sel = set(out["selected"])
    rec = len(sel & {i for i, v in truth.items() if v}) / sum(truth.values())
    assert rec > 0.85 and out["shield_blocked"] == 0
    # propose-and-test with the learned risk and pre-filter (live code path)
    out2 = asyncio.run(run_query(q, tiers, PrivacyPolicy(),
                                 ReplayParams(n_cal=600, cert="split", risk=True, prefilter=True),
                                 profiles=Profiles()))
    sel = set(out2["selected"])
    rec = len(sel & {i for i, v in truth.items() if v}) / sum(truth.values())
    assert rec > 0.85 and out2["shield_blocked"] == 0


def test_ppo_router_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    from ptsq.rl.ppo import PPOConfig, PPORouter, synthetic_item_pool, train_ppo
    from ptsq.simulator import CostWeights
    prof = mock_profiles()
    out = tmp_path / "r.pt"
    _, hist = train_ppo(synthetic_item_pool(500), prof.edge, prof.cloud, CostWeights(),
                        PPOConfig(episodes=8, batch_episodes=4), out_path=str(out), log_every=0)
    assert out.exists() and len(hist) == 2
    router = PPORouter(str(out))
    tr, rules = synthetic_trace(3000, 0.2, seed=9)
    base = replay(tr, rules, MethodConfig("f"), ReplayParams(n_cal=500), prof, seed=0)
    r = replay(tr, rules, MethodConfig("p", router=router), ReplayParams(n_cal=500), prof, seed=0)
    assert r["recall"] >= base["recall"] - 1e-12 and r["privacy_violations"] == 0


def test_shipped_configs_build():
    """Every key in the shipped configs must be accepted by the backend constructors."""
    from ptsq.config import build_params, build_policy, build_tiers, load_config
    for path in ("configs/default.yaml", "configs/mock.yaml"):
        cfg = load_config(path)
        cfg["cache"] = None
        build_tiers(cfg)
        build_params(cfg)
        build_policy(cfg)


def test_cascade_precheck():
    import numpy as np
    from ptsq.cascade import cascade_can_pay_off
    from ptsq.types import TierId
    chain = [TierId.DEVICE, TierId.EDGE, TierId.CLOUD]
    slow_ref = {TierId.DEVICE: 0.1, TierId.EDGE: 0.05, TierId.CLOUD: 0.9}
    fast_ref = {TierId.DEVICE: 0.1, TierId.EDGE: 0.05, TierId.CLOUD: 0.01}
    assert cascade_can_pay_off(chain, slow_ref, None, 2, 1500, 10000)
    assert not cascade_can_pay_off(chain, fast_ref, None, 2, 1500, 10000)
    keep = np.zeros((10000, 1), bool); keep[:1000] = True          # very selective rule
    assert cascade_can_pay_off(chain, fast_ref, keep, 2, 1500, 10000)
    pos = np.zeros((50, 1), bool)                                    # ... that drops all pilot positives
    assert not cascade_can_pay_off(chain, fast_ref, keep, 2, 1500, 10000, pos, 0.8)
    assert not cascade_can_pay_off([TierId.DEVICE], slow_ref, None, 2, 1500, 10000)


def test_learned_risk_keeps_guarantee():
    """Propose-and-test with a learned disagreement risk: the edge model is confidently wrong on a
    detectable subgroup; the guarantee must hold and the risk should be used to save cost."""
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(7)
    uc = {TierId.DEVICE: 0.5, TierId.EDGE: 0.02, TierId.CLOUD: 1.0}
    fails, used = 0, 0
    for t in range(30):
        N = 20000
        y = rng.random(N) < 0.25
        grp = rng.random(N) < 0.15                       # subgroup where the edge model is wrong
        flip = grp & (rng.random(N) < 0.7)
        ye = y ^ flip
        sm = np.where(ye, 0.99, 0.01)                    # confidently right or wrong
        sd = rng.random(N)
        X = csr_matrix(np.c_[grp + rng.normal(0, 0.3, N), sm])
        loc = rng.integers(0, N, size=3000)
        inp = StratumInput(level=PrivacyLevel.LOW, chain=[TierId.DEVICE, TierId.EDGE, TierId.CLOUD], N=N,
                           sample_idx=loc, y_sample=y[loc], s_dev_sample=sd[loc], s_mid_sample=sm[loc],
                           feat_sample=X[loc], feat_pop=X)
        plan = calibrate_stratum(inp, 0.9, 0.9, 0.05, uc, cert="split")
        used += plan.risk_pop is not None
        pred = np.zeros(N, bool)
        for i in range(N):
            if plan.direct or not plan.feasible:
                pred[i] = y[i]
                continue
            d = plan.route_device(sd[i])
            if d != "up":
                pred[i] = d == "acc"
                continue
            m = plan.route_mid(sm[i], plan.risk_pop[i] if plan.risk_pop is not None else None)
            pred[i] = y[i] if m == "up" else m == "acc"
        tp = (pred & y).sum()
        fails += (tp / y.sum() < 0.9) or (tp / max(pred.sum(), 1) < 0.9)
    assert used >= 15
    assert fails / 30 <= 0.1


def test_learned_prefilter_keeps_guarantee():
    """A learned text pre-filter (fitted on S_A) becomes a candidate pruning rule; certification
    on S_B must keep the guarantee, and the filter should prune obvious negatives."""
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(11)
    uc = {TierId.DEVICE: 0.5, TierId.EDGE: 0.3, TierId.CLOUD: 1.0}
    fails, used = 0, 0
    for t in range(20):
        N = 20000
        y = rng.random(N) < 0.2
        z = np.where(y, rng.normal(1.5, 1, N), rng.normal(-1, 1, N))        # informative feature
        sm = 1 / (1 + np.exp(-(3 * (2 * y - 1) + rng.normal(0, 1.5, N))))
        sd = rng.random(N)
        X = csr_matrix(np.c_[z, sm, np.zeros(N)])                         # [text..., s_E, len]
        loc = rng.integers(0, N, size=3000)
        inp = StratumInput(level=PrivacyLevel.MEDIUM, chain=[TierId.DEVICE, TierId.EDGE], N=N,
                           sample_idx=loc, y_sample=y[loc], s_dev_sample=sd[loc], feat_sample=X[loc], feat_pop=X)
        plan = calibrate_stratum(inp, 0.9, 0.9, 0.05, uc, cert="split", prefilter=True)
        pm = plan.pruned_mask(None)
        used += pm is not None and pm.any()
        if plan.direct or not plan.feasible:
            pred = y.copy()
        else:
            pred = np.where(pm, False, y) if pm is not None else y.copy()
            d = np.array([plan.route_device(v) for v in sd])
            pred = np.where((pm if pm is not None else False) | (d == "rej"), False, np.where(d == "acc", True, y))
        tp = (pred & y).sum()
        fails += (tp / y.sum() < 0.9) or (tp / max(pred.sum(), 1) < 0.9)
    assert used >= 10
    assert fails / 20 <= 0.1


def test_betting_test_validity_and_bargain_rt():
    """BARGAIN baseline: the betting test rejects a true null at most alpha of the time, and the
    recall-target threshold meets the target in (almost) all runs on a synthetic stratum."""
    from ptsq.bargain import bargain_rt, betting_reject
    rng = np.random.default_rng(3)
    rej = np.mean([betting_reject((rng.random(300) < 0.8).astype(float), 0.8, 0.05) for _ in range(1500)])
    assert rej <= 0.05 + 0.015
    fails = 0
    for _ in range(60):
        y = rng.random(20000) < 0.1
        A = np.clip(np.where(y, rng.beta(5, 2, 20000), rng.beta(2, 5, 20000)), 0, 1)
        tau, drawn = bargain_rt(A, y, 3000, 0.9, 0.05, rng)
        fails += (A[y] >= tau).mean() < 0.9
        assert len(drawn) == 3000
    assert fails / 60 <= 0.1


def test_bargain_replay_smoke():
    from ptsq.replay import MethodConfig, Profiles, ReplayParams, replay
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.random(n) < 0.2
    tr = pd.DataFrame({"rid": np.arange(n), "privacy": rng.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1]),
                       "tokens": 100, "nbytes": 400, "s_D": np.clip(y * 0.6 + rng.random(n) * 0.4, 0, 1),
                       "s_E": np.clip(y * 0.8 + rng.random(n) * 0.2, 0, 1), "y_E": y, "y_C": y, "t_D": 0.05})
    for beta in (0.0, 0.01):
        r = replay(tr, [], MethodConfig("B", kind="bargain", beta=beta), ReplayParams(n_cal=500), Profiles(), seed=1)
        assert r["privacy_violations"] == 0 and r["precision"] == 1.0


def test_threshold_certify_valid():
    """Single-threshold certificate: recall >= gamma w.p. >= 1 - delta (empirical, 400 draws)."""
    from ptsq.cascade import threshold_certify
    rng = np.random.default_rng(3)
    N = 20000
    y = rng.random(N) < 0.15
    A = np.clip(np.where(y, rng.beta(5, 2, N), rng.beta(1, 6, N)), 0, 1)
    uc = {TierId.DEVICE: 0.01, TierId.EDGE: 0.02, TierId.CLOUD: 1.0}
    chain = [TierId.DEVICE, TierId.EDGE, TierId.CLOUD]
    fails, n_ok = 0, 0
    for _ in range(400):
        S = rng.integers(0, N, 1500)
        p = threshold_certify(PrivacyLevel.LOW, chain, N, y[S], A[S], 0.9, 0.05, uc)
        if p is None:
            continue
        n_ok += 1
        rho = p.bounds["rho"]
        fails += (A[y] >= rho).mean() < 0.9
    assert n_ok > 300 and fails / n_ok <= 0.05 + 0.03


def test_hybrid_replay_smoke():
    from ptsq.replay import MethodConfig, Profiles, ReplayParams, replay
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.random(n) < 0.2
    tr = pd.DataFrame({"rid": np.arange(n), "privacy": rng.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1]),
                       "tokens": 100, "nbytes": 400, "s_D": np.clip(y * 0.6 + rng.random(n) * 0.4, 0, 1),
                       "s_E": np.clip(y * 0.8 + rng.random(n) * 0.2, 0, 1), "y_E": y, "y_C": y, "t_D": 0.05})
    r = replay(tr, [], MethodConfig("H"), ReplayParams(n_cal=500, cert="split", test="cond", hybrid=0.2),
               Profiles(), seed=1)
    assert r["privacy_violations"] == 0


def test_no_bottleneck_skips_calibration():
    """Without a bottleneck reference (e.g. no cloud rate limit) every stratum goes to its reference."""
    from ptsq.replay import calibration_cannot_pay_off, unit_costs, Profiles
    from ptsq.types import PrivacyLevel
    from ptsq.cascade import StratumPlan
    chains = {PrivacyLevel.LOW: [TierId.DEVICE, TierId.EDGE, TierId.CLOUD],
              PrivacyLevel.MEDIUM: [TierId.DEVICE, TierId.EDGE], PrivacyLevel.HIGH: [TierId.DEVICE]}
    cheap = {TierId.DEVICE: 0.13, TierId.EDGE: 0.07, TierId.CLOUD: 0.035}
    limited = {TierId.DEVICE: 0.13, TierId.EDGE: 0.07, TierId.CLOUD: 1.0}
    assert calibration_cannot_pay_off(list(chains), chains, cheap)
    assert not calibration_cannot_pay_off(list(chains), chains, limited)
