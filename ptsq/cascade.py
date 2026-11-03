"""Privacy-tiered cascade with probabilistic accuracy guarantees.

Per privacy stratum ``s`` (all tuples of one privacy level) the permitted tiers form a
chain, e.g. LOW: D -> E -> C, MEDIUM: D -> E, HIGH: D. The last tier of the chain is the
stratum's *permitted reference* R_s; guarantees are stated relative to R_s.

A configuration theta = (G, lo_D, hi_D[, lo_E, hi_E]) consists of a subset G of the
candidate symbolic rules plus score thresholds, and routes every tuple t:
    some rule in G rejects t            -> reject  (pruned, no LLM call)
    s_D <= lo_D / s_D >= hi_D           -> reject / accept at D
    otherwise, chain length 2           -> decided by R_s (exact)
    otherwise, s_E <= lo_E / >= hi_E    -> reject / accept at E
    otherwise                           -> decided by R_s = C (exact)
(For chain length 1 the device is the reference and only G can cause errors.)

From a uniform with-replacement calibration sample S (labelled by R_s, device-scored, and
E-scored for chain length 3) we compute Clopper-Pearson bounds, for every theta in a finite
grid Theta, on
    m(theta)  fraction of tuples that are positive but rejected below R_s
    f(theta)  fraction of tuples that are negative but accepted below R_s
    a(theta)  fraction of tuples accepted overall
    p         fraction of positives
and keep theta with  UCB(m) <= (1-gamma_r) LCB(p)  and  UCB(f) <= (1-gamma_p) LCB(a).
With a union bound over the 3|Theta|+1 bounds, every kept theta satisfies recall >= gamma_r
and precision >= gamma_p (w.r.t. R_s) simultaneously w.p. >= 1-delta_s (Theorem 1 of the
paper). Rule selection is part of Theta, so an LLM-proposed rule that is wrong can only make
theta infeasible (or expensive), never break the guarantee. We return the cheapest kept
theta; if none is feasible we fall back to "no rules, everything to R_s", which is exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .bounds import cp_lower, cp_upper
from .types import PrivacyLevel, TierId

INF = float("inf")


@dataclass
class StratumInput:
    level: PrivacyLevel
    chain: list[TierId]                  # permitted tiers, ascending; chain[-1] is reference
    N: int                               # stratum size
    sample_idx: np.ndarray               # int[n] positions (in the stratum) of S, with replacement
    y_sample: np.ndarray                 # bool[n] reference labels
    s_dev_sample: np.ndarray             # float[n] device scores
    s_mid_sample: np.ndarray | None = None    # float[n] middle-tier scores (chain len 3)
    rule_keep_sample: np.ndarray | None = None  # bool[n, R] keep-masks of candidate rules
    rule_keep_pop: np.ndarray | None = None     # bool[N, R] keep-masks on the whole stratum (local, cheap)
    feat_sample: object | None = None    # features of S (rows aligned with sample_idx), for the learned risk
    feat_pop: object | None = None       # features of the whole stratum (unlabelled)


@dataclass
class StratumPlan:
    level: PrivacyLevel
    chain: list[TierId]
    rules: tuple = ()                   # indices of candidate rules used for pruning
    lo_d: float = -INF
    hi_d: float = INF
    lo_m: float = -INF
    hi_m: float = INF
    feasible: bool = True               # False => fell back to "everything to reference"
    direct: bool = False                # True  => cheaper to send every tuple straight to R_s
    skip_device: bool = False           # True  => plan never uses s_D, so the device does not score
    est_cost: float = 0.0
    bounds: dict = field(default_factory=dict)
    n_configs: int = 1
    rho_rej: float = INF                # middle tier rejects only tuples with risk <= rho_rej
    rho_acc: float = INF                # ... and accepts only tuples with risk <= rho_acc
    risk_pop: np.ndarray | None = None  # learned risk of every tuple of the stratum (if used)
    n_sym: int = 10**9                  # rule indices >= n_sym are learned pre-filters
    learned_keep_pop: np.ndarray | None = None   # bool[N, L] keep-masks of the learned pre-filters
    risk_model: object | None = None    # callable features -> risk (live execution scores tuples lazily)

    @property
    def reference(self) -> TierId:
        return self.chain[-1]

    @property
    def prune_enabled(self) -> bool:
        return len(self.rules) > 0

    def pruned_mask(self, rule_keep: np.ndarray | None) -> np.ndarray | None:
        if not self.rules:
            return None
        n = rule_keep.shape[0] if rule_keep is not None else self.learned_keep_pop.shape[0]
        keep = np.ones(n, dtype=bool)
        for j in self.rules:
            keep &= rule_keep[:, j] if j < self.n_sym else self.learned_keep_pop[:, j - self.n_sym]
        return ~keep

    def route_device(self, s: float) -> str:
        """'rej', 'acc' or 'up' for a (non-pruned) tuple with device score s."""
        if len(self.chain) == 1:
            return "acc" if s >= 0.5 else "rej"   # device is the reference
        if s <= self.lo_d:
            return "rej"
        if s >= self.hi_d:
            return "acc"
        return "up"

    def route_mid(self, s: float, risk: float | None = None) -> str:
        if s <= self.lo_m and (risk is None or risk <= self.rho_rej):
            return "rej"
        if s >= self.hi_m and (risk is None or risk <= self.rho_acc):
            return "acc"
        return "up"


# Data-independent threshold grid (scores are probabilities). Fixing the grid BEFORE sampling is
# required for the union-bound argument of Theorem 1: thresholds must not depend on the sample.
DEV_LO = (-INF, 1e-4, 1e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
DEV_HI = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, INF)
MID_LO = (-INF, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
MID_HI = (0.5, 0.7, 0.8, 0.9, 0.95, INF)
# Learned-risk thresholds: quantiles of the (out-of-fold) risk on S_A. INF = off.
RISK_QUANTILES = (0.5, 0.75, 0.9)
# Learned pre-filter: prune the tuples whose predicted-positive score is below these population quantiles.
PREFILTER_QUANTILES = (0.25, 0.5)


def rule_subsets(n_rules: int, max_size: int = 2) -> list[tuple]:
    subs = [()]
    for k in range(1, min(max_size, n_rules) + 1):
        subs += list(combinations(range(n_rules), k))
    return subs


def calibrate_stratum(inp: StratumInput, gamma_r: float, gamma_p: float | None, delta_s: float,
                      unit_cost: dict[TierId, float], max_rule_subset: int = 2,
                      use_rules: bool = True, cert: str = "union", split: float = 0.5,
                      bonferroni: int = 0, prefilter: bool = False, test: str = "ratio",
                      eta: int = 0) -> StratumPlan:
    """Select the cheapest certified configuration for one stratum.

    ``unit_cost`` gives the cost of processing one tuple at each tier (TierId.DEVICE is the
    on-device scoring cost, which pruning saves).

    ``cert`` chooses how the selection over the grid is made valid:
      "union"  every configuration is bounded at level delta/(3K+1) (union bound over the grid);
      "split"  the first ``split`` fraction of the (i.i.d.) draws S_A only *proposes* candidates:
               the Pareto frontier of (estimated cost, estimated slack) among configurations that
               look feasible on S_A, ordered from the safest to the cheapest. The remaining draws
               S_B *test* them one by one in that order (fixed-sequence testing as in Learn-then-
               Test), each with four bounds at level delta/4, stopping at the first failure. All
               configurations that pass are valid simultaneously w.p. >= 1-delta, so returning the
               cheapest of them keeps Theorem 1, while each bound is taken at delta/4 instead of
               delta/(3K+1).

    ``test`` (split only) chooses how one configuration is tested on S_B:
      "ratio"  population fractions: UCB(misses/n) <= (1-gamma_r) LCB(positives/n), and likewise
               for precision -- four bounds at delta/4;
      "cond"   conditional on the sampled reference positives (answers), the number of them the
               configuration misses (gets wrong) is Binomial(#positives, 1-recall) (Binomial(#answers,
               1-precision)): UCB(misses, #pos) <= 1-gamma_r and UCB(fp, #answers) <= 1-gamma_p -- two
               exact bounds at delta/2 that test recall and precision directly.
    """
    chain = list(inp.chain)
    n = len(inp.sample_idx)
    plan = StratumPlan(level=inp.level, chain=chain)
    if n == 0:
        plan.lo_d, plan.hi_d, plan.feasible = (0.5 - 1e-12, 0.5, False) if len(chain) == 1 else (-INF, INF, False)
        return plan

    y = np.asarray(inp.y_sample, dtype=bool)
    sd = np.nan_to_num(np.asarray(inp.s_dev_sample, dtype=float), nan=0.0)
    R = 0 if (inp.rule_keep_sample is None or not use_rules) else inp.rule_keep_sample.shape[1]
    subsets = rule_subsets(R, max_rule_subset)
    rk_sample, rk_pop = inp.rule_keep_sample, inp.rule_keep_pop
    learned_pop = None
    # ---- optional learned pre-filter (propose-and-test only): a text classifier fitted on S_A
    # predicts the reference label; "keep if score >= population quantile q" becomes a candidate
    # pruning rule like the LLM-proposed ones (thresholds from unlabelled population quantiles).
    if cert == "split" and prefilter and len(chain) > 1 and inp.feat_sample is not None and inp.feat_pop is not None:
        from .risk import crossfit_risk
        nA_ = int(np.floor(split * n))
        nc = inp.feat_sample.shape[1]            # features: [TF-IDF | s_E | log length]
        cols = np.r_[0:nc - 2, nc - 1]
        FS, FP = inp.feat_sample[:, cols], inp.feat_pop[:, cols]     # drop s_E: the filter runs before E
        model, oof = crossfit_risk(FS[:nA_], y[:nA_], seed=1)
        if model is not None:
            sv_S = np.concatenate([oof, model(FS[nA_:])])
            sv_P = model(FP)
            thr = [float(np.quantile(sv_P, q)) for q in PREFILTER_QUANTILES]
            lk_S = np.stack([sv_S >= t for t in thr], 1)
            learned_pop = np.stack([sv_P >= t for t in thr], 1)
            base_S = rk_sample[:, :R] if R else np.zeros((n, 0), bool)
            base_P = rk_pop[:, :R] if (R and rk_pop is not None) else np.zeros((len(sv_P), 0), bool)
            rk_sample, rk_pop = np.hstack([base_S, lk_S]), np.hstack([base_P, learned_pop])
            # nested thresholds: never combine two learned filters; at most one with symbolic rules
            subsets = subsets + [G + (R + j,) for G in subsets if len(G) < max_rule_subset
                                 for j in range(len(thr))]
            plan.n_sym = R

    # ---- threshold grid (fixed, data-independent) ----------------------------------------
    if len(chain) == 1:
        LD = HD = LM = HM = np.array([np.nan])
    else:
        lo_d, hi_d = np.array(DEV_LO), np.array(DEV_HI)
        if len(chain) == 3:
            sm = np.asarray(inp.s_mid_sample, dtype=float)
            lo_m, hi_m = np.array(MID_LO), np.array(MID_HI)
        else:
            lo_m, hi_m = np.array([-INF]), np.array([INF])
        LD, HD, LM, HM = (a.ravel() for a in np.meshgrid(lo_d, hi_d, lo_m, hi_m, indexing="ij"))
        ok = (LD < HD) & (LM < HM)
        LD, HD, LM, HM = LD[ok], HD[ok], LM[ok], HM[ok]
    Kt = len(LD)
    K = Kt * len(subsets)
    plan.n_configs = K
    a = delta_s / (3 * K + 1)
    p_l = float(cp_lower(y.sum(), n, a))

    if len(chain) == 1:
        uses_dev = np.ones(Kt, dtype=bool)
    else:
        uses_dev = ~(np.isneginf(LD) & np.isposinf(HD))
    best = None
    n_feasible = 0
    nA = int(np.floor(split * n)) if cert == "split" else 0
    cand = []          # (cost[k], G, k, missA, fpA, accA, missB, fpB, accB) per subset, vectorised
    # ---- optional learned disagreement risk (propose-and-test only): fitted on S_A ------
    risk_S, risk_pop, risk_grid = None, None, [(INF, INF)]
    risk_model = None
    if cert == "split" and len(chain) == 3 and inp.feat_sample is not None:
        from .risk import crossfit_risk
        nA_ = int(np.floor(split * n))
        d_A = (sm[:nA_] >= 0.5) != y[:nA_]               # does the middle tier disagree with R?
        model, oof = crossfit_risk(inp.feat_sample[:nA_], d_A)
        if model is not None:
            # S_A: out-of-fold risks (honest proposal); S_B: the model fitted on all of S_A
            risk_S = np.concatenate([oof, model(inp.feat_sample[nA_:])])
            # population risks need s_E, which is only paid for on tuples that reach the edge;
            # they are used only there, and the thresholds are quantiles of the S_A risks
            risk_pop = model(inp.feat_pop) if inp.feat_pop is not None else None
            risk_model = model
            lv_r = [INF] + [float(np.quantile(oof, q)) for q in RISK_QUANTILES]
            risk_grid = [(r1, r2) for r1 in lv_r for r2 in lv_r]
    plan.n_configs = K * len(risk_grid)
    for rr, ra in risk_grid:
        # ---- threshold-dependent decisions (Kt x n), independent of the rule subset ---------
        if len(chain) == 1:
            rej_t = np.zeros((1, n), dtype=bool)          # device = reference: no errors
            acc_t = np.zeros((1, n), dtype=bool)
            acc_all_t = y[None, :].copy()
            up_t = np.zeros((1, n), dtype=bool)
            to_ref_t = up_t
        else:
            S = sd[None, :]
            rej_d = S <= LD[:, None]
            acc_d = S >= HD[:, None]
            up_t = ~rej_d & ~acc_d
            if len(chain) == 3:
                M = sm[None, :]
                rej_m = up_t & (M <= LM[:, None])
                acc_m = up_t & (M >= HM[:, None])
                if risk_S is not None:              # learned disagreement risk: escalate risky tuples
                    rej_m &= (risk_S <= rr)[None, :]
                    acc_m &= (risk_S <= ra)[None, :]
                to_ref_t = up_t & ~rej_m & ~acc_m
            else:
                rej_m = acc_m = np.zeros_like(up_t)
                to_ref_t = up_t
            rej_t = rej_d | rej_m
            acc_t = acc_d | acc_m
            acc_all_t = acc_t | (to_ref_t & y[None, :])

        for G in subsets:
            if G:
                keep = np.ones(n, dtype=bool)
                for j in G:
                    keep &= rk_sample[:, j]
                if rk_pop is not None:                 # exact joint keep fraction of the subset
                    kp = np.ones(rk_pop.shape[0], dtype=bool)
                    for j in G:
                        kp &= rk_pop[:, j]
                    keep_frac = float(kp.mean())
                else:
                    keep_frac = float(keep.mean())
            else:
                keep = np.ones(n, dtype=bool)
                keep_frac = 1.0
            pr = ~keep[None, :]
            miss_m = (pr | (rej_t & ~pr)) & y[None, :]
            fp_m = (acc_t & ~pr) & ~y[None, :]
            acc_m = acc_all_t & ~pr
            miss, fp, acc = miss_m.sum(1), fp_m.sum(1), acc_m.sum(1)
            if cert != "split":
                m_u = cp_upper(miss, n, a)
                f_u = cp_upper(fp, n, a)
                a_l = cp_lower(acc, n, a)
                feas = m_u <= (1 - gamma_r) * p_l
                if gamma_p is not None:
                    feas &= f_u <= (1 - gamma_p) * a_l
                n_feasible += int(feas.sum())
            # expected cost per configuration, extrapolated to the stratum. A configuration whose
            # device thresholds are (-inf, +inf) never looks at s_D, so the device is skipped.
            cost = np.where(uses_dev, inp.N * keep_frac * unit_cost[TierId.DEVICE], 0.0)
            if len(chain) > 1:
                up_live = (up_t & keep[None, :]).sum(1) / n
                cost = cost + inp.N * up_live * unit_cost[chain[1]]
                if len(chain) == 3:
                    ref_live = (to_ref_t & keep[None, :]).sum(1) / n
                    # if both edge thresholds are infinite, escalations skip E and go straight to C
                    mid_trivial = np.isneginf(LM) & np.isposinf(HM)
                    cost = cost + inp.N * np.where(mid_trivial, up_live * (unit_cost[chain[2]] - unit_cost[chain[1]]),
                                                    ref_live * unit_cost[chain[2]])
            if cert == "split":
                cand.append((cost, (G, rr, ra), miss_m[:, :nA].sum(1), fp_m[:, :nA].sum(1), acc_m[:, :nA].sum(1),
                             miss_m[:, nA:].sum(1), fp_m[:, nA:].sum(1), acc_m[:, nA:].sum(1)))
                continue
            if feas.any():
                idx = np.flatnonzero(feas)
                k = idx[np.argmin(cost[idx])]
                if best is None or cost[k] < best[0]:
                    best = (float(cost[k]), G, k, float(m_u[k]), float(f_u[k]), float(a_l[k]))
                    best_skip = not bool(uses_dev[k])

    opt_cost = None
    if cert == "split" and cand:
        _split_select.bonferroni = bonferroni
        _split_select.test = test
        _split_select.eta = eta
        best, n_feasible, p_l, a = _split_select(cand, uses_dev, y, nA, n, gamma_r, gamma_p, delta_s)
        opt_cost = _split_select.opt_cost
        if best is not None:
            G_, rr_, ra_ = best[1]
            best = (best[0], G_) + tuple(best[2:])
            plan.rho_rej, plan.rho_acc = rr_, ra_
            if risk_model is not None and (np.isfinite(rr_) or np.isfinite(ra_)):
                plan.risk_pop = risk_pop
                plan.risk_model = risk_model
        best_skip = (not bool(uses_dev[best[2]])) if best is not None else False

    direct_cost = inp.N * unit_cost[chain[-1]] if len(chain) > 1 else INF
    if best is not None and direct_cost < best[0]:
        # the cascade does not pay off for this stratum: route everything to the reference
        plan.direct = True
        plan.lo_d = plan.lo_m = -INF
        plan.hi_d = plan.hi_m = INF
        plan.est_cost = float(direct_cost)
        plan.bounds = {"p_lower": p_l, "n": n, "K": K, "alpha": a, "n_feasible": n_feasible}
    elif best is not None:
        cost, G, k, mu, fu, al = best
        plan.rules = tuple(G)
        if learned_pop is not None and any(j >= plan.n_sym for j in G):
            plan.learned_keep_pop = learned_pop
        plan.skip_device = best_skip
        if len(chain) == 1:
            plan.lo_d, plan.hi_d = 0.5 - 1e-12, 0.5
        else:
            plan.lo_d, plan.hi_d, plan.lo_m, plan.hi_m = map(float, (LD[k], HD[k], LM[k], HM[k]))
        plan.est_cost = cost
        plan.bounds = {"p_lower": p_l, "miss_upper": mu, "fp_upper": fu, "acc_lower": al,
                       "n": n, "K": K, "alpha": a, "n_feasible": n_feasible}
    else:
        plan.feasible = False
        plan.rules = ()
        if len(chain) == 1:
            plan.lo_d, plan.hi_d = 0.5 - 1e-12, 0.5
        else:
            plan.lo_d = plan.lo_m = -INF
            plan.hi_d = plan.hi_m = INF
        plan.est_cost = float(inp.N * unit_cost[chain[-1]])
        plan.bounds = {"p_lower": p_l, "n": n, "K": K, "alpha": a, "n_feasible": 0}
    if opt_cost is not None:
        plan.bounds["opt_cost"] = opt_cost
    return plan


def _split_select(cand, uses_dev, y, nA, n, gamma_r, gamma_p, delta_s):
    """Propose on S_A (first nA draws), fixed-sequence test on S_B (the rest)."""
    cond = getattr(_split_select, "test", "ratio") == "cond"
    nq = 2 if cond else 4                                # bounds per tested configuration
    nB = n - nA
    yA, yB = y[:nA], y[nA:]
    kA, kB = int(yA.sum()), int(yB.sum())                # sampled reference positives
    cost = np.concatenate([c[0] for c in cand])
    gi = np.concatenate([[i] * len(c[0]) for i, c in enumerate(cand)])
    ki = np.concatenate([np.arange(len(c[0])) for c in cand])
    mA, fA, aA, mB, fB, aB = (np.concatenate([c[j] for c in cand]) for j in range(2, 8))
    # point estimate of the best cost achievable at all (used to decide whether more samples pay off)
    pt = (mA + mB) <= (1 - gamma_r) * y.sum()
    if gamma_p is not None:
        pt &= (fA + fB) <= (1 - gamma_p) * (aA + aB)
    _split_select.opt_cost = float(cost[pt].min()) if pt.any() else float("inf")
    # proposal: plain (non-simultaneous) bounds at level delta_s on S_A -- only a heuristic
    if cond:
        slack = (1 - gamma_r) - cp_upper(mA, kA, delta_s)
        if gamma_p is not None:
            slack = np.minimum(slack, (1 - gamma_p) - cp_upper(fA, aA, delta_s))
    else:
        pA = float(cp_lower(kA, max(nA, 1), delta_s))
        muA, fuA, alA = cp_upper(mA, max(nA, 1), delta_s), cp_upper(fA, max(nA, 1), delta_s), cp_lower(aA, max(nA, 1), delta_s)
        slack = (1 - gamma_r) * pA - muA
        if gamma_p is not None:
            slack = np.minimum(slack, (1 - gamma_p) * alA - fuA)
    okA = slack >= 0
    if not okA.any() or nB == 0:
        return None, 0, float(cp_lower(kB, max(nB, 1), delta_s / nq)), delta_s / nq
    idx = np.flatnonzero(okA)
    idx = idx[np.lexsort((-slack[idx], cost[idx]))]      # by cost, ties: larger slack first
    front, best_slack = [], -np.inf
    for i in idx:                                        # Pareto frontier: cheaper => less slack
        if slack[i] > best_slack:
            front.append(i)
            best_slack = slack[i]
    order = front[::-1]                                  # safest (most expensive) first
    L = _split_select.bonferroni
    if L:                                                # Bonferroni over <= L frontier points
        pick = np.unique(np.round(np.linspace(0, len(order) - 1, min(L, len(order)))).astype(int))
        order = [order[j] for j in pick]
        a = delta_s / (nq * len(order))
    else:                                                # fixed-sequence testing (with tolerance eta:
        eta = getattr(_split_select, "eta", 0)           # continue past up to eta failed tests,
        a = delta_s / (nq * (eta + 1))                   # each at level delta / (eta + 1))
    p_l = float(cp_lower(kB, nB, a))

    def test(i):
        if cond:                                         # recall / precision tested directly
            mu = float(cp_upper(mB[i], kB, a))           # UCB on 1 - recall
            ok = mu <= 1 - gamma_r
            fu = float(cp_upper(fB[i], aB[i], a)) if gamma_p is not None else 0.0   # UCB on 1 - precision
            if gamma_p is not None:
                ok = ok and fu <= 1 - gamma_p
            return ok, mu, fu, float(aB[i]) / max(nB, 1)
        mu, fu, al = float(cp_upper(mB[i], nB, a)), float(cp_upper(fB[i], nB, a)), float(cp_lower(aB[i], nB, a))
        ok = mu <= (1 - gamma_r) * p_l
        if gamma_p is not None:
            ok = ok and fu <= (1 - gamma_p) * al
        return ok, mu, fu, al

    best, n_pass, n_fail = None, 0, 0
    for i in order:
        ok, mu, fu, al = test(i)
        if not ok:
            if L:
                continue                                 # Bonferroni: test every candidate
            n_fail += 1
            if n_fail > eta:
                break                                    # fixed-sequence testing stops here
            continue
        n_pass += 1
        if best is None or cost[i] < best[0]:
            best = (float(cost[i]), cand[gi[i]][1], int(ki[i]), mu, fu, al)
    return best, n_pass, p_l, a


def choose_n_cal(y_pilot, n_base: int, min_pos: int | None, n_max: int, p_floor: float = 0.005) -> int:
    """Calibration size from an INDEPENDENT pilot sample: enough draws that the calibration
    sample is expected to contain ``min_pos`` reference positives (recall bounds are driven by
    the number of positives; low-selectivity predicates need far more than n_base draws).
    Because the pilot is disjoint from the calibration draws, the CP bounds stay valid
    conditionally on n (Theorem 1 is unaffected)."""
    if not min_pos or y_pilot is None or len(y_pilot) == 0:
        return int(n_base)
    p = max(float(np.mean(y_pilot)), p_floor)
    return int(min(max(n_base, int(np.ceil(min_pos / p))), n_max))


def split_delta(delta: float, levels: list[PrivacyLevel], certified: list[PrivacyLevel] | None = None
                ) -> dict[PrivacyLevel, float]:
    """Union bound across strata. If ``certified`` is given, only those strata can fail (the others
    are answered exactly by their reference, e.g. a single-tier High stratum), so delta is split
    among them alone."""
    c = [lv for lv in (certified if certified is not None else levels)]
    d = delta / max(len(c), 1)
    return {lv: d for lv in levels}


def reference_is_bottleneck(chain, unit_cost: dict) -> bool:
    """True if the stratum's reference is costlier per tuple than some cheaper tier, i.e. a cascade
    can save by keeping tuples away from it (e.g. a rate-limited cloud API). The learned prefilter
    is enabled only then: its pruning is unknown before calibration, so with it the pre-check
    below cannot be used, and without a bottleneck reference calibration rarely pays off."""
    return len(chain) > 1 and unit_cost[chain[-1]] > min(unit_cost[t] for t in chain[:-1])


def cascade_can_pay_off(chain, unit_cost: dict, rule_keep_pop: np.ndarray | None, max_rule_subset: int,
                        n_overhead: int, N: int, rule_keep_pilot_pos: np.ndarray | None = None,
                        gamma_r: float = 0.0) -> bool:
    """Cheap pre-check, made after the pilot and BEFORE the calibration sample is drawn: can a
    cascade on this stratum be cheaper than sending every tuple straight to the reference?

    If the reference is the cheapest tier per tuple (e.g. an API without a rate limit vs. a
    slow device), only free symbolic pruning can save anything; the best conceivable cascade
    then pays the reference for the tuples kept by the rules plus the calibration labels. Rule
    subsets that already drop more than a (1 - gamma_r) fraction of the pilot's positives are
    ignored (they will not be certified). If no cascade can beat the direct plan, calibration is
    skipped. The direct plan is exact, so this decision affects cost only, never the guarantee."""
    if len(chain) < 2:
        return False
    u_ref = unit_cost[chain[-1]]
    if u_ref > min(unit_cost[t] for t in chain[:-1]):
        return True
    kmin = 1.0
    if rule_keep_pop is not None and rule_keep_pop.size:
        from itertools import combinations
        R = rule_keep_pop.shape[1]
        for k in range(1, max_rule_subset + 1):
            for G in combinations(range(R), k):
                G = list(G)
                if rule_keep_pilot_pos is not None and len(rule_keep_pilot_pos):
                    if rule_keep_pilot_pos[:, G].all(axis=1).mean() < gamma_r:
                        continue
                kmin = min(kmin, float(rule_keep_pop[:, G].all(axis=1).mean()))
    return kmin * N + n_overhead < N


# Fixed, data-independent candidate thresholds for the single-threshold certificate (scores are
# probabilities and highly polarised, so the grid is dense near 0).
THRESH_GRID = np.concatenate([[-INF], 10.0 ** np.linspace(-14, 0, 281)[:-1], np.linspace(0.05, 0.95, 19)])
THRESH_GRID = np.unique(THRESH_GRID)


def threshold_certify(level, chain, N: int, y: np.ndarray, proxy: np.ndarray, gamma_r: float, delta_s: float,
                      unit_cost: dict[TierId, float]) -> StratumPlan | None:
    """Single-threshold certificate (the family certified by SUPG/BARGAIN-style methods).

    The strongest non-reference permitted tier scores every tuple; tuples with score < rho are
    rejected and all others are decided by the reference, so precision is 1. On the WHOLE sample
    (no propose/test split), rho runs over the fixed grid from the safest (-inf) upwards and is
    tested by fixed-sequence testing with the direct recall bound UCB_delta(k_m(rho), k+) <= 1-gamma_r
    (k+: sampled reference positives, k_m: those with score < rho); the largest rho before the first
    failure is certified w.p. >= 1-delta. Returns the plan with its estimated cost, or None."""
    if len(chain) < 2:
        return None
    y = np.asarray(y, dtype=bool)
    A = np.nan_to_num(np.asarray(proxy, dtype=float), nan=0.5)
    kp = int(y.sum())
    if kp == 0:
        return None
    Ap = np.sort(A[y])
    k_m = np.searchsorted(Ap, THRESH_GRID, side="left")          # positives with score < rho
    ucb = cp_upper(k_m, kp, delta_s)
    ok = ucb <= 1 - gamma_r
    if not ok[0]:
        return None
    j = len(ok) if ok.all() else int(np.argmin(ok))               # first failure
    rho = float(THRESH_GRID[j - 1])
    frac_up = float((A >= rho).mean())
    proxy_tier = chain[-2]
    plan = StratumPlan(level=level, chain=list(chain))
    if proxy_tier == TierId.DEVICE:                                # chain D -> R
        plan.lo_d = float(np.nextafter(rho, -INF)) if np.isfinite(rho) else -INF
        plan.hi_d = INF
        cost = N * unit_cost[TierId.DEVICE] + N * frac_up * unit_cost[chain[-1]]
    else:                                                          # chain D -> E -> C: device skipped
        plan.skip_device = True
        plan.lo_d, plan.hi_d = -INF, INF
        plan.lo_m = float(np.nextafter(rho, -INF)) if np.isfinite(rho) else -INF
        plan.hi_m = INF
        cost = N * unit_cost[proxy_tier] + N * frac_up * unit_cost[chain[-1]]
    plan.est_cost = float(cost)
    plan.bounds = {"family": "threshold", "rho": rho, "k_pos": kp, "miss_upper": float(ucb[j - 1]),
                   "alpha": delta_s, "n": len(y)}
    return plan
