"""BARGAIN recall-target selection [Zeighami, Shankar, Parameswaran, SIGMOD 2025], re-implemented
from the paper for use as a per-stratum baseline.

Estimation uses the betting test of Waudby-Smith and Ramdas (hedged capital with a predictable
plug-in bet): for i.i.d. X_i in [0, 1] with mean mu, the capital
    K_t = prod_{i<=t} (1 + lam_i (X_i - m)),   0 <= lam_i < 1/m,
is a nonnegative supermartingale when mu <= m, so P(sup_t K_t >= 1/alpha) <= alpha (Ville). The test
declares mu > m (the target is met) when the capital reaches 1/alpha; it is anytime valid.

Two recall-target (RT) variants, as in the paper:
  * BARGAIN_R-U: uniform sample; the selected threshold is the largest candidate rho whose recall,
    tested on the sampled positives (X = 1[score >= rho]), passes at level delta. Exact guarantee.
  * BARGAIN_R-A: first a binary search (half of the budget and of delta) for a cutoff rho_P below
    which the positive density is estimated to be < beta; then R-U on the tuples above rho_P. With
    beta > 0 the guarantee only covers positives in the dense region (a relaxed guarantee).
Scores are rank-normalized to [0, 1] for the binary search (LLM scores are highly polarized).
"""
from __future__ import annotations

import numpy as np


def betting_reject(x: np.ndarray, m: float, alpha: float, c: float = 0.5) -> bool:
    """True if the betting test rejects H0: mean(x) <= m, i.e. concludes mean > m (prob <= alpha
    under H0). x in [0, 1], processed in the given (i.i.d.) order."""
    x = np.asarray(x, dtype=float)
    t = len(x)
    if t == 0 or m >= 1:
        return False
    if m <= 0:
        return bool((x > 0).any())
    i = np.arange(1, t + 1)
    mu_hat = (0.5 + np.cumsum(x)) / (i + 1)                   # mu_i
    sig2 = (0.25 + np.cumsum((x - mu_hat) ** 2)) / (i + 1)     # sigma_i^2
    var_prev = np.maximum(np.concatenate([[0.25], sig2[:-1]]), 1e-6)   # predictable: sigma_{i-1}^2
    lam = np.sqrt(2 * np.log(1 / alpha) / (var_prev * i * np.log(1 + i)))
    lam = np.minimum(lam, c / m)                     # keeps every factor > 0
    logk = np.cumsum(np.log1p(lam * (x - m)))
    return bool(logk.max() >= np.log(1 / alpha))


def _ru_threshold(A, y, S, gamma, alpha, cands):
    """Largest candidate rho whose recall passes the betting test on the sampled positives."""
    pos = S[y[S]]
    if len(pos) == 0:
        return -np.inf
    for rho in cands:                                # descending
        if betting_reject((A[pos] >= rho).astype(float), gamma, alpha):
            return rho
    return -np.inf


def bargain_rt(A: np.ndarray, y: np.ndarray, budget: int, gamma: float, delta: float, rng,
               beta: float = 0.0, r: float = 0.05, n_cands: int = 200):
    """Select a threshold tau on proxy scores A (recall target gamma, failure prob delta) with an
    oracle budget. Returns (tau, drawn): drawn are the labelled positions (with repetition)."""
    Ns = len(A)
    # rank-normalized scores with random tie breaking
    order = np.lexsort((rng.random(Ns), A))
    u = np.empty(Ns)
    u[order] = (np.arange(Ns) + 0.5) / Ns
    drawn = []
    lo_idx = np.arange(Ns)
    d_ru = delta
    k_ru = budget
    if beta > 0:
        k1, k_ru, d1, d_ru = budget // 2, budget - budget // 2, delta / 2, delta / 2
        rho, rho_p = 0.5, 0.0
        while k1 > 0 and rho < 1 - r / 2:
            win = np.flatnonzero((u >= rho) & (u < rho + r))
            if len(win) == 0:
                break
            xs, low = [], False
            while k1 > 0:
                j = int(win[rng.integers(0, len(win))])
                drawn.append(j); k1 -= 1
                xs.append(1.0 - float(y[j]))          # density < beta  <=>  mean(1 - y) > 1 - beta
                if betting_reject(np.array(xs), 1 - beta, d1):
                    low = True
                    break
            if not low:
                break
            rho_p, rho = rho, (1 + rho) / 2
        lo_idx = np.flatnonzero(u >= rho_p)
    S = lo_idx[rng.integers(0, len(lo_idx), size=max(k_ru, 0))] if len(lo_idx) else np.array([], int)
    drawn = np.concatenate([np.asarray(drawn, dtype=int), S.astype(int)])
    sub = A[lo_idx]
    cands = np.unique(np.quantile(sub, np.linspace(0, 1, n_cands + 1)))[::-1] if len(sub) else np.array([])
    tau = _ru_threshold(A, y, S.astype(int), gamma, d_ru, cands)
    if beta > 0 and np.isfinite(tau):
        pass                                          # returned set: A >= tau (within the dense region)
    return tau, drawn
