"""Exact binomial (Clopper-Pearson) confidence bounds, vectorised."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta


def cp_upper(k, n, alpha):
    """One-sided (1-alpha) upper confidence bound on a binomial proportion."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.where(k >= n, 1.0, beta.ppf(1 - alpha, k + 1, np.maximum(n - k, 1e-12)))
    return np.clip(np.nan_to_num(out, nan=1.0), 0.0, 1.0)


def cp_lower(k, n, alpha):
    """One-sided (1-alpha) lower confidence bound on a binomial proportion."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.where(k <= 0, 0.0, beta.ppf(alpha, np.maximum(k, 1e-12), n - k + 1))
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


def required_sample_size(selectivity: float, gamma_r: float, delta: float, n_configs: int) -> int:
    """Rough sample size so that a zero-miss configuration is certifiable:
    n >~ ln(1/a) / ((1-gamma_r) * selectivity), a = delta / (3K+1)."""
    a = delta / (3 * n_configs + 1)
    return int(np.ceil(np.log(1 / a) / max((1 - gamma_r) * selectivity, 1e-9)))
