"""Learned disagreement risk (component of propose-and-test certification).

A light model that predicts, for one tuple, whether the middle tier (the private edge model)
disagrees with the stratum's permitted reference. It is trained only on the proposal half S_A
of the calibration sample, so the certification half S_B stays independent of it, and it is
used only to decide which tuples the edge tier may finalize; every configuration that uses it
is certified on S_B like any other (Theorem 2). A useless risk model therefore costs nothing
in accuracy: the configurations that use it simply do not pass.

Features are computed on the device: TF-IDF of the text (fitted without labels on the
query's candidate tuples) plus the edge score and the log length in characters. The device
score is deliberately not a feature: plans for LOW/MEDIUM strata usually skip the device, so
using it would require scoring every tuple there. The learned pre-filter drops s_E as well,
so pruning needs no model call at all.
"""
from __future__ import annotations

import numpy as np


def text_features(texts, numeric: np.ndarray, max_features: int = 30000):
    """Sparse feature matrix [TF-IDF(text) | numeric]. Fitting the vocabulary uses no labels."""
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features, sublinear_tf=True)
    X = vec.fit_transform([t if isinstance(t, str) else "" for t in texts])
    return hstack([X, csr_matrix(np.asarray(numeric, dtype=float))]).tocsr()


def fit_risk(X, d, min_pos: int = 5, C: float = 4.0):
    """Logistic regression on (features, disagreement). Returns a callable X -> risk in [0, 1],
    or None if S_A contains too few disagreements to learn from."""
    d = np.asarray(d, dtype=bool)
    if d.sum() < min_pos or (~d).sum() < min_pos:
        return None
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X, d)
    return lambda Z: m.predict_proba(Z)[:, 1]


def crossfit_risk(X_A, d_A, n_folds: int = 2, seed: int = 0):
    """Fit on all of S_A (for S_B and the population) and return out-of-fold risks for S_A, so
    that the proposal step judges risk-based configurations on tuples the model has not seen
    (in-sample risks are optimistic and would make fixed-sequence testing stop early)."""
    model = fit_risk(X_A, d_A)
    if model is None:
        return None, None
    d_A = np.asarray(d_A, dtype=bool)
    oof = np.empty(len(d_A))
    fold = np.random.default_rng(seed).integers(0, n_folds, size=len(d_A))
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        m = fit_risk(X_A[np.flatnonzero(tr)], d_A[tr])
        oof[te] = m(X_A[np.flatnonzero(te)]) if m is not None else model(X_A[np.flatnonzero(te)])
    return model, oof



def query_features(texts, cache: str | None = None):
    """Label-free part of the features for the live executor: (TF-IDF matrix, log length).

    Fitting the bigram vocabulary needs ~1 GB of peak memory, which does not fit next to the
    device model in the Jetson's 8 GB. ``scripts/precompute_features.py`` therefore runs the
    same computation on the device in a separate process before the query and stores it in
    ``cache`` (an .npz keyed by a hash of the texts); it is used only if the texts match."""
    import hashlib
    t = [x if isinstance(x, str) else "" for x in texts]
    h = hashlib.sha1("\x00".join(t).encode("utf-8", "ignore")).hexdigest()
    loglen = np.log1p(np.array([len(x) for x in t], dtype=float))
    if cache:
        from pathlib import Path
        from scipy.sparse import load_npz
        p = Path(cache)
        if p.exists() and p.with_suffix(".sha1").exists() and p.with_suffix(".sha1").read_text().strip() == h:
            return load_npz(p).tocsr(), loglen
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    X = vec.fit_transform(t).tocsr()
    if cache:
        from pathlib import Path
        from scipy.sparse import save_npz
        p = Path(cache)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_npz(p, X)
        p.with_suffix(".sha1").write_text(h)
    return X, loglen


def assemble(Xt, loglen, s_mid, rows):
    """Rows of [TF-IDF | s_E | log length] (s_E may be a placeholder where it is not used)."""
    from scipy.sparse import csr_matrix, hstack
    rows = np.asarray(rows)
    num = np.c_[np.asarray(s_mid, dtype=float), loglen[rows]]
    return hstack([Xt[rows], csr_matrix(num)]).tocsr()
