"""Synthetic traces for tests and dry runs.

Everything produced here is SYNTHETIC. It exercises the code paths and lets one check that
the guarantee machinery behaves as the theory says; it is NOT an experimental result and
must never be reported as one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .llm.backends import MockBackend
from .types import PrivacyLevel


def synthetic_trace(n: int = 5000, selectivity: float = 0.2, privacy_mix=(0.6, 0.3, 0.1),
                    q_device: float = 1.2, q_edge: float = 2.5, edge_flip: float = 0.02,
                    seed: int = 0) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(seed)
    truth = rng.random(n) < selectivity
    tmap = {i: bool(t) for i, t in enumerate(truth)}
    dev = MockBackend("dev", tmap, quality=q_device, seed=seed)
    edg = MockBackend("edge", tmap, quality=q_edge, flip=edge_flip, seed=seed)
    privacy = rng.choice([0, 1, 2], size=n, p=list(privacy_mix))
    nbytes = rng.integers(300, 3000, size=n)
    tokens = (nbytes * 0.3).astype(int)
    s_D = np.array([dev.score_one(i).p_yes for i in range(n)])
    s_E = np.array([edg.score_one(i).p_yes for i in range(n)])
    t_D = 0.03 + 4e-5 * tokens
    # candidate rules: good (keeps all positives, prunes ~55% of negatives), leaky (drops 8%
    # of positives), useless (keeps everything)
    good = truth | (rng.random(n) < 0.45)
    leaky = (truth & (rng.random(n) > 0.08)) | (rng.random(n) < 0.3)
    useless = np.ones(n, dtype=bool)
    tr = pd.DataFrame({"rid": np.arange(n), "privacy": privacy, "tokens": tokens, "nbytes": nbytes,
                       "s_D": s_D, "y_D": s_D >= 0.5, "t_D": t_D, "s_E": s_E, "y_E": s_E >= 0.5,
                       "y_C": truth, "truth": truth,
                       "rule_0": good, "rule_1": leaky, "rule_2": useless})
    rules = [{"type": "keyword_any", "field": "text", "keywords": ["synthetic-good"]},
             {"type": "keyword_any", "field": "text", "keywords": ["synthetic-leaky"]},
             {"type": "keyword_any", "field": "text", "keywords": ["synthetic-useless"]}]
    return tr, rules


def mock_profiles():
    from .replay import Profiles
    from .simulator import TierProfile
    return Profiles(device_per_tuple_s=0.05,
                    edge=TierProfile(rtt_s=0.002, bw_mbps=100, fixed_s=0.08, per_token_s=2e-5, concurrency=2,
                                     batchable=True),
                    cloud=TierProfile(rtt_s=0.04, bw_mbps=20, fixed_s=0.4, per_token_s=3e-4, concurrency=16,
                                      batchable=False, price_in=4.0, price_out=12.0))


_ = PrivacyLevel
