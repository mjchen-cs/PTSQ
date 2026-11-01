#!/usr/bin/env python
"""Prepare a public text table and inject *synthetic* PII so that tuples get privacy levels.

Examples
  python scripts/prepare_data.py --hf Yelp/yelp_review_full --n 30000 --out data/yelp.csv
  python scripts/prepare_data.py --hf fancyzhx/amazon_polarity --text-fields title content --n 30000 --out data/amazon.csv
  python scripts/prepare_data.py --synthetic --n 5000 --out data/toy.csv      # offline dry run

PII is appended as natural sentences ("Contact me at 13812345678.") with configurable rates:
HIGH-inducing (ID card / bank card) and MEDIUM-inducing (phone / email). All generated values
are random and checksum-valid; they belong to no real person.
(Verify dataset names and licences on the HuggingFace hub before use.)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptsq.privacy import PrivacyPolicy, label_privacy  # noqa: E402

_ID_W = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_ID_C = "10X98765432"


def fake_cn_id(rng: random.Random) -> str:
    area = rng.choice(["110101", "310104", "440305", "330106", "510107"])
    birth = f"{rng.randint(1960, 2004)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
    body = area + birth + f"{rng.randint(0, 999):03d}"
    return body + _ID_C[sum(int(a) * b for a, b in zip(body, _ID_W)) % 11]


def fake_card(rng: random.Random) -> str:
    digits = [6, 2] + [rng.randint(0, 9) for _ in range(13)]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return "".join(map(str, digits)) + str((10 - total % 10) % 10)


def fake_phone(rng: random.Random) -> str:
    return rng.choice(["138", "139", "150", "186", "177"]) + "".join(str(rng.randint(0, 9)) for _ in range(8))


def fake_email(rng: random.Random) -> str:
    user = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(5, 9)))
    return f"{user}{rng.randint(1, 99)}@{rng.choice(['example.com', 'mail.example.org', 'test.example.net'])}"


HIGH_TEMPL = ["My ID number is {}.", "Refund to card {} please.", "身份证号 {}。", "Bank card: {}."]
MED_TEMPL = ["Contact me at {}.", "You can reach me via {}.", "联系电话 {}。", "Email me: {}."]


def inject(texts: list[str], high: float, medium: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for t in texts:
        u = rng.random()
        if u < high:
            v = fake_cn_id(rng) if rng.random() < 0.5 else fake_card(rng)
            t = f"{t} {rng.choice(HIGH_TEMPL).format(v)}"
        elif u < high + medium:
            v = fake_phone(rng) if rng.random() < 0.5 else fake_email(rng)
            t = f"{t} {rng.choice(MED_TEMPL).format(v)}"
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", help="HuggingFace dataset name, e.g. Yelp/yelp_review_full")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-fields", nargs="+", default=["text"])
    ap.add_argument("--csv", help="use a local CSV instead of --hf")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--high", type=float, default=0.10, help="fraction of tuples with HIGH PII")
    ap.add_argument("--medium", type=float, default=0.30, help="fraction of tuples with MEDIUM PII")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.synthetic:
        rng = np.random.default_rng(a.seed)
        words = ["food", "service", "delivery", "late", "dirty", "great", "staff", "gift", "broken", "price"]
        texts = [" ".join(rng.choice(words, size=rng.integers(8, 40))) for _ in range(a.n)]
        df = pd.DataFrame({"text": texts, "stars": rng.integers(1, 6, size=a.n)})
    elif a.csv:
        df = pd.read_csv(a.csv).sample(n=a.n, random_state=a.seed)
    else:
        from datasets import load_dataset
        # newer huggingface_hub requires 'namespace/name'; map the legacy canonical names
        a.hf = {"yelp_review_full": "Yelp/yelp_review_full",
                "amazon_polarity": "fancyzhx/amazon_polarity"}.get(a.hf, a.hf)
        ds = load_dataset(a.hf, split=a.split).shuffle(seed=a.seed).select(range(a.n))
        df = ds.to_pandas()
        if len(a.text_fields) > 1:
            df["text"] = df[a.text_fields].astype(str).agg(". ".join, axis=1)
        elif a.text_fields[0] != "text":
            df["text"] = df[a.text_fields[0]]
        if a.hf.endswith("yelp_review_full") and "label" in df:
            df["stars"] = df["label"] + 1
    df = df.reset_index(drop=True)
    df["text"] = inject(df["text"].astype(str).tolist(), a.high, a.medium, a.seed)
    df["rid"] = np.arange(len(df))
    df["privacy"] = label_privacy(df, PrivacyPolicy(), existing_col=None).to_numpy()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keep = ["rid", "text", "privacy"] + [c for c in ("stars",) if c in df]
    df[keep].to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}; privacy mix:",
          df["privacy"].value_counts(normalize=True).sort_index().round(3).to_dict())


if __name__ == "__main__":
    main()
