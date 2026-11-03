#!/usr/bin/env python
"""Data-driven illustration figures for Sections I, IV, V and VIII-B, computed from the PILOT
traces and the measured tier profiles (no synthetic numbers):

  python scripts/make_design_figs.py --pilot results/traces_q35 --vague14 results/traces \
         --vague32 results/traces_e32 --profiles results/profiles.json --out paper/figs

  fig_motivation.pdf   tier cost vs agreement (Q1) and the privacy mix with the tiers each level may reach
  fig_agreement.pdf    agreement per query, effect of predicate phrasing, confident errors
  fig_calibration.pdf  score distributions with the fixed threshold grid, and certification of the grid
  fig_rules.pdf        LLM-proposed rules: pruning power vs recall of reference positives
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figstyle import COLW, FULLW, TIER, plt, save, style, subcaption  # noqa: E402

from ptsq.bounds import cp_lower, cp_upper  # noqa: E402
from ptsq.cascade import DEV_HI, DEV_LO, MID_HI, MID_LO, StratumInput, calibrate_stratum, rule_subsets  # noqa: E402
from ptsq.config import load_profiles  # noqa: E402
from ptsq.replay import ReplayParams, unit_costs  # noqa: E402
from ptsq.types import PrivacyLevel, TierId  # noqa: E402

QNAME = {"yelp_delay": "Q1 delay", "yelp_hygiene": "Q2 hygiene", "yelp_staff_praise": "Q3 staff",
         "amazon_defect": "Q4 defect", "amazon_gift": "gift (vague pilot)"}
QORDER = ["yelp_delay", "yelp_hygiene", "yelp_staff_praise", "amazon_defect"]   # pilot of Q5 used an earlier phrasing


def rp(pred, y):
    tp = float((pred & y).sum())
    return tp / max(y.sum(), 1), tp / max(pred.sum(), 1)


def f1(r, p):
    return 2 * r * p / (r + p) if r + p else 0.0


# ----------------------------------------------------------------------------------------
def fig_motivation(tr: pd.DataFrame, prof, out: Path, main_csv: Path = Path("results/exp/main.csv")):
    """(a) the three tiers: time vs. agreement with the cloud (Q1 pilot), with the levels allowed to
    use each tier; (b) end-to-end cost vs. accuracy of the methods (all queries, main experiment)."""
    y = tr.y_C.astype(bool)
    tok = float(tr.tokens.mean())
    d_lat = float(tr.t_D.mean()) if "t_D" in tr else prof.device_per_tuple_s
    e, c = prof.edge, prof.cloud
    e_lat = (e.fixed_s + e.per_token_s * tok * 32) / 32
    c_lat = max(c.fixed_s + c.rtt_s + c.per_token_s * tok, (tok + 1) * 60 / c.tpm if c.tpm else 0)
    c_cny = c.price_in * tok / 1e6 * 1000
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 1.62))
    ax = axes[0]
    pts = [("D", "o", d_lat * 1000, f1(*rp(tr.s_D >= 0.5, y)), "Device 1.5B\nall levels", (1.0, -0.20), "center"),
           ("E", "s", e_lat * 1000, f1(*rp(tr.s_E >= 0.5, y)), "Edge 32B\nLow, Med", (1.0, -0.21), "center"),
           ("C", "D", c_lat * 1000, 1.0, f"Cloud 122B\nLow only\n+{c_cny:.2f} CNY", (0.62, -0.30), "center")]
    for t, mk, sec, f, lab, (fx, dy), ha in pts:
        ax.scatter([sec], [f], s=40, marker=mk, color=TIER[t], edgecolor="k", lw=0.4, zorder=3)
        ax.text(sec * fx, f + dy, lab, ha=ha, va="center", fontsize=5.4, color=TIER[t], linespacing=0.95)
    ax.set_xscale("log"); ax.set_xlim(40, 3000); ax.set_ylim(0.2, 1.1); ax.minorticks_off()
    ax.set_xticks([100, 1000], ["100", "1k"])
    ax.set_xlabel("s per 1,000 tuples"); ax.set_ylabel("F1 vs. cloud")
    subcaption(ax, "(a) Tiers (Q1, pilot)", y=-0.40)
    # (b) methods: cost relative to Static-Ref vs. F1 against the permitted reference
    ax = axes[1]
    m = pd.read_csv(main_csv)
    ref = m[m.method == "Static-Ref"].groupby("qid")["cost"].mean()
    g = m.groupby(["method", "qid"])[["cost", "recall", "precision", "privacy_violations"]].mean()
    rc = (g["cost"] / ref.reindex(g.index.get_level_values(1)).values).groupby(level=0).mean()
    acc = g.groupby(level=0)[["recall", "precision", "privacy_violations"]].mean()
    acc["F1"] = 2 * acc.recall * acc.precision / (acc.recall + acc.precision)
    ax.add_patch(plt.Rectangle((0, 0.8), 1.0, 0.32, color="#d5f5e3", lw=0, zorder=0))
    ax.text(0.55, 1.075, "target met, cheaper", fontsize=5.2, color="#2e8b57", va="center")
    lab = {"Cloud-All": ("Cloud-All\nviolates\nthe policy", 1.50, 0.68, "center"),
           "Static-Ref": ("Static-Ref", 1.06, 1.0, "left"),
           "Cert-2Tier": ("Cert-2Tier", 0.67, 0.70, "left"),
           "Conf-Cascade": ("Conf-Cascade", 0.04, 0.635, "left"),
           "Device-Only": ("Device-Only", 0.33, 0.47, "left"),
           "SUPG-JT": ("SUPG-JT\n(asymptotic)", 0.02, 0.83, "left"),
           "BARGAIN-JT": ("BARGAIN-JT", 0.47, 0.875, "left"),
           "PTSQ": ("PTSQ", 0.37, 1.025, "left")}
    for k, n in enumerate(["Cloud-All", "Static-Ref", "Cert-2Tier", "Conf-Cascade", "Device-Only", "SUPG-JT", "BARGAIN-JT", "PTSQ"]):
        if n not in rc.index:
            continue
        col, fill, ht, mk = style(n, k)
        big = n == "PTSQ"
        ax.scatter([rc[n]], [acc.F1[n]], s=70 if big else 26, marker="*" if big else mk, color=col,
                   edgecolor="k", lw=0.4, zorder=3)
        if n == "Cloud-All":
            ax.scatter([rc[n]], [acc.F1[n]], s=60, marker="x", color="k", lw=0.8, zorder=4)
        t, tx, ty, ha = lab[n]
        ax.text(tx, ty, t, fontsize=5.4, ha=ha, va="center", color=col, linespacing=0.95,
                weight="bold" if big else "normal")
    ax.set_xlim(0, 1.8); ax.set_ylim(0.35, 1.12)
    ax.set_xlabel("cost / Static-Ref"); ax.set_ylabel(r"F1 vs. permitted ref.")
    subcaption(ax, "(b) Methods (Q1-Q8)", y=-0.40)
    fig.tight_layout(w_pad=0.8)
    save(fig, out / "fig_motivation.pdf")


# ----------------------------------------------------------------------------------------
def fig_agreement(T: dict, vague: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(FULLW, 1.75), gridspec_kw={"width_ratios": [1.6, 1, 1.1]})
    # (a) per-query recall / precision of edge and device
    ax = axes[0]
    qs = [q for q in QORDER if q in T]
    w = 0.19
    series = [("Edge recall", "s_E", 0, TIER["E"], "#fdd0a2", "////"), ("Edge precision", "s_E", 1, TIER["E"], "white", "xxx"),
              ("Device recall", "s_D", 0, TIER["D"], "#dadaeb", "\\\\\\\\"), ("Device precision", "s_D", 1, TIER["D"], "white", "...")]
    x = np.arange(len(qs))
    for k, (lab, col, idx, ec, fc, ht) in enumerate(series):
        vals = [rp(T[q][col] >= 0.5, T[q].y_C.astype(bool))[idx] for q in qs]
        ax.bar(x + (k - 1.5) * w, vals, w, color=fc, edgecolor=ec, hatch=ht, linewidth=0.6, label=lab)
    ax.axhline(0.8, color="k", lw=0.6, ls=":")
    ax.text(len(qs) - 0.5, 0.81, r"$\gamma=0.8$", fontsize=6, ha="right", va="bottom")
    ax.set_xticks(x, [QNAME[q] for q in qs])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("vs. cloud reference")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.6, columnspacing=0.8,
              handlelength=1.2, frameon=False)
    subcaption(ax, "(a) Agreement of cheaper tiers (pilot)")

    # (b) predicate phrasing
    ax = axes[1]
    labs, vals = [], []
    for lab, tr in vague.items():
        r, p = rp(tr.s_E >= 0.5, tr.y_C.astype(bool))
        labs.append(lab); vals.append((r, p))
    x = np.arange(len(labs))
    ax.bar(x - 0.18, [v[0] for v in vals], 0.36, color="#fdd0a2", edgecolor=TIER["E"], hatch="////", lw=0.6, label="recall")
    ax.bar(x + 0.18, [v[1] for v in vals], 0.36, color="white", edgecolor=TIER["E"], hatch="xxx", lw=0.6, label="precision")
    for xi, v in zip(x, vals):
        ax.text(xi, 0.06, f"F1\n{f1(*v):.2f}", ha="center", fontsize=6, bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_xticks(x, labs, fontsize=6)
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8, frameon=False)
    subcaption(ax, "(b) Vague vs. precise predicate")

    # (c) confident errors
    ax = axes[2]
    bins = np.linspace(0, 1, 21)
    allv = pd.concat([T[q] for q in qs])
    dis = allv[(allv.s_E >= 0.5) != allv.y_C.astype(bool)].s_E
    agr = allv[(allv.s_E >= 0.5) == allv.y_C.astype(bool)].s_E
    ax.hist(agr, bins=bins, color="#c6dbef", edgecolor="#1f5fa8", lw=0.5, label="agrees with ref.")
    ax.hist(dis, bins=bins, color="#f5b7b1", edgecolor="#c0392b", lw=0.5, hatch="////", label="disagrees")
    ax.set_yscale("log")
    ax.set_xlabel(r"edge score $s_E$")
    ax.set_ylabel("tuples (log)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8, frameon=False)
    share = float(((dis <= 0.01) | (dis >= 0.99)).mean())
    ax.text(0.5, 0.62, f"{share:.0%} of disagreements\nhave $s_E<0.01$ or $>0.99$", transform=ax.transAxes,
            ha="center", fontsize=6, bbox=dict(fc="white", ec="0.6", lw=0.4, pad=1.5))
    subcaption(ax, "(c) Edge errors are confident")
    fig.tight_layout(w_pad=0.5)
    save(fig, out / "fig_agreement.pdf")


# ----------------------------------------------------------------------------------------
def certify_grid(inp, gamma, delta_s, uc, max_rule_subset=2):
    """Bounds and cost of every configuration (same arithmetic as cascade.calibrate_stratum)."""
    y = inp.y_sample
    sd, sm = inp.s_dev_sample, inp.s_mid_sample
    n = len(y)
    LD, HD, LM, HM = (a.ravel() for a in np.meshgrid(np.array(DEV_LO), np.array(DEV_HI), np.array(MID_LO),
                                                      np.array(MID_HI), indexing="ij"))
    ok = (LD < HD) & (LM < HM)
    LD, HD, LM, HM = LD[ok], HD[ok], LM[ok], HM[ok]
    R = inp.rule_keep_sample.shape[1]
    subsets = rule_subsets(R, max_rule_subset)
    K = len(LD) * len(subsets)
    a = delta_s / (3 * K + 1)
    p_l = float(cp_lower(y.sum(), n, a))
    S, M = sd[None, :], sm[None, :]
    rej_d, acc_d = S <= LD[:, None], S >= HD[:, None]
    up = ~rej_d & ~acc_d
    rej_m, acc_m = up & (M <= LM[:, None]), up & (M >= HM[:, None])
    to_ref = up & ~rej_m & ~acc_m
    rej, acc = rej_d | rej_m, acc_d | acc_m
    acc_all = acc | (to_ref & y[None, :])
    rows = []
    for G in subsets:
        keep = np.ones(n, bool)
        for j in G:
            keep &= inp.rule_keep_sample[:, j]
        pr = ~keep[None, :]
        miss = ((pr | (rej & ~pr)) & y[None, :]).sum(1)
        fp = ((acc & ~pr) & ~y[None, :]).sum(1)
        accn = (acc_all & ~pr).sum(1)
        m_u, f_u, a_l = cp_upper(miss, n, a), cp_upper(fp, n, a), cp_lower(accn, n, a)
        rec_lb = 1 - m_u / max(p_l, 1e-12)
        prec_lb = 1 - f_u / np.maximum(a_l, 1e-12)
        cert = (m_u <= (1 - gamma) * p_l) & (f_u <= (1 - gamma) * a_l)
        to_c = (to_ref & keep[None, :]).sum(1) / n
        rows.append(pd.DataFrame({"rec_lb": rec_lb, "prec_lb": prec_lb, "cert": cert, "to_cloud": to_c,
                                  "G": [G] * len(LD)}))
    return pd.concat(rows, ignore_index=True), K


def fig_calibration(tr: pd.DataFrame, rules: list, prof, out: Path, gamma=0.8, n=4000, seed=0):
    low = tr[tr.privacy == 0].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    loc = rng.integers(0, len(low), size=n)
    rk = low[[f"rule_{i}" for i in range(len(rules))]].to_numpy(bool)
    y = low.y_C.to_numpy(bool)
    inp = StratumInput(level=PrivacyLevel.LOW, chain=[TierId.DEVICE, TierId.EDGE, TierId.CLOUD], N=len(low) * 30,
                       sample_idx=loc, y_sample=y[loc], s_dev_sample=low.s_D.to_numpy()[loc],
                       s_mid_sample=low.s_E.to_numpy()[loc], rule_keep_sample=rk[loc], rule_keep_pop=rk)
    P = ReplayParams()
    uc = unit_costs(prof, float(low.tokens.mean()), float(low.nbytes.mean()), prof.device_per_tuple_s, P.weights)
    plan = calibrate_stratum(inp, gamma, gamma, 0.05 / 3, uc)
    grid, K = certify_grid(inp, gamma, 0.05 / 3, uc)

    fig, axes = plt.subplots(1, 3, figsize=(FULLW, 1.75), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    bins = np.linspace(0, 1, 26)
    for ax, col, lo_grid, hi_grid, lo, hi, lab, tag in (
            (axes[0], "s_D", DEV_LO, DEV_HI, plan.lo_d, plan.hi_d, r"device score $s_D$", "(a) Device scores and grid"),
            (axes[1], "s_E", MID_LO, MID_HI, plan.lo_m, plan.hi_m, r"edge score $s_E$", "(b) Edge scores and grid")):
        ax.hist(low[col][~y], bins=bins, color="#c6dbef", edgecolor="#1f5fa8", lw=0.5, label="ref. negative")
        ax.hist(low[col][y], bins=bins, color="#f5b7b1", edgecolor="#c0392b", lw=0.5, hatch="////", label="ref. positive")
        for t in list(lo_grid) + list(hi_grid):
            if np.isfinite(t):
                ax.axvline(t, color="0.55", lw=0.35, ls=":")
        for t, c in ((lo, "#2e8b57"), (hi, "#c0392b")):
            if np.isfinite(t):
                ax.axvline(t, color=c, lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.legend(loc="upper center", fontsize=5.6)
        subcaption(ax, tag)
    axes[0].set_ylabel("tuples (log)")
    ax = axes[2]
    g = grid.sample(n=min(len(grid), 6000), random_state=0)
    ax.scatter(g.to_cloud[~g.cert], g.rec_lb[~g.cert].clip(-0.05), s=2, color="0.75", label="not certified", rasterized=True)
    ax.scatter(g.to_cloud[g.cert], g.rec_lb[g.cert], s=3, color="#2e8b57", label="certified", rasterized=True)
    ch = grid[grid.cert]
    if plan.feasible and not plan.direct and len(ch):
        best = ch.loc[ch.to_cloud.idxmin()]
        ax.scatter([best.to_cloud], [best.rec_lb], s=60, marker="*", color="#c0392b", edgecolor="k", lw=0.4,
                   zorder=4, label="selected")
    ax.axhline(gamma, color="k", lw=0.6, ls=":")
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("fraction of tuples sent to cloud")
    ax.set_ylabel("certified recall")
    ax.legend(loc="lower right", fontsize=5.6, markerscale=1.5)
    subcaption(ax, f"(c) {K:,} configurations, $n$={n:,}")
    fig.tight_layout(w_pad=0.5)
    save(fig, out / "fig_calibration.pdf")
    return plan, K


# ----------------------------------------------------------------------------------------
def fig_rules(T: dict, R: dict, out: Path):
    fig, ax = plt.subplots(figsize=(COLW * 0.62, 1.55))
    marks = ["o", "s", "^", "D", "v"]
    cols = ["#c0392b", "#1f5fa8", "#2e8b57", "#e67e22", "#7d3c98"]
    ax.axhspan(0.95, 1.02, xmin=0, xmax=0.9 / 1.02, color="#e8f6ef", zorder=0)
    ax.text(0.03, 0.965, "useful", fontsize=6, color="#2e8b57", va="top")
    for k, q in enumerate([q for q in QORDER if q in T]):
        tr, y = T[q], T[q].y_C.astype(bool)
        pts = [(tr[f"rule_{i}"].mean(), tr.loc[y, f"rule_{i}"].mean()) for i in range(len(R[q]))]
        if pts:
            ax.scatter(*zip(*pts), marker=marks[k], color=cols[k], s=16, edgecolor="k", lw=0.3, label=QNAME[q], zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("fraction of tuples kept")
    ax.set_ylabel("recall of ref. positives")
    ax.legend(loc="lower right", fontsize=5.4, handletextpad=0.2, borderpad=0.25)
    save(fig, out / "fig_rules.pdf")


def load_dir(d: Path, only=None):
    T, R = {}, {}
    for f in sorted(d.glob("*.csv")):
        q = f.stem
        if only and q not in only:
            continue
        T[q] = pd.read_csv(f)
        rp_ = d / f"{q}.rules.json"
        R[q] = json.loads(rp_.read_text()) if rp_.exists() else []
    return T, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="results/traces_q35")
    ap.add_argument("--vague14", default="results/traces", help="pilot traces with the 14B edge (yelp_delivery)")
    ap.add_argument("--vague32", default="results/traces_e32", help="pilot traces with the 32B edge, 72B reference")
    ap.add_argument("--profiles", default="results/profiles.json")
    ap.add_argument("--out", default="paper/figs")
    ap.add_argument("--main", default="results/exp/main.csv", help="main-suite log for Fig. 1(b)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prof = load_profiles(a.profiles)
    T, R = load_dir(Path(a.pilot), QORDER)
    fig_motivation(T["yelp_delay"], prof, out, Path(a.main))
    vague = {}
    v14 = Path(a.vague14) / "yelp_delivery.csv"
    v32 = Path(a.vague32) / "yelp_delivery.csv"
    p32 = Path(a.vague32) / "yelp_delay.csv"
    for lab, f in (("vague\n14B", v14), ("vague\n32B", v32), ("precise\n32B", p32)):
        if f.exists():
            vague[lab] = pd.read_csv(f)
    fig_agreement(T, vague, out)
    plan, K = fig_calibration(T["yelp_delay"], R["yelp_delay"], prof, out)
    fig_rules(T, R, out)
    print(f"[design figs] written to {out}; calibration example: |Theta|={K}, plan rules={plan.rules}, "
          f"thresholds D=({plan.lo_d},{plan.hi_d}) E=({plan.lo_m},{plan.hi_m}), feasible={plan.feasible}, "
          f"direct={plan.direct}")


if __name__ == "__main__":
    main()
