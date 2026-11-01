#!/usr/bin/env python
"""Turn results/exp/*.csv into the paper's figures (paper/figs) and tables (paper/tables).

  python scripts/make_figures.py --exp results/exp --paper paper
  python scripts/make_figures.py --placeholders --paper paper     # regenerate empty placeholders

Every figure/table the paper includes is produced here, so the PDF is always in sync with the
CSV files. Numbers quoted in the text are marked \\tbd{} in the .tex and must be updated by hand
from tables/summary.tex, which this script also writes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figstyle import COLW, FULLW, PALETTE, TIER, bars, plt, save, style, subcaption  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK2 = "0.3"
MAIN_ORDER = ["Cloud-All", "Device-Only", "Static-Ref", "Conf-Cascade", "Cert-2Tier", "SUPG-JT",
              "BARGAIN-JT (relaxed)", "BARGAIN-JT", "PTSQ"]
COMPLIANT = ["Static-Ref", "Conf-Cascade", "Cert-2Tier", "SUPG-JT", "BARGAIN-JT", "PTSQ"]
QLABEL = {"yelp_delay": "Q1", "yelp_hygiene": "Q2", "yelp_staff_praise": "Q3", "amazon_defect": "Q4",
          "amazon_gift": "Q5"}
WATERMARK = None


def _wm(fig):
    if WATERMARK:
        fig.text(0.5, 0.5, WATERMARK, ha="center", va="center", fontsize=14, color="red", alpha=0.25,
                 rotation=12, weight="bold")


def placeholder_fig(path: Path, w: float, h: float, label: str):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.96, fill=False, ls="--", ec="0.4", lw=0.8,
                               transform=ax.transAxes))
    ax.text(0.5, 0.5, f"PLACEHOLDER\n{label}\n(run scripts/make_figures.py after the experiments)",
            ha="center", va="center", color="0.3", fontsize=8, transform=ax.transAxes)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def tex_table(path: Path, header: list[str], rows: list[list[str]], colspec: str, caption_note: str = ""):
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption_note:
        lines.append(caption_note)
    path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------- figures
def _qlab(q):
    return QLABEL.get(q, q)


def _grouped(ax, per_q: pd.DataFrame, names, ylabel=None, log=False, err: pd.DataFrame | None = None):
    """per_q: index=method, columns=query; grouped hatched bars, one group per query.
    err (same shape): whisker half-length, e.g. the standard deviation over seeds."""
    qs = sorted(per_q.columns, key=_qlab)
    w = 0.8 / max(len(names), 1)
    for k, n in enumerate(names):
        c, f, ht, _ = style(n, k)
        vals = per_q.loc[n].reindex(qs).values if n in per_q.index else np.full(len(qs), np.nan)
        ev = None
        if err is not None and n in err.index:
            ev = err.loc[n].reindex(qs).values
        ax.bar(np.arange(len(qs)) + (k - (len(names) - 1) / 2) * w, vals, w, color=f, edgecolor=c, hatch=ht,
               linewidth=0.6, label=n, yerr=ev, error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": "0.2"})
    ax.set_xticks(np.arange(len(qs)), [_qlab(q) for q in qs])
    if log:
        ax.set_yscale("log")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def fig_main(df: pd.DataFrame, path: Path):
    names = [m for m in COMPLIANT if m in set(df.method)]
    base = df[df.method == "Static-Ref"].groupby("qid")[["latency_s", "money", "cost"]].mean()
    fig, axes = plt.subplots(1, 4, figsize=(FULLW, 1.75), gridspec_kw={"width_ratios": [1, 1, 1, 0.95]})
    for ax, (m, lab, tag) in zip(axes[:3], [("cost", "total cost / Static-Ref", "(a) Total cost"),
                                             ("latency_s", "latency / Static-Ref", "(b) End-to-end latency"),
                                             ("money", "API money / Static-Ref", "(c) Cloud spending")]):
        per_q = df.groupby(["method", "qid"])[m].mean().unstack().div(base[m].replace(0, np.nan), axis=1)
        sd_q = df.groupby(["method", "qid"])[m].std().unstack().div(base[m].replace(0, np.nan), axis=1)
        _grouped(ax, per_q, names, lab, err=sd_q)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        subcaption(ax, tag)
    ax = axes[3]
    for k, n in enumerate(names + [x for x in ("Device-Only",) if x in set(df.method)]):
        c, f, ht, mk = style(n, k)
        d = df[df.method == n]
        ax.scatter(d.recall, d.precision, s=10, marker=mk, color=c, alpha=0.7, edgecolor="k", lw=0.2, label=n)
    ax.axhline(0.8, color="k", lw=0.5, ls=":"); ax.axvline(0.8, color="k", lw=0.5, ls=":")
    ax.set_xlabel(r"Rec$_R$"); ax.set_ylabel(r"Prec$_R$")
    subcaption(ax, "(d) Accuracy per run")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(names), bbox_to_anchor=(0.5, 1.07), frameon=False)
    fig.tight_layout(w_pad=0.6)
    _wm(fig)
    save(fig, path)


def fig_decisions(df: pd.DataFrame, path: Path):
    """Work done by PTSQ per query, as fractions of N (a tuple can visit several tiers)."""
    d = df[df.method == "PTSQ"].groupby("qid")[["N", "n_pruned", "n_labelled", "n_edge", "n_cloud"]].mean()
    qs = list(d.index)
    parts = [("pruned by rules", d.n_pruned / d.N, TIER["rule"], "#c7e9c0", "////"),
             ("calibration labels", d.n_labelled / d.N, "#555555", "#e0e0e0", "xxx"),
             ("sent to edge", d.n_edge / d.N, TIER["E"], "#fdd0a2", "\\\\\\\\"),
             ("sent to cloud", d.n_cloud / d.N, TIER["C"], "#c6dbef", "...")]
    fig, ax = plt.subplots(figsize=(COLW, 1.4))
    w = 0.2
    for k, (lab, v, c, f, ht) in enumerate(parts):
        ax.bar(np.arange(len(qs)) + (k - 1.5) * w, v.values, w, color=f, edgecolor=c, hatch=ht, lw=0.6, label=lab)
    ax.set_xticks(np.arange(len(qs)), [_qlab(q) for q in qs])
    ax.set_ylabel("fraction of $N$")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=5.8, columnspacing=0.8)
    ax.grid(axis="x", visible=False)
    _wm(fig)
    save(fig, path)


def fig_guarantee(df: pd.DataFrame, path: Path):
    """(a) per-stratum recall over all runs; (b) fraction of runs that miss a per-stratum
    recall or precision target, per query (one marker per query), against delta."""
    rc = [c for c in df.columns if c.startswith("recall_") and c != "recall_abs"]
    pc = [c for c in df.columns if c.startswith("precision_") and c != "precision_abs"]
    df = df.copy()
    df["fail"] = df[rc].lt(df.gamma, axis=0).any(axis=1) | df[pc].lt(df.gamma, axis=0).any(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 1.75), gridspec_kw={"width_ratios": [1.35, 1]})
    gammas = sorted(df.gamma.unique())
    meths = [m for m in ("PTSQ", "BARGAIN-JT", "SUPG-JT", "Conf-Cascade") if m in set(df.method)]
    ax = axes[0]
    bw = 0.8 / len(meths)
    for k, meth in enumerate(meths):
        c, f, ht, _ = style(meth, k)
        d = df[df.method == meth]
        data = [pd.concat([d[d.gamma == g][s] for s in rc]).dropna().values for g in gammas]
        pos = np.arange(len(gammas)) + (k - (len(meths) - 1) / 2) * bw
        bp = ax.boxplot(data, positions=pos, widths=bw * 0.85, patch_artist=True, showfliers=True,
                        flierprops={"marker": ".", "ms": 1.2, "mec": c},
                        medianprops={"color": c, "lw": 0.9}, whiskerprops={"lw": 0.6, "color": c},
                        capprops={"lw": 0.6, "color": c})
        for b in bp["boxes"]:
            b.set(facecolor=f, edgecolor=c, hatch=ht, lw=0.6)
    for i, g in enumerate(gammas):
        ax.hlines(g, i - 0.45, i + 0.45, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(gammas)), [f"{g:g}" for g in gammas])
    ax.set_xlabel(r"target $\gamma$"); ax.set_ylabel("per-stratum Rec$_R$")
    ax.set_ylim(0.2, 1.02)
    ax.grid(axis="x", visible=False)
    subcaption(ax, "(a) Per-stratum recall", y=-0.42)
    ax = axes[1]
    for k, meth in enumerate([m for m in ("PTSQ", "BARGAIN-JT", "SUPG-JT") if m in meths]):
        c, f, ht, mk = style(meth, k)
        fr = df[df.method == meth].groupby(["gamma", "qid"]).fail.mean().unstack()
        off = (k - 1) * 0.12
        for q in fr.columns:
            ax.scatter(np.arange(len(gammas)) + off, 100 * fr[q].reindex(gammas).values, s=9, marker=mk,
                       color=c, edgecolor="k", lw=0.2, alpha=0.85, zorder=3)
        ax.plot(np.arange(len(gammas)) + off, 100 * fr.reindex(gammas).max(axis=1).values, color=c, lw=0.8,
                label=meth)
    ax.axhline(100 * 0.05, color="k", lw=0.8, ls="--", label=r"$\delta=5\%$")
    ax.set_xticks(range(len(gammas)), [f"{g:g}" for g in gammas])
    ax.set_xlabel(r"target $\gamma$"); ax.set_ylabel("runs missing a target (%)")
    ax.set_ylim(-1, 17)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=5.4, handlelength=1.0, title="worst query", title_fontsize=5.4,
              borderpad=0.25, labelspacing=0.2, handletextpad=0.4)
    subcaption(ax, "(b) Violation rate per query", y=-0.42)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=style(m)[1], edgecolor=style(m)[0], hatch=style(m)[2]) for m in meths]
    handles.append(plt.Line2D([], [], color="k", ls="--", lw=0.8))
    fig.legend(handles, meths + ["target"], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08),
               fontsize=5.6, columnspacing=0.8, handlelength=1.4)
    fig.tight_layout(w_pad=0.5)
    _wm(fig)
    save(fig, path)


def fig_sensitivity(df: pd.DataFrame, path: Path):
    panels = [("gamma", r"target $\gamma_r=\gamma_p$", "(a) Accuracy target"),
              ("high_frac", "fraction of High tuples", "(b) Privacy mix"),
              ("tpm", "cloud rate limit (TPM)", "(c) Provider rate limit"),
              ("n_cal", r"calibration size $n$", "(d) Fixed sample size"),
              ("N", r"tuples per query $N$", "(e) Data size")]
    panels = [p for p in panels if (df.setting == p[0]).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(FULLW, 1.6))
    axes = np.atleast_1d(axes)
    for ax, (tag, xl, cap) in zip(axes, panels):
        d = df[df.setting == tag]
        if tag == "N":
            ref = d[d.method == "Static-Ref"].groupby(["qid", "x"])["cost"].mean()
            r = (d[d.method == "PTSQ"].groupby(["qid", "x"])["cost"].mean() / ref).unstack(0)
            r = r[sorted(r.columns, key=lambda q: _qlab(q))]
            for k, q in enumerate(r.columns):
                c, mk = ["#c0392b", "#1f5fa8", "#2e8b57", "#e67e22", "#7d3c98"][k % 5], "osD^v"[k % 5]
                ax.plot(r.index, r[q].values, marker=mk, color=c, mfc="white", label=_qlab(q))
            ax.set_xscale("log"); ax.set_ylim(0, 1.15); ax.set_ylabel("PTSQ / Static-Ref")
            ax.set_xticks(list(r.index), [f"{v / 1000:g}k" for v in r.index]); ax.minorticks_off()
            ax.legend(fontsize=5.4, loc="lower left", ncol=2, handlelength=1.0, columnspacing=0.6)
        elif tag == "tpm":
            xs = sorted([v for v in d.x.unique() if v > 0]) + ([0.0] if (d.x == 0).any() else [])
            lab = {19000.0: "19k", 28500.0: "28.5k", 38000.0: "38k", 950000.0: "950k", 0.0: r"$\infty$"}
            for k, meth in enumerate(("PTSQ", "Static-Ref")):
                c, f, ht, mk = style(meth, k)
                s_ = d[d.method == meth].groupby("x")["cost"].mean()
                ax.plot(range(len(xs)), [s_[v] for v in xs], marker=mk, color=c,
                        mfc="white" if meth != "PTSQ" else c, label=meth)
            ax.set_xticks(range(len(xs)), [lab.get(v, f"{v / 1000:g}k") for v in xs])
            ax.set_yscale("log"); ax.minorticks_off()
        else:
            for k, meth in enumerate(("PTSQ", "Static-Ref")):
                c, f, ht, mk = style(meth, k)
                s_ = d[d.method == meth].groupby("x")["cost"].mean()
                ax.plot(s_.index, s_.values, marker=mk, color=c, mfc="white" if meth != "PTSQ" else c, label=meth)
            if tag == "bandwidth":
                ax.set_xscale("log")
            if tag == "gamma":
                ax.set_ylabel("total cost")
                ax.legend(fontsize=5.8, loc="upper left")
        ax.set_xlabel(xl)
        subcaption(ax, cap)
    fig.tight_layout(w_pad=0.5)
    _wm(fig)
    save(fig, path)


def fig_device_model(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 1.55))
    g = df.groupby("device_model")
    order = list(g["dev_auc"].mean().sort_values().index)
    x = np.arange(len(order))
    hts = ["////", "\\\\\\\\", "xxx", "..."]
    for k, o in enumerate(order):
        axes[0].bar(k, g["dev_auc"].mean()[o], 0.6, color="#dadaeb", edgecolor=TIER["D"], hatch=hts[k % 4], lw=0.6)
        axes[1].bar(k, g["cost"].mean()[o] / g["cost"].mean().max(), 0.6, color="#f5b7b1", edgecolor="#c0392b",
                    hatch=hts[k % 4], lw=0.6)
    for ax, yl, cap in ((axes[0], "AUC vs. reference", "(a) Device-model quality"),
                        (axes[1], "PTSQ cost (norm.)", "(b) Cost of PTSQ")):
        ax.set_xticks(x, [{"base": "1.5B"}.get(o, o.upper()) for o in order]); ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
        subcaption(ax, cap, y=-0.36)
    axes[0].set_ylim(0.5, 1.0)
    fig.tight_layout(w_pad=0.6)
    _wm(fig)
    save(fig, path)


LIVE_FEAT_S = {"yelp_delay": 38.6, "amazon_defect": 23.0}     # TF-IDF precomputation on the Jetson (measured)


def fig_components(exp: Path, live_path: Path | None, path: Path):
    """(a,b) certification and learned components, (c) wrong rules, (d) live vs. replay time."""
    sens = pd.read_csv(exp / "sensitivity.csv")
    main = pd.read_csv(exp / "main.csv")
    ref = main[main.method == "Static-Ref"].groupby("qid")["cost"].mean()
    variants = [(0.0, "union bound, $k^+$=800"), (0.25, "propose-and-test (ratio)"),
                (0.5, "+ conditional test"), (0.75, "+ learned risk"), (1.0, "+ learned prefilter")]
    vsty = [("#555555", "#e0e0e0", "ooo"), ("#2e8b57", "#c7e9c0", "xxx"), ("#1f5fa8", "#c6dbef", "\\\\\\\\"), ("#e67e22", "#fdd0a2", "..."),
            ("#c0392b", "#f5b7b1", "////")]
    have_live = live_path is not None and Path(live_path).exists()
    fig, axes = plt.subplots(1, 4, figsize=(FULLW, 1.6), gridspec_kw={"width_ratios": [1, 1.25, 1, 1.05]})
    # (a) relative cost vs. target, per certification variant
    ax = axes[0]
    gs = [0.75, 0.8, 0.9]
    w = 0.8 / len(variants)
    for k, ((x, lab), (c, f, ht)) in enumerate(zip(variants, vsty)):
        vals = []
        for g in gs:
            d = sens[(sens.method == "PTSQ") & (sens.setting == f"cert{g:g}") & (sens.x == x)]
            vals.append((d.groupby("qid")["cost"].mean() / ref).mean() if len(d) else np.nan)
        ax.bar(np.arange(len(gs)) + (k - (len(variants) - 1) / 2) * w, vals, w, color=f, edgecolor=c, hatch=ht, lw=0.6, label=lab)
    ax.set_xticks(range(len(gs)), [f"{g:g}" for g in gs]); ax.set_xlabel(r"target $\gamma$")
    ax.set_ylabel("cost / Static-Ref"); ax.grid(axis="x", visible=False)
    subcaption(ax, "(a) Certification, all queries")
    # (b) per query at gamma = 0.8
    ax = axes[1]
    qs = [q for q in QLABEL if q in set(sens.qid)]
    for k, ((x, lab), (c, f, ht)) in enumerate(zip(variants, vsty)):
        d = sens[(sens.method == "PTSQ") & (sens.setting == "cert0.8") & (sens.x == x)]
        v = (d.groupby("qid")["cost"].mean() / ref).reindex(qs).values
        ax.bar(np.arange(len(qs)) + (k - (len(variants) - 1) / 2) * w, v, w, color=f, edgecolor=c, hatch=ht, lw=0.6)
    ax.set_xticks(range(len(qs)), [_qlab(q) for q in qs]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("cost / Static-Ref")
    subcaption(ax, r"(b) Per query, $\gamma=0.8$")
    # (c) wrong rules: how often the injected rule is used, and the worst stratum recall
    ax = axes[2]
    br = pd.read_csv(exp / "bad_rules.csv")
    g = br.groupby("leak")
    lk = list(g.groups)
    ax.bar(range(len(lk)), 100 * g["injected_used"].mean().values, 0.55, color="#c7e9c0", edgecolor="#2e8b57",
           hatch="xxx", lw=0.6, label="rule used (% runs)")
    ax.set_xticks(range(len(lk)), [f"{100 * v:g}%" for v in lk]); ax.set_xlabel("positives dropped by rule")
    ax.set_ylabel("runs using rule (%)"); ax.set_ylim(0, 160); ax.grid(axis="x", visible=False)
    a2 = ax.twinx()
    a2.plot(range(len(lk)), g["min_stratum_recall"].min().values, marker="o", color="#c0392b", label="min stratum Rec$_R$")
    a2.axhline(0.8, color="k", lw=0.5, ls=":")
    a2.set_ylim(0.5, 1.0); a2.set_ylabel(r"min. stratum Rec$_R$", color="#c0392b"); a2.grid(False)
    a2.tick_params(axis="y", direction="in", colors="#c0392b", labelsize=6.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, ["rule used", r"min. Rec$_R$"], fontsize=5.4, loc="lower right", handlelength=1.2)
    ax.set_yticks([0, 50, 100])
    subcaption(ax, "(c) Wrong LLM rules")
    # (d) live vs. replay of the same seed: where the time goes
    ax = axes[3]
    if have_live:
        lv = pd.read_csv(live_path)
        rows, labs = [], []
        for q in [q for q in ("yelp_delay", "amazon_defect") if q in set(lv.qid)]:
            l_ = lv[lv.qid == q].iloc[0]
            r_ = main[(main.method == "PTSQ") & (main.qid == q) & (main.seed == int(l_.seed))].iloc[0]
            s_ = main[(main.method == "Static-Ref") & (main.qid == q) & (main.seed == int(l_.seed))].iloc[0]
            rows += [(0, r_.t_calib, r_.t_stream), (l_.t_compile + LIVE_FEAT_S.get(q, 0), l_.t_calib, l_.t_stream),
                     (0, 0, s_.latency_s)]
            labs += ["rep.", f"live\n{_qlab(q)}", "S-R"]
        rows = np.array(rows) / 3600
        x = np.arange(len(rows)) + np.repeat(np.arange(len(rows) // 3) * 0.5, 3)
        parts = [("compile + features", "#555555", "#e0e0e0", "ooo"), ("calibration", "#e67e22", "#fdd0a2", "..."),
                 ("execution", "#1f5fa8", "#c6dbef", "\\\\\\\\")]
        bottom = np.zeros(len(rows))
        for j, (lab, c, f, ht) in enumerate(parts):
            ax.bar(x, rows[:, j], 0.75, bottom=bottom, color=f, edgecolor=c, hatch=ht, lw=0.6, label=lab)
            bottom += rows[:, j]
        ax.set_xticks(x, labs, fontsize=5.6)
        ax.set_ylabel("hours"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 10.5)
        ax.legend(fontsize=5.4, loc="upper right", handlelength=1.2, ncol=1)
    subcaption(ax, "(d) Live run vs. replay")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.40, 1.07), frameon=False)
    fig.tight_layout(w_pad=0.5)
    _wm(fig)
    save(fig, path)


def table_per_query(df: pd.DataFrame, path: Path):
    rows = []
    for qid, d in df.groupby("qid"):
        p, b = d[d.method == "PTSQ"], d[d.method == "Static-Ref"]
        if p.empty or b.empty:
            continue
        rows.append([qid.replace("_", r"\_"), f"{b.cost.mean() / p.cost.mean():.2f}$\\times$",
                     f"{b.money.mean() / max(p.money.mean(), 1e-9):.2f}$\\times$",
                     f"{b.latency_s.mean() / p.latency_s.mean():.2f}$\\times$",
                     f"{p.recall.mean():.3f}", f"{p.precision.mean():.3f}", f"{p.n_pruned.mean() / p.N.mean():.0%}".replace("%", r"\%")])
    tex_table(path, ["Query", "Cost", "Money", "Latency", "Rec.$_R$", "Prec.$_R$", "Pruned"], rows, "lrrrrrr")


def fig_federated(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 1.55))
    g = df.groupby("round")
    axes[0].plot(g["dev_auc"].mean().index, g["dev_auc"].mean().values, marker="o", color=TIER["D"], mfc="white")
    axes[0].set_ylabel("device AUC")
    c = g["cost"].mean()
    axes[1].plot(c.index, c.values / c.iloc[0], marker="s", color="#c0392b", mfc="white")
    axes[1].set_ylabel("PTSQ cost / round 0")
    for ax, cap in zip(axes, ("(a) Device-model AUC", "(b) Cost of PTSQ")):
        ax.set_xlabel("federated round")
        subcaption(ax, cap, y=-0.42)
    fig.tight_layout(w_pad=0.6)
    _wm(fig)
    save(fig, path)


# -------------------------------------------------------------------------------- tables
def table_main(df: pd.DataFrame, path: Path):
    names = [m for m in MAIN_ORDER if m in set(df.method)]
    rows = []
    for n in names:
        d = df[df.method == n]
        rows.append([n.replace("PTSQ", r"\textbf{PTSQ}"), f"{d.latency_s.mean():.0f}", f"{d.money.mean():.2f}",
                     f"{d.mb_up_cloud.mean():.1f}", f"{d.recall.mean():.3f}", f"{d.precision.mean():.3f}",
                     f"{d.recall_abs.mean():.3f}", f"{int(d.privacy_violations.mean())}"])
    tex_table(path, ["Method", "Lat. (s)", "Money", "Cloud MB", "Rec.$_R$", "Prec.$_R$", "Rec.$_C$", "Viol."],
              rows, "lrrrrrrr")


ABL_SENS = None


def table_ablation(df: pd.DataFrame, path: Path):
    order = ["PTSQ", "w/o Symbolic", "w/o Edge tier", "Heuristic router"]
    base = df[df.method == "PTSQ"].groupby("qid")["cost"].mean()
    rows = []
    for n in order:
        d = df[df.method == n]
        if d.empty:
            continue
        rel = (d.groupby("qid")["cost"].mean() / base).mean()
        rows.append([n, f"{rel:.2f}$\\times$", f"{d.latency_s.mean():.0f}", f"{d.money.mean():.2f}",
                     f"{d.n_device.mean():.0f}", f"{d.recall.mean():.3f}", f"{d.precision.mean():.3f}"])
        if n == "PTSQ":
            # certification variants come from the sensitivity suite (setting cert0.8, same gamma and seeds)
            sp = None          # certification variants are shown in fig_components (a, b)
            if sp is not None and sp.exists():
                s = pd.read_csv(sp)
                s = s[(s.method == "PTSQ") & (s.setting == "cert0.8")]
                for x, name in ((0.75, "w/o learned prefilter"), (0.5, "w/o learned risk, prefilter"),
                                (0.0, "union bound, $k^+{=}800$")):
                    d2 = s[s.x == x]
                    if d2.empty:
                        continue
                    rel2 = (d2.groupby("qid")["cost"].mean() / base).mean()
                    rows.append([name, f"{rel2:.2f}$\\times$", f"{d2.latency_s.mean():.0f}", f"{d2.money.mean():.2f}",
                                 f"{d2.n_device.mean():.0f}", f"{d2.recall.mean():.3f}", f"{d2.precision.mean():.3f}"])
    tex_table(path, ["Variant", "Cost", "Lat. (s)", "Money", r"\#Device", "Rec.$_R$", "Prec.$_R$"], rows, "lrrrrrr")


def table_overhead(df: pd.DataFrame, path: Path):
    rows = []
    for _, r in df.iterrows():
        extra = f" ($n$={int(r['n_cal'])}, $|\\Theta|$={int(r['configs'])})" if pd.notna(r.get("n_cal")) else ""
        rows.append([r["component"] + extra, f"{r['ms']:.3f}"])
    tex_table(path, ["Component", "Time (ms)"], rows, "lr")


def table_rules(main: pd.DataFrame, trace_dir: Path, path: Path, per_query: int = 2):
    import json
    from ptsq.symbolic import rule_to_str
    rows = []
    for rp in sorted(trace_dir.glob("*.rules.json")):
        qid = rp.name[: -len(".rules.json")]
        rules = json.loads(rp.read_text())
        d = main[(main.method == "PTSQ") & (main.qid == qid)] if main is not None else pd.DataFrame()
        used = d["rules_LOW"].fillna("").astype(str).tolist() if "rules_LOW" in d else []
        for j, rule in enumerate(rules[:per_query]):
            frac = np.mean([str(j) in u.split(";") for u in used]) if used else float("nan")
            txt = rule_to_str(rule).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
            mark = r"\checkmark" if frac >= 0.5 else ("--" if np.isnan(frac) else "")
            rows.append([qid.replace("_", r"\_"), r"\texttt{" + txt[:70] + "}", mark])
    tex_table(path, ["Query", "Proposed rule (truncated)", "Used"], rows, "llc")


def table_live(live: pd.DataFrame, main: pd.DataFrame, path: Path):
    """Live runs vs. trace replay of the SAME seeds (simulator fidelity)."""
    rows = []
    for qid, d in live.groupby("qid"):
        rp = main[(main.method == "PTSQ") & (main.qid == qid) & (main.seed.isin(d.seed))]
        def cell(col, fmt):
            lv, rv = d[col].mean(), rp[col].mean()
            return f"{fmt.format(lv)} / {fmt.format(rv)} ({100 * (lv / rv - 1):+.0f}\\%)"
        rows.append([_qlab(qid), cell("latency_s", "{:.0f}"), cell("money", "{:.2f}"), cell("n_cloud", "{:.0f}")])
    tex_table(path, ["Query", "Latency (s)", "Money (CNY)", "Cloud calls"], rows, "lrrr")


def table_bad_rules(df: pd.DataFrame, path: Path):
    rows = []
    for leak, d in df.groupby("leak"):
        rows.append([f"{leak:.0%}".replace("%", r"\%"), f"{d.injected_used.mean():.0%}".replace("%", r"\%"),
                     f"{d.min_stratum_recall.min():.3f}", f"{d.recall.mean():.3f}", f"{d.precision.mean():.3f}",
                     f"{d.cost.mean():.0f}"])
    tex_table(path, ["Leak", "Used", "min Rec.$_R$ (stratum)", "Rec.$_R$", "Prec.$_R$", "Cost"], rows, "rrrrrr")


def summary(exp: Path, path: Path):
    """Key numbers for the text (copy into the \\tbd{} macros)."""
    out = []
    if (exp / "main.csv").exists():
        df = pd.read_csv(exp / "main.csv")
        b = df[df.method == "Static-Ref"].groupby("qid")[["cost", "money", "mb_up_cloud", "latency_s"]].mean()
        p = df[df.method == "PTSQ"].groupby("qid")[["cost", "money", "mb_up_cloud", "latency_s"]].mean()
        for c in b.columns:
            r = (b[c] / p[c].replace(0, np.nan))
            out.append(f"% Static-Ref / PTSQ {c}: mean ratio {r.mean():.2f}x (min {r.min():.2f}, max {r.max():.2f})")
    if (exp / "guarantee.csv").exists():
        g = pd.read_csv(exp / "guarantee.csv")
        strata = [c.split("_", 1)[1] for c in g.columns if c.startswith("recall_") and c != "recall_abs"]
        for meth in ("PTSQ", "Conf-Cascade"):
            for gam in sorted(g.gamma.unique()):
                d = g[(g.method == meth) & (g.gamma == gam)]
                fails = []
                for s in strata:
                    v = d[[f"recall_{s}", f"precision_{s}"]].dropna()
                    fails.append(((v[f"recall_{s}"] < gam) | (v[f"precision_{s}"] < gam)).mean())
                out.append(f"% {meth} gamma={gam}: max per-stratum failure rate {np.nanmax(fails):.3f}")
    if (exp / "sensitivity.csv").exists():
        s = pd.read_csv(exp / "sensitivity.csv")
        s = s[s.setting == "N"]
        if len(s):
            out.append("% Data-size sweep: cost ratio PTSQ/Static-Ref = a + b/N fitted on the MEASURED")
            out.append("% subsets; the value at N=60000 is a PROJECTION from this model, not a measurement.")
            for q, d in s.groupby("qid"):
                rr = (d[d.method == "PTSQ"].groupby("x")["cost"].mean()
                      / d[d.method == "Static-Ref"].groupby("x")["cost"].mean()).dropna()
                if len(rr) >= 3:
                    A = np.vstack([np.ones(len(rr)), 1.0 / rr.index.values]).T
                    a, b = np.linalg.lstsq(A, rr.values, rcond=None)[0]
                    pred = A @ np.array([a, b])
                    r2 = 1 - ((rr.values - pred) ** 2).sum() / max(((rr.values - rr.values.mean()) ** 2).sum(), 1e-12)
                    out.append(f"% {q}: measured ratio N={int(rr.index.min())}: {rr.iloc[0]:.3f}, "
                               f"N={int(rr.index.max())}: {rr.iloc[-1]:.3f}; fit a={a:.3f}, b={b:.0f}, R2={r2:.3f}; "
                               f"PROJECTED at N=60000: {a + b / 60000:.3f}; limit N->inf: {a:.3f}")
    path.write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="results/exp")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--placeholders", action="store_true")
    ap.add_argument("--traces", default="results/traces")
    ap.add_argument("--live", default="results/live.csv")
    ap.add_argument("--watermark", default=None, help="stamp text on every figure (e.g. SYNTHETIC PREVIEW)")
    a = ap.parse_args()
    global WATERMARK
    WATERMARK = a.watermark
    exp, figs, tabs = Path(a.exp), Path(a.paper) / "figs", Path(a.paper) / "tables"
    global ABL_SENS
    ABL_SENS = exp / "sensitivity.csv"
    figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    specs = {
        "main": (fig_main, "fig_main.pdf", FULLW, 1.75, table_main, "main.tex"),
        "main ": (None, None, 0, 0, table_per_query, "per_query.tex"),
        "main  ": (fig_decisions, "fig_decisions.pdf", COLW, 1.45, None, None),
        "guarantee": (fig_guarantee, "fig_guarantee.pdf", COLW, 1.7, None, None),
        "sensitivity": (fig_sensitivity, "fig_sensitivity.pdf", FULLW, 1.6, None, None),
        "federated": (fig_federated, "fig_federated.pdf", COLW, 1.6, None, None),
        "ablation": (None, None, 0, 0, table_ablation, "ablation.tex"),
        "device_model": (fig_device_model, "fig_device_model.pdf", COLW, 1.6, None, None),
        "overhead": (None, None, 0, 0, table_overhead, "overhead.tex"),
        "bad_rules": (None, None, 0, 0, table_bad_rules, "bad_rules.tex"),
    }
    for name, (ff, fn, w, h, tf, tn) in specs.items():
        csv = exp / f"{name.strip()}.csv"
        have = csv.exists() and csv.stat().st_size > 5 and not a.placeholders
        df = pd.read_csv(csv) if have else None
        if ff:
            if have and len(df):
                ff(df, figs / fn)
            else:
                placeholder_fig(figs / fn, w, h, fn)
        if tf:
            if have and len(df):
                tf(df, tabs / tn)
            else:
                (tabs / tn).write_text(r"\begin{tabular}{c}\toprule \tbd{Placeholder: run the " + name.strip().replace("_", r"\_") +
                                       r" suite, then make\_figures} \\ \bottomrule\end{tabular}" + "\n")
        print(f"[fig] {name}: {'data' if have else 'placeholder'}")
    ph = r"\begin{tabular}{c}\toprule \tbd{Placeholder: %s} \\ \bottomrule\end{tabular}" + "\n"
    main_df = pd.read_csv(exp / "main.csv") if (exp / "main.csv").exists() and not a.placeholders else None
    if Path(a.traces).exists() and any(Path(a.traces).glob("*.rules.json")) and not a.placeholders:
        table_rules(main_df, Path(a.traces), tabs / "rules.tex")
    else:
        (tabs / "rules.tex").write_text(ph % r"collect traces, then make\_figures")
    if Path(a.live).exists() and main_df is not None:
        table_live(pd.read_csv(a.live), main_df, tabs / "live.tex")
    else:
        (tabs / "live.tex").write_text(ph % r"run run\_live.py, then make\_figures")
    if (exp / "sensitivity.csv").exists() and (exp / "bad_rules.csv").exists() and not a.placeholders:
        fig_components(exp, Path(a.live), figs / "fig_components.pdf")
    summary(exp, tabs / "summary.tex")


if __name__ == "__main__":
    main()
