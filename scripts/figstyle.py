"""Shared matplotlib style for all paper figures (IEEE two-column, Times-like fonts).

Conventions follow typical ICDE papers: boxed axes with inward ticks, serif fonts that match
the body text, hatched bars with one colour per method, line plots with distinct markers, and
sub-captions "(a) ...", "(b) ..." under each panel.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_SERIF = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((f for f in _SERIF if f in _avail), "DejaVu Serif")

plt.rcParams.update({
    "font.family": "serif", "font.serif": [SERIF], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.6",
    "legend.fancybox": False, "legend.borderpad": 0.3, "legend.handlelength": 1.6,
    "axes.linewidth": 0.6, "axes.edgecolor": "0.15", "axes.grid": True, "grid.color": "0.88",
    "grid.linestyle": "--", "grid.linewidth": 0.4, "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1, "lines.markersize": 3.5, "hatch.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

COLW, FULLW = 3.45, 7.1

# one entry per method: (edge/line colour, light fill, hatch, marker)
PALETTE = {
    "PTSQ": ("#c0392b", "#f5b7b1", "////", "o"),
    "Static-Ref": ("#1f5fa8", "#c6dbef", "\\\\\\\\", "s"),
    "Cert-2Tier": ("#2e8b57", "#c7e9c0", "xxx", "^"),
    "Conf-Cascade": ("#e67e22", "#fdd0a2", "...", "D"),
    "SUPG-JT": ("#8c564b", "#e8d5c9", "+++", "X"),
    "BARGAIN-JT": ("#17becf", "#d2f1f5", "---", "P"),
    "BARGAIN-JT (relaxed)": ("#7f7f7f", "#e6e6e6", "|||", "p"),
    "Device-Only": ("#7d3c98", "#dadaeb", "---", "v"),
    "Cloud-All": ("#555555", "#e0e0e0", "ooo", "P"),
    "w/o Symbolic": ("#2e8b57", "#c7e9c0", "xxx", "^"),
    "w/o Edge tier": ("#1f5fa8", "#c6dbef", "\\\\\\\\", "s"),
    "w/o RL (fixed)": ("#e67e22", "#fdd0a2", "...", "D"),
    "Heuristic router": ("#7d3c98", "#dadaeb", "---", "v"),
}
CYCLE = [("#c0392b", "o"), ("#1f5fa8", "s"), ("#2e8b57", "^"), ("#e67e22", "D"), ("#7d3c98", "v"),
         ("#555555", "P")]
TIER = {"D": "#7d3c98", "E": "#e67e22", "C": "#1f5fa8", "rule": "#2e8b57"}
LEVEL = {"LOW": "#d5f5e3", "MEDIUM": "#fdebd0", "HIGH": "#fadbd8"}


def style(name: str, i: int = 0):
    return PALETTE.get(name, (CYCLE[i % len(CYCLE)][0], "#eeeeee", "", CYCLE[i % len(CYCLE)][1]))


def bars(ax, x, heights, names, width=0.7, yerr=None):
    for k, (xi, h, n) in enumerate(zip(x, heights, names)):
        c, f, ht, _ = style(n, k)
        ax.bar(xi, h, width=width, color=f, edgecolor=c, hatch=ht, linewidth=0.7,
               yerr=None if yerr is None else yerr[k],
               error_kw={"elinewidth": 0.6, "ecolor": "0.25", "capsize": 1.5})


def subcaption(ax, text: str, y: float = -0.50):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def save(fig, path, **kw):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)
