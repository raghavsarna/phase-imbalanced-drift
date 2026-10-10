"""Figs. 1 and 2 of the letter.

Fig. 1 (fig_detect.pdf)
(a) Monte Carlo detection delay versus the share pi of the class whose
    residual mean rises (results/theory.json), with the bounds of Prop. 1;
(b) changes detected on SEA versus minority share for six detectors
    (results/detection/*.json).
Fig. 2 (fig_classify.pdf)
(a) prequential balanced accuracy over time on SEA-10% in a sliding window
    (results/timeline/*.npz, written by timeline.py), mean over five seeds;
(b) hold-out balanced accuracy versus SEA minority share (results/holdout/).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from summarize import ROOT, load_detection, load_runs

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
_PAPER = Path(os.environ.get("PHASE_PAPER_DIR", HERE.parent / "SPL_submission"))
PAPER_FIG = (_PAPER if _PAPER.exists() else ROOT / "latex") / "figures"

# validated categorical slots (dataviz reference palette), fixed order
C = {"ph": "#2a78d6", "adwin": "#eb6834", "ddm": "#1baf7a"}
MK = {"ph": "o", "adwin": "s", "ddm": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.3,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# Fig. 2: one fixed categorical slot per method (validated with the dataviz validator);
# PHASE variants share slot 1 and differ by line style and marker fill
M_COL = {"PHASE": "#2a78d6", "PHASE-pooled": "#2a78d6", "PHASE-noseg": MUTED,
         "ROSE": "#eb6834", "UOB": "#1baf7a", "ARF-US": "#eda100", "ARF": "#e87ba4"}
M_MK = {"PHASE": "o", "PHASE-pooled": "o", "PHASE-noseg": "D", "ROSE": "s", "UOB": "^", "ARF-US": "v", "ARF": "P"}
M_LS = {"PHASE-pooled": "--", "PHASE-noseg": ":"}
M_LAB = {"PHASE": "PHASE", "PHASE-pooled": "PHASE, pooled PH", "PHASE-noseg": "PHASE, no detector"}


def window_ba(y, pred, ends, width):
    """Balanced accuracy over the samples [t - width, t) for every t in ends."""
    out = []
    for t in ends:
        yy, pp = y[t - width:t], pred[t - width:t]
        out.append(np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)]))
    return np.asarray(out)


def fig_classify(stream="sea10", width=1500, step=250):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.5), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) prequential balanced accuracy over time -------------------------
    methods = ["PHASE", "PHASE-pooled", "PHASE-noseg", "ROSE"]
    cps = None
    for m in methods:
        runs = [np.load(f) for f in sorted((ROOT / "timeline").glob(f"{stream}__{m}__*.npz"))]
        if not runs:
            continue
        n = len(runs[0]["y"])
        ends = np.arange(1000 + width, n + 1, step)
        ba = np.mean([window_ba(r["y"], r["pred"], ends, width) for r in runs], axis=0)
        cps = runs[0]["change_points"]
        ax1.plot(ends / 1e3, 100 * ba, color=M_COL[m], ls=M_LS.get(m, "-"), lw=0.9 if m != "PHASE" else 1.2,
                 marker=M_MK[m], markevery=(4 * methods.index(m), 16), ms=2.6, mfc="white" if m == "PHASE-pooled" else M_COL[m],
                 label=M_LAB.get(m, m), zorder=4 if m == "PHASE" else 3)
    for c, lab in zip(cps, ["$8\\to9$\nmin.", "$9\\to7$\nmaj.", "$7\\to9.5$\nmin."]):
        ax1.axvline(c / 1e3, color=GRID, lw=0.8, zorder=0)
        ax1.text(c / 1e3 + 0.4, 67.8, lab, fontsize=5.4, color=MUTED, va="bottom", linespacing=0.95)
    ax1.set_xlim(2.5, 50)
    ax1.set_ylim(67, 93)
    ax1.set_xticks([10, 20, 30, 40, 50])
    ax1.set_xlabel("Time ($10^3$ samples)\n(a)", linespacing=1.6)
    ax1.set_ylabel("Prequential BA (%)", labelpad=1.5)

    # (b) hold-out balanced accuracy vs minority share ----------------------
    streams = ["sea", "sea20", "sea10", "sea5", "sea2", "sea1"]
    df = load_runs("holdout")
    x = np.arange(len(streams))
    for m in ["PHASE", "PHASE-pooled", "ROSE", "UOB", "ARF-US", "ARF"]:
        g = df[df.method == m]
        mu = [100 * g[g.dataset == s].ba.mean() for s in streams]
        sd = [100 * g[g.dataset == s].ba.std() for s in streams]
        ax2.errorbar(x, mu, yerr=sd, color=M_COL[m], ls=M_LS.get(m, "-"), marker=M_MK[m], ms=2.6,
                     mfc="white" if m == "PHASE-pooled" else M_COL[m], lw=0.9 if m != "PHASE" else 1.2,
                     capsize=1.2, elinewidth=0.5, label=M_LAB.get(m, m), zorder=4 if m == "PHASE" else 3)
    ax2.set_xticks(x, ["nat.", "20", "10", "5", "2", "1"])
    ax2.set_ylim(45, 95)
    ax2.set_yticks([50, 60, 70, 80, 90])
    ax2.set_xlabel("SEA minority share (%)\n(b)", linespacing=1.6)
    ax2.set_ylabel("Hold-out BA (%)", labelpad=1.5)

    handles, labels = {}, {}
    for ax in (ax1, ax2):
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    order = ["PHASE", "PHASE, pooled PH", "PHASE, no detector", "ROSE", "UOB", "ARF-US", "ARF"]
    fig.legend([handles[l] for l in order if l in handles], [l for l in order if l in handles],
               loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, handlelength=2.0,
               columnspacing=0.8, labelspacing=0.15, fontsize=5.8)
    fig.tight_layout(pad=0.25, w_pad=0.9)
    for out in (OUT, PAPER_FIG):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig_classify.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(out / "fig_classify.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("wrote fig_classify.pdf to", OUT, "and", PAPER_FIG)


def main():
    th = json.loads((ROOT / "theory.json").read_text())
    det = load_detection()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.5))

    # (a) delay vs pi --------------------------------------------------------
    rows = th["delay"]["rows"]
    pi = np.array([r["pi"] for r in rows])
    grid = np.logspace(np.log10(pi.min()), np.log10(pi.max()), 100)
    lam, dlt, Dl = th["lam"], th["delta"], th["delay"]["Delta"]
    ax1.plot(grid, (lam + 1) / (grid * (Dl - dlt)), color=C["ph"], lw=0.9, label="bound, class-wise")
    g2 = grid[grid * Dl > dlt * 1.02]
    ax1.plot(g2, (lam + 1) / (g2 * Dl - dlt), color=MUTED, lw=0.9, ls="--", label="bound, pooled")
    ax1.axvline(dlt / Dl, color=GRID, lw=0.8, zorder=0)
    ax1.text(dlt / Dl * 0.95, 48, "$\\pi_c\\Delta_c=\\delta$", fontsize=5.6, color=MUTED, rotation=90,
             va="bottom", ha="right")
    ax1.plot(pi, [r["class"]["mean_delay"] for r in rows], "o", color=C["ph"], ms=3.2, label="sim., class-wise")
    ok = [r for r in rows if r["pooled"]["detect_rate"] >= 0.99]
    ax1.plot([r["pi"] for r in ok], [r["pooled"]["mean_delay"] for r in ok], "o", mfc="white", mec=MUTED,
             ms=3.2, label="sim., pooled")
    for r in rows:
        if r["pooled"]["detect_rate"] < 0.99:
            ax1.annotate(f"{100 * r['pooled']['detect_rate']:.0f}%", (r["pi"], 4.2e4), fontsize=5.4,
                         color=MUTED, ha="center")
    ax1.text(0.0105, 7.5e4, "pooled PH detects:", fontsize=5.4, color=MUTED)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(40, 1.6e5)
    ax1.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ["1", "2", "5", "10", "20", "50"])
    ax1.minorticks_off()
    ax1.set_xlabel("Share $\\pi_c$ of changed class (%)\n(a)", linespacing=1.6)
    ax1.set_ylabel("Mean delay (samples)", labelpad=1.5)
    # direct labels (the caption gives the full key): solid/filled = class-wise, dashed/open = pooled
    ax1.text(0.0108, 2.6e3, "class-wise", fontsize=6.0, color=C["ph"], ha="left", va="top")
    ax1.text(0.125, 2.6e3, "pooled", fontsize=6.0, color=MUTED, ha="left", va="bottom")

    # (b) detected changes vs minority share -------------------------------
    streams = ["sea", "sea20", "sea10", "sea5", "sea2", "sea1"]
    xl = ["nat.", "20", "10", "5", "2", "1"]
    x = np.arange(len(streams))
    for test in ("ph", "adwin", "ddm"):
        for mode in ("class", "pooled"):
            g = det[det.detector == f"{mode}-{test}"]
            rate = [100 * g[g.stream == s].det.sum() / max(1, g[g.stream == s].n_true.sum()) for s in streams]
            ax2.plot(x, rate, ls="-" if mode == "class" else "--", marker=MK[test], ms=2.8, lw=1.0,
                     color=C[test], mfc=C[test] if mode == "class" else "white",
                     label=f"{test.upper() if test != 'ph' else 'PH'}, {'class-wise' if mode == 'class' else 'pooled'}")
    ax2.set_xticks(x, xl)
    ax2.set_ylim(-3, 105)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("SEA minority share (%)\n(b)", linespacing=1.6)
    ax2.set_ylabel("Changes detected (%)", labelpad=1.5)
    h, l = ax2.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, handlelength=2.0,
               columnspacing=0.9, labelspacing=0.15, fontsize=5.8)
    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25, w_pad=0.9)
    for out in (OUT, PAPER_FIG):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig_detect.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(out / "fig_detect.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("wrote fig_detect.pdf to", OUT, "and", PAPER_FIG)
    if (ROOT / "timeline").exists():
        fig_classify()


if __name__ == "__main__":
    main()
