"""Fig. 2 of the letter, computed from results/runs/*.json.

(a) detection rate of each SEA change (theta 8->9, 9->7, 7->9.5) by the
    class-conditional PH bank and by a pooled PH test, over the imbalanced
    SEA streams (minority share 10% and 5%; 5 seeds each);
(b) test balanced accuracy versus minority share for PHASE and baselines.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from summarize import load_runs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
_PAPER = Path(os.environ.get("PHASE_PAPER_DIR", ROOT.parent / "SPL_submission"))
PAPER_FIG = (_PAPER if _PAPER.exists() else ROOT / "results" / "latex") / "figures"

SEA = ["sea", "sea10", "sea5"]
XLAB = ["Natural", "10%", "5%"]
# validated categorical slots (dataviz reference palette), fixed order
C = {"PHASE": "#2a78d6", "PHASE-pooled": "#2a78d6", "UOB": "#eb6834", "OOB": "#1baf7a",
     "ARF": "#eda100", "HAT": "#e87ba4", "SRP": "#4a3aa7"}
MK = {"PHASE": "o", "PHASE-pooled": "o", "UOB": "s", "OOB": "^", "ARF": "D", "HAT": "v", "SRP": "P"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _caught(boundaries, cps=(10000, 20000, 30000), lo=500, hi=5000):
    hit, used = [], set()
    for c in cps:
        for i, x in enumerate(boundaries):
            if i not in used and c - lo <= x <= c + hi:
                used.add(i)
                hit.append(c)
                break
    return hit


def main():
    import json
    df = load_runs()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.62), gridspec_kw={"width_ratios": [0.95, 1.05]})

    # (a) which changes are detected --------------------------------------
    cps = (10000, 20000, 30000)
    labels = ["$8\\to9$\n(min.)", "$9\\to7$\n(maj.)", "$7\\to9.5$\n(min.)"]
    x = np.arange(3)
    for j, (m, lab, hatch) in enumerate([("PHASE", "Class-wise", None), ("PHASE-pooled", "Pooled", "////")]):
        hits, n = np.zeros(3), 0
        for d in ("sea10", "sea5"):
            for f in sorted((ROOT / "results" / "runs").glob(f"{d}__{m}__*.json")):
                b = json.loads(f.read_text())["boundaries"]
                hits += [c in _caught(b) for c in cps]
                n += 1
        rate = 100 * hits / n
        bars = ax1.bar(x + (j - 0.5) * 0.36, rate, 0.34, color=C["PHASE"] if j == 0 else "white",
                       edgecolor=C["PHASE"], hatch=hatch, linewidth=0.8, label=lab, zorder=3)
        for bar, r in zip(bars, rate):
            ax1.text(bar.get_x() + bar.get_width() / 2, r + 2, f"{r:.0f}", ha="center", va="bottom",
                     fontsize=5.6, color=INK)
    ax1.set_xticks(x, labels, fontsize=6.5)
    ax1.set_ylim(0, 138)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_ylabel("Detected (%)", labelpad=1.5)
    ax1.set_xlabel("SEA change ($\\theta$)\n(a)", linespacing=1.6)
    ax1.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax1.legend(loc="upper center", frameon=False, handlelength=1.2, borderaxespad=0.0,
               bbox_to_anchor=(0.5, 1.03), ncol=2, columnspacing=0.8)

    # (b) balanced accuracy vs minority share ------------------------------
    x = np.arange(len(SEA))
    for m in ["PHASE", "PHASE-pooled", "UOB", "OOB", "ARF", "SRP"]:
        g = df[df["dataset"].isin(SEA) & (df["method"] == m)]
        mu = [100 * g[g["dataset"] == d]["ba"].mean() for d in SEA]
        sd = [100 * g[g["dataset"] == d]["ba"].std() for d in SEA]
        ls = "--" if m == "PHASE-pooled" else "-"
        mfc = "white" if m == "PHASE-pooled" else C[m]
        lab = "PHASE, pooled" if m == "PHASE-pooled" else m
        ax2.errorbar(x, mu, yerr=sd, color=C[m], ls=ls, marker=MK[m], ms=3.0, mfc=mfc,
                     lw=1.1, capsize=1.5, elinewidth=0.6, label=lab, zorder=3)
    ax2.set_xticks(x, XLAB)
    ax2.set_xlim(-0.25, 2.25)
    ax2.set_ylim(15, 100)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_ylabel("Balanced accuracy (%)", labelpad=1.5)
    ax2.set_xlabel("Minority share\n(b)", linespacing=1.6)
    ax2.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax2.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.6, columnspacing=0.5,
               borderaxespad=0.1, labelspacing=0.15, fontsize=6.2)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25, w_pad=0.8)
    for out in (OUT, PAPER_FIG):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig_results.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(out / "fig_results.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("wrote fig_results.pdf to", OUT, "and", PAPER_FIG)


if __name__ == "__main__":
    main()
