"""Supplementary tables for the PHASE ensemble of the first version (logs in results/).

Fixes the points raised on the first version (R1 minor 5-7):
  * BA with two decimals, so ties are visible; ties share the bold face and the
    average rank uses mid-ranks;
  * ablation averages are computed from the displayed (rounded) entries;
  * Cohen's kappa is reported (it was only "in the logs").
Writes <paper>/tables/supp_phase_{ba,kappa,ablation}.tex
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_tables import ABL, NAMES, ORDER, friedman, holm_wilcoxon  # noqa: E402
from summarize import DATASETS, load_runs  # noqa: E402

PAPER = Path(os.environ.get("PAPER_DIR", HERE.parents[1] / "SPL_resubmission"))
OUT = PAPER / "tables"
HEAD = ["SEA", "SEA$_{10}$", "SEA$_{5}$", "Elec2", "Cover.", "Ins-A", "Ins-G", "Credit"]
NAMES = {k: v.split("~")[0] for k, v in NAMES.items()}


def table(metric, digits):
    lines = ["\\begin{tabular}{@{}l*{8}{c}c@{}}", "\\toprule", "Method & " + " & ".join(HEAD) + " & Rank \\\\"]
    tests = {}
    for protocol, title in (("holdout", "Temporal hold-out (train on first 80\\%, test on last 20\\%)"),
                            ("preq", "Prequential (test-then-train)")):
        df = load_runs(protocol)
        df = df[df.method.isin(ORDER)]
        g = df.groupby(["method", "dataset"])[metric]
        mu, sd = g.mean().unstack(), g.std().unstack()
        disp = (100 * mu).round(digits)                  # what is printed decides bold face and ranks
        M = disp.loc[[m for m in ORDER if m in disp.index], DATASETS].T
        ranks, fr = friedman(M)
        tests[protocol] = {"friedman": fr, "wilcoxon": holm_wilcoxon(M)}
        lines += ["\\midrule", f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\"]
        for m in ORDER:
            cells = []
            for d in DATASETS:
                v, s = disp.loc[m, d], 100 * sd.loc[m, d]
                txt = f"{v:.{digits}f}{{\\tiny$\\pm${s:.{digits}f}}}"
                cells.append(f"\\textbf{{{txt}}}" if v >= M.loc[d].max() else txt)
            r = ranks[m]
            rk = f"\\textbf{{{r:.2f}}}" if r <= ranks.min() + 1e-12 else f"{r:.2f}"
            lines.append(NAMES[m] + " & " + " & ".join(cells) + f" & {rk} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n", tests


def ablation():
    df = load_runs("holdout")
    mu = df.groupby(["method", "dataset"]).ba.mean().unstack()
    ref = mu.loc["PHASE", DATASETS]
    head = ["SEA", "S$_{10}$", "S$_{5}$", "El.", "Co.", "I-A", "I-G", "Cr."]
    lines = ["\\begin{tabular}{@{}l*{8}{r}r@{}}", "\\toprule", "Variant & " + " & ".join(head) + " & Avg. \\\\",
             "\\midrule"]
    shown = (100 * ref).round(1)
    lines.append("PHASE (BA) & " + " & ".join(f"{v:.1f}" for v in shown) + f" & {shown.mean():.2f} \\\\")
    for key, label in ABL:
        if label is None:
            lines.append(f"\\midrule\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{key}}}}}\\\\")
            continue
        diff = (100 * (mu.loc[key, DATASETS] - ref)).round(1)
        f = lambda v: "0.0" if abs(v) < 0.05 else f"${v:+.1f}$"
        lines.append(f"\\quad {label} & " + " & ".join(f(v) for v in diff) + f" & ${diff.mean():+.2f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ba, t1 = table("ba", 2)
    ka, t2 = table("kappa", 1)
    (OUT / "supp_phase_ba.tex").write_text(ba)
    (OUT / "supp_phase_kappa.tex").write_text(ka)
    (OUT / "supp_phase_ablation.tex").write_text(ablation())
    for name, t in (("BA", t1), ("kappa", t2)):
        for p, v in t.items():
            fr = v["friedman"]
            print(name, p, f"Friedman p_F={fr['p_F']:.2g} CD={fr['cd']:.2f}",
                  "min Holm p =", round(min(x["p_holm"] for x in v["wilcoxon"].values()), 3))


if __name__ == "__main__":
    main()
