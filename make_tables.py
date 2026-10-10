"""Write the LaTeX tables, number macros and statistical tests of the letter.

    python make_tables.py  ->  <paper>/tables/{tab_main,tab_ablation,tab_detect,numbers}.tex
                               results/stats.json
<paper> is $PHASE_PAPER_DIR, else ../SPL_submission if present, else results/latex.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from summarize import BASELINES, DATASETS, ROOT, load_detection, load_runs

HERE = Path(__file__).resolve().parent
_PAPER = Path(os.environ.get("PHASE_PAPER_DIR", HERE.parent / "SPL_submission"))
OUT = (_PAPER if _PAPER.exists() else ROOT / "latex") / "tables"
HEAD = ["SEA", "SEA$_{10}$", "SEA$_{5}$", "Elec2", "Cover.", "Ins-A", "Ins-G", "Credit"]
NAMES = {"HAT": "HAT~\\cite{bifet2009hat}", "ARF": "ARF~\\cite{gomes2017arf}", "SRP": "SRP~\\cite{gomes2019srp}",
         "OOB": "OOB~\\cite{wang2015oob}", "UOB": "UOB~\\cite{wang2015oob}", "ARF-US": "ARF-US",
         "ROSE": "ROSE~\\cite{cano2022rose}", "PHASE": "PHASE"}
ORDER = BASELINES + ["PHASE"]
ABL = [("Detector", None),
       ("PHASE-pooled", "PH, pooled"), ("PHASE-adwin-class", "ADWIN, class"), ("PHASE-adwin", "ADWIN, pooled"),
       ("PHASE-ddm-class", "DDM, class"), ("PHASE-ddm", "DDM, pooled"),
       ("PHASE-periodic", "periodic"), ("PHASE-noseg", "none"),
       ("Learner", None),
       ("PHASE-nocost", "no costs"), ("PHASE-norefine", "no SVM"),
       ("PHASE-nomemory", "no memory"), ("PHASE-stack", "+ stacking")]
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}  # Nemenyi


def mean_table(df, metric="ba"):
    g = df.groupby(["method", "dataset"])[metric]
    return g.mean().unstack(), g.std().unstack(), g.size().unstack()


def cell(m, s, bold=False):
    if pd.isna(m):
        return "--"
    txt = f"{100 * m:.1f}" + (f"{{\\tiny$\\pm${100 * s:.1f}}}" if not pd.isna(s) else "")
    return f"\\textbf{{{txt}}}" if bold else txt


def friedman(M):
    """M: datasets x methods (higher is better). Average ranks, Friedman/Iman-Davenport, Nemenyi CD."""
    R = M.rank(axis=1, ascending=False)
    N, k = M.shape
    chi2, p_chi = stats.friedmanchisquare(*[M[c].to_numpy() for c in M.columns])
    F = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_F = stats.f.sf(F, k - 1, (k - 1) * (N - 1))
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    return R.mean(), {"chi2": chi2, "p_chi2": p_chi, "F": F, "p_F": p_F, "cd": cd, "N": N, "k": k}


def holm_wilcoxon(M, ref="PHASE"):
    res = {}
    for c in M.columns:
        if c == ref:
            continue
        d = M[ref] - M[c]
        p = stats.wilcoxon(d, alternative="two-sided", zero_method="zsplit").pvalue if (d != 0).any() else 1.0
        res[c] = {"p": float(p), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    order = sorted(res, key=lambda c: res[c]["p"])
    m, run = len(order), 0.0
    for i, c in enumerate(order):
        run = max(run, min(1.0, (m - i) * res[c]["p"]))
        res[c]["p_holm"] = run
    return res


def main_table(dfs):
    lines = ["\\begin{tabular}{@{}l*{8}{c}c@{}}", "\\toprule",
             "Method & " + " & ".join(HEAD) + " & Rank \\\\"]
    tests = {}
    for protocol, title in (("holdout", "Temporal hold-out: trained on the first 80\\%, frozen, tested on the last 20\\%"),
                            ("preq", "Prequential (test-then-train) over the whole stream")):
        df = dfs[protocol]
        df = df[df.method.isin(ORDER)]
        mu, sd, _ = mean_table(df)
        ds = [d for d in DATASETS if d in mu.columns]
        M = mu.loc[[m for m in ORDER if m in mu.index], ds].T.dropna(axis=0)
        ranks, fr = friedman(M) if len(M) >= 2 else (pd.Series(dtype=float), {})
        tests[protocol] = {"friedman": fr, "ranks": ranks.to_dict(),
                           "wilcoxon": holm_wilcoxon(M) if "PHASE" in M and len(M) >= 2 else {}}
        lines += ["\\midrule", f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\"]
        for m in ORDER:
            if m not in mu.index:
                continue
            cells = []
            for d in DATASETS:
                best = d in mu.columns and not pd.isna(mu.loc[m, d]) and mu.loc[m, d] >= mu[d].loc[[x for x in ORDER if x in mu.index]].max() - 1e-12
                cells.append(cell(mu.loc[m, d], sd.loc[m, d], best) if d in mu.columns else "--")
            r = ranks.get(m, np.nan)
            rk = "--" if pd.isna(r) else (f"\\textbf{{{r:.2f}}}" if r <= ranks.min() + 1e-12 else f"{r:.2f}")
            lines.append(NAMES[m] + " & " + " & ".join(cells) + f" & {rk} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n", tests


def ablation_table(df):
    mu, _, _ = mean_table(df)
    head = ["SEA", "S$_{10}$", "S$_{5}$", "El.", "Co.", "I-A", "I-G", "Cr."]
    lines = ["\\begin{tabular}{@{}l*{8}{r}r@{}}", "\\toprule",
             "Variant & " + " & ".join(head) + " & Avg. \\\\", "\\midrule"]
    ref = mu.loc["PHASE"]
    lines.append("PHASE (BA) & " + " & ".join("--" if pd.isna(ref.get(d)) else f"{100 * ref[d]:.1f}" for d in DATASETS)
                 + f" & {100 * ref[DATASETS].mean():.1f} \\\\")
    for key, label in ABL:
        if label is None:
            lines.append(f"\\midrule\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{key}}}}}\\\\")
            continue
        if key not in mu.index:
            continue
        diff = 100 * (mu.loc[key] - ref)
        sgn = lambda v: "--" if pd.isna(v) else ("0.0" if abs(v) < 0.05 else f"${v:+.1f}$")
        cells = [sgn(diff.get(d)) for d in DATASETS]
        lines.append(f"\\quad {label} & " + " & ".join(cells) + f" & {sgn(diff[DATASETS].mean())} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines).replace("−", "-") + "\n"


def detect_table(det):
    streams = [("sea20", "20\\%"), ("sea10", "10\\%"), ("sea5", "5\\%"), ("sea2", "2\\%")]
    rows = [("class-ph", "PH, class-wise (ours)"), ("pooled-ph", "PH, pooled"),
            ("class-adwin", "ADWIN, class-wise"), ("pooled-adwin", "ADWIN, pooled"),
            ("class-ddm", "DDM, class-wise"), ("pooled-ddm", "DDM, pooled")]
    lines = ["\\begin{tabular}{@{}l*{4}{c}cc@{}}", "\\toprule",
             "& \\multicolumn{4}{c}{Detected (of 15) at minority share} & \\multicolumn{2}{c}{False alarms} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(l){6-7}",
             "Detector & " + " & ".join(s for _, s in streams) + " & drift & null \\\\", "\\midrule"]
    for key, label in rows:
        g = det[det.detector == key]
        cells = [str(int(g[g.stream == s].det.sum())) for s, _ in streams]
        fa_drift = int(g[g.stream.isin(["sea", "sea20", "sea10", "sea5", "sea2", "sea1"])].fa.sum())
        fa_null = int(g[g.stream.isin(["seanull", "seanull5"])].fa.sum())
        lines.append(f"{label} & " + " & ".join(cells) + f" & {fa_drift} & {fa_null} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def macro(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}"


def numbers(dfs, det, tests, theory):
    N = []
    hold, preq = dfs["holdout"], dfs["preq"]
    mu_h, _, _ = mean_table(hold)
    mu_p, _, _ = mean_table(preq) if not preq.empty else (pd.DataFrame(),) * 3
    tag = {"sea": "Sea", "sea10": "SeaTen", "sea5": "SeaFive", "elec2": "Elec", "covtype": "Cov",
           "insects_abrupt": "InsA", "insects_gradual": "InsG", "creditcard": "Credit"}
    for d, t in tag.items():
        for proto, mu, P in (("holdout", mu_h, "H"), ("preq", mu_p, "P")):
            if mu.empty or d not in mu.columns or "PHASE" not in mu.index:
                continue
            base = mu.loc[[b for b in BASELINES if b in mu.index], d]
            N.append(macro(f"nPhase{P}{t}", f"{100 * mu.loc['PHASE', d]:.1f}"))
            N.append(macro(f"nBest{P}{t}", base.idxmax()))
            N.append(macro(f"nBestBA{P}{t}", f"{100 * base.max():.1f}"))
            N.append(macro(f"nGain{P}{t}", f"{100 * (mu.loc['PHASE', d] - base.max()):.1f}"))
    for proto, P in (("holdout", "H"), ("preq", "P")):
        T = tests.get(proto, {})
        fr = T.get("friedman", {})
        if fr:
            N += [macro(f"nFriedP{P}", f"{fr['p_F']:.2g}"), macro(f"nCD{P}", f"{fr['cd']:.2f}"),
                  macro(f"nRankPhase{P}", f"{T['ranks']['PHASE']:.2f}")]
            others = {m: r for m, r in T["ranks"].items() if m != "PHASE"}
            bm = min(others, key=others.get)
            N += [macro(f"nRankBestName{P}", bm), macro(f"nRankBest{P}", f"{others[bm]:.2f}"),
                  macro(f"nNemSig{P}", str(sum(1 for r in others.values() if r - T['ranks']['PHASE'] > fr['cd']))),
                  macro(f"nRawP{P}", f"{min(v['p'] for v in T['wilcoxon'].values()):.3f}"),
                  macro(f"nHolmMin{P}", f"{min(v['p_holm'] for v in T['wilcoxon'].values()):.3f}")]
            W = T["wilcoxon"]
            nsig = sum(1 for v in W.values() if v["p_holm"] < 0.05)
            N += [macro(f"nWilcSig{P}", str(nsig)),
                  macro(f"nWins{P}", str(sum(v['wins'] for v in W.values()))),
                  macro(f"nPairs{P}", str(sum(v['wins'] + v['losses'] for v in W.values())))]
    # detection study
    if not det.empty:
        def s(stream, d, col="det"):
            g = det[(det.stream == stream) & (det.detector == d)]
            return g[col].sum() if col in ("det", "fa") else g[col].mean()
        for stream, t in (("sea", "Nat"), ("sea20", "Twenty"), ("sea10", "Ten"), ("sea5", "Five"), ("sea2", "Two"), ("sea1", "One")):
            for d, dt in (("class-ph", "Cls"), ("pooled-ph", "Pool"), ("class-adwin", "ClsAd"), ("class-ddm", "ClsDdm")):
                N.append(macro(f"nDet{dt}{t}", str(int(s(stream, d)))))
                dl = s(stream, d, "delay")
                N.append(macro(f"nDelay{dt}{t}", "--" if pd.isna(dl) else f"{dl:.0f}"))
        for d, dt in (("class-ph", "Cls"), ("pooled-ph", "Pool"), ("class-adwin", "ClsAd"), ("pooled-adwin", "PoolAd"),
                      ("class-ddm", "ClsDdm"), ("pooled-ddm", "PoolDdm")):
            g = det[det.detector == d]
            N.append(macro(f"nFA{dt}", str(int(g.fa.sum()))))
            N.append(macro(f"nFANull{dt}", str(int(g[g.stream.str.startswith('seanull')].fa.sum()))))
            mins = g[g.stream.isin(["sea20", "sea10", "sea5"])]
            N.append(macro(f"nMinHits{dt}", str(int(mins.hit_10000.sum() + mins.hit_30000.sum()))))
    # theory simulation
    if theory:
        for row in theory["delay"]["rows"]:
            t = {0.01: "One", 0.02: "Two", 0.05: "Five", 0.1: "Ten", 0.2: "Twenty", 0.5: "Half"}[row["pi"]]
            N.append(macro(f"nSimDetPool{t}", f"{100 * row['pooled']['detect_rate']:.0f}"))
            N.append(macro(f"nSimDelayCls{t}", f"{row['class']['mean_delay']:.0f}"))
            N.append(macro(f"nSimBoundCls{t}", f"{row['bound_class']:.0f}"))
        fa = {r["lam"]: r for r in theory["false_alarm"]["rows"]}
        N.append(macro("nSimFACls", f"{100 * fa[25]['class']:.1f}"))
        N.append(macro("nSimFAPool", f"{100 * fa[25]['pooled']:.1f}"))
    # PHASE diagnostics
    ph = hold[hold.method == "PHASE"]
    for d, t in tag.items():
        g = ph[ph.dataset == d]
        if len(g):
            N.append(macro(f"nSeg{t}", f"{g.n_boundaries.mean() + 1:.0f}"))
            N.append(macro(f"nLastSeg{t}", f"{g.last_seg_len.mean():.0f}"))
            N.append(macro(f"nMem{t}", f"{g.n_memory.mean():.0f}"))
    if "PHASE-nomemory" in mu_h.index and "covtype" in mu_h.columns:
        N.append(macro("nNoMemCov", f"{100 * mu_h.loc['PHASE-nomemory', 'covtype']:.1f}"))
        N.append(macro("nNoSegCov", f"{100 * mu_h.loc['PHASE-noseg', 'covtype']:.1f}"))
    tp = preq[preq.dataset == "covtype"].groupby("method").time_s.mean() if not preq.empty else pd.Series(dtype=float)
    if "PHASE" in tp and "ARF" in tp:
        N += [macro(f"nTime{m.replace('-', '')}PCov", f"{tp[m] / 60:.0f}") for m in ("PHASE", "ARF", "SRP", "ROSE")]
    # Fig. 2(a): prequential BA on SEA-10% in the 5000 samples after the minority-only change 7 -> 9.5
    tl = ROOT / "timeline"
    if tl.exists():
        def seg_ba(f, a, b):
            r = np.load(f)
            yy, pp = r["y"][a:b], r["pred"][a:b]
            return np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])
        for m, t in (("PHASE", "Phase"), ("PHASE-pooled", "Pool"), ("PHASE-noseg", "Noseg"), ("ROSE", "Rose")):
            files = sorted(tl.glob(f"sea10__{m}__*.npz"))
            if files:
                N.append(macro(f"nPost{t}", f"{100 * np.mean([seg_ba(f, 30000, 35000) for f in files]):.1f}"))
        # the other minority-only change (8 -> 9, samples 10k-15k) and the same window on SEA-5%
        post = lambda d, m, lo: np.mean([seg_ba(f, lo, lo + 5000) for f in sorted(tl.glob(f"{d}__{m}__*.npz"))])
        if list(tl.glob("sea10__PHASE-pooled__*.npz")):
            N.append(macro("nPostEarlyPhase", f"{100 * post('sea10', 'PHASE', 10000):.1f}"))
            N.append(macro("nPostEarlyPool", f"{100 * post('sea10', 'PHASE-pooled', 10000):.1f}"))
        if list(tl.glob("sea5__PHASE-pooled__*.npz")):
            gain = post("sea5", "PHASE", 30000) - post("sea5", "PHASE-pooled", 30000)
            N.append(macro("nPostGainFive", f"{100 * gain:.1f}"))
        hits = 0
        for seed in range(5):
            f = ROOT / "cache" / f"preq__sea10__class-ph__{seed}.json"
            if f.exists():
                b = json.loads(f.read_text())["boundaries"]
                hits += any(10000 - 500 <= x <= 10000 + 5000 for x in b)
        N.append(macro("nEightNineHits", str(hits)))
    # Fig. 2(b): hold-out BA at minority shares 2% and 1%
    for d, t in (("sea2", "Two"), ("sea1", "One")):
        for m, mt in (("PHASE", "Phase"), ("UOB", "Uob"), ("ARF-US", "ArfUs"), ("ROSE", "Rose")):
            g = hold[(hold.dataset == d) & (hold.method == m)]
            if len(g):
                N.append(macro(f"nShare{t}{mt}", f"{100 * g.ba.mean():.1f}"))
    return "% generated by make_tables.py from the experiment logs -- do not edit\n" + "\n".join(N) + "\n"


def main():
    dfs = {p: load_runs(p) for p in ("holdout", "preq")}
    det = load_detection()
    theory = json.loads((ROOT / "theory.json").read_text()) if (ROOT / "theory.json").exists() else None
    OUT.mkdir(parents=True, exist_ok=True)
    tab, tests = main_table(dfs)
    (OUT / "tab_main.tex").write_text(tab)
    (OUT / "tab_ablation.tex").write_text(ablation_table(dfs["holdout"]))
    if not det.empty:
        (OUT / "tab_detect.tex").write_text(detect_table(det))
    (OUT / "numbers.tex").write_text(numbers(dfs, det, tests, theory))
    (ROOT / "stats.json").write_text(json.dumps(tests, indent=1, default=float))
    print(json.dumps(tests, indent=1, default=float))
    print("wrote tables to", OUT)


if __name__ == "__main__":
    main()
