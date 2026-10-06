"""Write the LaTeX tables of the letter from results/runs/*.json.

    python make_tables.py  ->  ../SPL_submission/tables/tab_main.tex
                               ../SPL_submission/tables/tab_ablation.tex
                               ../SPL_submission/tables/numbers.tex  (\\newcommand macros used in the text)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from summarize import load_runs

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "SPL_submission" / "tables"
DS = ["sea", "sea10", "sea5", "elec2", "covtype"]
HEAD = ["SEA", "SEA-10\\%", "SEA-5\\%", "Elec2", "Covert."]
BASE = [("HAT", "HAT"), ("ARF", "ARF~\\cite{gomes2017arf}"), ("SRP", "SRP~\\cite{gomes2019srp}"),
        ("OOB", "OOB~\\cite{wang2015oob}"), ("UOB", "UOB~\\cite{wang2015oob}"), ("PHASE", "PHASE")]
ABL = [("PHASE", "PHASE (full)"),
       None, ("PHASE-pooled", "I: pooled PH test"), ("PHASE-periodic", "I: periodic, every 5000"),
       ("PHASE-noseg", "I: no segmentation"),
       None, ("PHASE-nocost", "II: no class costs"), ("PHASE-norefine", "III: no SVM refinement"),
       None, ("PHASE-nometa", "IV: never stack"), ("PHASE-alwaysmeta", "IV: always stack")]


def stats(df, method, ds, metric):
    v = df[(df.method == method) & (df.dataset == ds)][metric].to_numpy(dtype=float)
    return (np.nan, np.nan, 0) if len(v) == 0 else (v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan, len(v))


def cell(mean, std, bold=False, show_std=True):
    if np.isnan(mean):
        return "--"
    s = f"{100 * mean:.1f}"
    if show_std and not np.isnan(std):
        s += f"{{\\tiny$\\pm${100 * std:.1f}}}"
    return f"\\textbf{{{s}}}" if bold else s


def main_table(df):
    lines = [r"\begin{tabular}{l" + "c" * len(DS) + "}", r"\toprule",
             " & " + " & ".join(HEAD) + r" \\"]
    for metric, title in [("ba", "Balanced accuracy"), ("kappa", "Cohen's $\\kappa$")]:
        lines += [r"\midrule", rf"\multicolumn{{{len(DS) + 1}}}{{l}}{{\emph{{{title}}}}} \\"]
        best = {d: max(stats(df, m, d, metric)[0] for m, _ in BASE if not np.isnan(stats(df, m, d, metric)[0]))
                for d in DS}
        for m, name in BASE:
            cells = []
            for d in DS:
                mu, sd, _ = stats(df, m, d, metric)
                cells.append(cell(mu, sd, bold=(not np.isnan(mu) and np.isclose(mu, best[d]))))
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def ablation_table(df):
    lines = [r"\begin{tabular}{l" + "c" * len(DS) + "}", r"\toprule",
             "Variant & " + " & ".join(HEAD) + r" \\", r"\midrule"]
    ref = {d: stats(df, "PHASE", d, "ba")[0] for d in DS}
    for row in ABL:
        if row is None:
            lines.append(r"\addlinespace[1pt]")
            continue
        m, name = row
        cells = []
        for d in DS:
            mu = stats(df, m, d, "ba")[0]
            if np.isnan(mu):
                cells.append("--")
            elif m == "PHASE":
                cells.append(f"{100 * mu:.1f}")
            else:
                diff = 100 * (mu - ref[d])
                d_str = f"{diff:+.1f}".replace("-", "$-$")
                cells.append(f"{100 * mu:.1f}{{\\tiny\\,({d_str})}}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def numbers(df):
    """Macros for numbers quoted in the text (so text and tables cannot disagree)."""
    import json
    macros = {}
    for d, tag in zip(DS, ["Sea", "SeaTen", "SeaFive", "Elec", "Cov"]):
        for m, mt in [("PHASE", "Phase"), ("ARF", "Arf"), ("UOB", "Uob"), ("OOB", "Oob"),
                      ("SRP", "Srp"), ("HAT", "Hat")]:
            for metric, mm in [("ba", "BA"), ("kappa", "Kappa"), ("acc", "Acc"), ("min_recall", "MinRec"),
                               ("gmean", "Gmean")]:
                mu = stats(df, m, d, metric)[0]
                if not np.isnan(mu):
                    macros[f"{mt}{mm}{tag}"] = f"{100 * mu:.1f}"
            t = df[(df.method == m) & (df.dataset == d)]["time_s"]
            if len(t):
                macros[f"{mt}Time{tag}"] = f"{t.mean():.0f}"
    det = df[df["det"].notna()]
    for d, tag in zip(["sea", "sea10", "sea5"], ["Sea", "SeaTen", "SeaFive"]):
        for m, mt in [("PHASE", "Cls"), ("PHASE-pooled", "Pool")]:
            g = det[(det.dataset == d) & (det.method == m)]
            macros[f"Det{mt}{tag}"] = f"{int(g['det'].sum())}"
            macros[f"FA{mt}{tag}"] = f"{int(g['fa'].sum())}"
            macros[f"Delay{mt}{tag}"] = f"{g['delay'].mean():.0f}"
            macros[f"Loc{mt}{tag}"] = f"{g['loc'].mean():.0f}"
    seg = df[df.method == "PHASE"].groupby("dataset")[["n_segments", "stacked", "svm_frac", "last_seg_len"]].mean()
    for d, tag in zip(DS, ["Sea", "SeaTen", "SeaFive", "Elec", "Cov"]):
        if d in seg.index:
            macros[f"Segs{tag}"] = f"{seg.loc[d, 'n_segments']:.0f}" if d == "covtype" else f"{seg.loc[d, 'n_segments']:.1f}"
            macros[f"Stack{tag}"] = f"{100 * seg.loc[d, 'stacked']:.0f}"
            macros[f"LastSeg{tag}"] = f"{seg.loc[d, 'last_seg_len']:.0f}"
    # paired seed wins of PHASE over the best baseline (balanced accuracy)
    for d, tag in zip(["sea", "sea10", "sea5", "elec2"], ["Sea", "SeaTen", "SeaFive", "Elec"]):
        means = {m: stats(df, m, d, "ba")[0] for m, _ in BASE[:-1]}
        best = max(means, key=means.get)
        a = df[(df.method == "PHASE") & (df.dataset == d)].set_index("seed")["ba"]
        b = df[(df.method == best) & (df.dataset == d)].set_index("seed")["ba"]
        macros[f"Wins{tag}"] = f"{int((a > b).sum())}"
        macros[f"Gain{tag}"] = f"{100 * (a - b).mean():.1f}"
        macros[f"Best{tag}"] = best
    for m, mt in [("PHASE-noseg", "NoSeg"), ("PHASE-pooled", "Pooled"), ("PHASE-periodic", "Periodic"),
                  ("PHASE-nocost", "NoCost"), ("PHASE-nometa", "Never"), ("PHASE-alwaysmeta", "Always")]:
        for d, tag in zip(DS, ["Sea", "SeaTen", "SeaFive", "Elec", "Cov"]):
            mu = stats(df, m, d, "ba")[0]
            if not np.isnan(mu):
                macros[f"{mt}BA{tag}"] = f"{100 * mu:.1f}"
    lines = [f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    return "\n".join(lines) + "\n"


def main():
    df = load_runs()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_main.tex").write_text(main_table(df))
    (OUT / "tab_ablation.tex").write_text(ablation_table(df))
    (OUT / "numbers.tex").write_text(numbers(df))
    print("wrote", *(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
