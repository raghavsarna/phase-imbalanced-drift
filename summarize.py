"""Aggregate results/runs/*.json into summary tables (CSV + LaTeX rows).

    python summarize.py            -> results/summary.csv, results/detection.csv,
                                      results/tables.tex, printed overview
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
DATASETS = ["sea", "sea10", "sea5", "elec2", "covtype"]
LABEL = {"sea": "SEA", "sea10": "SEA-10\\%", "sea5": "SEA-5\\%", "elec2": "Elec2", "covtype": "Covertype"}
BASELINES = ["HAT", "ARF", "SRP", "OOB", "UOB"]
ABLATIONS = ["PHASE-pooled", "PHASE-periodic", "PHASE-noseg", "PHASE-nocost", "PHASE-norefine",
             "PHASE-nometa", "PHASE-alwaysmeta"]


def load_runs():
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text())
        m = r["metrics"]
        row = {"dataset": r["dataset"], "method": r["method"], "seed": r["seed"],
               "acc": m["acc"], "ba": m["ba"], "gmean": m["gmean"], "kappa": m["kappa"],
               "mcc": m["mcc"], "macro_f1": m["macro_f1"], "min_recall": m["minority_recall"],
               "time_s": r["time_total_s"]}
        if "segments" in r:
            row["n_segments"] = len(r["segments"])
            row["svm_frac"] = np.mean([s["svm_admitted"] for s in r["segments"]])
            row["cost_on_frac"] = np.mean([np.mean(list(s["b_star"].values())) for s in r["segments"]])
            row["last_seg_len"] = r["segments"][-1]["n"]
        if "stacked" in r:
            row["stacked"] = float(r["stacked"])
        if r.get("gate"):
            row["gate_meta"], row["gate_fused"] = r["gate"]["ba_meta"], r["gate"]["ba_fused"]
        if "meta_importance" in r:
            for k, v in r["meta_importance"].items():
                row[f"imp_{k}"] = v
        if "detection" in r:
            d = r["detection"]
            row.update({"det": d["detected"], "n_true": d["n_true"], "fa": d["false_alarms"],
                        "delay": d["mean_delay"], "loc": d["mean_abs_loc_error"]})
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean, std, best=False, scale=100, digits=1):
    s = f"{scale * mean:.{digits}f}"
    if not np.isnan(std) and std > 0:
        s += f"\\,{{\\scriptsize$\\pm${scale * std:.{digits}f}}}"
    return f"\\textbf{{{s}}}" if best else s


def main():
    df = load_runs()
    if df.empty:
        print("no runs yet")
        return
    agg = df.groupby(["dataset", "method"]).agg(["mean", "std", "count"])
    agg.to_csv(ROOT / "results" / "summary.csv")

    # ---- overview
    for metric in ["gmean", "kappa", "acc", "ba", "min_recall", "time_s"]:
        piv = df.pivot_table(index="method", columns="dataset", values=metric, aggfunc="mean")
        piv = piv[[d for d in DATASETS if d in piv.columns]]
        print(f"\n== {metric} (mean over seeds)")
        print((piv if metric == "time_s" else 100 * piv).round(2).to_string())
    if "stacked" in df:
        st = df[df["method"] == "PHASE"].groupby("dataset")[["stacked", "n_segments", "svm_frac", "cost_on_frac", "last_seg_len"]].mean()
        print("\n== PHASE: fraction stacked by the gate, segments, SVM admitted, costs on, last segment length\n", st.round(2).to_string())
    n = df.groupby(["dataset", "method"]).size().unstack(0)
    print("\n== runs per cell\n", n.to_string())

    # ---- detection on SEA streams
    det = df[df["det"].notna()] if "det" in df else pd.DataFrame()
    if not det.empty:
        dsum = det.groupby(["dataset", "method"]).agg(
            detected=("det", "sum"), n_true=("n_true", "sum"), fa=("fa", "sum"),
            delay=("delay", "mean"), loc=("loc", "mean"), seeds=("seed", "count"))
        dsum.to_csv(ROOT / "results" / "detection.csv")
        print("\n== detection (SEA, summed over seeds)\n", dsum.round(1).to_string())

    # ---- LaTeX rows
    lines = []
    for metric in ["gmean", "kappa"]:
        lines.append(f"% ---- {metric}: methods x datasets (mean +- std over seeds), best in bold")
        g = df.groupby(["method", "dataset"])[metric]
        piv_m = g.mean().unstack()
        piv_s = g.std().unstack().reindex_like(piv_m)
        methods = [m for m in BASELINES + ["PHASE"] if m in piv_m.index]
        for m in methods + [a for a in ABLATIONS if a in piv_m.index]:
            cells = []
            for d in DATASETS:
                if d not in piv_m.columns or m not in piv_m.index or np.isnan(piv_m.loc[m, d]):
                    cells.append("--")
                    continue
                best = m in methods and piv_m.loc[methods, d].max() == piv_m.loc[m, d]
                cells.append(fmt(piv_m.loc[m, d], piv_s.loc[m, d], best))
            lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    (ROOT / "results" / "tables.tex").write_text("\n".join(lines) + "\n")
    print("\nwrote results/summary.csv, results/detection.csv, results/tables.tex")


if __name__ == "__main__":
    main()
