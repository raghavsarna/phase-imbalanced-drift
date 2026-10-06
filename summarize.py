"""Load the experiment logs into data frames and print an overview.

    python summarize.py   ->  results/summary_<protocol>.csv, results/detection_summary.csv
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PHASE_RESULTS", Path(__file__).resolve().parent / "results"))
DATASETS = ["sea", "sea10", "sea5", "elec2", "covtype", "insects_abrupt", "insects_gradual", "creditcard"]
BASELINES = ["HAT", "ARF", "SRP", "OOB", "UOB", "ARF-US", "ROSE"]
MAIN = ["PHASE"] + BASELINES


def load_runs(protocol="holdout"):
    rows = []
    for f in sorted((ROOT / protocol).glob("*.json")):
        r = json.loads(f.read_text())
        m = r["metrics"]
        row = {"protocol": protocol, "dataset": r["dataset"], "method": r["method"], "seed": r["seed"],
               "acc": m["acc"], "ba": m["ba"], "gmean": m["gmean"], "kappa": m["kappa"],
               "mcc": m["mcc"], "macro_f1": m["macro_f1"], "min_recall": m["minority_recall"],
               "time_s": r.get("time_total_s", np.nan),
               "n_boundaries": len(r["boundaries"]) if "boundaries" in r else np.nan}
        if r.get("segments"):
            last = r["segments"][-1]
            row.update(last_seg_len=last["n"], n_memory=last.get("n_memory", 0),
                       last_seg_classes=len(last["class_counts"]),
                       svm=float(last.get("svm_admitted", False)))
        if "detection" in r:
            d = r["detection"]
            row.update(det=d["detected"], n_true=d["n_true"], fa=d["false_alarms"], delay=d["mean_delay"])
        if "n_refits" in r:
            row["n_refits"] = r["n_refits"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_detection():
    rows = []
    for f in sorted((ROOT / "detection").glob("*.json")):
        r = json.loads(f.read_text())
        rows.append({"stream": r["stream"], "detector": f"{r['mode']}-{r['test']}", "seed": r["seed"],
                     "share": r["minority_share"], "det": r["detected"], "n_true": r["n_true"],
                     "fa": r["false_alarms"], "delay": r["mean_delay"], "loc": r["mean_abs_loc_error"],
                     **{f"hit_{c}": int(c in r["hits"]) for c in (10000, 20000, 30000)}})
    return pd.DataFrame(rows)


def main():
    for protocol in ("holdout", "preq"):
        df = load_runs(protocol)
        if df.empty:
            continue
        df.drop(columns="protocol").groupby(["dataset", "method"]).agg(["mean", "std", "count"]).to_csv(ROOT / f"summary_{protocol}.csv")
        for metric in ("ba", "kappa"):
            piv = 100 * df.pivot_table(index="method", columns="dataset", values=metric, aggfunc="mean")
            print(f"\n== {protocol}: {metric}\n", piv[[d for d in DATASETS if d in piv]].round(1).to_string())
        print(f"\n== {protocol}: runs per cell\n",
              df.groupby(["method", "dataset"]).size().unstack()[[d for d in DATASETS if d in set(df.dataset)]].to_string())
    det = load_detection()
    if not det.empty:
        s = det.groupby(["stream", "detector"]).agg(det=("det", "sum"), n_true=("n_true", "sum"), fa=("fa", "sum"),
                                                    delay=("delay", "mean"), loc=("loc", "mean"))
        s.to_csv(ROOT / "detection_summary.csv")
        print("\n== detection\n", s.round(0).to_string())


if __name__ == "__main__":
    main()
