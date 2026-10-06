"""Run PHASE, its ablations and the stream baselines.

Protocol: temporal hold-out.  Every method sees the first 80% of the stream
(stream learners in a single pass), is then frozen, and predicts the last
20% without access to its labels.

    python run_experiments.py --datasets sea sea10 sea5 elec2 --seeds 0 1 2 3 4 --jobs 10
    python run_experiments.py --datasets covtype --seeds 0 --jobs 6

Each run writes results/runs/<dataset>__<method>__<seed>.json; existing files
are skipped, so the script can be interrupted and resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import baselines
from data import load
from detector import match_change_points
from metrics import evaluate
from phase import PHASE, PhaseConfig

OUT = Path(__file__).resolve().parent / "results" / "runs"
TRAIN_FRACTION = 0.8

PHASE_VARIANTS = {
    "PHASE":          {},
    "PHASE-pooled":   {"detector": "pooled"},
    "PHASE-periodic": {"detector": "periodic"},
    "PHASE-noseg":    {"detector": "none"},
    "PHASE-nocost":   {"use_costs": False},
    "PHASE-norefine": {"use_refine": False},
    "PHASE-nometa":   {"meta": "never"},
    "PHASE-alwaysmeta": {"meta": "always"},
}
BASELINES = ["HAT", "ARF", "SRP", "OOB", "UOB"]
ALL_METHODS = list(PHASE_VARIANTS) + BASELINES


def run_one(dataset, method, seed):
    path = OUT / f"{dataset}__{method}__{seed}.json"
    if path.exists():
        return str(path), "skipped"
    X, y, meta = load(dataset, seed)
    n_tr = int(TRAIN_FRACTION * len(y))
    X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    rec = {"dataset": dataset, "name": meta["name"], "method": method, "seed": seed,
           "n_train": n_tr, "n_test": len(y_te), "n_features": X.shape[1],
           "n_classes": meta["n_classes"],
           "train_class_counts": np.bincount(y_tr, minlength=meta["n_classes"]).tolist(),
           "test_class_counts": np.bincount(y_te, minlength=meta["n_classes"]).tolist()}
    t0 = time.time()
    if method in PHASE_VARIANTS:
        cfg = replace(PhaseConfig(seed=seed), **PHASE_VARIANTS[method])
        model = PHASE(cfg, meta["n_classes"]).fit(X_tr, y_tr)
        t_fit = time.time() - t0
        y_pred = model.predict(X_te)
        rec["config"] = asdict(cfg)
        rec["boundaries"], rec["alarms"] = model.boundaries, model.alarms
        rec["segments"] = [asdict(s) for s in model.segments]
        rec["stacked"] = bool(model.stack)
        rec["gate"] = model.gate
        if model.stack:
            rec["meta_importance"] = model.meta_importance_by_learner()
        cps = meta.get("change_points")
        if cps:
            rec["true_change_points"] = [c for c in cps if c < n_tr]
            rec["detection"] = match_change_points(rec["true_change_points"], model.boundaries, model.alarms)
    else:
        majority = int(np.bincount(y_tr).argmax())
        y_pred = baselines.run(method, X_tr, y_tr, X_te, seed, fallback=majority)
        t_fit = time.time() - t0
    rec["time_total_s"] = time.time() - t0
    rec["time_fit_s"] = t_fit
    rec["metrics"] = evaluate(y_te, y_pred)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return str(path), f"{rec['metrics']['acc']:.4f} acc, {rec['metrics']['gmean']:.4f} gmean, {rec['time_total_s']:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sea", "sea10", "sea5", "elec2"])
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    tasks = [(d, m, s) for d in args.datasets for s in args.seeds for m in args.methods]
    print(f"{len(tasks)} runs on {args.jobs} workers", flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(run_one, *t): t for t in tasks}
        for f in as_completed(futs):
            t = futs[f]
            try:
                path, msg = f.result()
                print(f"[done] {t}: {msg}", flush=True)
            except Exception:
                print(f"[FAIL] {t}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
