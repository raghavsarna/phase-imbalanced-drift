"""Per-sample prequential predictions for the accuracy-over-time plot (Fig. 2(a)).

Re-runs the prequential protocol of run_experiments.py for a few streams and
methods with identical settings and seeds, and stores the predictions in
results/timeline/<dataset>__<method>__<seed>.npz.  The overall balanced
accuracy of every run is checked against the logged run in results/preq/.

    python timeline.py --datasets sea10 --jobs 8
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import baselines
import prequential
from data import load
from metrics import evaluate
from phase import PhaseConfig
from run_experiments import PHASE_VARIANTS, ROOT, _detect

OUT = ROOT / "timeline"
METHODS = ["PHASE", "PHASE-pooled", "PHASE-noseg", "ROSE", "UOB", "ARF-US"]


def run(dataset, method, seed):
    path = OUT / f"{dataset}__{method}__{seed}.npz"
    if path.exists():
        return path.name, "skipped"
    X, y, meta = load(dataset, seed)
    K = meta["n_classes"]
    if method in PHASE_VARIANTS:
        cfg = replace(PhaseConfig(seed=seed), **PHASE_VARIANTS[method])
        pred, _ = prequential.phase_prequential(X, y, K, cfg, detection=_detect(dataset, "preq", cfg, K, X, y))
    elif method == "ROSE":
        pred = baselines.run_rose(X, y, K, 0, seed)
    else:
        pred = prequential.baseline_prequential(method, X, y, seed)
    w = prequential.WARMUP
    ba = evaluate(y[w:], pred[w:])["ba"]
    logged = json.loads((ROOT / "preq" / f"{dataset}__{method}__{seed}.json").read_text())["metrics"]["ba"]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y=y.astype(np.int8), pred=pred.astype(np.int8),
                        change_points=np.asarray(meta.get("change_points") or [], dtype=int))
    return path.name, f"BA {ba:.4f} (logged {logged:.4f}){'' if abs(ba - logged) < 1e-9 else '  MISMATCH'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sea10"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    tasks = [(d, m, s) for d in args.datasets for m in args.methods for s in args.seeds]
    with ProcessPoolExecutor(args.jobs) as pool:
        for name, msg in pool.map(run, *zip(*tasks)):
            print(name, msg, flush=True)


if __name__ == "__main__":
    main()
