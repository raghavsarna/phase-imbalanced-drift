"""Run PHASE, its ablations and the stream baselines under two protocols.

holdout  temporal hold-out: every method sees the first 80% of the stream
         (stream learners in a single pass), is then frozen, and predicts the
         last 20% without access to its labels (deployment without labels);
preq     prequential, test-then-train over the whole stream (prequential.py).

    python run_experiments.py --protocol holdout --datasets sea sea10 sea5 elec2 --seeds 0 1 2 3 4
    python run_experiments.py --protocol preq --datasets covtype --methods PHASE ARF --seeds 0

Each run writes results/<protocol>/<dataset>__<method>__<seed>.json; existing
files are skipped, so the script can be interrupted and resumed.  Detector
outputs are cached in results/cache/ and shared by variants with the same
detector.
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
import prequential
from data import load
from detector import match_change_points
from metrics import evaluate
from phase import PHASE, PhaseConfig

ROOT = Path(os.environ.get("PHASE_RESULTS", Path(__file__).resolve().parent / "results"))
CACHE = ROOT / "cache"
TRAIN_FRACTION = 0.8
DATASETS = ["sea", "sea10", "sea5", "elec2", "covtype", "insects_abrupt", "insects_gradual", "creditcard"]

PHASE_VARIANTS = {
    "PHASE":              {},
    # Phase I: detector
    "PHASE-pooled":       {"detector": "pooled"},
    "PHASE-ddm":          {"detector": "pooled", "test": "ddm"},
    "PHASE-ddm-class":    {"detector": "class", "test": "ddm"},
    "PHASE-adwin":        {"detector": "pooled", "test": "adwin"},
    "PHASE-adwin-class":  {"detector": "class", "test": "adwin"},
    "PHASE-periodic":     {"detector": "periodic"},
    "PHASE-noseg":        {"detector": "none"},
    # Phases II-III and class memory
    "PHASE-nocost":       {"use_costs": False},
    "PHASE-norefine":     {"use_refine": False},
    "PHASE-nomemory":     {"memory": 0},
    # optional stacked meta-learner over all segments
    "PHASE-stack":        {"meta": "always"},
}
BASELINES = ["HAT", "ARF", "SRP", "OOB", "UOB", "ARF-US", "ROSE"]
MAIN = ["PHASE"] + BASELINES
PREQ_VARIANTS = ["PHASE", "PHASE-pooled", "PHASE-noseg"]


def _detect(dataset, protocol, cfg, K, X, y):
    """Detector output, cached per (protocol, dataset, detector, seed)."""
    if cfg.detector in ("none", "periodic"):
        return PHASE(cfg, K).detect(X, y)
    key = f"{protocol}__{dataset}__{cfg.detector}-{cfg.test}__{cfg.seed}.json"
    path = CACHE / key
    if path.exists():
        d = json.loads(path.read_text())
        return d["boundaries"], d["alarms"]
    t0 = time.time()
    b, a = PHASE(cfg, K).detect(X, y)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"boundaries": list(map(int, b)), "alarms": list(map(int, a)),
                               "time_s": time.time() - t0}))
    tmp.replace(path)
    return b, a


def run_one(protocol, dataset, method, seed):
    out = ROOT / protocol
    path = out / f"{dataset}__{method}__{seed}.json"
    if path.exists():
        return str(path), "skipped"
    X, y, meta = load(dataset, seed)
    K = meta["n_classes"]
    rec = {"protocol": protocol, "dataset": dataset, "name": meta["name"], "method": method, "seed": seed,
           "n": len(y), "n_features": X.shape[1], "n_classes": K}
    t0 = time.time()
    cps = meta.get("change_points")
    if protocol == "holdout":
        n_tr = int(TRAIN_FRACTION * len(y))
        X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
        rec.update(n_train=n_tr, n_test=len(y_te),
                   train_class_counts=np.bincount(y_tr, minlength=K).tolist(),
                   test_class_counts=np.bincount(y_te, minlength=K).tolist())
        if method in PHASE_VARIANTS:
            cfg = replace(PhaseConfig(seed=seed), **PHASE_VARIANTS[method])
            b, a = _detect(dataset, protocol, cfg, K, X_tr, y_tr)
            model = PHASE(cfg, K).fit_segments(X_tr, y_tr, b, a)
            y_pred = model.predict(X_te)
            rec["config"] = asdict(cfg)
            rec["boundaries"], rec["alarms"] = list(map(int, model.boundaries)), list(map(int, model.alarms))
            rec["segments"] = [asdict(s) for s in model.segments]
            rec["stacked"] = bool(model.stack)
            if cps:
                rec["true_change_points"] = [c for c in cps if c < n_tr]
                rec["detection"] = match_change_points(rec["true_change_points"], model.boundaries, model.alarms)
        elif method == "ROSE":
            y_pred = baselines.run_rose(X, y, K, n_tr, seed)
        else:
            majority = int(np.bincount(y_tr).argmax())
            y_pred = baselines.run(method, X_tr, y_tr, X_te, seed, fallback=majority)
        y_eval, start = y_te, n_tr
    else:
        w = prequential.WARMUP
        if method in PHASE_VARIANTS:
            cfg = replace(PhaseConfig(seed=seed), **PHASE_VARIANTS[method])
            det = _detect(dataset, protocol, cfg, K, X, y)
            pred, info = prequential.phase_prequential(X, y, K, cfg, detection=det)
            rec["config"] = asdict(cfg)
            rec.update(boundaries=info["boundaries"], alarms=info["alarms"], R=info["R"], W_max=info["W_max"],
                       n_refits=len(info["refits"]),
                       svm_frac=float(np.mean([r["svm"] for r in info["refits"]])) if info["refits"] else None)
            if cps:
                rec["true_change_points"] = list(cps)
                rec["detection"] = match_change_points(cps, info["boundaries"], info["alarms"])
        elif method == "ROSE":
            pred = baselines.run_rose(X, y, K, 0, seed)
        else:
            pred = prequential.baseline_prequential(method, X, y, seed)
        y_pred, y_eval, start = pred[w:], y[w:], w
        rec["windowed_ba"] = prequential.windowed_ba(y, pred, w)
        rec["warmup"] = w
    rec["time_total_s"] = time.time() - t0
    rec["metrics"] = evaluate(y_eval, y_pred)
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return str(path), f"BA {rec['metrics']['ba']:.4f}, kappa {rec['metrics']['kappa']:.4f}, {rec['time_total_s']:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["holdout", "preq"], default="holdout")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    methods = args.methods or (list(PHASE_VARIANTS) + BASELINES if args.protocol == "holdout"
                               else PREQ_VARIANTS + BASELINES)
    size = {"covtype": 9, "insects_abrupt": 8, "creditcard": 7, "insects_gradual": 6}
    slow = {"SRP": 3, "PHASE-noseg": 2, "PHASE-stack": 2, "PHASE": 1, "ARF": 1}
    if args.protocol == "preq":       # detector variants only on the smaller streams
        methods_for = lambda d: [m for m in methods if not (size.get(d) and m in ("PHASE-pooled", "PHASE-noseg"))]
    else:
        methods_for = lambda d: methods
    tasks = [(args.protocol, d, m, s) for d in args.datasets for s in args.seeds for m in methods_for(d)]
    tasks.sort(key=lambda t: (-size.get(t[1], 0), -slow.get(t[2], 0)))      # longest first
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
