"""Detector-triggered retraining with a fixed learner (R1-7).

The deployed classifier is the frozen class-balanced random forest that also
serves as reference model.  When the detector alarms at time a, the forest is
refitted on the W_ref labelled samples that follow a and redeployed at
a + W_ref; detection restarts on the new model (new calibration window, same
threshold).  Every detector uses the same learner and the same rule, so
differences in balanced accuracy (BA) are due to detection only.

Policies: the detectors of bench.py at their calibrated thresholds, 'none'
(never retrain), 'periodic' (retrain every 20 000 samples) and 'oracle'
(retrain once at the true change).  BA is computed prequentially over the
50 000 samples after the change, and over the whole monitored stream.
Output: results_v2/retrain/<family>.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import bench as B
from streams import load

OUT_DIR = B.ROOT / "retrain"
SEEDS = range(10)
POLICIES = ["bank", "pooled", "ph-pooled", "ddm-pooled", "adwin-pooled", "ddm-oci", "lfr", "rbmim",
            "none", "periodic", "oracle"]
WIN = 50_000


def balanced_acc(y, yp):
    cls = np.unique(y)
    return float(np.mean([np.mean(yp[y == c] == c) for c in cls]))


def run(family, seed, policy, res):
    X, y, info = load(family, seed, True)
    n, Wr, Wc, tau = len(y), info["W_ref"], info["W_cal"], info["tau"]
    o = res["oracle"]
    yp = np.empty(n, dtype=np.int64)
    model = B.reference(X, y, Wr, seed)
    start = Wr                                    # first sample predicted by the current model
    yp[:Wr] = y[:Wr]                              # training window: not scored
    retrains = []
    while start < n:
        pred = np.concatenate([model.predict(X[s:min(n, s + 50_000)]) for s in range(start, n, 50_000)])
        yp[start:] = pred
        if policy == "none":
            break
        if policy == "periodic":
            a = start + 20_000 - Wr
        elif policy == "oracle":
            a = tau if start <= tau else n
        else:
            seg = slice(start - Wr, n)
            inf = dict(info, W_ref=Wr, W_cal=Wc, tau=None)
            yy, pp = y[seg], yp[seg].copy()
            pp[:Wr] = model.predict(X[start - Wr:start])   # residuals of the training window: unused
            ctx = B.context(yy.astype(np.int64), pp.astype(np.int64), inf)
            ctx["src"], ctx["seed"] = None, seed
            if policy == "rbmim":
                ctx["rbm"] = _rbm(X[seg], yy, ctx, seed)
            al = B.DETECTORS[policy][0](ctx, res["det"][policy]["knob"], o)
            a = int(al[0]) + start - Wr if len(al) else n
        if a + Wr >= n:
            break
        model = B.reference(X[a:a + Wr], y[a:a + Wr], Wr, seed)
        retrains.append(int(a))
        start = a + Wr
    t0 = Wr + Wc
    post = slice(tau, min(n, tau + WIN))
    return {"ba_post": balanced_acc(y[post], yp[post]), "ba_all": balanced_acc(y[t0:], yp[t0:]),
            "retrains": retrains}


def _rbm(X, y, ctx, seed):
    from baselines import RBMIM
    r = RBMIM(X.shape[1], ctx["K"], seed=seed)
    r.fit_reference(X[:ctx["t_w"]], y[:ctx["t_w"]])
    return r.stream_errors(X, y, ctx["t_w"])


def _run_args(a):
    return run(*a)


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*", default=["sea5", "gauss5", "gmm5", "credit", "covtype"])
    ap.add_argument("--jobs", type=int, default=1, help="seeds run in parallel")
    a = ap.parse_args()
    fams = a.families
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fam in fams:
        OUT = OUT_DIR / f"{fam}.json"
        res = json.loads((B.ROOT / "bench" / f"{fam}.json").read_text())
        out = {fam: json.loads(OUT.read_text())} if OUT.exists() else {fam: {}}
        for pol in POLICIES:
            if pol in out[fam]:
                continue
            with ProcessPoolExecutor(a.jobs) as ex:
                runs = list(ex.map(_run_args, [(fam, s, pol, res) for s in SEEDS]))
            out[fam][pol] = runs
            print(fam, pol, f"BA_post={np.mean([r['ba_post'] for r in runs]):.3f} "
                  f"BA_all={np.mean([r['ba_all'] for r in runs]):.3f} "
                  f"retrains={np.mean([len(r['retrains']) for r in runs]):.1f}", flush=True)
            OUT.write_text(json.dumps(out[fam], indent=1))


if __name__ == "__main__":
    main()
