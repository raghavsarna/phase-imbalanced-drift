"""Two small studies for the supplement.

elec2   Dependence of the residuals on a real autocorrelated stream (R1-4b).
        Elec2 (Harries, 1999) in its original order, block-shuffled (blocks of
        1000 samples permuted: short-range dependence kept, long-range trend
        broken) and fully shuffled (Assumption 1 holds).  For each order: the
        lag-1 autocorrelation of the residuals within each class sequence, and
        the alarm rate of the uncalibrated LR bank (Clopper--Pearson rates,
        h = log(2n/alpha)) and of the pooled LR-CUSUM with the same guarantee,
        with restarts after every alarm.
timing  Cost per sample of every detector (median of 3 runs) on one S5 stream
        with its calibrated threshold; RBM-IM includes its online training.
Output: results_v2/extras.json
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

import bench as B
import detectors as D

OUT = B.ROOT / "extras.json"


def _elec2():
    sys.path.insert(0, str(B.ROOT.parent))
    from data import elec2
    X, y, _ = elec2()
    return X, y


def lag1_within_class(y, e):
    r = []
    for k in np.unique(y):
        s = e[y == k].astype(float)
        if len(s) > 10 and s.std() > 0:
            r.append(np.corrcoef(s[:-1], s[1:])[0, 1])
    return float(np.mean(r))


def elec2(seeds=range(10), W_ref=3000, W_cal=3000, alpha=0.05):
    X0, y0 = _elec2()
    n = len(y0)
    rows = {}
    for order in ("natural", "block", "shuffled"):
        out = []
        for s in (seeds if order != "natural" else [0]):
            rng = np.random.default_rng(s)
            if order == "natural":
                idx = np.arange(n)
            elif order == "block":
                blocks = np.array_split(np.arange(n), n // 1000)
                idx = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
            else:
                idx = rng.permutation(n)
            X, y = X0[idx], y0[idx]
            rf = B.reference(X, y, W_ref, s)
            yhat = rf.predict(X).astype(np.int64)
            ctx = B.context(y.astype(np.int64), yhat, dict(K=2, W_ref=W_ref, W_cal=W_cal, tau=None, target=0))
            t0, e = ctx["t0"], ctx["e"]
            n_mon = n - t0
            h = math.log(2 * n_mon / alpha)
            a_bank = D.bank_lr(ctx["y"], e, ctx["L_theory"], h, t0)
            q = D.odds_design(np.array([ctx["pool_t"]]), B.ODDS)
            a_pool = D.pooled_lr(e, D.llr_table([ctx["pool_t"]], q), h, t0)
            out.append({"lag1": lag1_within_class(ctx["y"][t0:], e[t0:]),
                        "bank_alarms_per_1e4": len(a_bank) / n_mon * 1e4,
                        "pooled_alarms_per_1e4": len(a_pool) / n_mon * 1e4,
                        "bank_any": bool(len(a_bank)), "pooled_any": bool(len(a_pool))})
        rows[order] = {k: float(np.mean([o[k] for o in out])) for k in out[0]}
        print(order, rows[order], flush=True)
    rows["n"] = int(n)
    return rows


def timing(family="sea5", seed=0):
    res = json.loads((B.ROOT / "bench" / f"{family}.json").read_text())
    c = B.prepare(family, seed, True)
    o = res["oracle"]
    names = [d for d in B.DETECTORS if d in res["det"] and not d.startswith("bank-r")]
    out = {}
    for d in names:
        fn, k = B.DETECTORS[d][0], res["det"][d]["knob"]
        fn(c, k, o)                                   # compile / warm-up
        ts = []
        for _ in range(3):
            c.pop("rbm", None)
            t = time.perf_counter()
            fn(c, k, o)
            ts.append(time.perf_counter() - t)
        out[d] = float(np.median(ts)) / (c["n"] - c["t_w"]) * 1e6      # microseconds per sample
        print(f"{d:14s} {out[d]:8.3f} us/sample", flush=True)
    return out


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for part in sys.argv[1:] or ["elec2", "timing"]:
        res[part] = globals()[part]()
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
