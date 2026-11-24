"""Streams with several successive changes, with the full detect -> refit -> recalibrate loop.

After every alarm at time a, the reference model is refitted on [a, a + W_ref),
the rates are refrozen on the next W_cal samples and monitoring resumes; the
detector is blind during these W_ref + W_cal samples.  Thresholds are those
calibrated at ARL 5e4 on the corresponding single-change family (bench.py).

Streams
  seamulti    SEA, 5% minority, theta 8 -> 9.5 -> 7 -> 9.5 -> 8 every 40k samples:
              the changes alternately raise the minority (class 0) and the
              majority (class 1) residuals of the current model.
  gaussmulti  binary Gaussian, 5% minority whose mean moves 3.0 -> 2.2 -> 1.6 -> 1.2 -> 0.9
              (along the diagonal, distance from the majority mean) every 40k samples, so every
              change raises the minority error of a model fitted on the previous concept.
  insectsnat  INSECTS abrupt-imbalanced in its original order: real temperature changes at
              the five positions of Souza et al. (2020), Table 2 (83859, 128651, 182320,
              242883, 268380), affecting all six classes.

A change tau_j is detected by the first alarm in [tau_j, tau_{j+1}); a miss counts
the whole interval as delay; every other alarm is a false alarm.
Output: results_v2/multichange/<stream>.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import bench as B
from baselines import RBMIM
from streams import _gauss_stream, _real, _sea_block

OUT = B.ROOT / "multichange"
INSECT_CPS = [83859, 128651, 182320, 242883, 268380]
POLICIES = ["bank", "pooled", "ph-bank", "ph-pooled", "ra", "ddm-bank", "ddm-pooled", "adwin-bank",
            "adwin-pooled", "ddm-oci", "lfr", "rbmim"]
CAL_FAMILY = {"seamulti": "sea5", "gaussmulti": "gauss5", "insectsnat": "insects"}


def seamulti(seed, seg=40_000, thetas=(8.0, 9.5, 7.0, 9.5, 8.0)):
    rng = np.random.default_rng(seed)
    blocks = [_sea_block(rng, seg, th, 0.05) for th in thetas]
    X, y = np.vstack([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks])
    return X, y, dict(K=2, taus=[seg * j for j in range(1, len(thetas))], W_ref=2000, W_cal=3000)


def gaussmulti(seed, seg=40_000, pos=(3.0, 2.2, 1.6, 1.2, 0.9), d=5):
    rng = np.random.default_rng(seed)
    u = np.ones(d) / np.sqrt(d)
    Xs, ys = [], []
    for p in pos:
        X, y = _gauss_stream(rng, np.array([0.95, 0.05]), np.vstack([np.zeros(d), p * u]), seg, None, 1, None)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys), dict(K=2, taus=[seg * j for j in range(1, len(pos))],
                                                   W_ref=2000, W_cal=3000)


def insectsnat(seed):
    X, y = _real("insects")                      # standardised, original order
    return X.astype(float), y, dict(K=int(y.max()) + 1, taus=list(INSECT_CPS), W_ref=10_000, W_cal=20_000)


STREAMS = {"seamulti": seamulti, "gaussmulti": gaussmulti, "insectsnat": insectsnat}


def run(stream, seed, policy, knob, o):
    X, y, info = STREAMS[stream](seed)
    n, Wr, Wc, K = len(y), info["W_ref"], info["W_cal"], info["K"]
    fn = B.DETECTORS[policy][0]
    alarms, start = [], 0
    while start + Wr + Wc < n:
        model = B.reference(X[start:start + Wr], y[start:start + Wr], Wr, seed)
        seg = slice(start, n)
        yy = y[seg].astype(np.int64)
        pp = np.concatenate([model.predict(X[s:min(n, s + 50_000)]) for s in range(start, n, 50_000)]).astype(np.int64)
        ctx = B.context(yy, pp, dict(K=K, W_ref=Wr, W_cal=Wc, tau=None, target=0))
        ctx["src"], ctx["seed"] = None, seed
        if policy == "rbmim":
            r = RBMIM(X.shape[1], K, seed=seed)
            r.fit_reference(X[start:start + Wr], yy[:Wr])
            ctx["rbm"] = r.stream_errors(X[seg], yy, Wr)
        al = fn(ctx, knob, o)
        if not len(al):
            break
        a = start + int(al[0])
        alarms.append(a)
        start = a
    taus = info["taus"]
    bounds = taus + [n]
    det, delays, used = [], [], set()
    for j, t in enumerate(taus):
        hit = [a for a in alarms if t <= a < bounds[j + 1]]
        det.append(bool(hit))
        delays.append(hit[0] - t if hit else bounds[j + 1] - t)
        if hit:
            used.add(hit[0])
    return {"seed": seed, "alarms": alarms, "detected": det, "delays": delays,
            "false_alarms": len([a for a in alarms if a not in used]), "n": n}


def _job(a):
    return run(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("streams", nargs="*", default=list(STREAMS))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.streams:
        f = OUT / f"{s}.json"
        res = json.loads(f.read_text()) if f.exists() else {}
        cal = json.loads((B.ROOT / "bench" / f"{CAL_FAMILY[s]}.json").read_text())
        nseeds = 10 if s == "insectsnat" else a.seeds          # natural order: seeds only vary the forest
        for pol in POLICIES:
            if pol in res:
                continue
            t = time.time()
            knob = cal["det"][pol]["knob"]
            with ProcessPoolExecutor(a.jobs) as ex:
                runs = list(ex.map(_job, [(s, sd, pol, knob, cal["oracle"]) for sd in range(nseeds)]))
            res[pol] = runs
            f.write_text(json.dumps(res, indent=1))
            det = np.mean([np.mean(r["detected"]) for r in runs])
            dl = np.mean([np.mean(r["delays"]) for r in runs])
            fa = np.mean([r["false_alarms"] / r["n"] * 1e5 for r in runs])
            print(f"{s:11s} {pol:13s} detected={det:.2f} delay={dl:8.0f} FA/1e5={fa:5.2f} ({time.time() - t:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
