"""Prequential (test-then-train) evaluation.

Every sample is first predicted and then revealed to the learner.  Stream
baselines are updated after every sample.  PHASE runs its detector causally
on the whole stream and refits its base layer on the current segment (at most
the last ``W_max`` samples, plus the class memory) every ``R`` samples and
right after every alarm; between refits the model is fixed.  Predictions of
the first ``warmup`` samples are not scored, for any method.
"""
from __future__ import annotations

import numpy as np

import baselines
from phase import PHASE, memory_indices

R_REFIT = 1000
W_MAX = 10_000
WARMUP = 1000
N_MIN = 200          # smallest new-segment window that triggers a refit


def phase_prequential(X, y, K, cfg, R=R_REFIT, W_max=W_MAX, warmup=WARMUP, detection=None):
    n = len(y)
    if detection is None:
        detection = PHASE(cfg, K).detect(X, y)
    boundaries, alarms = detection
    events = sorted(set(range(warmup, n, R)) | {a + 1 for a in alarms if a + 1 < n})
    pred = np.full(n, -1, dtype=int)
    model, log = None, []
    for i, e in enumerate(events):
        seg_start = max([b for b, a in zip(boundaries, alarms) if a < e], default=0)
        a = max(seg_start, e - W_max)
        if e - a < N_MIN and model is not None:
            a = None
        if a is not None:
            idx = np.concatenate([memory_indices(y, a, e, cfg.memory), np.arange(a, e)])
            if len(np.unique(y[idx])) >= 2:
                try:
                    new = PHASE(cfg, K).fit_window(X[idx], y[idx], seed=cfg.seed + i)
                except ValueError:       # a cross-fitting fold holds a single class: keep the model
                    new = None
            else:
                new = None
            if new is not None:
                model = new
                info = model.segments[0]
                log.append({"t": int(e), "start": int(a), "n": int(e - a), "n_memory": int(len(idx) - (e - a)),
                            "svm": bool(info.svm_admitted), "costs_on": float(np.mean(list(info.b_star.values())))})
        hi = events[i + 1] if i + 1 < len(events) else n
        if model is not None:
            pred[e:hi] = model.predict(X[e:hi])
        else:
            pred[e:hi] = np.bincount(y[:e]).argmax()
    return pred, {"boundaries": list(map(int, boundaries)), "alarms": list(map(int, alarms)),
                  "refits": log, "R": R, "W_max": W_max}


def baseline_prequential(name, X, y, seed):
    model = baselines.make(name, seed)
    n = len(y)
    pred = np.empty(n, dtype=int)
    majority = {}
    for t in range(n):
        x = dict(enumerate(X[t]))
        p = model.predict_one(x)
        pred[t] = max(majority, key=majority.get) if (p is None and majority) else (0 if p is None else int(p))
        yt = int(y[t])
        majority[yt] = majority.get(yt, 0) + 1
        model.learn_one(x, yt)
    return pred


def windowed_ba(y, pred, start, width=2000):
    """Balanced accuracy in consecutive windows (for plots)."""
    out = []
    for a in range(start, len(y) - width + 1, width):
        yy, pp = y[a:a + width], pred[a:a + width]
        rec = [np.mean(pp[yy == c] == c) for c in np.unique(yy)]
        out.append(float(np.mean(rec)))
    return out
