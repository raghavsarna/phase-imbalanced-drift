"""Benchmark streams with a single known change that affects one minority class.

Every generator returns (X, y, info) where info holds
  K        number of classes
  tau      change time (None for a drift-free stream)
  target   the class whose conditional law changes (the 'changed class')
  W_ref    samples used to fit the frozen reference model
  W_cal    following samples used to freeze the pre-change residual rates
  n        stream length
Monitoring starts at t0 = W_ref + W_cal and the change happens at tau > t0.
``change=False`` gives the drift-free (null) version used to calibrate and
measure the average run length.

Families
  sea      SEA (Street & Kim, 2001): y = 1{x1 + x2 > theta}, 10% label noise,
           class-0 share fixed by rejection; theta 8 -> 9.5 enlarges the
           class-0 region, so only class-0 residuals of a frozen model rise.
  gauss    binary Gaussian, d = 5: the minority mean moves towards the majority.
  gmm5     five Gaussian classes, priors .45/.30/.15/.07/.03; the 3% class moves.
  credit   CreditCard (Dal Pozzolo et al., 2015), 0.17% fraud, shuffled;
  covtype  Covertype (Blackard & Dean, 1999), 7 classes, shuffled;
  insects  INSECTS abrupt-imbalanced (Souza et al., 2020), 6 classes, shuffled;
           for the three real streams the order is randomised (removing the
           natural drift) and, after tau, the features of the smallest class
           are moved a fraction s of the way to the centroid of the other
           classes (standardised units): a known minority-only covariate change.
  priorshift   SEA with fixed theta whose class-0 share jumps 5% -> 15% at tau
           (no change of P(x|y) and of the class-conditional residuals).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RIVER = Path(os.environ.get("RIVER_DATA", Path.home() / "river_data"))
COVTYPE_CSV = Path(os.environ.get("PHASE_COVTYPE_CSV",
                                  Path(__file__).resolve().parents[1] / "data" / "covtype.csv"))


# --------------------------------------------------------------------------- #
def _sea_block(rng, n, theta, share, noise=0.1):
    """n SEA samples with threshold theta and class-0 share 'share' (rejection)."""
    X, y, got = [], [], 0
    while got < n:
        m = 4 * (n - got) + 1000
        x = rng.uniform(0, 10, size=(m, 3))
        lab = (x[:, 0] + x[:, 1] > theta).astype(int)
        flip = rng.random(m) < noise
        lab = np.where(flip, 1 - lab, lab)
        X.append(x)
        y.append(lab)
        got += m
    X, y = np.vstack(X), np.concatenate(y)
    if share is None:
        return X[:n], y[:n]
    n0 = int(round(share * n))
    i0, i1 = np.flatnonzero(y == 0)[:n0], np.flatnonzero(y == 1)[:n - n0]
    if len(i0) < n0 or len(i1) < n - n0:
        return _sea_block(rng, 2 * n, theta, share, noise)[0][:n], None  # pragma: no cover
    idx = np.concatenate([i0, i1])
    idx = idx[rng.permutation(n)]
    return X[idx], y[idx]


def sea(seed, share=0.05, change=True, theta0=8.0, theta1=9.5,
        W_ref=2000, W_cal=3000, pre=10_000, post=200_000, ramp=0, target=0):
    """ramp > 0: theta moves linearly from theta0 to theta1 over 'ramp' samples after tau
    (gradual drift).  theta1 < theta0 shrinks the class-0 region, so only class-1 residuals rise
    (target=1, a majority-only change when class 0 is the minority)."""
    rng = np.random.default_rng(seed)
    t0 = W_ref + W_cal
    tau = t0 + pre
    n = tau + post
    Xa, ya = _sea_block(rng, tau, theta0, share)
    if change and ramp > 0:
        blocks, step = [], 1000
        for i in range(0, post, step):
            th = theta0 + (theta1 - theta0) * min(1.0, (i + step / 2) / ramp)
            blocks.append(_sea_block(rng, min(step, post - i), th, share))
        Xb, yb = np.vstack([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks])
    else:
        Xb, yb = _sea_block(rng, post, theta1 if change else theta0, share)
    return (np.vstack([Xa, Xb]), np.concatenate([ya, yb]),
            dict(K=2, tau=tau if change else None, target=target, W_ref=W_ref, W_cal=W_cal, n=n,
                 family="sea", share=share))


def priorshift(seed, share0=0.05, share1=0.15, W_ref=2000, W_cal=3000, pre=10_000, post=20_000):
    rng = np.random.default_rng(seed)
    t0 = W_ref + W_cal
    tau = t0 + pre
    Xa, ya = _sea_block(rng, tau, 8.0, share0)
    Xb, yb = _sea_block(rng, post, 8.0, share1)
    return (np.vstack([Xa, Xb]), np.concatenate([ya, yb]),
            dict(K=2, tau=tau, target=0, W_ref=W_ref, W_cal=W_cal, n=tau + post,
                 family="priorshift", share=share0))


# --------------------------------------------------------------------------- #
def _gauss_stream(rng, priors, means, n, tau, target, new_mean, ramp=0, extra=()):
    """Class 'target' moves to new_mean at tau (linearly over 'ramp' samples if ramp > 0);
    extra: further (class, new_mean) pairs that move at the same time."""
    K, d = means.shape
    y = rng.choice(K, size=n, p=priors)
    X = rng.standard_normal((n, d)) + means[y]
    if tau is not None:
        t = np.arange(n)
        frac = np.clip((t - tau + 1) / ramp, 0, 1) if ramp > 0 else (t >= tau).astype(float)
        for k, nm in ((target, new_mean), *extra):
            sel = y == k
            X[sel] += frac[sel, None] * (nm - means[k])
    return X, y


def gauss(seed, share=0.05, change=True, d=5, sep=2.5, moved=1.5,
          W_ref=2000, W_cal=3000, pre=10_000, post=200_000, ramp=0, majority=False):
    """The minority mean moves from sep*u to moved*u (or, with majority=True, the majority
    mean moves from 0 to (sep - moved)*u, i.e. towards the minority by the same distance)."""
    rng = np.random.default_rng(seed)
    t0 = W_ref + W_cal
    tau = t0 + pre
    n = tau + post
    u = np.ones(d) / np.sqrt(d)
    means = np.vstack([np.zeros(d), sep * u])
    tgt, nm = (0, (sep - moved) * u) if majority else (1, moved * u)
    X, y = _gauss_stream(rng, np.array([1 - share, share]), means, n,
                         tau if change else None, tgt, nm, ramp=ramp)
    return X, y, dict(K=2, tau=tau if change else None, target=tgt, W_ref=W_ref, W_cal=W_cal, n=n,
                      family="gauss", share=share)


def gmm5(seed, change=True, d=5, W_ref=3000, W_cal=5000, pre=15_000, post=200_000, two=False):
    priors = np.array([0.45, 0.30, 0.15, 0.07, 0.03])
    geo = np.random.default_rng(777)                       # fixed geometry for all seeds
    means = geo.standard_normal((5, d))
    means = 3.0 * means / np.linalg.norm(means, axis=1, keepdims=True)
    new_mean = 0.5 * (means[4] + means[0])                 # the 3% class moves half-way to class 0
    rng = np.random.default_rng(seed)
    t0 = W_ref + W_cal
    tau = t0 + pre
    n = tau + post
    extra = ((3, 0.5 * (means[3] + means[0])),) if two else ()   # two=True: the 7% class moves too
    X, y = _gauss_stream(rng, priors, means, n, tau if change else None, 4, new_mean, extra=extra)
    return X, y, dict(K=5, tau=tau if change else None, target=4, W_ref=W_ref, W_cal=W_cal, n=n,
                      family="gmm5", share=0.03)


# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _real(name):
    if name == "credit":
        df = pd.read_csv(RIVER / "CreditCard" / "creditcard.csv")
        X = df.drop(columns=["Time", "Class"]).to_numpy(float)
        y = df["Class"].to_numpy(int)
    elif name == "covtype":
        df = pd.read_csv(COVTYPE_CSV)
        X = df.iloc[:, :-1].to_numpy(float)
        y = df.iloc[:, -1].to_numpy(int) - 1
    elif name == "insects":
        df = pd.read_csv(RIVER / "Insects" / "abrupt_imbalanced.csv", header=None)
        X = df.iloc[:, :-1].to_numpy(float)
        y = np.unique(df.iloc[:, -1].to_numpy(), return_inverse=True)[1]
    else:
        raise ValueError(name)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    X = (X - mu) / sd
    return X.astype(np.float32), y.astype(np.int64)


REAL_CFG = {  # W_ref, W_cal, tau-offset from t0, post, shift fraction s
    "credit":  dict(W_ref=30_000, W_cal=60_000, pre=20_000, post=180_000, s=0.3),
    "covtype": dict(W_ref=20_000, W_cal=40_000, pre=40_000, post=400_000, s=0.15),
    "insects": dict(W_ref=10_000, W_cal=20_000, pre=30_000, post=290_000, s=0.15),
}


def semi(name, seed, change=True, s=None):
    X0, y0 = _real(name)
    cfg = REAL_CFG[name]
    s = cfg["s"] if s is None else s
    rng = np.random.default_rng(seed)
    t0 = cfg["W_ref"] + cfg["W_cal"]
    tau = t0 + cfg["pre"]
    n = min(len(y0), tau + cfg["post"])
    idx = rng.permutation(len(y0))[:n]
    X, y = X0[idx].astype(float), y0[idx].copy()
    K = int(y0.max()) + 1
    cnt = np.bincount(y0, minlength=K)
    target = int(np.argmin(cnt))
    if change:
        c_t = X0[y0 == target].mean(0)
        c_r = X0[y0 != target].mean(0)
        post = (np.arange(n) >= tau) & (y == target)
        X[post] += s * (c_r - c_t)
    return X, y, dict(K=K, tau=tau if change else None, target=target, W_ref=cfg["W_ref"],
                      W_cal=cfg["W_cal"], n=n, family=name, share=float(cnt[target] / cnt.sum()))


# --------------------------------------------------------------------------- #
FAMILIES = {
    # name: (generator, kwargs); synthetic generators accept 'post' (long null streams)
    "sea20":  (sea, dict(share=0.20)),
    "sea10":  (sea, dict(share=0.10)),
    "sea5":   (sea, dict(share=0.05)),
    "sea2":   (sea, dict(share=0.02, W_ref=5000)),
    "sea1":   (sea, dict(share=0.01, W_ref=10_000)),
    "gauss5": (gauss, dict(share=0.05)),
    "gauss1": (gauss, dict(share=0.01, W_ref=10_000)),
    "gmm5":   (gmm5, dict()),
    "credit": (semi, dict(name="credit")),
    "covtype": (semi, dict(name="covtype")),
    "insects": (semi, dict(name="insects")),
    # additional scenarios (supplement)
    "sea5grad":  (sea, dict(share=0.05, ramp=20_000)),            # gradual minority drift
    "gauss5grad": (gauss, dict(share=0.05, ramp=20_000)),
    "gmm5two":   (gmm5, dict(two=True)),                          # 3% and 7% classes change
    "sea5maj":   (sea, dict(share=0.05, theta1=6.5, target=1)),   # majority-only change
    "gauss5maj": (gauss, dict(share=0.05, majority=True)),
    "sea5wc1k":  (sea, dict(share=0.05, W_cal=1000)),             # calibration-window size
    "sea5wc10k": (sea, dict(share=0.05, W_cal=10_000)),
    "priorshift": (priorshift, dict()),
    "priorctrl": (priorshift, dict(share1=0.05)),      # control: same stream without the shift
}
SYNTHETIC = {"sea20", "sea10", "sea5", "sea2", "sea1", "gauss5", "gauss1", "gmm5",
             "sea5grad", "gauss5grad", "gmm5two", "sea5maj", "gauss5maj", "sea5wc1k", "sea5wc10k"}


def load(family, seed, change=True, null_post=None):
    """null_post: for synthetic drift-free streams, the monitored length after tau."""
    gen, kw = FAMILIES[family]
    kw = dict(kw)
    if family in ("priorshift", "priorctrl"):
        return gen(seed, **kw)
    if gen is semi:
        return semi(kw.pop("name"), seed, change)
    if null_post is not None and not change:
        kw["post"] = null_post
    return gen(seed, change=change, **kw)
