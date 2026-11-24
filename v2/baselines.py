"""Detectors that are not plain residual recursions: ADWIN (River) and RBM-IM.

RBM-IM (Korycki & Krawczyk, ICDE 2021) -- reimplementation
------------------------------------------------------------
The authors' code is no longer available (github.com/lkorycki/rbm-im returns
404), so we follow Sec. V of the paper and fix the unspecified choices:

* Architecture (Eqs. 8-12): visible units v = features min-max scaled on the
  reference window and clipped to [0, 1] (Bernoulli means); a class layer z
  (softmax) connected to the hidden layer; H = max(4, V/2) hidden units.
* Training: CD-1 (k = 1), learning rate 0.05, mini-batch M = 50, effective-
  number class weights (1 - b) / (1 - b^{n_m}) with b = 0.999 (Cui et al., 2019,
  the weighting cited by the paper) multiplying each sample's gradient; trained
  for 5 epochs on the reference window, then one CD step per incoming batch
  (the detector keeps learning online, as in the paper).
* Reconstruction error (Eq. 26) per instance with the hidden layer at its
  mean-field value given (x, 1_y); per class and batch, the mean over that
  class's instances in the batch (Eq. 27).
* Trend test: the paper applies a Granger-causality test (lag, level and window
  unspecified) to the least-squares trend of the per-class error.  We use the
  least-squares slope of the last 30 per-class batch errors and alarm when its
  t-statistic exceeds kappa; kappa is calibrated to the common average run
  length like every other detector's threshold.  After an alarm all trend
  windows are cleared.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
def adwin(route, e, K, log10_inv_delta, t0):
    from river.drift import ADWIN
    delta = 10.0 ** (-log10_inv_delta)
    chans = [ADWIN(delta=delta) for _ in range(K)]
    alarms = []
    for t in range(t0, len(e)):
        k = route[t]
        chans[k].update(float(e[t]))
        if chans[k].drift_detected:
            alarms.append(t)
            chans = [ADWIN(delta=delta) for _ in range(K)]
    return np.asarray(alarms, dtype=np.int64)


# --------------------------------------------------------------------------- #
def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))


def _softmax(a):
    a = a - a.max(1, keepdims=True)
    p = np.exp(a)
    return p / p.sum(1, keepdims=True)


class RBMIM:
    def __init__(self, V, K, H=None, lr=0.05, batch=50, beta=0.999, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.K = V, K
        self.H = H or max(4, V // 2)
        self.W = 0.01 * rng.standard_normal((V, self.H))
        self.U = 0.01 * rng.standard_normal((K, self.H))
        self.a, self.b, self.c = np.zeros(V), np.zeros(self.H), np.zeros(K)
        self.lr, self.batch, self.beta, self.rng = lr, batch, beta, rng
        self.counts = np.zeros(K)

    def scale_fit(self, X):
        self.lo, self.hi = X.min(0), X.max(0)
        self.hi[self.hi == self.lo] = self.lo[self.hi == self.lo] + 1

    def _v(self, X):
        return np.clip((X - self.lo) / (self.hi - self.lo), 0, 1)

    def _weights(self, y):
        self.counts += np.bincount(y, minlength=self.K)
        w = (1 - self.beta) / (1 - self.beta ** np.maximum(self.counts, 1))
        w = w[y]
        return w / w.mean()

    def _cd(self, v, Z, w):
        h0 = _sigmoid(v @ self.W + Z @ self.U + self.b)
        hs = (self.rng.random(h0.shape) < h0).astype(float)
        v1 = _sigmoid(hs @ self.W.T + self.a)
        z1 = _softmax(hs @ self.U.T + self.c)
        h1 = _sigmoid(v1 @ self.W + z1 @ self.U + self.b)
        n = len(v)
        wv, wv1 = v * w[:, None], v1 * w[:, None]
        self.W += self.lr * (wv.T @ h0 - wv1.T @ h1) / n
        self.U += self.lr * ((Z * w[:, None]).T @ h0 - (z1 * w[:, None]).T @ h1) / n
        self.a += self.lr * (wv - wv1).mean(0)
        self.b += self.lr * ((h0 - h1) * w[:, None]).mean(0)
        self.c += self.lr * ((Z - z1) * w[:, None]).mean(0)

    def recon_error(self, v, Z):
        h = _sigmoid(v @ self.W + Z @ self.U + self.b)
        vr = _sigmoid(h @ self.W.T + self.a)
        zr = _softmax(h @ self.U.T + self.c)
        return np.sqrt(((v - vr) ** 2).sum(1) + ((Z - zr) ** 2).sum(1))

    def fit_reference(self, X, y, epochs=5):
        self.scale_fit(X)
        v, Z = self._v(X), np.eye(self.K)[y]
        w = self._weights(y)
        for _ in range(epochs):
            idx = self.rng.permutation(len(y))
            for s in range(0, len(y), self.batch):
                j = idx[s:s + self.batch]
                self._cd(v[j], Z[j], w[j])

    def stream_errors(self, X, y, t0):
        """Per-class batch errors from t0 on: list over classes of (end_time, error) arrays."""
        T = [[] for _ in range(self.K)]
        R = [[] for _ in range(self.K)]
        for s in range(t0, len(y), self.batch):
            Xb, yb = X[s:s + self.batch], y[s:s + self.batch]
            v, Z = self._v(Xb), np.eye(self.K)[yb]
            err = self.recon_error(v, Z)
            end = s + len(yb) - 1
            for k in np.unique(yb):
                T[k].append(end)
                R[k].append(err[yb == k].mean())
            self._cd(v, Z, self._weights(yb))
        return [(np.asarray(T[k], dtype=np.int64), np.asarray(R[k])) for k in range(self.K)]


def rbmim_alarms(seqs, kappa, t_start, window=30, min_pts=10):
    """Alarm times from per-class error sequences with the slope t-test (resets after alarms)."""
    ev = sorted((int(t), k, float(r)) for k, (T, R) in enumerate(seqs) for t, r in zip(T, R) if t >= t_start)
    K = len(seqs)
    buf = [[] for _ in range(K)]
    alarms = []
    for t, k, r in ev:
        b = buf[k]
        b.append(r)
        if len(b) > window:
            b.pop(0)
        m = len(b)
        if m < min_pts:
            continue
        x = np.arange(m, dtype=float)
        yv = np.asarray(b)
        xm, ym = x.mean(), yv.mean()
        sxx = ((x - xm) ** 2).sum()
        slope = ((x - xm) * (yv - ym)).sum() / sxx
        resid = yv - ym - slope * (x - xm)
        se = np.sqrt(max((resid ** 2).sum() / (m - 2), 1e-12) / sxx)
        if slope / se > kappa:
            alarms.append(t)
            buf = [[] for _ in range(K)]
    return np.asarray(alarms, dtype=np.int64)
