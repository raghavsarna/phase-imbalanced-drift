"""Change detection on the residuals of a frozen reference model.

At the start of every segment a class-balanced random forest is fitted on the
first ``W_ref`` samples and frozen; its error indicators
e_t = 1{h(x_t) != y_t} are i.i.d. while the concept is unchanged.

Two choices define a detector:

* ``mode``  -- how residuals are routed to tests:
    "class"    one test per class, fed only with that class's residuals
               (a bank of K tests; the proposed detector),
    "pooled"   a single test on all residuals,
    "periodic" fixed boundaries every ``period`` samples (no test),
    "none"     no boundaries;
* ``test``  -- the sequential test run on each channel:
    "ph"       Page-Hinkley (Eq. (3) of the letter),
    "ddm"      DDM (Gama et al., 2004; River implementation, defaults),
    "adwin"    ADWIN (Bifet & Gavalda, 2007; River implementation, defaults).

After an alarm the change time is localised by the least-squares single-change
estimator on the alarming channel's residuals, and the detector restarts from
that estimate with a new reference model.  Every step uses only samples up to
the alarm time, so the procedure is causal and can run online.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class PageHinkley:
    """One-sided PH test for an increase in the mean of a scalar sequence."""

    __slots__ = ("n", "mean", "U", "M", "t_min", "burn_in")

    def __init__(self, burn_in: int = 0):
        self.n, self.mean, self.U, self.M, self.t_min = 0, 0.0, 0.0, 0.0, -1
        self.burn_in = burn_in            # first samples only initialise the mean

    def update(self, e: float, t: int, delta: float) -> float:
        self.n += 1
        self.mean += (e - self.mean) / self.n
        if self.n <= self.burn_in:
            self.t_min = t
            return 0.0
        self.U += e - self.mean - delta
        if self.U < self.M or self.n == self.burn_in + 1:
            self.M, self.t_min = min(self.U, 0.0), t
        return self.U - self.M


class _PHChannel:
    def __init__(self, lam, delta, min_inst):
        self.ph, self.lam, self.delta, self.min_inst = PageHinkley(), lam, delta, min_inst

    @property
    def n(self):
        return self.ph.n

    def update(self, e, t):
        stat = self.ph.update(e, t, self.delta)
        return self.ph.n >= self.min_inst and stat > self.lam


class _RiverChannel:
    def __init__(self, test):
        from river import drift
        self.d = drift.binary.DDM() if test == "ddm" else drift.ADWIN()
        self.n = 0

    def update(self, e, t):
        self.n += 1
        self.d.update(e)
        return bool(self.d.drift_detected)


def make_channel(test, lam=25.0, delta=0.02, min_inst=30):
    if test == "ph":
        return _PHChannel(lam, delta, min_inst)
    if test in ("ddm", "adwin"):
        return _RiverChannel(test)
    raise ValueError(test)


def ls_change_point(e):
    """Least-squares estimate of a single mean change in e (index of first new sample).

    k* = argmax_k  k (n - k) / n * (mean(e[:k]) - mean(e[k:]))^2,
    the maximum-likelihood estimate for a mean shift in i.i.d. Gaussian noise.
    """
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n < 2:
        return 0
    c = np.cumsum(e)[:-1]
    k = np.arange(1, n)
    gain = (c - k * c[-1] / n - k * e[-1] / n) ** 2 * n / (k * (n - k))
    return int(np.argmax(gain)) + 1


def segment(X, y, n_classes, mode="class", test="ph", lam=25.0, delta=0.02,
            min_inst=30, W_ref=1000, L_min=1000, period=5000, seed=0,
            probe_trees=100, chunk=4096):
    """Return (boundaries, alarms) for the labelled stream (X, y).

    boundaries: estimated change points t_1 < ... < t_{S-1} (segment starts)
    alarms:     times at which the corresponding alarm was raised
    """
    n = len(y)
    if mode == "none":
        return [], []
    if mode == "periodic":
        b = list(range(period, n - period + 1, period))
        return b, list(b)

    boundaries, alarms = [], []
    start = 0
    while start + W_ref < n:
        ref = slice(start, start + W_ref)
        cls, cnt = np.unique(y[ref], return_counts=True)
        sw = (W_ref / (len(cls) * cnt))[np.searchsorted(cls, y[ref])]      # class costs, Eq. (6)
        probe = RandomForestClassifier(probe_trees, min_samples_leaf=5, random_state=seed, n_jobs=1)
        probe.fit(X[ref], y[ref], sample_weight=sw)
        t0 = start + W_ref
        err = np.empty(n - t0)
        done = 0                                                         # residuals computed lazily
        bank = [make_channel(test, lam, delta, min_inst) for _ in range(n_classes if mode == "class" else 1)]
        restart = None
        for i in range(n - t0):
            if i == done:
                hi = min(n - t0, done + chunk)
                err[done:hi] = probe.predict(X[t0 + done:t0 + hi]) != y[t0 + done:t0 + hi]
                done = hi
            t = t0 + i
            k = int(y[t]) if mode == "class" else 0
            if bank[k].update(err[i], t) and bank[k].n >= min_inst:
                # localise the change on the alarming channel's residuals since t0
                idx = np.flatnonzero(y[t0:t + 1] == y[t]) if mode == "class" else np.arange(t + 1 - t0)
                tau = t0 + int(idx[ls_change_point(err[idx])])
                last = boundaries[-1] if boundaries else 0
                if tau - last >= L_min and n - tau >= L_min:
                    boundaries.append(tau)
                    alarms.append(t)
                    restart = tau
                    break
                bank[k] = make_channel(test, lam, delta, min_inst)   # too close: discard and continue
        if restart is None:
            break
        start = restart
    return boundaries, alarms


def match_change_points(true_cps, boundaries, alarms, tol_before=500, tol_after=5000):
    """Detection statistics against ground truth (synthetic streams only).

    A true change point tau is detected by the first estimate in
    [tau - tol_before, tau + tol_after]; remaining estimates are false alarms.
    """
    used, delays, loc_err, hits = set(), [], [], []
    for tau in true_cps:
        for i, b in enumerate(boundaries):
            if i not in used and tau - tol_before <= b <= tau + tol_after:
                used.add(i)
                hits.append(int(tau))
                loc_err.append(abs(b - tau))
                delays.append(alarms[i] - tau)
                break
    return {
        "detected": len(used), "n_true": len(true_cps),
        "false_alarms": len(boundaries) - len(used),
        "mean_abs_loc_error": float(np.mean(loc_err)) if loc_err else None,
        "mean_delay": float(np.mean(delays)) if delays else None,
        "hits": hits,
    }
