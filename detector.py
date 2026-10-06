"""Phase I: Page-Hinkley segmentation of the labelled training stream.

The original detector ran a "multivariate" PH test on the norm of the
feature deviation, with two defects: the running mean used the global
time index instead of the time since the last reset, and unscaled features
made the increment ||x - mean|| - delta always positive.  The statistic
therefore grew linearly and fired at every multiple of the minimum segment
length.  It also cannot see real drift such as SEA's, where P(x) is fixed
and only P(y|x) changes.

Here PH monitors residuals of a reference model, as in classical change
detection.  At the start of every segment a class-balanced random forest is
fitted on the first ``W_ref`` samples and frozen; its error indicators
e_t = 1{h(x_t) != y_t} are i.i.d. while the concept is unchanged.  In
``mode="class"`` one PH statistic is run per class on that class's errors (a
bank of K tests); ``mode="pooled"`` uses a single statistic on all errors.
After an alarm the change time is localised by the least-squares (maximum-
likelihood) single-change estimator on the alarming channel's residuals,
and the detector restarts from that estimate with a new reference model.
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


def ls_change_point(e):
    """Least-squares estimate of a single mean change in e (index of first new sample).

    k* = argmax_k  k (n - k) / n * (mean(e[:k]) - mean(e[k:]))^2,
    the maximum-likelihood estimate for a mean shift in i.i.d. noise.
    """
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n < 2:
        return 0
    c = np.cumsum(e)[:-1]
    k = np.arange(1, n)
    gain = (c - k * c[-1] / n - k * e[-1] / n) ** 2 * n / (k * (n - k))
    return int(np.argmax(gain)) + 1


def segment(X, y, n_classes, mode="class", lam=25.0, delta=0.02,
            min_inst=30, W_ref=1000, L_min=1000, period=5000, seed=0, burn_in=0,
            probe_trees=100):
    """Return (boundaries, alarms) for the training stream (X, y).

    boundaries: estimated change points t_1 < ... < t_{S-1} (segment starts)
    alarms:     times at which the corresponding alarm was raised
    mode:       "class" | "pooled" | "periodic" | "none"
    """
    n = len(y)
    if mode == "none":
        return [], []
    if mode == "periodic":            # reproduces the original behaviour
        b = list(range(period, n - period + 1, period))
        return b, list(b)

    boundaries, alarms = [], []
    start = 0
    while start + W_ref < n:
        ref = slice(start, start + W_ref)
        cls, cnt = np.unique(y[ref], return_counts=True)
        sw = (W_ref / (len(cls) * cnt))[np.searchsorted(cls, y[ref])]      # Eq. (6) costs
        probe = RandomForestClassifier(probe_trees, min_samples_leaf=5, random_state=seed, n_jobs=1)
        probe.fit(X[ref], y[ref], sample_weight=sw)
        t0 = start + W_ref
        err = (probe.predict(X[t0:]) != y[t0:]).astype(float)
        bank = [PageHinkley(burn_in) for _ in range(n_classes if mode == "class" else 1)]
        restart = None
        for i, e in enumerate(err):
            t = t0 + i
            k = int(y[t]) if mode == "class" else 0
            stat = bank[k].update(e, t, delta)
            if bank[k].n >= min_inst and stat > lam:
                # localise the change on the alarming channel's residuals since t0
                idx = np.flatnonzero(y[t0:t + 1] == y[t]) if mode == "class" else np.arange(t + 1 - t0)
                tau = t0 + int(idx[ls_change_point(err[idx])])
                last = boundaries[-1] if boundaries else 0
                if tau - last >= L_min and n - tau >= L_min:
                    boundaries.append(tau)
                    alarms.append(t)
                    restart = tau
                    break
                bank[k] = PageHinkley(burn_in)            # too close: discard and continue
        if restart is None:
            break
        start = restart
    return boundaries, alarms


def match_change_points(true_cps, boundaries, alarms, tol_before=500, tol_after=5000):
    """Detection statistics against ground truth (synthetic streams only).

    A true change point tau is detected by the first estimate in
    [tau - tol_before, tau + tol_after]; remaining estimates are false alarms.
    """
    used, delays, loc_err = set(), [], []
    for tau in true_cps:
        for i, b in enumerate(boundaries):
            if i not in used and tau - tol_before <= b <= tau + tol_after:
                used.add(i)
                loc_err.append(abs(b - tau))
                delays.append(alarms[i] - tau)
                break
    return {
        "detected": len(used), "n_true": len(true_cps),
        "false_alarms": len(boundaries) - len(used),
        "mean_abs_loc_error": float(np.mean(loc_err)) if loc_err else None,
        "mean_delay": float(np.mean(delays)) if delays else None,
    }
