"""Sequential detectors on the residuals of a frozen reference classifier.

Every detector consumes the stream from index ``t0`` on and returns the array
of alarm times.  After an alarm all statistics are reset (the reference model
and any frozen pre-change estimates are kept), so the same routine measures
the average run length on drift-free streams and the detection delay on
streams with a change.

Residual-based detectors (numba):
  bank_lr      label-routed Bernoulli log-likelihood-ratio CUSUM bank (proposed)
  pooled_lr    Bernoulli LLR-CUSUM on the pooled residuals
  ra_cusum     risk-adjusted (label-adjusted) CUSUM: one statistic, per-class baselines
  ph           Page--Hinkley, class-routed or pooled (running mean, as in v1)
  ddm          DDM, class-routed or pooled
  ddm_oci      DDM-OCI: DDM on the time-decayed recall of the minority class
  lfr          Linear Four Rates (binary: minority vs rest)
ADWIN (River) and RBM-IM live in baselines.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.stats import beta as beta_dist


# --------------------------------------------------------------------------- #
# Pre-change estimates and design of the LLR increments
# --------------------------------------------------------------------------- #
def cp_upper(k, m, level):
    """Clopper--Pearson upper confidence bound for a binomial proportion."""
    if m == 0:
        return 1.0
    if k >= m:
        return 1.0
    return float(beta_dist.ppf(1 - level, k + 1, m - k))


def frozen_rates(y, e, K, beta=0.05, ucb=True):
    """Per-class residual rates from a calibration window.

    Returns (mu_plug, mu_tilde, counts): mu_plug = (s + 1/2)/(m + 1) and
    mu_tilde the Clopper--Pearson upper bound at level beta/K (Proposition 1
    requires mu_k <= mu_tilde_k).  Classes absent from the window get 1 (off).
    """
    mu_hat, mu_t, m = np.ones(K), np.ones(K), np.zeros(K, dtype=np.int64)
    for k in range(K):
        sel = y == k
        m[k] = int(sel.sum())
        if m[k] > 0:
            s = int(e[sel].sum())
            mu_hat[k] = (s + 0.5) / (m[k] + 1)
            mu_t[k] = cp_upper(s, m[k], beta / K) if ucb else mu_hat[k]
    return mu_hat, mu_t, m


def _unused_frozen_rates(y, e, K, beta=0.05, ucb=True):
    """Per-class residual rates from a calibration window.

    Returns (mu_hat, mu_tilde, counts); mu_tilde is the Clopper--Pearson upper
    bound at level beta/K (Proposition 1 requires mu_k <= mu_tilde_k).
    """
    mu_hat, mu_t, m = np.ones(K), np.ones(K), np.zeros(K, dtype=np.int64)
    for k in range(K):
        sel = y == k
        m[k] = int(sel.sum())
        if m[k] > 0:
            s = int(e[sel].sum())
            mu_hat[k] = s / m[k]
            mu_t[k] = cp_upper(s, m[k], beta / K) if ucb else mu_hat[k]
    return mu_hat, mu_t, m


def odds_design(mu_t, odds=2.0):
    """Post-change design rate with odds ratio ``odds`` against mu_tilde."""
    mu_t = np.asarray(mu_t, dtype=float)
    return odds * mu_t / (1.0 - mu_t + odds * mu_t)


def llr_table(mu_t, q):
    """LLR increments l(e) for e = 0, 1; channels with mu_t >= q are disabled (nan)."""
    mu_t, q = np.atleast_1d(mu_t).astype(float), np.atleast_1d(q).astype(float)
    L = np.full((len(mu_t), 2), np.nan)
    ok = (mu_t > 0) & (mu_t < q) & (q < 1)
    L[ok, 1] = np.log(q[ok] / mu_t[ok])
    L[ok, 0] = np.log((1 - q[ok]) / (1 - mu_t[ok]))
    # mu_t == 0 cannot occur with the CP bound; guard anyway
    return L


# --------------------------------------------------------------------------- #
# CUSUM family
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _bank(route, e, L, h, t0, out_zero):
    """Max-of-channels CUSUM; channel route[t] receives increment L[route[t], e[t]].

    out_zero[i] receives the last-zero time of the alarming channel (Page's
    change-point estimate) for alarm i.
    """
    K = L.shape[0]
    W = np.zeros(K)
    last0 = np.full(K, t0)
    alarms = np.empty(len(e), np.int64)
    na = 0
    for t in range(t0, len(e)):
        k = route[t]
        inc = L[k, e[t]]
        if inc != inc:          # nan: channel disabled
            continue
        w = W[k] + inc
        if w <= 0.0:
            W[k] = 0.0
            last0[k] = t
        else:
            W[k] = w
            if w >= h:
                alarms[na] = t
                out_zero[na] = last0[k] + 1
                na += 1
                for j in range(K):
                    W[j] = 0.0
                    last0[j] = t
    return alarms[:na]


def bank_lr(y, e, L, h, t0, return_loc=False):
    loc = np.empty(len(e), np.int64)
    a = _bank(y.astype(np.int64), e.astype(np.int64), L, float(h), int(t0), loc)
    return (a, loc[:len(a)]) if return_loc else a


def pooled_lr(e, L1, h, t0):
    """L1: llr table of shape (1, 2) for the pooled residual."""
    loc = np.empty(len(e), np.int64)
    return _bank(np.zeros(len(e), np.int64), e.astype(np.int64), L1, float(h), int(t0), loc)


@njit(cache=True)
def _ra(y, e, L, h, t0):
    alarms = np.empty(len(e), np.int64)
    na, W = 0, 0.0
    for t in range(t0, len(e)):
        inc = L[y[t], e[t]]
        if inc != inc:
            continue
        W = max(0.0, W + inc)
        if W >= h:
            alarms[na] = t
            na += 1
            W = 0.0
    return alarms[:na]


def ra_cusum(y, e, L, h, t0):
    """Risk-adjusted CUSUM (Steiner et al., 2000) with per-class baselines and odds-ratio design."""
    return _ra(y.astype(np.int64), e.astype(np.int64), L, float(h), int(t0))


# --------------------------------------------------------------------------- #
# Page--Hinkley (v1 detector), class-routed or pooled
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _ph(route, e, K, delta, lam, t0, min_inst):
    n = np.zeros(K)
    mean = np.zeros(K)
    U = np.zeros(K)
    M = np.zeros(K)
    alarms = np.empty(len(e), np.int64)
    na = 0
    for t in range(t0, len(e)):
        k = route[t]
        n[k] += 1
        mean[k] += (e[t] - mean[k]) / n[k]
        U[k] += e[t] - mean[k] - delta[k]
        if U[k] < M[k]:
            M[k] = U[k]
        if n[k] >= min_inst and U[k] - M[k] > lam:
            alarms[na] = t
            na += 1
            for j in range(K):
                n[j] = 0.0
                mean[j] = 0.0
                U[j] = 0.0
                M[j] = 0.0
    return alarms[:na]


def ph(route, e, K, delta, lam, t0, min_inst=30):
    delta = np.broadcast_to(np.asarray(delta, float), (K,)).copy()
    return _ph(route.astype(np.int64), e.astype(np.float64), K, delta, float(lam), int(t0), min_inst)


# --------------------------------------------------------------------------- #
# DDM (Gama et al., 2004), class-routed or pooled; River defaults: warm-up 30
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _ddm(route, e, K, c, t0, warm):
    n = np.zeros(K)
    p = np.zeros(K)
    pmin = np.full(K, np.inf)
    smin = np.full(K, np.inf)
    alarms = np.empty(len(e), np.int64)
    na = 0
    for t in range(t0, len(e)):
        k = route[t]
        n[k] += 1
        p[k] += (e[t] - p[k]) / n[k]
        if n[k] < warm:
            continue
        s = math.sqrt(p[k] * (1 - p[k]) / n[k])
        if p[k] + s < pmin[k] + smin[k]:
            pmin[k] = p[k]
            smin[k] = s
        if p[k] + s > pmin[k] + c * smin[k]:
            alarms[na] = t
            na += 1
            for j in range(K):
                n[j] = 0.0
                p[j] = 0.0
                pmin[j] = np.inf
                smin[j] = np.inf
    return alarms[:na]


def ddm(route, e, K, c, t0, warm=30):
    return _ddm(route.astype(np.int64), e.astype(np.float64), K, float(c), int(t0), warm)


# --------------------------------------------------------------------------- #
# DDM-OCI (Wang et al., 2013): DDM on the time-decayed minority-class recall
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _ddm_oci(y, e, m, c, eta, t0, warm):
    R = 1.0
    n = 0.0
    best = -np.inf
    Rb = 0.0
    sb = 0.0
    alarms = np.empty(len(e), np.int64)
    na = 0
    for t in range(t0, len(e)):
        if y[t] != m:
            continue
        n += 1
        if n == 1:
            R = 1.0 - e[t]
        else:
            R = eta * R + (1 - eta) * (1 - e[t])   # decayed recall
        if n < warm:
            continue
        s = math.sqrt(max(R * (1 - R), 1e-12) / n)
        # DDM on the minority error p = 1 - R: keep the state minimising p + s,
        # i.e. maximising R - s, and alarm when p + s > p_min + c s_min.
        if R - s > best:
            best = R - s
            Rb = R
            sb = s
        if R - s < Rb - c * sb:
            alarms[na] = t
            na += 1
            n = 0.0
            best = -np.inf
    return alarms[:na]


def ddm_oci(y, e, minority, c, t0, eta=0.9, warm=30):
    return _ddm_oci(y.astype(np.int64), e.astype(np.float64), int(minority), float(c), float(eta), int(t0), warm)


# --------------------------------------------------------------------------- #
# Linear Four Rates (Wang & Abraham, 2015), Menelaus conventions
# --------------------------------------------------------------------------- #
LFR_NB = np.array(list(range(0, 31)) + [40, 60, 100, 150, 200, 400])   # buckets for the number of updates u (next bucket >= u)
LFR_PG = np.linspace(0, 1, 101)                            # rate grid


@njit(parallel=True, cache=True)
def _lfr_tails(eta, nmc, NB, PG, ntail, seed, maxterms):
    """Lowest and highest ntail of nmc null draws of R after u updates:
    R = 0.5 eta^u + (1-eta) sum_{j<u} eta^j I_j,  I_j ~ Bernoulli(p).

    The 0.5 eta^u term is the statistic's initial value (Menelaus and Wang &
    Abraham simulate without it, so minority-class rates exceed their bounds
    right after every reset); terms with j >= maxterms (weight < 1e-7) are dropped.
    """
    A, Bn = len(NB), len(PG)
    lo = np.zeros((A, Bn, ntail))
    hi = np.zeros((A, Bn, ntail))
    for cell in prange(A * Bn):
        a, b = cell // Bn, cell % Bn
        np.random.seed(seed + cell)
        u, p = NB[a], PG[b]
        terms = min(u, maxterms)
        sims = np.empty(nmc)
        for r in range(nmc):
            s = 0.5 * eta ** u
            w = 1.0 - eta
            for j in range(terms):
                if np.random.random() < p:
                    s += w
                w *= eta
            sims[r] = s
        srt = np.sort(sims)
        lo[a, b, :] = srt[:ntail]
        hi[a, b, :] = srt[nmc - ntail:][::-1]
    return lo, hi


LFR_TABLES = {"fine": (2_000_000, 2000, 2024), "coarse": (200_000, 2000, 2025)}   # nmc, ntail, seed
_TAILS: dict = {}


def _tails(which):
    if which not in _TAILS:
        nmc, ntail, seed = LFR_TABLES[which]
        f = Path(__file__).resolve().parents[1] / "results_v2" / ("lfr_tails.npz" if which == "fine" else "lfr_tails_coarse.npz")
        if f.exists():
            z = np.load(f)
            _TAILS[which] = (z["lo"], z["hi"])
        else:
            _TAILS[which] = _lfr_tails(0.9, nmc, LFR_NB, LFR_PG, ntail, seed, 150)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, lo=_TAILS[which][0], hi=_TAILS[which][1])
    return _TAILS[which]


def lfr_bounds(alpha, eta=0.9):
    """Two-sided bounds at level alpha: 2e6 null draws per cell for alpha <= 1e-3,
    2e5 draws for 1e-3 < alpha <= 1e-2."""
    assert eta == 0.9
    which = "fine" if alpha <= 1e-3 else "coarse"
    nmc, ntail, _ = LFR_TABLES[which]
    lo, hi = _tails(which)
    i = min(ntail - 1, int(math.floor(alpha * (nmc - 1))))
    return lo[:, :, i].copy(), hi[:, :, i].copy()


@njit(cache=True)
def _bucket(N, NB):
    for a in range(len(NB)):
        if NB[a] >= N:
            return a
    return len(NB) - 1


@njit(cache=True)
def _lfr(ytrue, ypred, lb, ub, eta, t0, burn, NB):
    alarms = np.empty(len(ytrue), np.int64)
    na = 0
    C = np.ones((2, 2))          # C[yhat, y]
    R = np.full(4, 0.5)          # tpr, tnr, ppv, npv
    P = np.full(4, 0.5)
    U = np.zeros(4)              # number of updates of each rate since the reset
    since = 0
    for t in range(t0, len(ytrue)):
        yt, yp = ytrue[t], ypred[t]
        C[yp, yt] += 1
        TP, TN, FP, FN = C[1, 1], C[0, 0], C[1, 0], C[0, 1]
        Pn = np.array([TP / (TP + FN), TN / (TN + FP), TP / (TP + FP), TN / (TN + FN)])
        Nn = np.array([TP + FN, TN + FP, TP + FP, TN + FN])
        corr = 1.0 if yt == yp else 0.0
        since += 1
        alarm = False
        for r in range(4):
            if Pn[r] != P[r]:
                R[r] = eta * R[r] + (1 - eta) * corr
                U[r] += 1
            P[r] = Pn[r]
            if since > burn:
                a = _bucket(int(U[r]), NB)
                # conservative grid lookup: lower bound at floor(P), upper bound at ceil(P)
                b_lo = int(math.floor(P[r] * 100 + 1e-9))
                b_hi = int(math.ceil(P[r] * 100 - 1e-9))
                if R[r] < lb[a, b_lo] - 5e-7 or R[r] > ub[a, b_hi] + 5e-7:   # tol > truncated weight 0.9^150
                    alarm = True
        if alarm:
            alarms[na] = t
            na += 1
            C[:, :] = 1.0
            R[:] = 0.5
            P[:] = 0.5
            U[:] = 0
            since = 0
    return alarms[:na]


def lfr(y, yhat, minority, alpha, t0, eta=0.9, burn=50):
    """LFR on the binary problem 'minority' vs rest (positive = minority)."""
    lb, ub = lfr_bounds(alpha, eta)
    yt = (y == minority).astype(np.int64)
    yp = (yhat == minority).astype(np.int64)
    return _lfr(yt, yp, lb, ub, float(eta), int(t0), burn, LFR_NB)


# --------------------------------------------------------------------------- #
# Multi-chart bank: M designs per class (max over K x M CUSUMs)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _bank_multi(y, e, L, h, t0):
    """L: (K, M, 2) increments; every class-k sample updates the M charts of class k."""
    K, M = L.shape[0], L.shape[1]
    W = np.zeros((K, M))
    alarms = np.empty(len(e), np.int64)
    na = 0
    for t in range(t0, len(e)):
        k = y[t]
        hit = False
        for j in range(M):
            inc = L[k, j, e[t]]
            if inc != inc:
                continue
            w = max(0.0, W[k, j] + inc)
            W[k, j] = w
            if w >= h:
                hit = True
        if hit:
            alarms[na] = t
            na += 1
            W[:, :] = 0.0
    return alarms[:na]


def bank_multi(y, e, mu, odds_list, h, t0):
    L = np.stack([llr_table(mu, odds_design(mu, r)) for r in odds_list], axis=1)
    return _bank_multi(y.astype(np.int64), e.astype(np.int64), L, float(h), int(t0))
