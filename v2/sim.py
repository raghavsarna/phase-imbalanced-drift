"""Monte Carlo study of the theory on Bernoulli residuals (Section III of the letter).

S1  delay versus minority share at a common ARL (thresholds calibrated per method)
S2  delay versus ARL at pi_c = 5%
S3  false-alarm probability within n samples versus h: i.i.d. labels, prior
    shift, Markov labels, autocorrelated residuals, and the bound n e^{-h}
S4  frozen estimate of mu_c: plug-in versus Clopper--Pearson bound, versus the
    number m_c of calibration residuals of the class

Methods: bank-LR (odds-ratio design 2 on the pre-change rate), bank-LR with the
oracle post-change rate, pooled LR-CUSUM with the oracle pooled post-change
rate, risk-adjusted CUSUM, and class-routed / pooled Page--Hinkley with oracle
tolerance delta = Delta/2 and pi_c Delta/2.  Output: results_v2/sim.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.stats import beta as beta_dist

OUT = Path(__file__).resolve().parents[1] / "results_v2"
MU_C, MU_O, DELTA_C = 0.10, 0.05, 0.15          # changed class 0.10 -> 0.25, other class 0.05


def kl(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def odds(mu, r=2.0):
    return r * mu / (1 - mu + r * mu)


# --------------------------------------------------------------------------- #
# One generic kernel: statistic type 0 = routed LR bank, 1 = single LR CUSUM on
# per-sample increments (pooled or risk-adjusted), 2 = PH (routed or pooled).
# Labels: class 1 = changed class with probability pi.  Change at nu (or never).
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _run(kind, route_pooled, L, delta, h, pi, mu, nu, p_post, max_t, seed, label_mode, pi_after, rho_res):
    np.random.seed(seed)
    W = np.zeros(2)
    n = np.zeros(2)
    mean = np.zeros(2)
    U = np.zeros(2)
    M = np.zeros(2)
    prev_y = 1 if np.random.random() < pi else 0
    prev_e = np.zeros(2)
    for t in range(max_t):
        # labels
        p1 = pi if (label_mode != 1 or t < max_t // 2) else pi_after
        if label_mode == 2:              # Markov labels with stationary share pi, persistence 0.9
            if np.random.random() < 0.9:
                y = prev_y
            else:
                y = 1 if np.random.random() < pi else 0
        else:
            y = 1 if np.random.random() < p1 else 0
        prev_y = y
        m = mu[y]
        if t >= nu and y == 1:
            m = p_post
        # residuals (optionally autocorrelated within class: repeat previous w.p. rho_res)
        if rho_res > 0 and np.random.random() < rho_res:
            e = int(prev_e[y])
        else:
            e = 1 if np.random.random() < m else 0
        prev_e[y] = e
        if kind == 0:
            k = 0 if route_pooled else y
            w = max(0.0, W[k] + L[k, e])
            W[k] = w
            if w >= h:
                return t
        elif kind == 1:
            w = max(0.0, W[0] + L[y, e])
            W[0] = w
            if w >= h:
                return t
        else:
            k = 0 if route_pooled else y
            n[k] += 1
            mean[k] += (e - mean[k]) / n[k]
            U[k] += e - mean[k] - delta[k]
            if U[k] < M[k]:
                M[k] = U[k]
            if n[k] >= 30 and U[k] - M[k] > h:
                return t
    return max_t


@njit(parallel=True, cache=True)
def _many(kind, route_pooled, L, delta, h, pi, mu, nu, p_post, max_t, seed0, R, label_mode, pi_after, rho_res):
    out = np.empty(R, np.int64)
    for r in prange(R):
        out[r] = _run(kind, route_pooled, L, delta, h, pi, mu, nu, p_post, max_t, seed0 + r,
                      label_mode, pi_after, rho_res)
    return out


def method_spec(name, pi, mu=(MU_O, MU_C), dc=DELTA_C):
    """(kind, pooled, L, delta) for a method; class index 1 = changed class."""
    mu = np.asarray(mu, float)
    p = mu[1] + dc
    mbar = (1 - pi) * mu[0] + pi * mu[1]
    L = np.zeros((2, 2))
    delta = np.zeros(2)
    if name in ("bank", "bank-oracle", "ra"):
        q = odds(mu) if name in ("bank", "ra") else np.array([odds(mu[0]), p])
        L[:, 1] = np.log(q / mu)
        L[:, 0] = np.log((1 - q) / (1 - mu))
        return (0 if name != "ra" else 1), False, L, delta
    if name == "pooled":
        q = mbar + pi * dc
        L[0, 1], L[0, 0] = math.log(q / mbar), math.log((1 - q) / (1 - mbar))
        return 0, True, L, delta
    if name == "ph-bank":
        delta[:] = dc / 2
        return 2, False, L, delta
    if name == "ph-pooled":
        delta[0] = pi * dc / 2
        return 2, True, L, delta
    raise ValueError(name)


METHODS = ["bank", "bank-oracle", "pooled", "ra", "ph-bank", "ph-pooled"]


def arl(name, pi, h, R=1000, max_t=2_000_000, seed=1):
    kind, pooled, L, delta = method_spec(name, pi)
    T = _many(kind, pooled, L, delta, h, pi, np.array([MU_O, MU_C]), 1 << 62, 0.0, max_t, seed, R, 0, pi, 0.0)
    return T.mean() + 1


def calibrate(name, pi, gamma, R=1000):
    lo, hi = (0.5, 40.0) if not name.startswith("ph") else (0.5, 200.0)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if arl(name, pi, mid, R=R) < gamma:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3 * hi:
            break
    return hi


def delay(name, pi, h, R=1000, nu=5000, max_extra=3_000_000, seed=7):
    """Delay after a change at nu; runs with a false alarm before nu are restarted."""
    kind, pooled, L, delta = method_spec(name, pi)
    mu = np.array([MU_O, MU_C])
    p = MU_C + DELTA_C
    out = []
    s = seed * 1_000_003
    need = R
    while need > 0:
        T = _many(kind, pooled, L, delta, h, pi, mu, nu, p, nu + max_extra, s, need, 0, pi, 0.0)
        s += need
        ok = T >= nu
        out.extend((T[ok] - nu).tolist())
        need -= int(ok.sum())
    return np.asarray(out, float)


def theory(pi, gamma):
    mbar = (1 - pi) * MU_O + pi * MU_C
    IR = pi * kl(MU_C + DELTA_C, MU_C)
    IP = kl(mbar + pi * DELTA_C, mbar)
    return {"I_R": IR, "I_P": IP, "lower_pool": math.log(gamma) / IP, "lower_route": math.log(gamma) / IR}


# --------------------------------------------------------------------------- #
def s1(gamma=1e5, pis=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5), R=600):
    rows = []
    for pi in pis:
        row = {"pi": pi, **theory(pi, gamma)}
        for mname in METHODS:
            h = calibrate(mname, pi, gamma)
            d = delay(mname, pi, h, R=R)
            row[mname] = {"h": h, "mean": d.mean(), "se": d.std(ddof=1) / math.sqrt(len(d)),
                          "median": float(np.median(d))}
        # Proposition 2 bound for the bank at its calibrated h (odds design, true mu)
        p, mu = MU_C + DELTA_C, MU_C
        q = odds(mu)
        g = p * math.log(q / mu) + (1 - p) * math.log((1 - q) / (1 - mu))
        row["bound_bank"] = (row["bank"]["h"] + math.log(q / mu)) / (pi * g)
        rows.append(row)
        print(json.dumps({k: (v if not isinstance(v, dict) else round(v["mean"])) for k, v in row.items()}), flush=True)
    return rows


def s2(pi=0.05, gammas=(1e3, 3e3, 1e4, 3e4, 1e5), R=600):
    rows = []
    for gm in gammas:
        row = {"gamma": gm, **theory(pi, gm)}
        for mname in METHODS:
            h = calibrate(mname, pi, gm, R=600)
            d = delay(mname, pi, h, R=R)
            row[mname] = {"h": h, "mean": d.mean(), "se": d.std(ddof=1) / math.sqrt(len(d))}
        rows.append(row)
        print(json.dumps({k: (v if not isinstance(v, dict) else round(v["mean"])) for k, v in row.items()}), flush=True)
    return rows


def s3(n=10_000, hs=(4, 6, 8, 10, 12, 14), R=20_000):
    """P(T <= n) without change, reviewer's setting: shares (0.9, 0.1), rates (0.05, 0.2)."""
    mu = np.array([0.05, 0.20])              # class 1 = 10% class with rate 0.2
    pi = 0.10
    rows = []
    q = odds(mu)
    Lb = np.zeros((2, 2))
    Lb[:, 1], Lb[:, 0] = np.log(q / mu), np.log((1 - q) / (1 - mu))
    mbar = 0.9 * 0.05 + 0.1 * 0.2
    qp = odds(mbar)
    Lp = np.zeros((2, 2))
    Lp[0, 1], Lp[0, 0] = math.log(qp / mbar), math.log((1 - qp) / (1 - mbar))
    for h in hs:
        row = {"h": h, "bound": min(1.0, n * math.exp(-h))}
        for tag, label_mode, pi_after, rho in [("iid", 0, pi, 0.0), ("prior_shift", 1, 0.3, 0.0),
                                               ("markov_labels", 2, pi, 0.0), ("autocorr_res", 0, pi, 0.5)]:
            Tb = _many(0, False, Lb, np.zeros(2), h, pi, mu, 1 << 62, 0.0, n, 11, R, label_mode, pi_after, rho)
            Tp = _many(0, True, Lp, np.zeros(2), h, pi, mu, 1 << 62, 0.0, n, 11, R, label_mode, pi_after, rho)
            row[f"bank_{tag}"] = float((Tb < n).mean())
            row[f"pooled_{tag}"] = float((Tp < n).mean())
        rows.append(row)
        print(row, flush=True)
    return rows


def s4(ms=(25, 50, 100, 200, 400, 1000), h=None, n=10_000, R=4000, pi=0.05, alpha=0.05):
    """Frozen mu_c estimated from m_c calibration residuals: plug-in versus CP bound.

    Reports the false-alarm probability within n samples (bound alpha: h = log(2n/alpha) and
    Clopper--Pearson level alpha/4 per class) and the delay after a change at nu = 2000.
    """
    h = math.log(2 * n / alpha) if h is None else h   # n e^{-h} = alpha/2, CP level alpha/2 split over K = 2
    rng = np.random.default_rng(5)
    mu_true = np.array([MU_O, MU_C])
    rows = []
    for m in ms:
        res = {"m": m, "h": h}
        for est in ("plugin", "cp"):
            fa, dl = [], []
            for r in range(200):
                mt = np.empty(2)
                for k, mk in enumerate((int(m / pi * (1 - pi)), m)):
                    s = rng.binomial(mk, mu_true[k])
                    if est == "plugin":
                        mt[k] = max(s, 0.5) / mk
                    else:
                        mt[k] = beta_dist.ppf(1 - alpha / 4, s + 1, mk - s) if s < mk else 1.0
                q = odds(mt)
                L = np.zeros((2, 2))
                L[:, 1], L[:, 0] = np.log(q / mt), np.log((1 - q) / (1 - mt))
                T0 = _many(0, False, L, np.zeros(2), h, pi, mu_true, 1 << 62, 0.0, n, 1000 * r, R // 200, 0, pi, 0.0)
                fa.extend((T0 < n).tolist())
                T1 = _many(0, False, L, np.zeros(2), h, pi, mu_true, 2000, MU_C + DELTA_C, 2_000_000, 7000 * r,
                           R // 200, 0, pi, 0.0)
                dl.extend((T1[T1 >= 2000] - 2000).tolist())
            res[est] = {"fa": float(np.mean(fa)), "delay": float(np.mean(dl))}
        rows.append(res)
        print(res, flush=True)
    return rows


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT / "sim.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for name in sys.argv[1:] or ["s3", "s4", "s1", "s2"]:
        res[name] = globals()[name]()
        res["params"] = {"mu_c": MU_C, "mu_o": MU_O, "Delta_c": DELTA_C, "odds": 2.0}
        path.write_text(json.dumps(res, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
