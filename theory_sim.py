"""Monte Carlo check of Propositions 1-2 of the letter on Bernoulli residuals.

(a) Detection delay versus the share pi of the class whose residual mean
    rises by Delta at tau: class-conditional bank vs pooled PH test, with the
    upper bounds (lambda+1)/(pi (Delta-delta)) and (lambda+1)/(pi Delta - delta).
(b) Probability of a false alarm within n samples versus lambda for a
    drift-free stream, with the bound n * sum_c pi_c exp(-theta_c lambda).

The PH tests are the ones of detector.py (running mean, min_inst = 30),
vectorised over Monte Carlo replicates.  Output: results/theory.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path(os.environ.get("PHASE_RESULTS", Path(__file__).resolve().parent / "results")) / "theory.json"
DELTA, LAM, MIN_INST = 0.02, 25.0, 30


def theta_star(mu, delta=DELTA):
    """Positive root of E[exp(theta (e - mu - delta))] = 1 for e ~ Bernoulli(mu)."""
    if mu <= 0:
        return np.inf
    f = lambda th: np.log1p(mu * np.expm1(th)) - th * (mu + delta)
    return brentq(f, 1e-9, 200.0)


class Bank:
    """Vectorised PH tests: R replicates x C channels."""

    def __init__(self, R, C):
        self.n = np.zeros((R, C))
        self.mean = np.zeros((R, C))
        self.U = np.zeros((R, C))
        self.M = np.zeros((R, C))

    def update(self, ch, e, lam, delta=DELTA):
        r = np.arange(len(ch))
        n = self.n[r, ch] + 1
        mean = self.mean[r, ch] + (e - self.mean[r, ch]) / n
        U = self.U[r, ch] + e - mean - delta
        M = np.minimum(self.M[r, ch], U)
        self.n[r, ch], self.mean[r, ch], self.U[r, ch], self.M[r, ch] = n, mean, U, M
        return (n >= MIN_INST) & (U - M > lam)

    def reset(self, rows):
        for a in (self.n, self.mean, self.U, self.M):
            a[rows] = 0.0


def delay_experiment(pis, mu=0.1, Delta=0.3, tau=20_000, horizon=30_000, R=300, seed=0):
    rng = np.random.default_rng(seed)
    res = []
    for pi in pis:
        out = {}
        for mode in ("class", "pooled"):
            bank = Bank(R, 2 if mode == "class" else 1)
            first = np.full(R, -1)
            for t in range(tau + horizon):
                c = (rng.random(R) < pi).astype(int)             # 1 = the affected class
                p = mu + Delta * ((t >= tau) & (c == 1))
                e = (rng.random(R) < p).astype(float)
                ch = c if mode == "class" else np.zeros(R, dtype=int)
                alarm = bank.update(ch, e, LAM)
                if t < tau:
                    bank.reset(np.flatnonzero(alarm))             # false alarm: restart
                else:
                    new = alarm & (first < 0)
                    first[new] = t - tau
            det = first >= 0
            out[mode] = {"detect_rate": float(det.mean()),
                         "mean_delay": float(first[det].mean()) if det.any() else None,
                         "median_delay": float(np.median(first[det])) if det.any() else None}
        out["pi"] = pi
        out["bound_class"] = (LAM + 1) / (pi * (Delta - DELTA))
        out["bound_pooled"] = (LAM + 1) / (pi * Delta - DELTA) if pi * Delta > DELTA else None
        res.append(out)
        print(out, flush=True)
    return {"mu": mu, "Delta": Delta, "tau": tau, "horizon": horizon, "R": R, "rows": res}


def false_alarm_experiment(lams, pis=(0.1, 0.9), mus=(0.2, 0.05), n=10_000, R=2000, seed=1):
    rng = np.random.default_rng(seed)
    pis, mus = np.asarray(pis), np.asarray(mus)
    th = np.array([theta_star(m) for m in mus])
    mu_pool = float(pis @ mus)
    th_pool = theta_star(mu_pool)
    rows = []
    for lam in lams:
        hit = {}
        for mode in ("class", "pooled"):
            bank = Bank(R, len(pis) if mode == "class" else 1)
            fired = np.zeros(R, bool)
            for t in range(n):
                c = rng.choice(len(pis), size=R, p=pis)
                e = (rng.random(R) < mus[c]).astype(float)
                fired |= bank.update(c if mode == "class" else np.zeros(R, dtype=int), e, lam)
            hit[mode] = float(fired.mean())
        rows.append({"lam": lam, "class": hit["class"], "pooled": hit["pooled"],
                     "bound_class": float(min(1.0, n * np.sum(pis * np.exp(-th * lam)))),
                     "bound_pooled": float(min(1.0, n * np.exp(-th_pool * lam)))})
        print(rows[-1], flush=True)
    return {"pis": pis.tolist(), "mus": mus.tolist(), "theta": th.tolist(), "theta_pooled": th_pool,
            "n": n, "R": R, "rows": rows}


def main():
    res = {"delta": DELTA, "lam": LAM,
           "delay": delay_experiment([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
           "false_alarm": false_alarm_experiment([5, 10, 15, 20, 25, 30, 35])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
