"""Detection benchmark at a common average run length (ARL).

For every stream family:
 1. oracle quantities (post-change shift of the changed class and of the pooled
    residual) are measured on 5 change streams with calibration seeds; they
    tune the pooled LR-CUSUM and both PH variants in their favour;
 2. the threshold of every detector is calibrated by bisection on drift-free
    streams with calibration seeds so that its false-alarm rate equals 1/ARL
    (with restarts after each alarm);
 3. detectors run at these thresholds on 30 evaluation streams with a change
    (seeds 0..29), and on 30 drift-free streams of the same length.

    python bench.py --families sea5 gmm5 --jobs 6
Output: results_v2/bench/<family>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import detectors as D
from baselines import RBMIM, adwin, rbmim_alarms
from streams import SYNTHETIC, load

ROOT = Path(__file__).resolve().parents[1] / "results_v2"
CACHE = ROOT / "cache"
ARL = 50_000
ALPHA = 0.05                      # level of the uncalibrated 'theory' threshold
ODDS = 2.0
EVAL_SEEDS = range(30)
CAL_SEEDS = range(1000, 1030)
ORACLE_SEEDS = range(2000, 2005)
NULL_POST = 100_000               # monitored length after tau of synthetic calibration streams


# --------------------------------------------------------------------------- #
def reference(X, y, W, seed):
    cls, cnt = np.unique(y[:W], return_counts=True)
    sw = (W / (len(cls) * cnt))[np.searchsorted(cls, y[:W])]
    rf = RandomForestClassifier(100, min_samples_leaf=5, random_state=seed, n_jobs=1)
    rf.fit(X[:W], y[:W], sample_weight=sw)
    return rf


def prepare(family, seed, change, null_post=None, need_X=False):
    """Stream + frozen reference predictions (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{family}__{seed}__{int(change)}__{null_post or 0}"
    f = CACHE / f"{tag}.npz"
    z = None
    if f.exists():
        try:
            z = np.load(f, allow_pickle=True)
            y, yhat, info = z["y"], z["yhat"], z["info"].item()
        except Exception:                       # partially written by a concurrent job: rebuild
            z = None
    if z is None:
        X, y, info = load(family, seed, change, null_post)
        rf = reference(X, y, info["W_ref"], seed)
        yhat = np.concatenate([rf.predict(X[s:s + 50_000]) for s in range(0, len(y), 50_000)]).astype(np.int16)
        tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, y=y.astype(np.int16), yhat=yhat, info=np.array(info, dtype=object))
        os.replace(tmp, f)                      # atomic: readers never see a partial file
    ctx = context(y.astype(np.int64), yhat.astype(np.int64), info)
    ctx["src"] = (family, seed, change, null_post)     # features are reloaded only for RBM-IM
    ctx["seed"] = seed
    return ctx


def context(y, yhat, info):
    K, Wr, Wc = info["K"], info["W_ref"], info["W_cal"]
    t0 = Wr + Wc
    e = (yhat != y).astype(np.int64)
    cal = slice(Wr, t0)
    mu_hat, mu_t, m = D.frozen_rates(y[cal], e[cal], K, beta=ALPHA / 2)
    pool_hat = (e[cal].sum() + 0.5) / (Wc + 1)
    pool_t = D.cp_upper(int(e[cal].sum()), Wc, ALPHA / 2)
    counts = np.bincount(y[:Wr], minlength=K)
    present = np.flatnonzero(counts > 0)
    minority = int(present[np.argmin(counts[present])])
    return dict(y=y, yhat=yhat, e=e, K=K, t_w=Wr, t0=t0, n=len(y), tau=info["tau"], target=info["target"],
                info=info, mu_hat=mu_hat, mu_t=mu_t, m=m, pool_hat=pool_hat, pool_t=pool_t, minority=minority,
                L_bank=D.llr_table(mu_hat, D.odds_design(mu_hat, ODDS)),       # calibrated thresholds
                L_theory=D.llr_table(mu_t, D.odds_design(mu_t, ODDS)))         # h = log(2n/alpha)


# --------------------------------------------------------------------------- #
# Detectors: name -> (run(ctx, knob, oracle) -> alarm times >= t0, knob bracket)
# --------------------------------------------------------------------------- #
def _after(a, t0):
    a = np.asarray(a, dtype=np.int64)
    return a[a >= t0]


def run_bank(c, h, o):
    return D.bank_lr(c["y"], c["e"], c["L_bank"], h, c["t0"])


MULTI_ODDS = (1.25, 2.0, 4.0)


def run_bank_multi(c, h, o):
    return D.bank_multi(c["y"], c["e"], c["mu_hat"], MULTI_ODDS, h, c["t0"])


def _bank_r(r):
    def run(c, h, o):
        return D.bank_lr(c["y"], c["e"], D.llr_table(c["mu_hat"], D.odds_design(c["mu_hat"], r)), h, c["t0"])
    return run


def run_bank_theory(c, h, o):
    return D.bank_lr(c["y"], c["e"], c["L_theory"], h, c["t0"])


def run_bank_oracle(c, h, o):
    q = D.odds_design(c["mu_hat"], ODDS)
    k = c["target"]
    q[k] = min(0.999, c["mu_hat"][k] + max(o["dc"], 1e-3))
    return D.bank_lr(c["y"], c["e"], D.llr_table(c["mu_hat"], q), h, c["t0"])


def run_pooled(c, h, o):
    q = min(0.999, c["pool_hat"] + max(o["dpool"], 1e-4))
    return D.pooled_lr(c["e"], D.llr_table([c["pool_hat"]], [q]), h, c["t0"])


def run_ra(c, h, o):
    return D.ra_cusum(c["y"], c["e"], c["L_bank"], h, c["t0"])


def run_ph_bank(c, lam, o):
    return _after(D.ph(c["y"], c["e"], c["K"], max(o["dc"], 1e-3) / 2, lam, c["t_w"]), c["t0"])


def run_ph_pooled(c, lam, o):
    return _after(D.ph(np.zeros_like(c["y"]), c["e"], 1, max(o["dpool"], 1e-4) / 2, lam, c["t_w"]), c["t0"])


def run_ddm_bank(c, k, o):
    return _after(D.ddm(c["y"], c["e"], c["K"], k, c["t_w"]), c["t0"])


def run_ddm_pooled(c, k, o):
    return _after(D.ddm(np.zeros_like(c["y"]), c["e"], 1, k, c["t_w"]), c["t0"])


def run_adwin_bank(c, k, o):
    return _after(adwin(c["y"], c["e"], c["K"], k, c["t_w"]), c["t0"])


def run_adwin_pooled(c, k, o):
    return _after(adwin(np.zeros_like(c["y"]), c["e"], 1, k, c["t_w"]), c["t0"])


def run_ddm_oci(c, k, o):
    return _after(D.ddm_oci(c["y"], c["e"], c["minority"], k, c["t_w"]), c["t0"])


def run_lfr(c, k, o):
    return _after(D.lfr(c["y"], c["yhat"], c["minority"], 10.0 ** (-k), c["t_w"]), c["t0"])


def rbm_seqs(c):
    if "rbm" not in c:
        X, y = load(*c["src"])[0], c["y"]
        r = RBMIM(X.shape[1], c["K"], seed=c["seed"])
        r.fit_reference(X[:c["t_w"]], y[:c["t_w"]])
        c["rbm"] = r.stream_errors(X, y, c["t_w"])
    return c["rbm"]


def run_rbmim(c, k, o):
    return _after(rbmim_alarms(rbm_seqs(c), k, c["t_w"]), c["t0"])


DETECTORS = {
    "bank":         (run_bank, (0.5, 40.0)),
    "bank-oracle":  (run_bank_oracle, (0.5, 40.0)),
    "bank-multi":   (run_bank_multi, (0.5, 40.0)),
    "bank-r1.5":    (_bank_r(1.5), (0.5, 40.0)),
    "bank-r3":      (_bank_r(3.0), (0.5, 40.0)),
    "pooled":       (run_pooled, (1e-3, 40.0)),
    "ra":           (run_ra, (0.5, 40.0)),
    "ph-bank":      (run_ph_bank, (0.5, 400.0)),
    "ph-pooled":    (run_ph_pooled, (0.5, 400.0)),
    "ddm-bank":     (run_ddm_bank, (1.0, 400.0)),
    "ddm-pooled":   (run_ddm_pooled, (1.0, 400.0)),
    "adwin-bank":   (run_adwin_bank, (0.3, 40.0)),
    "adwin-pooled": (run_adwin_pooled, (0.3, 40.0)),
    "ddm-oci":      (run_ddm_oci, (1.0, 400.0)),
    "lfr":          (run_lfr, (2.0, 6.3)),
    "rbmim":        (run_rbmim, (0.0, 60.0)),
    "bank-theory":  (run_bank_theory, None),        # not calibrated: h = log(2 n_mon / alpha)
}


# --------------------------------------------------------------------------- #
def oracle(family):
    dc, dp = [], []
    for s in ORACLE_SEEDS:
        c = prepare(family, s, True)
        tau, k, y, e = c["tau"], c["target"], c["y"], c["e"]
        pre, post = slice(c["t0"], tau), slice(tau, None)
        dc.append(e[post][y[post] == k].mean() - e[pre][y[pre] == k].mean())
        dp.append(e[post].mean() - e[pre].mean())
    return {"dc": float(np.mean(dc)), "dpool": float(np.mean(dp))}


def calibrate(name, ctxs, o, target_rate):
    fn, (lo, hi) = DETECTORS[name]
    mon = sum(c["n"] - c["t0"] for c in ctxs)

    def rate(k):
        return sum(len(fn(c, k, o)) for c in ctxs) / mon

    r_hi = rate(hi)
    if r_hi > target_rate:                      # cannot reach the target within the bracket
        return hi, r_hi, False
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if rate(mid) > target_rate:
            lo = mid
        else:
            hi = mid
    return hi, rate(hi), True


def evaluate(name, ctx, k, o):
    fn = DETECTORS[name][0]
    a = fn(ctx, k, o)
    tau = ctx["tau"]
    if tau is None:
        return {"fa": int(len(a)), "any": bool(len(a) > 0)}
    pre = a[a < tau]
    post = a[a >= tau]
    rec = {"fa": int(len(pre)), "detected": bool(len(post) > 0),
           "delay": int(post[0] - tau) if len(post) else None, "horizon": int(ctx["n"] - tau)}
    if name in ("bank", "bank-theory") and len(post):
        a2, loc = D.bank_lr(ctx["y"], ctx["e"], ctx["L_bank"] if name == "bank" else ctx["L_theory"],
                            k, ctx["t0"], return_loc=True)
        i = int(np.flatnonzero(a2 == post[0])[0])
        rec["loc_err"] = int(loc[i] - tau)
    return rec


def run_family(family, names=None, arl=ARL, out_dir=None, n_eval=30, n_cal=None, null_post=None):
    """Calibrate and evaluate every detector on one family.

    For ARL above 5e4 the drift-free calibration data grow with the ARL so that
    every threshold still rests on about 60 false alarms: longer synthetic
    streams, more permutations of the real streams (at most 100).
    """
    t_start = time.time()
    names = [x for x in (names or list(DETECTORS)) if x != "bank-theory" and not x.startswith("bank-r")] \
        if names is None else [x for x in names if x != "bank-theory"]
    scale = max(1.0, arl / ARL)
    if null_post is None:
        null_post = int(min(400_000, NULL_POST * scale)) if family in SYNTHETIC else None
    else:
        null_post = null_post if family in SYNTHETIC else None
    if n_cal is None:
        n_cal = 30 if family in SYNTHETIC else int(min(100, math.ceil(30 * scale)))
    out_dir = Path(out_dir) if out_dir else ROOT / "bench"
    out_f = out_dir / f"{family}.json"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out_f.read_text()) if out_f.exists() else {"family": family, "arl": arl, "det": {}}
    o = res.get("oracle") or oracle(family)
    res["oracle"] = o
    need_X = "rbmim" in names
    cal = [prepare(family, s, False, null_post, need_X) for s in range(1000, 1000 + n_cal)]
    ev = [prepare(family, s, True, None, need_X) for s in range(n_eval)]
    nul = [prepare(family, s, False, None, need_X) for s in range(n_eval)]
    res.update(share=float(np.mean([np.mean(c["y"] == c["target"]) for c in ev])),
               mu_pre_target=float(np.mean([c["mu_hat"][c["target"]] for c in ev])),
               n_mon=int(ev[0]["n"] - ev[0]["t0"]), n_cal=n_cal, null_post=null_post, n_eval=n_eval)
    for name in names:
        if name in res["det"]:
            continue
        t = time.time()
        k, r, ok = calibrate(name, cal, o, 1.0 / arl)
        res["det"][name] = {"knob": k, "cal_rate": r, "cal_ok": ok,
                            "runs": [evaluate(name, c, k, o) for c in ev],
                            "null": [evaluate(name, c, k, o) for c in nul]}
        print(f"{family:10s} {name:13s} arl_target={arl:.0e} knob={k:9.4f} arl={1 / max(r, 1e-12):9.0f} ok={ok} "
              f"det={np.mean([x['detected'] for x in res['det'][name]['runs']]):.2f} ({time.time() - t:.0f}s)",
              flush=True)
        out_f.write_text(json.dumps(res, indent=1))
    h = math.log(2 * res["n_mon"] / ALPHA)
    res["det"]["bank-theory"] = {"knob": h, "cal_rate": None, "cal_ok": True,
                                 "runs": [evaluate("bank-theory", c, h, o) for c in ev],
                                 "null": [evaluate("bank-theory", c, h, o) for c in nul]}
    out_f.write_text(json.dumps(res, indent=1))
    print(f"{family} done in {time.time() - t_start:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--detectors", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--arl", type=float, default=ARL)
    ap.add_argument("--out", default=None, help="output folder (default results_v2/bench)")
    ap.add_argument("--n-eval", type=int, default=30)
    a = ap.parse_args()
    n = len(a.families)
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_family, f, a.detectors, a.arl, a.out, a.n_eval) for f in a.families]
        for f in futs:
            f.result()


if __name__ == "__main__":
    main()
