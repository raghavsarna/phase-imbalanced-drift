"""Robustness of the detectors to the modelling assumptions (R1-4, R2-3, R2-4).

labels    labels missing completely at random (labelled fraction rho) and
          delayed by d samples; thresholds are those calibrated with full,
          immediate labels in bench.py.  The detector sees the labelled
          samples in label-arrival order; delays are in stream time.
refmodel  quality of the frozen reference model on SEA_5 and Gauss_5: the
          pre-change residual rate of the changed class is varied through the
          model class and training size; the bank is recalibrated to the
          common ARL for each model and its delay is compared with Prop. 2.
prior     pure prior shift (SEA, class-0 share 5% -> 15%, no concept change):
          probability of an alarm within 20 000 samples after the shift, with
          the thresholds calibrated on SEA_5.
Output: results_v2/robust.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import bench as B
import detectors as D
from streams import load

OUT = B.ROOT / "robust.json"
SEEDS = range(30)


def _bench(family):
    return json.loads((B.ROOT / "bench" / f"{family}.json").read_text())


def sub_ctx(ctx, rho, d, rng):
    """Context restricted to labelled samples (MCAR rho), in label-arrival order (delay d)."""
    n = ctx["n"]
    lab = rng.random(n) < rho
    lab[:ctx["t_w"]] = True                       # the reference window is labelled
    idx = np.flatnonzero(lab)
    info = dict(ctx["info"])
    t0_sub = int(np.searchsorted(idx, ctx["t0"]))
    info["W_ref"] = int(np.searchsorted(idx, ctx["t_w"]))
    info["W_cal"] = t0_sub - info["W_ref"]
    c = B.context(ctx["y"][idx], ctx["yhat"][idx], info)
    c["tau"] = None                               # handled in stream time
    c["arrival"] = idx + d
    c["src"], c["seed"] = ctx["src"], ctx["seed"]
    return c


def labels(families=("sea5", "gmm5", "credit"), rhos=(1.0, 0.5, 0.2), ds=(0, 1000, 5000),
           names=("bank", "pooled", "ph-bank", "ddm-oci", "lfr")):
    rows = []
    for fam in families:
        res = _bench(fam)
        o = res["oracle"]
        ctxs = [B.prepare(fam, s, True) for s in SEEDS]
        for rho in rhos:
            for d in ds:
                rng = np.random.default_rng(int(rho * 1000) + d)
                subs = [sub_ctx(c, rho, d, rng) for c in ctxs]
                for name in names:
                    k = res["det"][name]["knob"]
                    fn = B.DETECTORS[name][0]
                    delays, fas = [], []
                    for c, full in zip(subs, ctxs):
                        a = c["arrival"][fn(c, k, o)]
                        tau, H = full["tau"], full["n"] - full["tau"]
                        post = a[a >= tau]
                        delays.append(int(post[0] - tau) if len(post) else H)
                        fas.append(int((a < tau).sum()))
                    rows.append({"family": fam, "rho": rho, "d": d, "det": name,
                                 "delay": float(np.mean(delays)), "delays": delays,
                                 "fa": float(np.mean(fas))})
                    print(fam, rho, d, name, round(np.mean(delays)), flush=True)
    return rows


REFS = {
    "RF":          lambda s: RandomForestClassifier(100, min_samples_leaf=5, random_state=s, n_jobs=1),
    "RF-500":      "RF trained on the first 500 samples only",
    "tree-d2":     lambda s: DecisionTreeClassifier(max_depth=2, random_state=s),
    "logistic":    lambda s: LogisticRegression(max_iter=1000),
    "naive-bayes": lambda s: GaussianNB(),
}


def _ref_ctx(fam, seed, change, ref):
    X, y, info = load(fam, seed, change, B.NULL_POST if not change else None)
    W = info["W_ref"]
    Wfit = 500 if ref == "RF-500" else W
    m = REFS["RF"](seed) if ref == "RF-500" else REFS[ref](seed)
    cls, cnt = np.unique(y[:Wfit], return_counts=True)
    sw = (Wfit / (len(cls) * cnt))[np.searchsorted(cls, y[:Wfit])]
    m.fit(X[:Wfit], y[:Wfit], sample_weight=sw)
    yhat = np.concatenate([m.predict(X[s:s + 100_000]) for s in range(0, len(y), 100_000)])
    return B.context(y.astype(np.int64), yhat.astype(np.int64), info)


def refmodel(families=("sea5", "gauss5")):
    rows = []
    for fam in families:
        for ref in REFS:
            cal = [_ref_ctx(fam, s, False, ref) for s in range(1000, 1003)]
            ev = [_ref_ctx(fam, s, True, ref) for s in range(10)]
            h, r, ok = B.calibrate("bank", cal, {}, 1.0 / B.ARL)
            k = ev[0]["target"]
            mu = float(np.mean([c["mu_hat"][k] for c in ev]))
            p = float(np.mean([c["e"][c["tau"]:][c["y"][c["tau"]:] == k].mean() for c in ev]))
            pi = float(np.mean([np.mean(c["y"] == k) for c in ev]))
            ba = float(np.mean([1 - np.mean([c["e"][c["t0"]:c["tau"]][c["y"][c["t0"]:c["tau"]] == j].mean()
                                             for j in range(c["K"])]) for c in ev]))
            delays = []
            for c in ev:
                a = B.run_bank(c, h, {})
                post = a[a >= c["tau"]]
                delays.append(int(post[0] - c["tau"]) if len(post) else c["n"] - c["tau"])
            hm, _, _ = B.calibrate("bank-multi", cal, {}, 1.0 / B.ARL)
            dm = []
            for c in ev:
                a = B.run_bank_multi(c, hm, {})
                post = a[a >= c["tau"]]
                dm.append(int(post[0] - c["tau"]) if len(post) else c["n"] - c["tau"])
            q = D.odds_design(np.array([mu]), B.ODDS)[0]
            g = p * math.log(q / mu) + (1 - p) * math.log((1 - q) / (1 - mu))
            bound = (h + math.log(q / mu)) / (pi * g) if g > 0 else None
            rows.append({"family": fam, "ref": ref, "ba_pre": ba, "mu": mu, "p": p, "pi": pi, "h": h,
                         "drift": g, "bound": bound, "delay": float(np.mean(delays)), "delays": delays,
                         "delay_multi": float(np.mean(dm)), "h_multi": hm})
            print(rows[-1]["family"], ref, f"BA={ba:.3f} mu={mu:.3f} p={p:.3f} g={g:.4f} "
                  f"bound={bound} delay={np.mean(delays):.0f} multi={np.mean(dm):.0f}", flush=True)
    return rows


def prior(names=None, window=20_000):
    res = _bench("sea5")
    o = res["oracle"]
    names = names or [n for n in B.DETECTORS if n != "bank-theory" and n in res["det"]]
    rows = []
    sets = {fam: [B.prepare(fam, s, True, need_X="rbmim" in names) for s in SEEDS]
            for fam in ("priorshift", "priorctrl")}
    for name in names:
        k = res["det"][name]["knob"]
        fn = B.DETECTORS[name][0]
        out = {}
        for fam, ctxs in sets.items():
            hit = []
            for c in ctxs:
                a = fn(c, k, o)
                tau = c["tau"]
                hit.append(bool(((a >= tau) & (a < tau + window)).any()))
            out[fam] = float(np.mean(hit))
        rows.append({"det": name, "p_alarm_after": out["priorshift"], "p_alarm_control": out["priorctrl"],
                     "null_reference": 1 - math.exp(-window / B.ARL)})
        print(name, rows[-1], flush=True)
    return rows


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for part in sys.argv[1:] or ["prior", "labels", "refmodel"]:
        res[part] = globals()[part]()
        OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
