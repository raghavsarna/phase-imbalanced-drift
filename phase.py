"""PHASE: Page-Hinkley-segmented Adaptive Stacked Ensemble.

Phase I   segmentation of the training stream (detector.py)
Phase II  cost-sensitive base learners DT, LR, KM; cost on/off per learner
          chosen by cross-fitted balanced accuracy           (Eqs. (6)-(8))
Phase III likelihood-proportional fusion weights; RBF-SVM admitted through
          the nested convex reweighting if the fused balanced accuracy is
          below tau                                          (Eqs. (9)-(11))
Phase IV  one random-forest meta-learner trained on the cross-fitted,
          weighted posteriors of all segments (Eq. (12)); with meta="gate"
          it is kept only if its cross-validated balanced accuracy on the
          most recent segment exceeds that of the fused rule argmax p
Inference routes a query to the most recent segment's base layer (Eq. (13)).

Design notes:
  * posteriors used for the weights, the refinement decision and the
    meta-learner are out-of-fold (cross-fitted), not in-sample;
  * class costs follow Eq. (6); the cost on/off choice uses balanced accuracy;
  * k-means posteriors are cost-weighted cluster histograms (Eq. (7));
  * SVM posteriors are Platt-calibrated on out-of-fold decision values;
  * all methods use the raw input features (no hand-crafted features).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from detector import segment

BASE = ("DT", "LR", "KM")


@dataclass
class PhaseConfig:
    detector: str = "class"      # class | pooled | periodic | none
    lam: float = 25.0             # calibrated on SEA seeds 100-102 (not used for evaluation)
    delta: float = 0.02
    min_inst: int = 30
    W_ref: int = 1000
    L_min: int = 1000
    period: int = 5000
    use_costs: bool = True
    use_refine: bool = True
    meta: str = "gate"           # gate | always | never   (Phase IV)
    gate_trees: int = 100
    tau: float = 0.9
    folds: int = 5
    svm_max: int = 10_000        # SVM fitted on at most this many samples per segment
    km_k: int = 10
    rf_trees: int = 500
    seed: int = 0
    n_jobs: int = 1


def _to_K(model_classes, P, K):
    out = np.zeros((P.shape[0], K))
    out[:, np.asarray(model_classes, dtype=int)] = P
    return out


def _balanced_acc(y, P):
    return balanced_accuracy_score(y, P.argmax(1))


class _KMPosterior:
    """k-means posterior: cost-weighted, Laplace-smoothed class histograms (Eq. (7))."""

    def __init__(self, k, K, seed):
        self.k, self.K, self.seed = k, K, seed

    def fit(self, Z, y, w):
        return self.fit_clusters(Z).fit_hist(y, w)

    def fit_clusters(self, Z):
        self.km = KMeans(n_clusters=min(self.k, len(Z)), n_init=10, random_state=self.seed).fit(Z)
        return self

    def fit_hist(self, y, w):
        present = np.unique(y)
        H = np.zeros((self.km.n_clusters, self.K))
        np.add.at(H, (self.km.labels_, y), w)
        H[:, present] += 1.0
        self.table = H / H.sum(1, keepdims=True)
        return self

    def predict_proba(self, Z):
        return self.table[self.km.predict(Z)]


def _svm_scores(svm, Z, K):
    """Decision values mapped to K columns (one-vs-rest layout)."""
    d = svm.decision_function(Z)
    cls = svm.classes_.astype(int)
    D = np.zeros((len(Z), K))
    if d.ndim == 1:
        D[:, cls[1]], D[:, cls[0]] = d, -d
    else:
        D[:, cls] = d
    return D


@dataclass
class SegmentModel:
    start: int
    end: int
    n: int
    class_counts: dict
    b_star: dict = field(default_factory=dict)
    W: dict = field(default_factory=dict)          # final weights tilde W
    ba_fused: float = 0.0
    ba_oof: dict = field(default_factory=dict)
    svm_admitted: bool = False
    alpha: float = 1.0


class PHASE:
    def __init__(self, cfg: PhaseConfig, n_classes: int):
        self.cfg, self.K = cfg, n_classes

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        cfg = self.cfg
        self.boundaries, self.alarms = segment(
            X, y, self.K, mode=cfg.detector, lam=cfg.lam, delta=cfg.delta,
            min_inst=cfg.min_inst, W_ref=cfg.W_ref, L_min=cfg.L_min, period=cfg.period, seed=cfg.seed)
        edges = [0, *self.boundaries, len(y)]
        self.segments, self._models, metas, labels = [], [], [], []
        for s, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            info, models, F, folds = self._fit_segment(X[a:b], y[a:b], a, b, seed=cfg.seed + 7919 * s)
            self.segments.append(info)
            self._models.append(models)
            metas.append(F)
            labels.append(y[a:b])
        self.gate = None
        if cfg.meta == "gate":
            self.gate = self._gate(metas, labels, folds)
            self.stack = self.gate["ba_meta"] > self.gate["ba_fused"]
        else:
            self.stack = cfg.meta == "always"
        if self.stack:
            self.meta = self._rf(cfg.rf_trees).fit(np.vstack(metas), np.concatenate(labels))
        return self

    def _rf(self, trees):
        return RandomForestClassifier(
            n_estimators=trees, max_depth=20, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced" if self.cfg.use_costs else None,
            n_jobs=self.cfg.n_jobs, random_state=self.cfg.seed)

    def _gate(self, metas, labels, folds):
        """Cross-validated balanced accuracy of stacking vs. the fused rule on the last segment.

        The meta-learner is refitted F times on all earlier segments plus F-1 folds
        of the last one and evaluated on the held-out fold; only training data are used.
        """
        F_last, y_last = metas[-1], labels[-1]
        F_prev = np.vstack(metas[:-1]) if len(metas) > 1 else F_last[:0]
        y_prev = np.concatenate(labels[:-1]) if len(labels) > 1 else y_last[:0]
        pred = np.zeros_like(y_last)
        for tr, te in folds:
            g = self._rf(self.cfg.gate_trees).fit(np.vstack([F_prev, F_last[tr]]),
                                                  np.concatenate([y_prev, y_last[tr]]))
            pred[te] = g.classes_[g.predict_proba(F_last[te]).argmax(1)]
        fused = F_last.reshape(len(y_last), 4, self.K).sum(1)        # additive decomposition of p
        return {"ba_meta": float(balanced_accuracy_score(y_last, pred)),
                "ba_fused": float(balanced_accuracy_score(y_last, fused.argmax(1)))}

    def _fit_segment(self, Xs, ys, a, b, seed):
        cfg, K = self.cfg, self.K
        rng = np.random.default_rng(seed)
        N = len(ys)
        present, counts = np.unique(ys, return_counts=True)
        info = SegmentModel(a, b, N, {int(c): int(n) for c, n in zip(present, counts)})

        scaler = StandardScaler().fit(Xs)
        Z = scaler.transform(Xs)
        costs = np.ones(K)
        if cfg.use_costs:
            costs[present] = N / (len(present) * counts)               # Eq. (6)
        w = costs[ys]

        if counts.min() >= cfg.folds:
            splitter = StratifiedKFold(cfg.folds, shuffle=True, random_state=seed)
        else:
            splitter = KFold(cfg.folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(Z, ys))

        # ---- Phase II: cross-fitted posteriors for cost off/on ------------
        settings = (0, 1) if cfg.use_costs else (0,)
        oof = {(M, bb): np.zeros((N, K)) for M in BASE for bb in settings}
        for tr, te in folds:
            km = _KMPosterior(cfg.km_k, K, seed).fit_clusters(Z[tr])
            for bb in settings:
                sw = w[tr] if bb else None
                dt = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=20,
                                            min_samples_leaf=10, random_state=seed)
                dt.fit(Z[tr], ys[tr], sample_weight=sw)
                oof[("DT", bb)][te] = _to_K(dt.classes_, dt.predict_proba(Z[te]), K)
                lr = LogisticRegression(C=1.0, max_iter=2000)
                lr.fit(Z[tr], ys[tr], sample_weight=sw)
                oof[("LR", bb)][te] = _to_K(lr.classes_, lr.predict_proba(Z[te]), K)
                km.fit_hist(ys[tr], w[tr] if bb else np.ones(len(tr)))
                oof[("KM", bb)][te] = km.predict_proba(Z[te])
        for M in BASE:                                                  # Eq. (8), decoupled
            scores = {bb: _balanced_acc(ys, oof[(M, bb)]) for bb in settings}
            info.ba_oof[M] = scores
            info.b_star[M] = max(settings, key=lambda bb: (scores[bb], bb))
        P = {M: oof[(M, info.b_star[M])] for M in BASE}

        # ---- Phase III: likelihood weights, fusion, conditional refinement -
        S = {M: float(np.sum(w * P[M][np.arange(N), ys])) for M in BASE}   # Eq. (9)
        Wb = {M: S[M] / sum(S.values()) for M in BASE}
        pB = sum(Wb[M] * P[M] for M in BASE)
        info.ba_fused = _balanced_acc(ys, pB)
        W = dict(Wb)
        svm = platt = None
        sub = np.arange(N) if N <= cfg.svm_max else np.sort(rng.choice(N, cfg.svm_max, replace=False))
        if cfg.use_refine and info.ba_fused < cfg.tau:
            insub = np.zeros(N, bool)
            insub[sub] = True
            D = np.zeros((N, K))
            for tr, te in folds:
                trs = tr[insub[tr]]
                m = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed)
                m.fit(Z[trs], ys[trs], sample_weight=w[trs] if cfg.use_costs else None)
                D[te] = _svm_scores(m, Z[te], K)
            platt = LogisticRegression(C=1.0, max_iter=2000)
            platt.fit(D, ys, sample_weight=w if cfg.use_costs else None)
            P["SVM"] = _to_K(platt.classes_, platt.predict_proba(D), K)
            S_svm = float(np.sum(w * P["SVM"][np.arange(N), ys]))
            S_B = sum(Wb[M] * S[M] for M in BASE)
            alpha = S_B / (S_B + S_svm)                                     # Eq. (10)
            W = {M: alpha * Wb[M] for M in BASE}                            # Eq. (11)
            W["SVM"] = 1 - alpha
            info.alpha, info.svm_admitted = alpha, True
            svm = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed)
            svm.fit(Z[sub], ys[sub], sample_weight=w[sub] if cfg.use_costs else None)
        info.W = W

        F = self._meta_features(P, W, N)                                    # Eq. (12), out-of-fold

        # ---- refit the base layer on the whole segment for deployment ------
        final = {"scaler": scaler, "svm": svm, "platt": platt, "W": W}
        sw = lambda M: w if info.b_star[M] else None
        final["DT"] = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=20,
                                             min_samples_leaf=10, random_state=seed).fit(Z, ys, sample_weight=sw("DT"))
        final["LR"] = LogisticRegression(C=1.0, max_iter=2000).fit(Z, ys, sample_weight=sw("LR"))
        final["KM"] = _KMPosterior(cfg.km_k, K, seed).fit(Z, ys, w if info.b_star["KM"] else np.ones(N))
        return info, final, F, folds

    def _meta_features(self, P, W, N):
        blocks = [W[M] * P[M] for M in BASE]
        blocks.append(W["SVM"] * P["SVM"] if "SVM" in W else np.zeros((N, self.K)))
        return np.hstack(blocks)

    # -------------------------------------------------------------- predict
    def _base_posteriors(self, models, X):
        Z = models["scaler"].transform(X)
        P = {M: _to_K(models[M].classes_, models[M].predict_proba(Z), self.K) for M in ("DT", "LR")}
        P["KM"] = models["KM"].predict_proba(Z)
        if models["svm"] is not None:
            D = _svm_scores(models["svm"], Z, self.K)
            P["SVM"] = _to_K(models["platt"].classes_, models["platt"].predict_proba(D), self.K)
        return P

    def predict_proba(self, X):
        """Queries beyond the training horizon are routed to the last segment."""
        models = self._models[-1]
        P = self._base_posteriors(models, X)
        W = models["W"]
        if not self.stack:
            return sum(W[M] * P[M] for M in W)
        F = self._meta_features(P, W, len(X))
        return _to_K(self.meta.classes_, self.meta.predict_proba(F), self.K)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def meta_importance_by_learner(self):
        imp = self.meta.feature_importances_
        K = self.K
        names = [*BASE, "SVM"]
        return {names[i]: float(imp[i * K:(i + 1) * K].sum()) for i in range(4)}
