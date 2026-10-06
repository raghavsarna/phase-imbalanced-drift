"""Stream-learning baselines (River). Each is trained in one pass over the
training prefix and then frozen, exactly like PHASE (no test labels used)."""
from __future__ import annotations

import numpy as np
from river import ensemble, forest, tree


class ResamplingOnlineBagging:
    """OOB / UOB of Wang, Minku and Yao (IEEE TKDE 2015).

    Time-decayed class shares w_k(t) = theta w_k(t-1) + (1 - theta) 1{y_t = k};
    each member learns the sample k ~ Poisson(lambda) times, with
    lambda = w_max / w_y (oversampling, OOB) or w_min / w_y (undersampling, UOB).
    """

    def __init__(self, mode="over", n_models=10, theta=0.9, seed=0):
        self.mode, self.theta = mode, theta
        self.rng = np.random.default_rng(seed)
        self.models = [tree.HoeffdingTreeClassifier() for _ in range(n_models)]
        self.share = {}

    def learn_one(self, x, y):
        for k in self.share:
            self.share[k] *= self.theta
        self.share[y] = self.share.get(y, 0.0) + (1 - self.theta)
        ref = max(self.share.values()) if self.mode == "over" else min(self.share.values())
        lam = ref / self.share[y]
        for m in self.models:
            k = self.rng.poisson(lam)
            if k > 0:
                m.learn_one(x, y, w=float(k))

    def predict_one(self, x):
        votes = {}
        for m in self.models:
            for c, p in m.predict_proba_one(x).items():
                votes[c] = votes.get(c, 0.0) + p
        return max(votes, key=votes.get) if votes else None


def make(name: str, seed: int):
    if name == "HAT":
        return tree.HoeffdingAdaptiveTreeClassifier(seed=seed)
    if name == "ARF":
        return forest.ARFClassifier(n_models=10, seed=seed)
    if name == "SRP":
        return ensemble.SRPClassifier(n_models=10, seed=seed)
    if name == "OOB":
        return ResamplingOnlineBagging("over", seed=seed)
    if name == "UOB":
        return ResamplingOnlineBagging("under", seed=seed)
    raise ValueError(name)


def run(name, X_tr, y_tr, X_te, seed, fallback):
    model = make(name, seed)
    for row, label in zip(X_tr, y_tr):
        model.learn_one(dict(enumerate(row)), int(label))
    pred = []
    for row in X_te:
        p = model.predict_one(dict(enumerate(row)))
        pred.append(fallback if p is None else int(p))
    return np.asarray(pred)
