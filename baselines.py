"""Stream-learning baselines.

River: HAT, ARF, SRP.  Own implementations: OOB/UOB (Wang et al., 2015) and
ARF-US, an ARF trained on the undersampled stream of UOB.  ROSE (Cano and
Krawczyk, 2022) runs the authors' Java code inside MOA (rose/, see build.sh).
In the hold-out protocol each learner makes one pass over the training prefix
and is then frozen, exactly like PHASE (no test labels used).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from river import ensemble, forest, tree

ROSE_DIR = Path(__file__).resolve().parent / "rose"
_BREW_JAVA = "/opt/homebrew/opt/openjdk/bin/java"
JAVA = os.environ.get("JAVA", _BREW_JAVA if os.path.exists(_BREW_JAVA) else shutil.which("java") or "java")


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


class UnderSampled:
    """Wraps a River classifier: each sample is learned with probability
    min(1, w_min / w_y), using the time-decayed class shares of UOB."""

    def __init__(self, model, theta=0.9, seed=0):
        self.model, self.theta = model, theta
        self.rng = np.random.default_rng(seed)
        self.share = {}

    def learn_one(self, x, y):
        for k in self.share:
            self.share[k] *= self.theta
        self.share[y] = self.share.get(y, 0.0) + (1 - self.theta)
        if self.rng.random() < min(self.share.values()) / self.share[y]:
            self.model.learn_one(x, y)

    def predict_one(self, x):
        return self.model.predict_one(x)


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
    if name == "ARF-US":
        return UnderSampled(forest.ARFClassifier(n_models=10, seed=seed), seed=seed)
    raise ValueError(name)


def write_arff(path, X, y, K):
    with open(path, "w") as f:
        f.write("@relation stream\n")
        for j in range(X.shape[1]):
            f.write(f"@attribute x{j} numeric\n")
        f.write("@attribute class {" + ",".join(str(c) for c in range(K)) + "}\n@data\n")
        np.savetxt(f, np.column_stack([X, y]), fmt=["%.10g"] * X.shape[1] + ["%d"], delimiter=",")


def run_rose(X, y, K, n_train, seed):
    """ROSE predictions: hold-out (n_train > 0, predicts y[n_train:]) or prequential (n_train = 0)."""
    cp = os.pathsep.join([str(ROSE_DIR / "build" / "classes"), str(ROSE_DIR / "lib" / "moa.jar")])
    with tempfile.TemporaryDirectory() as tmp:
        arff, out = Path(tmp) / "s.arff", Path(tmp) / "pred.txt"
        write_arff(arff, X, y, K)
        subprocess.run([JAVA, "-Xmx3g", "-cp", cp, "RunROSE", str(arff), str(n_train), str(seed), str(out)],
                       check=True, stdout=subprocess.DEVNULL)
        return np.loadtxt(out, dtype=int).reshape(-1)


def run(name, X_tr, y_tr, X_te, seed, fallback):
    model = make(name, seed)
    for row, label in zip(X_tr, y_tr):
        model.learn_one(dict(enumerate(row)), int(label))
    pred = []
    for row in X_te:
        p = model.predict_one(dict(enumerate(row)))
        pred.append(fallback if p is None else int(p))
    return np.asarray(pred)
