"""Stream loaders.

Every loader returns ``(X, y, meta)`` where ``X`` is a float array of shape
(n, m) in stream order, ``y`` holds integer labels 0..K-1 and ``meta`` is a
dict with at least ``name``, ``n_classes`` and (for synthetic streams) the
ground-truth ``change_points``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from river.datasets import Elec2, synth

ROOT = Path(__file__).resolve().parent
COVTYPE_CSV = Path(os.environ.get("PHASE_COVTYPE_CSV", ROOT / "data" / "covtype.csv"))

# River SEA variants -> decision threshold theta (y = 1{x1 + x2 > theta})
SEA_THETA = {0: 8.0, 1: 9.0, 2: 7.0, 3: 9.5}


def _sea_class0_share(theta: float, noise: float) -> float:
    """P(y = 0) for SEA with x ~ U[0,10]^3 and label-flip probability noise."""
    clean = theta ** 2 / 200.0            # area of {x1 + x2 <= theta} / 100
    return (1 - noise) * clean + noise * (1 - clean)


def sea_stream(seed: int, minority_share: float | None = None,
               variants=(0, 1, 2, 3), lengths=(10_000, 10_000, 10_000, 20_000),
               noise: float = 0.1):
    """SEA stream with abrupt real drifts between consecutive concepts.

    Each concept is drawn from its own River generator and the concepts are
    concatenated, giving abrupt real drifts at the concept boundaries.

    ``minority_share`` (e.g. 0.1) imposes a fixed class-0 share in every
    concept by rejection sampling of class-0 samples; ``None`` keeps the
    natural SEA priors (class-0 share 0.30-0.46 depending on theta).
    """
    rng = np.random.default_rng(seed)
    X, y, change_points = [], [], []
    for j, (v, length) in enumerate(zip(variants, lengths)):
        if j > 0:
            change_points.append(int(sum(lengths[:j])))
        gen = iter(synth.SEA(variant=v, noise=noise, seed=10_000 * seed + j))
        p0 = _sea_class0_share(SEA_THETA[v], noise)
        if minority_share is None:
            keep0 = 1.0
        else:
            keep0 = min(1.0, minority_share * (1 - p0) / (p0 * (1 - minority_share)))
        count = 0
        while count < length:
            x, label = next(gen)
            label = int(label)
            if label == 0 and rng.random() >= keep0:
                continue
            X.append([x[0], x[1], x[2]])
            y.append(label)
            count += 1
    name = "SEA" if minority_share is None else f"SEA-{minority_share:g}"
    return (np.asarray(X, dtype=float), np.asarray(y, dtype=int),
            {"name": name, "n_classes": 2, "change_points": change_points,
             "thetas": [SEA_THETA[v] for v in variants]})


def elec2():
    """Elec2 (Harries, 1999) via River; the time index 'date' is dropped."""
    cols = ["day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    X, y = [], []
    for x, label in Elec2():
        X.append([float(x[c]) for c in cols])
        y.append(int(label))
    return (np.asarray(X), np.asarray(y, dtype=int),
            {"name": "Elec2", "n_classes": 2, "change_points": None})


RIVER_DATA = Path(os.environ.get("RIVER_DATA", Path.home() / "river_data"))


def _csv_stream(path, label_col=-1, drop=(), header="infer"):
    df = pd.read_csv(path, header=header)
    lab = df.iloc[:, label_col]
    feats = df.drop(columns=[df.columns[label_col], *drop])
    classes = np.unique(lab)
    y = np.searchsorted(classes, lab.to_numpy())
    return feats.to_numpy(dtype=float), y.astype(int), len(classes)


def insects(variant):
    """INSECTS (Souza et al., 2020), imbalanced variants, via River's download."""
    from river.datasets import Insects
    ds = Insects(variant=variant)
    if not ds.is_downloaded:
        ds.download()
    X, y, K = _csv_stream(ds.path, header=None)
    name = {"abrupt_imbalanced": "INSECTS-abrupt", "gradual_imbalanced": "INSECTS-gradual"}[variant]
    return X, y, {"name": name, "n_classes": K, "change_points": None}


def creditcard():
    """Credit-card fraud stream (0.17% fraud) in time order; the 'Time' index is dropped."""
    from river.datasets import CreditCard
    ds = CreditCard()
    if not ds.is_downloaded:
        ds.download()
    X, y, K = _csv_stream(ds.path, drop=("Time",))
    return X, y, {"name": "CreditCard", "n_classes": K, "change_points": None}


def covtype():
    """Forest Covertype (Blackard & Dean, 1999) in its standard stream order."""
    df = pd.read_csv(COVTYPE_CSV)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=int) - 1
    return X, y, {"name": "Covertype", "n_classes": 7, "change_points": None}


LOADERS = {
    "sea":    lambda seed: sea_stream(seed, None),
    "sea10":  lambda seed: sea_stream(seed, 0.10),
    "sea5":   lambda seed: sea_stream(seed, 0.05),
    # detection study only: further minority shares and drift-free streams
    "sea20":  lambda seed: sea_stream(seed, 0.20),
    "sea2":   lambda seed: sea_stream(seed, 0.02),
    "sea1":   lambda seed: sea_stream(seed, 0.01),
    "seanull":   lambda seed: sea_stream(seed, None, variants=(0,), lengths=(50_000,)),
    "seanull5":  lambda seed: sea_stream(seed, 0.05, variants=(0,), lengths=(50_000,)),
    "insects_abrupt":  lambda seed: insects("abrupt_imbalanced"),
    "insects_gradual": lambda seed: insects("gradual_imbalanced"),
    "creditcard": lambda seed: creditcard(),
    "elec2":  lambda seed: elec2(),
    "covtype": lambda seed: covtype(),
}


def load(name: str, seed: int = 0):
    return LOADERS[name](seed)
