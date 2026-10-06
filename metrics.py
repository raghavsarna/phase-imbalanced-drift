"""Imbalance-aware evaluation metrics (computed on classes present in y_true)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, matthews_corrcoef)


def evaluate(y_true, y_pred):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recall = cm.diagonal() / cm.sum(1)
    support = cm.sum(1)
    minority = int(labels[np.argmin(support)])
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "ba": float(recall.mean()),
        "gmean": float(np.exp(np.mean(np.log(np.maximum(recall, 1e-12)))) if recall.min() > 0 else 0.0),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": {int(c): float(r) for c, r in zip(labels, recall)},
        "support": {int(c): int(s) for c, s in zip(labels, support)},
        "minority_class": minority,
        "minority_recall": float(recall[np.argmin(support)]),
        "confusion": cm.tolist(),
        "labels": labels.tolist(),
    }
