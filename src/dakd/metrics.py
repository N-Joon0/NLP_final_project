from __future__ import annotations

import numpy as np


def reliability(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> list[dict]:
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "conf": float(conf[m].mean()) if m.any() else None,
                    "acc": float(acc[m].mean()) if m.any() else None})
    return out


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    n = len(labels)
    return float(sum(b["n"] / n * abs(b["acc"] - b["conf"]) for b in reliability(probs, labels, bins) if b["n"]))


def compute_metrics(probs: np.ndarray, labels: np.ndarray, primary: str, bins: int = 15) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score
    pred = probs.argmax(1)
    K = probs.shape[1]
    p = np.clip(probs, 1e-12, 1.0)
    onehot = np.eye(K)[labels]
    out = {
        "accuracy": float((pred == labels).mean()),
        "macro_f1": float(f1_score(labels, pred, average="macro", labels=list(range(K)), zero_division=0)),
        "per_class_f1": [float(x) for x in f1_score(labels, pred, average=None, labels=list(range(K)), zero_division=0)],
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(K))).tolist(),
        "ece": ece(probs, labels, bins),
        "nll": float(-np.log(p[np.arange(len(labels)), labels]).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "mean_confidence": float(probs.max(1).mean()),
        "n": int(len(labels)),
    }
    out["reliability"] = reliability(probs, labels, bins)
    out["primary"] = out[primary]
    out["primary_name"] = primary
    return out
