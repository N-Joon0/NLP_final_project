from __future__ import annotations

import numpy as np


def _metric_batch(y: np.ndarray, pred: np.ndarray, I: np.ndarray, metric: str, K: int) -> np.ndarray:
    yy, pp = y[I], pred[I]
    if metric == "accuracy":
        return (yy == pp).mean(1)
    f1s = []
    for k in range(K):
        tp = ((yy == k) & (pp == k)).sum(1)
        fp = ((yy != k) & (pp == k)).sum(1)
        fn = ((yy == k) & (pp != k)).sum(1)
        den = 2 * tp + fp + fn
        f1s.append(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0))
    return np.mean(f1s, axis=0)


def paired_bootstrap(y: np.ndarray, preds_a: list[np.ndarray], preds_b: list[np.ndarray], metric: str, K: int,
                     B: int = 10000, seed: int = 0, chunk: int = 500) -> dict:
    rng = np.random.RandomState(seed)
    N = len(y)
    obs = np.mean([_metric_batch(y, p, np.arange(N)[None], metric, K)[0] for p in preds_a]) - \
        np.mean([_metric_batch(y, p, np.arange(N)[None], metric, K)[0] for p in preds_b])
    diffs = []
    done = 0
    while done < B:
        b = min(chunk, B - done)
        I = rng.randint(0, N, size=(b, N))
        a = np.mean([_metric_batch(y, p, I, metric, K) for p in preds_a], axis=0)
        c = np.mean([_metric_batch(y, p, I, metric, K) for p in preds_b], axis=0)
        diffs.append(a - c)
        done += b
    diffs = np.concatenate(diffs)
    return {"observed": float(obs), "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5)),
            "p_one_sided": float((diffs <= 0).mean()), "B": B}


def mean_std(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0
