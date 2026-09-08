"""Paired statistical comparisons.

DeLong's test is used for AUROC, since it accounts for the correlation
between two areas computed on the same sample. The area under the
precision-recall curve has no closed-form variance, so paired
differences are assessed by percentile bootstrap. Within each declared
family of comparisons, p values are adjusted by the Benjamini-Hochberg
procedure.

A rank test across seeds was considered and rejected: with five paired
observations the smallest attainable p value is 0.0625, so such a test
cannot reach conventional significance regardless of effect size.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score


def _midrank(x):
    """Ranks with ties resolved to their midpoint, as DeLong requires."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out


def delong_test(y_true, p1, p2):
    """Two-sided DeLong test for two correlated ROC curves.

    Returns the two areas and the p value for their difference.
    """
    y_true = np.asarray(y_true)
    pos = y_true == 1
    X = np.vstack([np.asarray(p1)[pos],  np.asarray(p2)[pos]])
    Y = np.vstack([np.asarray(p1)[~pos], np.asarray(p2)[~pos]])
    m, n = X.shape[1], Y.shape[1]

    tx = np.array([_midrank(X[r]) for r in range(2)])
    ty = np.array([_midrank(Y[r]) for r in range(2)])
    tz = np.array([_midrank(np.concatenate([X[r], Y[r]])) for r in range(2)])

    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    Sc = (np.cov((tz[:, :m] - tx) / n) / m +
          np.cov(1 - (tz[:, m:] - ty) / m) / n)

    L = np.array([[1.0, -1.0]])
    var = float((L @ Sc @ L.T).item())
    if var <= 0:
        return float(auc[0]), float(auc[1]), 1.0
    z = float((L @ auc).item()) / np.sqrt(var)
    return float(auc[0]), float(auc[1]), float(2 * stats.norm.sf(abs(z)))


def bootstrap_auprc(y_true, p1, p2, n_boot=2000, seed=42):
    """Percentile interval for the paired difference in AUPRC.

    Resampling is paired: the same patient indices are drawn for both
    models, so the interval reflects the difference rather than the
    variability of each model separately.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p1, p2 = np.asarray(p1), np.asarray(p2)
    observed = (average_precision_score(y_true, p1) -
                average_precision_score(y_true, p2))

    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if 0 < y_true[i].sum() < len(i):
            diffs.append(average_precision_score(y_true[i], p1[i]) -
                         average_precision_score(y_true[i], p2[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(observed), float(lo), float(hi)


def benjamini_hochberg(p_values):
    """False discovery rate control within a declared test family."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        running = min(running, p[order[rank]] * n / (rank + 1))
        adjusted[order[rank]] = running
    return adjusted
