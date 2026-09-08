"""Patient-level discrimination and calibration metrics.

Kept separate from the training code so that CPU-only scripts, such as
the baseline and table builders, need not import TensorFlow.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)


def compute_metrics(y_true, y_score, is_probability=True):
    """Discrimination always; calibration only for probability outputs.

    Raw clinical scores such as SOFA are not probabilities, so the Brier
    score and calibration terms are undefined for them and returned as
    NaN rather than computed on an inappropriate scale.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    out = {
        'auroc': float(roc_auc_score(y_true, y_score)),
        'auprc': float(average_precision_score(y_true, y_score)),
    }
    if not is_probability:
        out.update(brier=np.nan, cal_slope=np.nan, cal_intercept=np.nan)
        return out

    out['brier'] = float(brier_score_loss(y_true, y_score))

    # Logistic recalibration: a slope of one and an intercept of zero
    # indicate perfect calibration. Slope alone is not sufficient, since
    # a well-sloped model can still be shifted systematically.
    eps = 1e-7
    p = np.clip(y_score, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        lr.fit(logit, y_true)
        out['cal_slope'] = float(lr.coef_[0][0])
        out['cal_intercept'] = float(lr.intercept_[0])
    except Exception:
        out['cal_slope'] = out['cal_intercept'] = np.nan
    return out


def metrics_at_threshold(y_true, y_score, threshold):
    """Threshold-dependent metrics, including the alert counts used in
    the discussion of review burden."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        'threshold': float(threshold),
        'sensitivity': tp / (tp + fn) if tp + fn else np.nan,
        'specificity': tn / (tn + fp) if tn + fp else np.nan,
        'ppv': tp / (tp + fp) if tp + fp else np.nan,
        'npv': tn / (tn + fn) if tn + fn else np.nan,
        'flagged': tp + fp,
        'false_alerts': fp,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }
