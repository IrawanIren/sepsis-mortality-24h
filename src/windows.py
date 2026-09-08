"""Sequence construction and fold-wise preprocessing.

Implements the locked experimental protocol:

  * Windows start at 0, S, 2S, ... while start + W <= n_hours. If the
    final window does not end at the last hour, a right-aligned window is
    appended at start = n_hours - W.
  * That right-aligned window is the only one used at inference, so every
    configuration predicts from the same clinical moment and produces
    exactly one score per patient.
  * Sliding windows therefore act purely as training-time augmentation.
  * Winsorisation and standardisation are fitted on training patients
    only, so no distributional information crosses into validation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

N_HOURS = 24


# ----------------------------------------------------------------------
# Window geometry
# ----------------------------------------------------------------------

def build_windows(window_size, step, n_hours=N_HOURS):
    """Return (start_indices, right_aligned_start)."""
    if window_size > n_hours:
        raise ValueError(f"window_size {window_size} exceeds {n_hours} hours")

    starts = list(range(0, n_hours - window_size + 1, step))
    right_aligned = n_hours - window_size
    if starts[-1] != right_aligned:
        starts.append(right_aligned)
    return starts, right_aligned


# ----------------------------------------------------------------------
# Array construction (built once per horizon, then cached)
# ----------------------------------------------------------------------

def build_arrays(df, cfg, horizon=N_HOURS):
    """Convert the long dataframe into dense arrays.

    Parameters
    ----------
    horizon : int
        Hours of observation retained. Hours >= horizon are discarded,
        which is how the early-prediction axis is implemented.

    Returns
    -------
    dyn  : (n_patients, horizon, 2 * n_dynamic)   absolute then delta
    stat : (n_patients, n_static)                 numeric then one-hot
    y    : (n_patients,)
    pids : (n_patients,)
    """
    id_col, hour_col = cfg['id_col'], cfg['hour_col']
    dyn_cols = cfg['dynamic_features']

    df = df.sort_values([id_col, hour_col]).copy()

    # Deltas are computed on the full record before truncation, so the
    # first retained hour still carries a meaningful change value except
    # at hour 0, where no predecessor exists.
    deltas = df.groupby(id_col)[dyn_cols].diff().fillna(0.0)
    deltas.columns = [f'{c}_delta' for c in dyn_cols]
    df = pd.concat([df, deltas], axis=1)

    if horizon < N_HOURS:
        df = df[df[hour_col] < horizon]

    feat_cols = dyn_cols + list(deltas.columns)
    n_pat = df[id_col].nunique()
    dyn = (df[feat_cols].to_numpy(dtype=np.float32)
             .reshape(n_pat, horizon, len(feat_cols)))

    # Static features are constant within a patient; take the first row.
    first = df.groupby(id_col, sort=True).first().reset_index()
    num = first[cfg['static_numeric']].to_numpy(dtype=np.float32)
    enc = OneHotEncoder(sparse_output=False, dtype=np.float32,
                        handle_unknown='ignore')
    cat = enc.fit_transform(first[cfg['static_categorical']])
    stat = np.hstack([num, cat])

    y = first[cfg['label_col']].to_numpy(dtype=np.float32)
    pids = first[id_col].to_numpy()

    return dyn, stat, y, pids


# ----------------------------------------------------------------------
# Fold-wise preprocessing
# ----------------------------------------------------------------------

def fit_transform_fold(dyn, stat, train_mask, percentiles=(1, 99)):
    """Winsorise and standardise, fitting on training patients only.

    Hard physiological clipping was already applied cohort-wide; it
    encodes domain knowledge rather than distributional information.
    Percentile limits are estimated from data and so must never see the
    validation or test partitions.
    """
    n_feat = dyn.shape[2]
    train_flat = dyn[train_mask].reshape(-1, n_feat)

    lo = np.percentile(train_flat, percentiles[0], axis=0)
    hi = np.percentile(train_flat, percentiles[1], axis=0)
    dyn_w = np.clip(dyn, lo, hi)

    flat = dyn_w[train_mask].reshape(-1, n_feat)
    mu, sd = flat.mean(axis=0), flat.std(axis=0)
    sd[sd == 0] = 1.0
    dyn_s = ((dyn_w - mu) / sd).astype(np.float32)

    stat_s = StandardScaler().fit(stat[train_mask]).transform(stat)

    return dyn_s, stat_s.astype(np.float32)


# ----------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------

def make_windows(dyn, stat, y, idx, window_size, step,
                 training, n_hours=N_HOURS):
    """Expand patients into windows.

    training=True  -> every window (augmentation)
    training=False -> the right-aligned window only (one score per patient)
    """
    starts, right_aligned = build_windows(window_size, step, n_hours)
    if not training:
        starts = [right_aligned]

    Xd = np.concatenate([dyn[idx, s:s + window_size, :] for s in starts])
    Xs = np.concatenate([stat[idx] for _ in starts])
    yy = np.concatenate([y[idx] for _ in starts])
    pid = np.concatenate([idx for _ in starts])

    if not training:
        # One prediction per patient is the point of the protocol; fail
        # loudly if that invariant is ever broken.
        assert len(pid) == len(np.unique(pid)), \
            "More than one inference window per patient"

    return Xd, Xs, yy, pid
