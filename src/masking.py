"""Build observation masks and time-since-last-observation tensors.

Paper B imputed missing hours by carry-forward before computing hourly
differences, which leaves the difference channels encoding the
measurement schedule alongside the physiological rate of change. This
module preserves the schedule explicitly instead: it records which hours
carried an observation, and how long ago the last one was, before any
imputation is applied.

The output is a set of aligned tensors sharing the patient ordering of
the Paper B arrays, so that representations can be assembled from them
without re-deriving the cohort.
"""

import numpy as np
import pandas as pd

N_HOURS = 24

# Laboratory extracts encode missing values as zero. Zero is
# physiologically impossible for every variable used here, so it can be
# restored to missing without ambiguity. Variables that do admit a valid
# zero, such as band cells and eosinophil counts, are not in this set.
ZERO_MEANS_MISSING = [
    'platelets', 'wbc', 'aniongap', 'bicarbonate', 'bun', 'calcium',
    'chloride', 'creatinine', 'glucose', 'sodium', 'potassium',
    'inr', 'pt', 'ptt', 'albumin', 'fibrinogen', 'thrombin',
    'hematocrit', 'hemoglobin',
]


def to_hourly(df, intimes, value_cols, id_col='subject_id',
              time_col='charttime', n_hours=N_HOURS):
    """Resample irregular records onto an hourly grid.

    Records are placed in the hour they fall into, counted from ICU
    admission. Where an hour holds several measurements of the same
    variable, the mean is taken. Hours with no measurement remain NaN,
    which is what the mask is built from.

    Parameters
    ----------
    intimes : Series indexed by patient identifier, giving ICU admission.

    Returns
    -------
    DataFrame indexed by (patient, hour) covering every hour 0..n_hours-1
    for every patient in `intimes`, with NaN where nothing was recorded.
    """
    d = df[[id_col, time_col] + value_cols].copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d[d[id_col].isin(intimes.index)]

    offset = d[id_col].map(intimes)
    d['hour'] = ((d[time_col] - offset).dt.total_seconds() // 3600).astype('Int64')
    d = d[d.hour.between(0, n_hours - 1)]

    hourly = d.groupby([id_col, 'hour'])[value_cols].mean()

    # Reindex onto the complete grid so that unobserved hours appear as
    # rows of NaN rather than being absent.
    full = pd.MultiIndex.from_product(
        [intimes.index, range(n_hours)], names=[id_col, 'hour'])
    return hourly.reindex(full)


def build_mask_tensors(hourly, features, id_col='subject_id',
                       n_hours=N_HOURS):
    """Derive the observation mask and time since last observation.

    Returns
    -------
    mask : (n_patients, n_hours, n_features) float32
        One where the hour carried an observation, zero otherwise.
    delta_t : (n_patients, n_hours, n_features) float32
        Hours elapsed since the most recent observation of that
        variable. Hours preceding the first observation are assigned the
        full window length, since the true interval is unknown and
        capping keeps the channel bounded.
    pids : (n_patients,) patient identifiers in tensor order
    """
    obs = hourly[features].notna()
    pids = hourly.index.get_level_values(id_col).unique().to_numpy()
    n_pat, n_feat = len(pids), len(features)

    mask = obs.to_numpy(dtype=np.float32).reshape(n_pat, n_hours, n_feat)

    # Elapsed time is accumulated forward: it resets to zero at an
    # observation and increments by one otherwise.
    delta_t = np.empty_like(mask)
    for t in range(n_hours):
        if t == 0:
            # No history at hour zero; unobserved variables start at the
            # window length rather than at zero.
            delta_t[:, 0, :] = np.where(mask[:, 0, :] == 1, 0.0, n_hours)
        else:
            prev = delta_t[:, t - 1, :]
            delta_t[:, t, :] = np.where(
                mask[:, t, :] == 1, 0.0,
                np.minimum(prev + 1.0, n_hours))

    return mask, delta_t.astype(np.float32), pids


def observed_only_delta(hourly, features, id_col='subject_id',
                        n_hours=N_HOURS):
    """Hourly difference computed across observed values only.

    Carry-forward imputation makes the naive difference zero at every
    imputed hour. Here the difference is taken between consecutive
    *observations* and held constant until the next one, so that it
    reflects change in the variable rather than the timing of the
    measurement. Hours before the first observation are zero.
    """
    pids = hourly.index.get_level_values(id_col).unique().to_numpy()
    n_pat, n_feat = len(pids), len(features)
    vals = hourly[features].to_numpy(dtype=np.float64).reshape(
        n_pat, n_hours, n_feat)

    out = np.zeros((n_pat, n_hours, n_feat), dtype=np.float32)
    last = np.full((n_pat, n_feat), np.nan)
    carried = np.zeros((n_pat, n_feat), dtype=np.float32)

    for t in range(n_hours):
        cur = vals[:, t, :]
        seen = ~np.isnan(cur)
        both = seen & ~np.isnan(last)

        # A new observation updates the carried difference; otherwise the
        # previous difference is retained.
        carried = np.where(both, (cur - last).astype(np.float32), carried)
        out[:, t, :] = carried
        last = np.where(seen, cur, last)

    return out


def summarise(mask, features, hourly=None):
    """Report observation frequency per feature, for the manuscript."""
    rows = []
    for j, f in enumerate(features):
        m = mask[:, :, j]
        rows.append({
            'feature': f,
            'hours observed (%)': round(m.mean() * 100, 1),
            'patients with >=1 observation (%)':
                round((m.sum(axis=1) > 0).mean() * 100, 1),
            'patients with >=2 observations (%)':
                round((m.sum(axis=1) > 1).mean() * 100, 1),
            'median observations per patient': float(np.median(m.sum(axis=1))),
        })
    return (pd.DataFrame(rows)
              .sort_values('hours observed (%)', ascending=False)
              .reset_index(drop=True))
