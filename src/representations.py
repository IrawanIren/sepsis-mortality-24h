"""Assemble the five input representations compared in this study.

Each representation is built from the same raw hourly grid and differs
only in which channels are supplied to the model. The architecture,
window configuration, and data partition are held fixed throughout, so
that any difference in performance is attributable to the representation
alone.

    R1  absolute values only
    R2  absolute values and carry-forward differences   (Paper B)
    R3  absolute values and observation mask
    R4  absolute values, mask, and time since last observation
    R5  R4 plus differences taken across observed values only

R2 reproduces the configuration used previously and serves as the
reference point: if it does not recover the earlier figure, the pipeline
is at fault rather than the representation.
"""

import numpy as np
import pandas as pd

N_HOURS = 24

REPRESENTATIONS = {
    'R1': ('absolute',),
    'R2': ('absolute', 'delta_ffill'),
    'R3': ('absolute', 'mask'),
    'R4': ('absolute', 'mask', 'delta_t'),
    'R5': ('absolute', 'mask', 'delta_t', 'delta_observed'),
}


def impute_forward_backward(hourly, features, id_col='stay_id',
                            n_hours=N_HOURS):
    """Carry-forward then carry-backward within each patient.

    This is the imputation used in the earlier study, reproduced here so
    that the absolute channels are identical across representations and
    the comparison isolates the added channels.
    """
    filled = (hourly[features]
              .groupby(level=id_col, group_keys=False)
              .apply(lambda g: g.ffill().bfill()))

    # Variables never measured in a patient remain missing; the cohort
    # median is used, computed once over all patients.
    medians = filled.median()
    filled = filled.fillna(medians)

    n_pat = len(hourly.index.get_level_values(id_col).unique())
    return filled.to_numpy(dtype=np.float32).reshape(
        n_pat, n_hours, len(features))


def delta_from_filled(absolute):
    """Hour-to-hour difference of the imputed series.

    On imputed hours this is identically zero, which is the property the
    present study sets out to examine.
    """
    d = np.zeros_like(absolute)
    d[:, 1:, :] = absolute[:, 1:, :] - absolute[:, :-1, :]
    return d


def build_representation(name, absolute, mask, delta_t, delta_observed,
                         urine=None):
    """Concatenate the channels a representation calls for.

    Parameters
    ----------
    absolute : (n_patients, n_hours, n_masked)
    mask, delta_t, delta_observed : same shape as `absolute`
    urine : (n_patients, n_hours, 1) or None
        Appended to every representation as an absolute channel, since
        an unrecorded hour indicates no output rather than no
        measurement and so carries no meaningful mask.
    """
    if name not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {name!r}")

    available = {
        'absolute': absolute,
        'delta_ffill': delta_from_filled(absolute),
        'mask': mask,
        'delta_t': delta_t,
        'delta_observed': delta_observed,
    }
    parts = [available[c] for c in REPRESENTATIONS[name]]
    if urine is not None:
        parts.append(urine)
    return np.concatenate(parts, axis=2).astype(np.float32)


def channel_counts(n_masked, n_unmasked=1):
    """Channel count per representation, for the methods table."""
    rows = []
    for name, channels in REPRESENTATIONS.items():
        n = len(channels) * n_masked + n_unmasked
        rows.append({
            'Representation': name,
            'Channels': ', '.join(c.replace('_', ' ') for c in channels),
            'Dimensions per hour': n,
        })
    return pd.DataFrame(rows)
