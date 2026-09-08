# Changes made during revision

This note records what changed between the first submission and the
revised manuscript, so that readers comparing the two are not left to
infer the reasons.

## Rebuilt feature extraction

Several static features in the original extract could not be reproduced
from any documented source. Minimum Glasgow Coma Scale contained values
of 2, which the scale does not admit since its floor is 3, and neither
that variable nor first-day SOFA matched the standard MIMIC-IV derived
tables.

The entire feature extraction was therefore rebuilt from the queries in
`sql/`, and every experiment repeated. The cohort of 19,902 patients,
the frozen partition, the architecture, and the evaluation protocol are
unchanged.

| Variable | Original | Rebuilt | Reference tables |
|---|---|---|---|
| Minimum GCS, mean | 11.26 | 13.33 | 13.28 and 13.32 |
| First-day SOFA, mean | 5.91 | 5.52 | 5.53 over hours 0-23 |
| Vasopressor use | 34.2% | 32.0% | — |
| Invasive ventilation | 63.4% | 63.0% | — |

Discrimination was almost unaffected: hyperparameter search reached
0.9118 on the rebuilt data against 0.9115 originally, and selected
identical hyperparameters.

## Single aggregation convention

The first submission averaged AUROC across seeds in some tables while
computing AUROC from averaged probabilities in others, and in one case
compared against a differently constructed baseline. Both are now
avoided: probabilities are averaged across seeds first, and out-of-fold
predictions are pooled, so every figure derives from one prediction
vector per model.

## Horizon features restricted

Static severity and treatment features previously drew on the full 24
hours even when the dynamic input was truncated. They are now recomputed
from hours 0 to H-1.

| Horizon | Original | Corrected |
|---|---|---|
| 6 hours | 0.8612 | 0.8068 |
| 12 hours | 0.8987 | 0.8393 |

The claim that twelve hours recovers 88% of the twenty-four-hour benefit
is withdrawn; the corrected figure is 32%.

## Withdrawn claims

- **Training time.** The convolutional block was reported as 7.8 times
  faster. Measured on one device within a single session the factor is
  1.38; the original figures were collected across different hardware.
- **Seed stability.** Reported as a ninefold reduction in variance. The
  quantity was a standard deviation, and the corrected factor is 1.7.
- **Alert burden.** Reported as a 52% increase; the correct statement is
  a 36% reduction relative to the boosting baseline.
- **Early prediction.** No longer claimed to be clinically viable at six
  or twelve hours.
- **Matched sensitivity.** Reported at 0.80; that is the target set on
  development data, and the realised test-set value is 0.771.

## Added experiments

- Controlled augmentation, separating replication from positional
  variety
- Evaluation at every inference window position
- Position-aware input channel
- Bi-LSTM and Transformer comparators under matched conditions
- Equal tuning budget for XGBoost and CatBoost, which raised both by
  0.0145 and narrowed the reported margin

Final figures are in `docs/RESULTS.md`.
