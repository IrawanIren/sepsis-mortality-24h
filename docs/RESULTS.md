# Reported results

Every figure below is reproducible from the code in this repository
given credentialed access to MIMIC-IV. Aggregate result files are in
`results/`; per-patient predictions are not redistributed.

All metrics follow one convention: probabilities are averaged across
seeds before any metric is computed, and out-of-fold predictions are
pooled so that each patient contributes one score. Differences between
models can therefore be verified by direct subtraction.

## Cohort

19,902 Sepsis-3 patients, 3,036 deaths within 30 days (15.25%).
Development set 15,921; held-out test set 3,981.

## Held-out test set

Platt-scaled probabilities, five seeds.

| Model | AUROC | AUPRC | Brier | Cal. slope | Cal. intercept |
|---|---|---|---|---|---|
| Proposed (CNN + Bi-GRU + static) | 0.9020 | 0.7030 | 0.0761 | 0.998 | +0.018 |
| CatBoost | 0.8688 | 0.6616 | 0.0826 | 1.006 | +0.030 |
| XGBoost | 0.8654 | 0.6571 | 0.0833 | 0.958 | −0.049 |
| Bi-LSTM | 0.8554 | 0.6301 | 0.0873 | 0.991 | +0.013 |
| Transformer | 0.8508 | 0.6265 | 0.0875 | 0.957 | −0.046 |
| First-day SOFA | 0.5832 | 0.2449 | — | — | — |

Paired comparisons against the proposed model, DeLong with
Benjamini-Hochberg correction: all significant at adjusted p < 0.001.
XGBoost against CatBoost did not differ (adjusted p = 0.110).

## Cross-validation, pooled out-of-fold

| Model | AUROC | AUPRC | Brier | Cal. slope |
|---|---|---|---|---|
| Proposed | 0.9128 | 0.7185 | 0.1102 | 0.998 |
| CatBoost | 0.8856 | 0.6675 | 0.0845 | 0.885 |
| XGBoost | 0.8836 | 0.6639 | 0.0964 | 1.109 |
| Bi-LSTM | 0.8711 | 0.6443 | 0.1332 | 1.040 |
| Transformer | 0.8681 | 0.6392 | 0.1372 | 1.066 |
| First-day SOFA | 0.6389 | 0.2822 | — | — |

The three sequence models show calibration intercepts near −1.5, a
systematic shift from the class weighting used in training. Platt
scaling removes it; the test-set figures above are recalibrated.

## Design factors

Pooled out-of-fold predictions, DeLong with Benjamini-Hochberg. The
reference differs between rows because experiments used either three or
five seeds; each comparison uses the same seeds on both sides.

| Factor | Reference | Reduced | Difference | Adjusted p |
|---|---|---|---|---|
| Observation shortened to 6 h | 0.9115 | 0.8101 | 0.1014 | <0.001 |
| Observation shortened to 12 h | 0.9115 | 0.8419 | 0.0696 | <0.001 |
| Both pathways removed | 0.9128 | 0.8508 | 0.0621 | <0.001 |
| Static pathway removed | 0.9128 | 0.8521 | 0.0608 | <0.001 |
| Encoder replaced by Transformer | 0.9128 | 0.8681 | 0.0447 | <0.001 |
| Encoder replaced by Bi-LSTM | 0.9128 | 0.8711 | 0.0417 | <0.001 |
| Augmentation removed | 0.9115 | 0.8784 | 0.0331 | <0.001 |
| Positional variety removed | 0.9115 | 0.9043 | 0.0072 | <0.001 |
| Convolutional block removed | 0.9128 | 0.9144 | −0.0015 | 0.081 |

The convolutional block is the only factor without a measurable effect
on discrimination.

## Controlled augmentation

Window size, inference window, and architecture identical; only the
training windows differ.

| Condition | Training examples per patient | Positional variety | AUROC |
|---|---|---|---|
| A1 one right-aligned window | 1 | No | 0.8720 (0.0053) |
| A2 same window repeated | 19 | No | 0.9008 (0.0003) |
| A3 sliding windows | 19 | Yes | 0.9094 (0.0005) |

Of the total effect of 0.0374, replication accounts for 0.0288 and
positional variety for 0.0085. Seed-to-seed standard deviation falls
tenfold through replication alone.

## Observation horizon

All features restricted to the information available at each horizon.

| Horizon | Seeds | AUROC | AUPRC |
|---|---|---|---|
| 6 h | 3 | 0.8068 (0.0035) | 0.4821 (0.0112) |
| 12 h | 3 | 0.8393 (0.0005) | 0.5746 (0.0010) |
| 24 h | 5 | 0.9095 (0.0010) | 0.7125 (0.0035) |

Twelve hours recovers 32% of the improvement the full period provides
over six. Retaining full-window static features at shorter horizons
inflates these figures substantially.

## Architecture ablation

Five seeds, timings measured on one device within a single session.

| Configuration | Seq. length at Bi-GRU | AUROC | Seed SD | Time (s) |
|---|---|---|---|---|
| Bi-GRU only | 6 | 0.8466 | 0.0007 | 931 |
| CNN + Bi-GRU | 3 | 0.8493 | 0.0008 | 610 |
| Bi-GRU + static | 6 | 0.9108 | 0.0017 | 1107 |
| CNN + Bi-GRU + static | 3 | 0.9094 | 0.0010 | 804 |

The proposed configuration trains 1.38 times faster than the variant
without the block, despite carrying 33% more parameters (50,865 against
38,257), because pooling halves the sequence length.

## Window position

Models trained once and scored at each of the nineteen inference
positions. AUROC rose smoothly from 0.8970 at hours 0-5 to 0.9094 at
hours 18-23, a range of 0.0124. Adding a normalised hour index as an
input channel changed AUROC by 0.0007, within the seed-to-seed standard
deviation of 0.0014.

## Operating points

Thresholds set on development data, applied unchanged to the test set.

| Model | Target | Realised sens. | Spec. | PPV | Flagged | False alerts |
|---|---|---|---|---|---|---|
| Proposed | 0.80 | 0.771 | 0.853 | 0.485 | 965 | 497 |
| XGBoost | 0.80 | 0.779 | 0.770 | 0.378 | 1,250 | 777 |
| CatBoost | 0.80 | 0.783 | 0.773 | 0.383 | 1,240 | 765 |
| Bi-LSTM | 0.80 | 0.774 | 0.761 | 0.368 | 1,277 | 807 |
| Transformer | 0.80 | 0.746 | 0.763 | 0.361 | 1,254 | 801 |

At comparable case detection the proposed model produced 36% fewer
false alerts than XGBoost.

## Tuning

Fifteen random-search trials per model on the same three folds.

| Model | Spread across trials | Before tuning | After |
|---|---|---|---|
| Proposed | 0.0050 | — | 0.9118 |
| XGBoost | 0.0243 | 0.8690 | 0.8829 |
| CatBoost | 0.0307 | 0.8694 | 0.8818 |

Boosting is five to six times more sensitive to tuning than the
proposed model. Equalising the budget raised both baselines by 0.0145
and narrowed the margin accordingly.
