# Sepsis Mortality Prediction from the First 24 Hours of ICU Care

Analysis code for a study of 30-day mortality prediction in sepsis using
the first 24 hours of intensive care data from MIMIC-IV.

The work examines which design decisions carry measurable weight:
observation window length, input representation, sliding-window
augmentation, and the choice of sequence encoder. It is not a proposal
for a novel architecture.

> **Status:** manuscript under review at the International Journal of
> Intelligent Engineering and Systems. A citation and DOI will be added
> on publication.

## Headline result

On a held-out test set of 3,981 patients, the proposed model reached an
AUROC of 0.902, against 0.865 for XGBoost, 0.869 for CatBoost, 0.855 for
a bidirectional LSTM, and 0.851 for a Transformer encoder, all trained
under an identical cohort, input representation, data partition, and
tuning budget.

Among the design factors examined, observation length carried the
largest effect (0.101 in AUROC), followed by the static feature pathway
(0.061) and the choice of sequence encoder (0.042 to 0.045). The
convolutional block contributed nothing measurable.

All reported figures are listed in [docs/RESULTS.md](docs/RESULTS.md).

---

## Data availability

This study uses **MIMIC-IV v2.1**, governed by the
[PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/view-dua/2.1/).

**No patient-level data is redistributed here.** Aggregate results are
provided in `results/` and `tables/`, and the figures in `figures/`, so
that every number reported in the manuscript can be traced to its source
without database access. Per-patient predictions, cached feature arrays,
and the cohort table itself are excluded, as these are derived patient
data.

To reproduce the analysis from scratch:

1. Complete the CITI *Data or Specimens Only Research* training
2. Obtain credentialed access to MIMIC-IV through PhysioNet
3. Build the derived concept tables from
   [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code)
4. Run the queries in `sql/` against your own database instance
5. Follow the pipeline below

The data partition is not distributed but is fully reproducible:
`src/make_splits.py` regenerates it deterministically from seed 42.

---

## Repository layout

```
.
├── sql/                     Extraction queries (MySQL)
│   ├── cohort_sepsis3.sql       Cohort definition and outcome
│   ├── hourly_vitalsign.sql     Vital signs, NULLs preserved
│   ├── hourly_labs.sql          Laboratory results, NULLs preserved
│   ├── hourly_sofa.sql          SOFA, GCS, and vasopressor rates by hour
│   ├── ventilation.sql          Ventilation episodes
│   ├── urine_output.sql         Recorded urine output
│   ├── lactate.sql              Lactate with specimen type
│   └── first_day_sofa.sql       First-day SOFA and components
├── src/
│   ├── inspect_data.py          Dataset structure and integrity checks
│   ├── make_splits.py           Frozen data partition (run once)
│   ├── masking.py               Hourly grid, observation masks, elapsed time
│   ├── representations.py       Input representations R1 to R5
│   ├── windows.py               Window construction, fold-wise preprocessing
│   ├── train.py                 Proposed architecture and fold training
│   ├── architectures.py         Bi-LSTM and Transformer comparators
│   ├── metrics.py               Discrimination and calibration
│   ├── stats.py                 DeLong, bootstrap, Benjamini-Hochberg
│   ├── runner.py                Experiment queue with automatic resume
│   ├── revision_tables.py       Manuscript tables
│   ├── revision_figures.py      Result figures
│   ├── fig_flow.py              Research workflow diagram
│   ├── fig_cohort.py            Patient selection flow
│   └── fig_arch.py              Model architecture diagram
├── docs/
│   ├── RESULTS.md               All reported figures, for verification
│   └── REVISION.md              What changed during revision, and why
├── results/                 Experiment summaries (aggregate only)
├── tables/                  Manuscript tables
├── figures/                 Manuscript figures
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/IrawanIren/sepsis-mortality-24h.git
cd sepsis-mortality-24h
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Pipeline

```bash
# 1. Inspect the extracted cohort table
python src/inspect_data.py --data data/sepsis_rebuilt_24h.parquet

# 2. Create the frozen data partition (run once)
python src/make_splits.py \
    --data data/sepsis_rebuilt_24h.parquet \
    --id-col subject_id --label-col label \
    --outdir splits/
```

Experiments are driven through `src/runner.py`, which appends each
completed run to a CSV and skips runs already recorded, so an
interrupted session resumes by rerunning the same command.

Tables and figures are generated afterwards from the saved predictions:

```python
import revision_tables, revision_figures

revision_tables.build_all(base=BASE, res_dir=RES, outdir='tables/',
                          folds=folds)
revision_figures.build_all(base=BASE, res_dir=RES, pred_dir=PRED,
                           outdir='figures/')
```

---

## Experimental protocol

Fixed before any model was trained, and applied to every model.

| Item | Specification |
|---|---|
| Time anchor | ICU admission; observation window hours 0-23 |
| Evaluation unit | Patient (one risk score per patient) |
| Aggregation | Final right-aligned window |
| Partition | 20% held-out test set; 5-fold CV on the remaining 80% |
| Folds | Identical across all models, generated once from seed 42 |
| Primary metric | AUROC |
| Co-primary metric | AUPRC |
| Statistical testing | DeLong for AUROC; 2,000-resample paired bootstrap for AUPRC; Benjamini-Hochberg within each declared family |
| Repetitions | 5 random seeds |
| Tuning budget | 15 random-search trials on 3 development folds, applied equally to every fitted model |

### Aggregation convention

Where several seeds were trained, **predicted probabilities are averaged
across seeds before any metric is computed**, and out-of-fold
predictions are pooled so that each development patient contributes
exactly one score. All reported metrics and all paired tests therefore
derive from the same prediction vector, and differences between models
in the tables can be verified by direct subtraction.

Prediction files are named to make their provenance unambiguous:

```
E1_W6_S1_s{seed}.npz              proposed model, out-of-fold
E2_{variant}_W6_S1_s{seed}.npz    ablation variants, out-of-fold
E3_h{H}_W6_S1_s{seed}.npz         horizon experiments
AUG_{A1,A2}_s{seed}.npz           controlled augmentation
{model}_f{fold}_s{seed}.npz       boosting baselines, out-of-fold
E5_{model}_{calibration}_s{seed}.npz   held-out test set
```

### Window construction

Windows start at indices 0, S, 2S, ... while `start + W <= 24`. If the
last such window does not end exactly at hour 23, one additional
right-aligned window is appended at `start = 24 - W`. That right-aligned
window is the only one used at inference, so sliding windows act purely
as training-time augmentation and evaluation remains at patient level.

### Horizon restriction

For the 6-hour and 12-hour experiments, every static feature that
derives from a time range is recomputed from hours 0 to H-1: first-day
SOFA and its six components, minimum GCS, vasopressor use, invasive
ventilation, and the three reassigned laboratory variables. SOFA at
sepsis onset enters only when onset falls within the horizon. Age, sex,
race, and comorbidity index are fixed at admission and are unchanged.

### Leakage control

Percentile winsorisation and standardisation are fitted on the training
partition of each fold only. Physiological bound clipping encodes
clinical knowledge rather than distributional information and is applied
cohort-wide. Imputation is carry-forward then carry-backward within each
patient; since patients are assigned to partitions as whole units, this
is independent of the fold structure.

The constancy criterion used to move three laboratory variables to the
static pathway depends only on whether a value changes and makes no
reference to the outcome. Recomputed within each training fold
separately, the same three variables exceed the threshold in all five
folds and no other variable approaches it.

---

## Citation

```bibtex
@article{TBD,
  title   = {Observation Window Design Outweighs Architectural Complexity
             in Deep Learning for Sepsis Mortality Prediction},
  author  = {TBD},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {TBD}
}
```

## License

Code is released under the MIT License. MIMIC-IV data is governed by
PhysioNet terms and is not covered by this license.
