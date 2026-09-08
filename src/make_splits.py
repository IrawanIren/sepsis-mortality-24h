"""Create the frozen data partition shared by every model.

    Full cohort
    |-- Test set 20%       frozen; touched only for the final evaluation
    `-- Development 80%    5-fold CV for tuning and cross-validated reporting

Run once. All subsequent experiments read the resulting files, so that
every model is compared on identical partitions and paired statistical
tests are valid.

Usage:
    python src/make_splits.py --data data/sepsis_merged_pure_24h.csv \
        --id-col stay_id --label-col label --outdir splits/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Collapse to one row per patient. The outcome is constant within a
    # patient, so taking the first value is safe.
    df = pd.read_csv(args.data, usecols=[args.id_col, args.label_col])
    patients = (df.groupby(args.id_col)[args.label_col]
                  .first()
                  .reset_index()
                  .sort_values(args.id_col)
                  .reset_index(drop=True))

    ids = patients[args.id_col].values
    y = patients[args.label_col].values
    print(f"Cohort: {len(ids):,} patients | "
          f"{int(y.sum()):,} events ({y.mean() * 100:.2f}%)")

    # Held-out test set, stratified on the outcome.
    dev_ids, test_ids, y_dev, y_test = train_test_split(
        ids, y, test_size=0.2, stratify=y, random_state=SEED
    )
    pd.DataFrame({args.id_col: np.sort(test_ids)}).to_csv(
        os.path.join(args.outdir, "test_set_ids.csv"), index=False
    )
    print(f"Test set: {len(test_ids):,} patients "
          f"({y_test.mean() * 100:.2f}% events)")

    # Cross-validation folds within the development set.
    # Splitting at the patient level means no patient can span two folds,
    # so StratifiedKFold is sufficient here. Sliding windows generated
    # later inherit the fold assignment of the patient they come from,
    # which prevents overlapping windows from leaking across folds.
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                          random_state=SEED)
    assignments = []
    for fold, (_, val_idx) in enumerate(skf.split(dev_ids, y_dev), start=1):
        for i in val_idx:
            assignments.append({args.id_col: dev_ids[i], "fold": fold})

    folds = pd.DataFrame(assignments).sort_values(args.id_col)
    folds.to_csv(os.path.join(args.outdir, "cv_folds.csv"), index=False)

    print(f"Development set: {len(dev_ids):,} patients "
          f"across {args.n_splits} folds")
    summary = folds.merge(patients, on=args.id_col)
    for fold, group in summary.groupby("fold"):
        print(f"  fold {fold}: n={len(group):,}  "
              f"events={group[args.label_col].mean() * 100:.2f}%")

    # Fail loudly rather than silently producing a leaking partition.
    overlap = set(test_ids) & set(dev_ids)
    assert not overlap, f"Leakage: {len(overlap)} patients in both sets"
    assert len(test_ids) + len(dev_ids) == len(ids), "Patient count mismatch"
    print("\nSanity check passed: test and development sets are disjoint.")

    metadata = {
        "seed": SEED,
        "n_patients": int(len(ids)),
        "n_events": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "test_size": 0.2,
        "n_splits": args.n_splits,
        "id_col": args.id_col,
        "label_col": args.label_col,
        "source_file": os.path.basename(args.data),
    }
    with open(os.path.join(args.outdir, "split_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Written to {args.outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--id-col", default="stay_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--outdir", default="splits")
    main(parser.parse_args())
