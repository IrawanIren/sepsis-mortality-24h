"""Inspect the structure of the 24-hour sepsis cohort file.

Detects the table layout (long vs. wide), identifies key columns, and
reports basic cohort characteristics. Run this before anything else so
that column names used by downstream scripts are known to be correct.

Usage:
    python src/inspect_data.py --data data/sepsis_merged_pure_24h.csv
"""

import argparse
import sys

import pandas as pd

ID_CANDIDATES = ["stay_id", "icustay_id", "hadm_id", "subject_id"]
TIME_CANDIDATES = ["hour", "hr", "hour_idx", "time_idx", "charttime", "timestep"]
LABEL_KEYWORDS = ["label", "mortal", "death", "died", "outcome", "target"]


def detect_column(df, candidates, kind):
    """Return the first matching column name, or None if none is present."""
    found = [c for c in candidates if c in df.columns]
    if not found:
        print(f"  [!] No {kind} column found. Searched: {candidates}")
        return None
    if len(found) > 1:
        print(f"  [i] Multiple {kind} candidates {found} -> using '{found[0]}'")
    return found[0]


def main(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"File not found: {path}")

    print("=" * 64)
    print("1. DIMENSIONS")
    print("=" * 64)
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]:,}")
    print(f"  Memory  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print("\n" + "=" * 64)
    print("2. KEY COLUMNS")
    print("=" * 64)
    id_col = detect_column(df, ID_CANDIDATES, "patient identifier")
    time_col = detect_column(df, TIME_CANDIDATES, "time index")
    label_cols = [c for c in df.columns
                  if any(k in c.lower() for k in LABEL_KEYWORDS)]
    print(f"  Patient ID : {id_col}")
    print(f"  Time index : {time_col}")
    print(f"  Label(s)   : {label_cols if label_cols else 'NONE DETECTED'}")

    if id_col is None:
        sys.exit("\nCannot continue without a patient identifier column.")

    print("\n" + "=" * 64)
    print("3. TABLE LAYOUT")
    print("=" * 64)
    n_patients = df[id_col].nunique()
    rows_per_patient = df.groupby(id_col).size()
    print(f"  Unique patients        : {n_patients:,}")
    print(f"  Rows per patient (med) : {rows_per_patient.median():.0f}")
    print(f"  Rows per patient (rng) : "
          f"{rows_per_patient.min()}-{rows_per_patient.max()}")

    median_rows = rows_per_patient.median()
    if median_rows == 1:
        layout = "WIDE - one row per patient (features already aggregated)"
    elif median_rows == 24:
        layout = "LONG - 24 rows per patient (hourly time series)"
    else:
        layout = f"LONG, irregular - median {median_rows:.0f} rows per patient"
    print(f"  Detected layout        : {layout}")

    # Uneven row counts break fixed-shape sequence construction downstream.
    if (rows_per_patient != median_rows).any():
        n_irregular = int((rows_per_patient != median_rows).sum())
        print(f"  [!] {n_irregular} patients deviate from the modal row count")

    if time_col and pd.api.types.is_numeric_dtype(df[time_col]):
        values = sorted(df[time_col].dropna().unique())
        print(f"  {time_col} range          : {values[0]} .. {values[-1]} "
              f"({len(values)} distinct values)")

    print("\n" + "=" * 64)
    print("4. OUTCOME")
    print("=" * 64)
    if label_cols:
        for col in label_cols:
            per_patient = df.groupby(id_col)[col].first()
            n_pos = int(per_patient.sum())
            print(f"  {col}: {n_pos:,} positive of {len(per_patient):,} "
                  f"({n_pos / len(per_patient) * 100:.2f}%)")
            # A label must be constant within a patient; if not, the
            # outcome was likely merged at the wrong granularity.
            if df.groupby(id_col)[col].nunique().max() > 1:
                print(f"  [!] '{col}' varies within patients - please check")
    else:
        print("  No outcome column detected.")

    print("\n" + "=" * 64)
    print("5. MISSINGNESS (top 10 columns)")
    print("=" * 64)
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing[missing > 0].head(10)
    if len(top_missing) == 0:
        print("  No missing values present.")
        print("  [i] The file appears fully imputed. Observed-vs-imputed")
        print("      indicators are therefore unavailable, and any masking")
        print("      based representation cannot be evaluated from this file.")
    else:
        for col, pct in top_missing.items():
            print(f"  {col:<32} {pct:6.2f}%")

    print("\n" + "=" * 64)
    print("6. ALL COLUMNS")
    print("=" * 64)
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col:<34} {str(df[col].dtype):<10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the CSV file")
    main(parser.parse_args().data)
