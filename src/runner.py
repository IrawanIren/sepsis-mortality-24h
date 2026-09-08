"""Experiment queue with automatic resume.

Free-tier Colab sessions are interrupted regularly, so every completed
run is appended to a JSONL file immediately. On restart the queue skips
runs whose identifier is already present, and work continues where it
stopped rather than from the beginning.
"""

import json
import os
import time

import numpy as np
import pandas as pd

from train import DEFAULT_HP, train_fold
from windows import build_arrays

# ----------------------------------------------------------------------
# Experiment definitions
# ----------------------------------------------------------------------

# E1: window configuration grid. W=24 admits a single window, so the
# stride is irrelevant there.
GRID_WS = ([(6, s) for s in (1, 3, 6)] +
           [(12, s) for s in (1, 2, 3, 4, 6)] +
           [(18, s) for s in (1, 3, 6)] +
           [(24, 1)])

# E2: architecture ablation, evaluated at the winning (W, S).
VARIANTS = ['full', 'no_mlp', 'no_cnn', 'gru_only']

# E3: early-prediction horizons. Observation is truncated to the first
# H hours, so the window cannot be longer than the horizon.
HORIZONS = [6, 12, 24]


def run_id(**kw):
    """Stable identifier used to detect already-completed runs."""
    return "|".join(f"{k}={kw[k]}" for k in sorted(kw))


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)['run_id'] for line in f if line.strip()}


def append_result(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record) + "\n")


# ----------------------------------------------------------------------
# Queue
# ----------------------------------------------------------------------

def run_queue(df, cfg, folds, results_path, jobs, cache_dir,
              epochs=40, batch_size=512):
    """Execute pending jobs, skipping those already recorded.

    jobs : list of dicts with keys
           experiment, window_size, step, variant, seed, horizon
    """
    done = load_done(results_path)
    pending = [j for j in jobs if run_id(**j) not in done]
    print(f"{len(done)} runs already complete | {len(pending)} pending")

    id_col = cfg['id_col']
    arrays = {}   # horizon -> (dyn, stat, y, pids)

    for n, job in enumerate(pending, 1):
        rid = run_id(**job)
        horizon = job['horizon']

        # Arrays depend only on the horizon, so build each one once and
        # reuse it across every window configuration and seed.
        if horizon not in arrays:
            cache = os.path.join(cache_dir, f"arrays_h{horizon}.npz")
            if os.path.exists(cache):
                z = np.load(cache, allow_pickle=True)
                arrays[horizon] = (z['dyn'], z['stat'], z['y'], z['pids'])
            else:
                built = build_arrays(df, cfg, horizon=horizon)
                np.savez_compressed(cache, dyn=built[0], stat=built[1],
                                    y=built[2], pids=built[3])
                arrays[horizon] = built
            print(f"  arrays for horizon {horizon}: {arrays[horizon][0].shape}")

        dyn, stat, y, pids = arrays[horizon]
        pid_to_row = {p: i for i, p in enumerate(pids)}

        t0 = time.time()
        fold_metrics = []
        for k in sorted(folds['fold'].unique()):
            val_pids = folds.loc[folds.fold == k, id_col].values
            val_idx = np.array([pid_to_row[p] for p in val_pids
                                if p in pid_to_row])
            dev_pids = folds[id_col].values
            train_idx = np.array([pid_to_row[p] for p in dev_pids
                                  if p in pid_to_row
                                  and p not in set(val_pids)])

            m, _, _ = train_fold(
                dyn, stat, y, train_idx, val_idx,
                window_size=job['window_size'], step=job['step'],
                hp=DEFAULT_HP, variant=job['variant'], seed=job['seed'],
                epochs=epochs, batch_size=batch_size, n_hours=horizon)
            fold_metrics.append(m)

        record = {'run_id': rid, **job,
                  'elapsed_s': round(time.time() - t0, 1)}
        for key in fold_metrics[0]:
            vals = [m[key] for m in fold_metrics]
            record[f'{key}_mean'] = float(np.mean(vals))
            record[f'{key}_std'] = float(np.std(vals))
        record['auroc_folds'] = [m['auroc'] for m in fold_metrics]

        append_result(results_path, record)
        print(f"[{n}/{len(pending)}] {rid} -> "
              f"AUROC {record['auroc_mean']:.4f} ± {record['auroc_std']:.4f} "
              f"({record['elapsed_s']:.0f}s)")

    return pd.read_json(results_path, lines=True)


# ----------------------------------------------------------------------
# Job builders
# ----------------------------------------------------------------------

def jobs_e1_screen():
    """Window grid, one seed per configuration."""
    return [{'experiment': 'E1', 'window_size': w, 'step': s,
             'variant': 'full', 'seed': 42, 'horizon': 24}
            for w, s in GRID_WS]


def jobs_e1_confirm(best_w, best_s, seeds=(1, 2, 3, 4)):
    """Additional seeds for the winning configuration."""
    return [{'experiment': 'E1b', 'window_size': best_w, 'step': best_s,
             'variant': 'full', 'seed': sd, 'horizon': 24} for sd in seeds]


def jobs_e2(best_w, best_s, seeds=(42, 1, 2)):
    """Architecture ablation at the winning configuration."""
    return [{'experiment': 'E2', 'window_size': best_w, 'step': best_s,
             'variant': v, 'seed': sd}
            | {'horizon': 24}
            for v in VARIANTS if v != 'full' for sd in seeds]


def jobs_e3(best_s, seeds=(42, 1, 2)):
    """Early-prediction horizons; window length is capped by the horizon."""
    return [{'experiment': 'E3', 'window_size': h, 'step': min(best_s, h),
             'variant': 'full', 'seed': sd, 'horizon': h}
            for h in HORIZONS if h < 24 for sd in seeds]
