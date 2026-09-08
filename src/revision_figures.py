"""Result figures for the revised submission.

Legends are placed outside the data area wherever a curve or bar could
reach the corner they would otherwise occupy, since an obscured legend
was one of the presentation problems raised in review.

Colours use the Okabe-Ito palette, which is colour-blind safe and
separable in greyscale; line styles and markers vary alongside colour so
that the figures survive black-and-white printing.
"""

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)

OKABE = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'purple': '#CC79A7',
    'sky':    '#56B4E9',
    'yellow': '#F0E442',
    'grey':   '#666666',
}
COLOURS = [OKABE['blue'], OKABE['orange'], OKABE['green'],
           OKABE['purple'], OKABE['sky'], OKABE['red']]
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
LINES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

SINGLE = (3.27, 2.6)      # one column
WIDE = (6.77, 3.0)        # full text width


def apply_style():
    for family in ('Times New Roman', 'Liberation Serif', 'DejaVu Serif'):
        if any(family in f.name for f in mpl.font_manager.fontManager.ttflist):
            serif = family
            break
    else:
        serif = 'serif'

    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': [serif], 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8.5, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'lines.linewidth': 1.4,
        'lines.markersize': 4, 'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })
    return serif


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'))
    plt.close(fig)


def _legend_below(ax, ncol=2, y=-0.32):
    """Place the legend under the axes, which keeps it clear of curves
    that approach the corners."""
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol,
              handlelength=2.2, columnspacing=1.4, borderaxespad=0)


# ----------------------------------------------------------------------
# Window configuration
# ----------------------------------------------------------------------

def fig_grid(res_dir, outdir):
    e1 = pd.read_csv(os.path.join(res_dir, 'e1_window_grid.csv'))
    g = e1[e1.experiment == 'E1']

    fig, ax = plt.subplots(figsize=SINGLE)
    for k, (W, sub) in enumerate(g.groupby('window_size')):
        sub = sub.sort_values('step')
        ax.errorbar(sub.step, sub.auroc_mean, yerr=sub.auroc_std,
                    marker=MARKERS[k], linestyle=LINES[k],
                    color=COLOURS[k % len(COLOURS)], capsize=2,
                    label=f'W = {W} h')
    ax.set_xlabel('Stride S (hours)')
    ax.set_ylabel('AUROC')
    _legend_below(ax, ncol=4, y=-0.30)
    save(fig, outdir, 'fig_window_grid')


# ----------------------------------------------------------------------
# Observation horizon
# ----------------------------------------------------------------------

def fig_horizon(res_dir, base, outdir):
    bw = json.load(open(os.path.join(base, 'best_window_rebuilt.json')))
    W, S = bw['window_size'], bw['step']
    e1 = pd.read_csv(os.path.join(res_dir, 'e1_window_grid.csv'))
    e3 = pd.read_csv(os.path.join(res_dir, 'e3_horizon.csv'))
    h24 = e1[(e1.window_size == W) & (e1.step == S)].assign(horizon=24)
    allh = pd.concat([e3, h24], ignore_index=True)

    a = allh.groupby('horizon').auroc_mean.agg(['mean', 'std']).reset_index()
    p = allh.groupby('horizon').auprc_mean.agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.errorbar(a.horizon, a['mean'], yerr=a['std'], marker=MARKERS[0],
                linestyle=LINES[0], color=COLOURS[0], capsize=3,
                label='AUROC')
    ax.errorbar(p.horizon, p['mean'], yerr=p['std'], marker=MARKERS[1],
                linestyle=LINES[1], color=COLOURS[1], capsize=3,
                label='AUPRC')
    ax.set_xlabel('Observation window (hours)')
    ax.set_ylabel('Performance')
    ax.set_xticks(sorted(allh.horizon.unique()))
    _legend_below(ax, ncol=2, y=-0.24)
    save(fig, outdir, 'fig_horizon')


# ----------------------------------------------------------------------
# Architecture ablation
# ----------------------------------------------------------------------

def fig_ablation(res_dir, outdir):
    e2 = pd.read_csv(os.path.join(res_dir, 'e2_ablation.csv'))
    order = ['gru_only', 'no_mlp', 'no_cnn', 'full']
    labels = ['Bi-GRU\nonly', 'CNN and\nBi-GRU', 'Bi-GRU and\nstatic',
              'Proposed']

    stats = (e2.groupby('variant').auroc_mean.agg(['mean', 'std'])
               .reindex([v for v in order if v in e2.variant.values]))

    fig, ax = plt.subplots(figsize=SINGLE)
    colours = [OKABE['sky']] * (len(stats) - 1) + [OKABE['blue']]
    hatches = [''] * (len(stats) - 1) + ['///']
    bars = ax.bar(range(len(stats)), stats['mean'], yerr=stats['std'],
                  capsize=3, color=colours, edgecolor='black',
                  linewidth=0.6, hatch=hatches)

    lo = stats['mean'].min() - 0.02
    hi = stats['mean'].max() + 0.02
    for b, v in zip(bars, stats['mean']):
        ax.text(b.get_x() + b.get_width() / 2, v + (hi - lo) * 0.03,
                f'{v:.3f}', ha='center', fontsize=8)

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(labels[:len(stats)], fontsize=8)
    ax.set_ylabel('AUROC')
    ax.set_ylim(lo, hi)
    save(fig, outdir, 'fig_ablation')


# ----------------------------------------------------------------------
# Controlled augmentation
# ----------------------------------------------------------------------

def fig_augmentation(res_dir, base, outdir):
    """Three conditions differing only in the training windows."""
    bw = json.load(open(os.path.join(base, 'best_window_rebuilt.json')))
    W, S = bw['window_size'], bw['step']
    aug = pd.read_csv(os.path.join(res_dir, 'augmentation_control.csv'))
    e1 = pd.read_csv(os.path.join(res_dir, 'e1_window_grid.csv'))
    seeds = sorted(aug.seed.unique())
    a3 = e1[(e1.window_size == W) & (e1.step == S) &
            (e1.seed.isin(seeds))].assign(condition='A3')
    allc = pd.concat([aug, a3], ignore_index=True)

    order = ['A1', 'A2', 'A3']
    labels = ['One window\nper patient', 'Same window\nrepeated',
              'Sliding\nwindows']
    stats = (allc.groupby('condition').auroc_mean.agg(['mean', 'std'])
                 .reindex([c for c in order if c in allc.condition.values]))

    fig, ax = plt.subplots(figsize=SINGLE)
    colours = [OKABE['sky'], OKABE['orange'], OKABE['blue']][:len(stats)]
    bars = ax.bar(range(len(stats)), stats['mean'], yerr=stats['std'],
                  capsize=3, color=colours, edgecolor='black', linewidth=0.6)

    lo = stats['mean'].min() - 0.015
    hi = stats['mean'].max() + 0.015
    for b, v in zip(bars, stats['mean']):
        ax.text(b.get_x() + b.get_width() / 2, v + (hi - lo) * 0.03,
                f'{v:.3f}', ha='center', fontsize=8)

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(labels[:len(stats)], fontsize=8)
    ax.set_ylabel('AUROC')
    ax.set_ylim(lo, hi)
    save(fig, outdir, 'fig_augmentation')


# ----------------------------------------------------------------------
# Window position
# ----------------------------------------------------------------------

def fig_position(res_dir, outdir):
    p = os.path.join(res_dir, 'window_position.csv')
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    g = (d.groupby(['start', 'right_aligned']).auroc
           .agg(['mean', 'std']).reset_index())

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ra = g[g.right_aligned == 1]
    ax.errorbar(g.start, g['mean'], yerr=g['std'], marker='o',
                linestyle='-', color=COLOURS[0], capsize=2, markersize=3.5,
                label='Inference window')
    ax.scatter(ra.start, ra['mean'], s=55, facecolor='none',
               edgecolor=OKABE['red'], linewidth=1.4, zorder=5,
               label='Right-aligned (used at inference)')

    ax.set_xlabel('First hour of the inference window')
    ax.set_ylabel('AUROC')
    ax.set_xticks(range(0, 19, 3))
    _legend_below(ax, ncol=1, y=-0.30)
    save(fig, outdir, 'fig_window_position')


# ----------------------------------------------------------------------
# Test-set curves
# ----------------------------------------------------------------------

def _load_models(pred_dir, seeds, cal='platt'):
    spec = [
        ('Proposed model', f'E5_full_{cal}_s{{seed}}.npz'),
        ('XGBoost',        f'E5_XGBoost_{cal}_s{{seed}}.npz'),
        ('CatBoost',       f'E5_CatBoost_{cal}_s{{seed}}.npz'),
        ('Bi-LSTM',        f'E5_BiLSTM_{cal}_s{{seed}}.npz'),
        ('Transformer',    f'E5_Transformer_{cal}_s{{seed}}.npz'),
    ]
    out = []
    for k, (label, pat) in enumerate(spec):
        scores = []
        try:
            for s in seeds:
                z = np.load(os.path.join(pred_dir, pat.format(seed=s)))
                scores.append(z['y_score'])
        except FileNotFoundError:
            continue
        out.append((label, z['y_true'], np.mean(scores, axis=0), k))

    sofa = os.path.join(pred_dir, 'E5_SOFA.npz')
    if os.path.exists(sofa):
        z = np.load(sofa)
        out.append(('First-day SOFA', z['y_true'], z['y_score'], 5))
    return out


def fig_roc(pred_dir, seeds, outdir):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for label, y, p, k in _load_models(pred_dir, seeds):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, linestyle=LINES[k % len(LINES)],
                color=COLOURS[k % len(COLOURS)],
                label=f'{label} ({auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], color='0.6', linewidth=0.8, linestyle=':')
    ax.set_xlabel('1 - specificity')
    ax.set_ylabel('Sensitivity')
    # Curves fill the lower right, so the legend goes underneath.
    _legend_below(ax, ncol=2, y=-0.28)
    save(fig, outdir, 'fig_roc')


def fig_pr(pred_dir, seeds, outdir):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    prev = None
    for label, y, p, k in _load_models(pred_dir, seeds):
        pre, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, pre, linestyle=LINES[k % len(LINES)],
                color=COLOURS[k % len(COLOURS)],
                label=f'{label} ({average_precision_score(y, p):.3f})')
        prev = y.mean()
    if prev is not None:
        ax.axhline(prev, color='0.6', linewidth=0.8, linestyle=':')
        ax.text(0.98, prev + 0.03, f'Prevalence {prev:.3f}',
                fontsize=8, color='0.4', ha='right')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim(0, 1.05)
    _legend_below(ax, ncol=2, y=-0.28)
    save(fig, outdir, 'fig_pr')


def fig_calibration(pred_dir, seeds, outdir, n_bins=10):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for label, y, p, k in _load_models(pred_dir, seeds):
        if 'SOFA' in label:
            continue
        obs, pred = calibration_curve(y, p, n_bins=n_bins,
                                      strategy='quantile')
        ax.plot(pred, obs, marker=MARKERS[k % len(MARKERS)],
                linestyle=LINES[k % len(LINES)],
                color=COLOURS[k % len(COLOURS)], label=label)
    ax.plot([0, 1], [0, 1], color='0.6', linewidth=0.8, linestyle=':',
            label='Perfect calibration')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    _legend_below(ax, ncol=2, y=-0.28)
    save(fig, outdir, 'fig_calibration')


# ----------------------------------------------------------------------

def build_all(base, res_dir, pred_dir, outdir, seeds=(42, 1, 2, 3, 4)):
    os.makedirs(outdir, exist_ok=True)
    serif = apply_style()
    print(f"Font in use: {serif}")

    fig_grid(res_dir, outdir)
    fig_horizon(res_dir, base, outdir)
    fig_ablation(res_dir, outdir)
    fig_augmentation(res_dir, base, outdir)
    fig_position(res_dir, outdir)
    fig_roc(pred_dir, seeds, outdir)
    fig_pr(pred_dir, seeds, outdir)
    fig_calibration(pred_dir, seeds, outdir)

    made = sorted(f for f in os.listdir(outdir) if f.endswith('.png'))
    print(f"\n{len(made)} figures written to {outdir}")
    for f in made:
        print(f"  {f}")
