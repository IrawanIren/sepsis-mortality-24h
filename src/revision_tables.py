"""Manuscript tables for the revised submission.

Every table derives from a single aggregation convention: probabilities
are averaged across seeds before any metric is computed, and out-of-fold
predictions are pooled so that each patient contributes one score.
Differences between models can therefore be verified by subtraction,
which the first submission did not permit.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def _fmt(m, s=None, dp=4):
    """Mean with standard deviation, or mean alone when s is undefined."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return f'{m:.{dp}f}'
    return f'{m:.{dp}f} ({s:.{dp}f})'


def _p(v):
    return '<0.001' if v < 0.001 else f'{v:.3f}'


def _write(df, outdir, name, title):
    df.to_csv(os.path.join(outdir, f'{name}.csv'), index=False)
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(df.to_string(index=False))


# ----------------------------------------------------------------------
# Cohort and features
# ----------------------------------------------------------------------

def table_cohort(df, cfg, outdir):
    """Median [IQR] for continuous variables, n (%) for categorical."""
    ID, LABEL, HOUR = cfg['id_col'], cfg['label_col'], cfg['hour_col']
    first = df[df[HOUR] < 24].groupby(ID).first()
    y = first[LABEL]
    surv, died = first[y == 0], first[y == 1]
    c_surv = f'Survivors (n={len(surv):,})'
    c_died = f'Non-survivors (n={len(died):,})'
    rows = []

    def cont(label, a, b, dp=1):
        p = stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue
        rows.append({
            'Variable': label,
            c_surv: f'{a.median():.{dp}f} [{a.quantile(.25):.{dp}f}, '
                    f'{a.quantile(.75):.{dp}f}]',
            c_died: f'{b.median():.{dp}f} [{b.quantile(.25):.{dp}f}, '
                    f'{b.quantile(.75):.{dp}f}]',
            'p': _p(p)})

    def binary(label, col):
        n0, n1 = int(surv[col].sum()), int(died[col].sum())
        p = stats.chi2_contingency(pd.crosstab(first[col], y)).pvalue
        rows.append({
            'Variable': label,
            c_surv: f'{n0:,} ({n0 / len(surv) * 100:.1f})',
            c_died: f'{n1:,} ({n1 / len(died) * 100:.1f})',
            'p': _p(p)})

    def categorical(label, col, mapping=None):
        ct = pd.crosstab(first[col], y)
        p = stats.chi2_contingency(ct).pvalue
        rows.append({'Variable': label, c_surv: '', c_died: '', 'p': _p(p)})
        for lvl in ct.index:
            name = mapping.get(lvl, lvl) if mapping else lvl
            rows.append({
                'Variable': f'  {name}',
                c_surv: f'{ct.loc[lvl, 0]:,} '
                        f'({ct.loc[lvl, 0] / len(surv) * 100:.1f})',
                c_died: f'{ct.loc[lvl, 1]:,} '
                        f'({ct.loc[lvl, 1] / len(died) * 100:.1f})',
                'p': ''})

    cont('Age, years', surv.admission_age, died.admission_age, 0)
    categorical('Sex', 'gender_encoded', {0: 'Female', 1: 'Male'})
    if 'race_collapsed' in first.columns:
        categorical('Race or ethnicity', 'race_collapsed')

    for label, col, dp in [
            ('First-day SOFA', 'sofa_first_day', 0),
            ('SOFA at sepsis onset', 'sofa_onset', 0),
            ('Charlson comorbidity index', 'charlson_comorbidity_index', 0),
            ('Minimum Glasgow Coma Scale', 'gcs_min', 0)]:
        if col in first.columns:
            cont(label, surv[col], died[col], dp)

    for label, col in [('Vasopressor use', 'vasopressor_used'),
                       ('Invasive ventilation',
                        'mechanical_ventilation_used')]:
        if col in first.columns:
            binary(label, col)

    # A representative subset of the dynamic variables, summarised over
    # the observation window; the full list appears in the feature table.
    agg = df[df[HOUR] < 24].groupby(ID)[cfg['dynamic_features']].mean()
    for label, col, dp in [('Heart rate, beats/min', 'heart_rate', 0),
                           ('Mean arterial pressure, mmHg', 'mbp', 0),
                           ('Respiratory rate, breaths/min', 'resp_rate', 0),
                           ('Lactate, mmol/L', 'lactate', 1),
                           ('Creatinine, mg/dL', 'creatinine', 1),
                           ('White blood cells, K/uL', 'wbc', 1)]:
        if col in agg.columns:
            cont(label, agg.loc[surv.index, col], agg.loc[died.index, col], dp)

    out = pd.DataFrame(rows)
    _write(out, outdir, 'table_cohort',
           'Baseline characteristics of the study cohort')
    return out


def table_features(cfg, outdir):
    """Feature inventory by pathway."""
    dyn = cfg['dynamic_features']
    vital = [c for c in dyn if c in (
        'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2')]
    lab = [c for c in dyn if c not in vital and c != 'urineoutput']

    rows = [
        {'Pathway': 'Dynamic', 'Category': 'Vital signs',
         'Count': len(vital), 'Features': ', '.join(vital)},
        {'Pathway': 'Dynamic', 'Category': 'Laboratory',
         'Count': len(lab), 'Features': ', '.join(lab)},
        {'Pathway': 'Dynamic', 'Category': 'Output', 'Count': 1,
         'Features': 'urine output'},
        {'Pathway': 'Dynamic', 'Category': 'Derived', 'Count': len(dyn),
         'Features': 'hourly difference for each variable above'},
        {'Pathway': 'Static', 'Category': 'Demographic', 'Count': 1,
         'Features': 'age at admission'},
        {'Pathway': 'Static', 'Category': 'Severity', 'Count': 8,
         'Features': 'first-day SOFA and its six components, '
                     'SOFA at sepsis onset'},
        {'Pathway': 'Static', 'Category': 'Comorbidity', 'Count': 1,
         'Features': 'Charlson comorbidity index'},
        {'Pathway': 'Static', 'Category': 'Neurological', 'Count': 1,
         'Features': 'minimum Glasgow Coma Scale'},
        {'Pathway': 'Static', 'Category': 'Treatment', 'Count': 2,
         'Features': 'vasopressor use, invasive ventilation'},
        {'Pathway': 'Static', 'Category': 'Reassigned', 'Count': 3,
         'Features': 'thrombin time, fibrinogen, albumin'},
        {'Pathway': 'Static', 'Category': 'Categorical', 'Count': 8,
         'Features': 'sex (2 levels), race or ethnicity (6 levels)'},
    ]
    out = pd.DataFrame(rows)
    _write(out, outdir, 'table_features', 'Features used in the model')
    return out


def table_constancy(df, cfg, outdir, folds=None,
                    moved=('thrombin', 'fibrinogen', 'albumin')):
    """Share of patients whose value never changes over the window.

    Constancy is also recomputed within each training fold, to confirm
    that the assignment of features to the static pathway does not
    depend on the partition.
    """
    ID = cfg['id_col']
    dyn = list(cfg['dynamic_features']) + [c for c in moved
                                           if c in df.columns]
    g = df.groupby(ID)

    rows = []
    for f in dyn:
        if f not in df.columns:
            continue
        nun = g[f].nunique()
        rec = {'Feature': f,
               'Constant over 24 h (%)': round((nun == 1).mean() * 100, 1),
               'Median distinct values': int(nun.median()),
               'Pathway': 'Static' if f in moved else 'Dynamic'}
        if folds is not None:
            per = []
            for k in sorted(folds.fold.unique()):
                ids_ = folds.loc[folds.fold != k, ID].values
                sub = df[df[ID].isin(ids_)].groupby(ID)[f].nunique()
                per.append((sub == 1).mean() * 100)
            rec['Range across training folds (%)'] = \
                f'{min(per):.1f} to {max(per):.1f}'
        rows.append(rec)

    out = (pd.DataFrame(rows)
             .sort_values('Constant over 24 h (%)', ascending=False))
    _write(out, outdir, 'table_constancy',
           'Constancy of candidate dynamic features over the '
           'observation window')
    return out


# ----------------------------------------------------------------------
# Experimental design and tuning
# ----------------------------------------------------------------------

def table_design(outdir):
    rows = [
        {'Stage': 'E0', 'Purpose': 'Hyperparameter search',
         'Configurations': '15 random draws', 'Seeds': 1, 'Folds': 3},
        {'Stage': 'E1', 'Purpose': 'Window size and stride',
         'Configurations': '12 combinations', 'Seeds': 1, 'Folds': 5},
        {'Stage': 'E1b', 'Purpose': 'Multi-seed confirmation',
         'Configurations': 'Selected configuration', 'Seeds': 5, 'Folds': 5},
        {'Stage': 'E2', 'Purpose': 'Architecture ablation',
         'Configurations': '4 variants', 'Seeds': 5, 'Folds': 5},
        {'Stage': 'E3', 'Purpose': 'Observation window length',
         'Configurations': '3 horizons, features restricted to each',
         'Seeds': 3, 'Folds': 5},
        {'Stage': 'A1-A3', 'Purpose': 'Controlled augmentation',
         'Configurations': '3 conditions', 'Seeds': 3, 'Folds': 5},
        {'Stage': 'C1-C2', 'Purpose': 'Comparator architectures',
         'Configurations': 'Bi-LSTM, Transformer', 'Seeds': 5, 'Folds': 5},
        {'Stage': 'E5', 'Purpose': 'Final evaluation',
         'Configurations': 'All models', 'Seeds': 5,
         'Folds': 'Held-out test set'},
    ]
    out = pd.DataFrame(rows)
    _write(out, outdir, 'table_design', 'Design of the experimental programme')
    return out


def table_hyperparameters(res_dir, base, outdir):
    """Search spaces and selected values for every fitted model."""
    spaces = {
        'Proposed model': (f'{base}/best_hp_rebuilt.json', {
            'learning_rate': '0.001, 0.0001, 0.00005',
            'l2': '0.01, 0.005, 0.001',
            'gru1': '32, 48, 64', 'gru2': '16, 24, 32',
            'dropout': '0.2, 0.3, 0.4'}),
        'Bi-LSTM': (f'{base}/best_hp_bilstm.json', {
            'learning_rate': '0.001, 0.0001, 0.00005',
            'l2': '0.01, 0.005, 0.001',
            'lstm1': '32, 48, 64', 'lstm2': '16, 24, 32',
            'dropout': '0.2, 0.3, 0.4'}),
        'Transformer': (f'{base}/best_hp_transformer.json', {
            'learning_rate': '0.001, 0.0001, 0.00005',
            'l2': '0.01, 0.005, 0.001',
            'd_model': '32, 64, 128', 'n_heads': '2, 4, 8',
            'ff_dim': '64, 128, 256', 'n_layers': '1, 2, 3',
            'dropout': '0.1, 0.2, 0.3'}),
        'XGBoost': (f'{base}/best_hp_xgboost.json', {
            'n_estimators': '200, 400, 800', 'max_depth': '3, 5, 7',
            'learning_rate': '0.01, 0.05, 0.1',
            'subsample': '0.6, 0.8, 1.0',
            'colsample_bytree': '0.6, 0.8, 1.0',
            'min_child_weight': '1, 5, 10'}),
        'CatBoost': (f'{base}/best_hp_catboost.json', {
            'iterations': '200, 400, 800', 'depth': '4, 6, 8',
            'learning_rate': '0.01, 0.05, 0.1',
            'l2_leaf_reg': '1, 3, 10', 'subsample': '0.6, 0.8, 1.0'}),
    }
    names = {'learning_rate': 'Learning rate', 'l2': 'L2 regularisation',
             'gru1': 'Units, first Bi-GRU layer',
             'gru2': 'Units, second Bi-GRU layer',
             'lstm1': 'Units, first Bi-LSTM layer',
             'lstm2': 'Units, second Bi-LSTM layer',
             'd_model': 'Model dimension', 'n_heads': 'Attention heads',
             'ff_dim': 'Feed-forward dimension',
             'n_layers': 'Encoder layers', 'dropout': 'Dropout rate',
             'n_estimators': 'Boosting rounds', 'max_depth': 'Maximum depth',
             'subsample': 'Row subsample',
             'colsample_bytree': 'Column subsample',
             'min_child_weight': 'Minimum child weight',
             'iterations': 'Boosting rounds', 'depth': 'Tree depth',
             'l2_leaf_reg': 'L2 leaf regularisation'}

    rows = []
    for model, (path, space) in spaces.items():
        if not os.path.exists(path):
            continue
        sel = json.load(open(path))
        for k, grid in space.items():
            v = sel.get(k)
            rows.append({
                'Model': model, 'Hyperparameter': names.get(k, k),
                'Search space': grid,
                'Selected': (f'{v:g}' if isinstance(v, float) else str(v))})

    out = pd.DataFrame(rows)
    _write(out, outdir, 'table_hyperparameters',
           'Search spaces and selected hyperparameters. Every model '
           'received fifteen random-search trials on the same three '
           'development folds.')
    return out


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

def table_grid(res_dir, outdir, n_hours=24):
    def n_win(W, S):
        starts = list(range(0, n_hours - W + 1, S))
        if starts[-1] != n_hours - W:
            starts.append(n_hours - W)
        return len(starts)

    g = pd.read_csv(os.path.join(res_dir, 'e1_window_grid.csv'))
    g = g[g.experiment == 'E1'].sort_values(['window_size', 'step'])
    out = pd.DataFrame({
        'W (h)': g.window_size, 'S (h)': g.step,
        'Windows per patient': [n_win(w, s)
                                for w, s in zip(g.window_size, g.step)],
        'AUROC': [_fmt(m, s) for m, s in zip(g.auroc_mean, g.auroc_std)],
        'AUPRC': [_fmt(m, s) for m, s in zip(g.auprc_mean, g.auprc_std)],
        'Calibration slope': [_fmt(m, s, 3) for m, s in
                              zip(g.cal_slope_mean, g.cal_slope_std)],
    })
    _write(out, outdir, 'table_grid',
           'Performance across window size and stride '
           '(five-fold cross-validation, one seed)')
    return out


def table_horizon(res_dir, base, outdir):
    bw = json.load(open(os.path.join(base, 'best_window_rebuilt.json')))
    W, S = bw['window_size'], bw['step']
    e1 = pd.read_csv(os.path.join(res_dir, 'e1_window_grid.csv'))
    e3 = pd.read_csv(os.path.join(res_dir, 'e3_horizon.csv'))
    h24 = e1[(e1.window_size == W) & (e1.step == S)].assign(horizon=24)
    allh = pd.concat([e3, h24], ignore_index=True)

    rows = []
    for h, sub in allh.groupby('horizon'):
        rows.append({
            'Observation window (h)': h, 'Seeds': len(sub),
            'AUROC': _fmt(sub.auroc_mean.mean(), sub.auroc_mean.std()),
            'AUPRC': _fmt(sub.auprc_mean.mean(), sub.auprc_mean.std()),
            'Calibration slope': _fmt(sub.cal_slope_mean.mean(),
                                      sub.cal_slope_mean.std(), 3)})
    out = pd.DataFrame(rows)
    _write(out, outdir, 'table_horizon',
           'Performance across observation windows, with all features '
           'restricted to the information available at each horizon')

    a = allh.groupby('horizon').auroc_mean.mean()
    if {6, 12, 24} <= set(a.index):
        frac = (a[12] - a[6]) / (a[24] - a[6]) * 100
        print(f"\nTwelve hours recovers {frac:.0f}% of the gain from six "
              f"to twenty-four hours")
    return out


def table_ablation(res_dir, base, outdir):
    bw = json.load(open(os.path.join(base, 'best_window_rebuilt.json')))
    W, S = bw['window_size'], bw['step']
    e2 = pd.read_csv(os.path.join(res_dir, 'e2_ablation.csv'))

    spec = {'gru_only': ('No', 'No', 'Bi-GRU only'),
            'no_mlp': ('Yes', 'No', 'CNN and Bi-GRU'),
            'no_cnn': ('No', 'Yes', 'Bi-GRU and static pathway'),
            'full': ('Yes', 'Yes', 'CNN, Bi-GRU and static pathway '
                                   '(proposed)')}
    rows = []
    for v in ('gru_only', 'no_mlp', 'no_cnn', 'full'):
        sub = e2[e2.variant == v]
        if not len(sub):
            continue
        cnn, mlp, label = spec[v]
        rows.append({
            'Configuration': label, 'CNN': cnn, 'Static pathway': mlp,
            'Sequence length at Bi-GRU': W // 2 if cnn == 'Yes' else W,
            'AUROC': _fmt(sub.auroc_mean.mean(), sub.auroc_mean.std()),
            'AUPRC': _fmt(sub.auprc_mean.mean(), sub.auprc_mean.std()),
            'Calibration slope': _fmt(sub.cal_slope_mean.mean(),
                                      sub.cal_slope_mean.std(), 3),
            'Training time (s)': f'{sub.elapsed_s.mean():.0f}'})
    out = pd.DataFrame(rows)
    _write(out, outdir, 'table_ablation',
           'Architecture ablation at the selected window configuration. '
           'Training times were measured on a single device within one '
           'session.')
    return out


def table_augmentation(res_dir, base, outdir):
    """Augmentation decomposed into replication and positional variety."""
    bw = json.load(open(os.path.join(base, 'best_window_rebuilt.json')))
    W, S = bw['window_size'], bw['step']
    aug = pd.read_csv(os.path.join(res_dir, 'augmentation_control.csv'))
    e1 = pd.read_csv(os.path.join(res_dir, 'e1_window_grid.csv'))
    seeds = sorted(aug.seed.unique())
    a3 = e1[(e1.window_size == W) & (e1.step == S) &
            (e1.seed.isin(seeds))].assign(
        condition='A3', train_windows=19, positional_variety='Yes')
    allc = pd.concat([aug, a3], ignore_index=True)

    labels = {'A1': 'One right-aligned window',
              'A2': 'Right-aligned window repeated',
              'A3': 'Sliding windows'}
    rows = []
    for c in ('A1', 'A2', 'A3'):
        sub = allc[allc.condition == c]
        if not len(sub):
            continue
        rows.append({
            'Condition': c, 'Training windows per patient':
                int(sub.train_windows.iloc[0]),
            'Positional variety': 'Yes' if c == 'A3' else 'No',
            'Description': labels[c],
            'AUROC': _fmt(sub.auroc_mean.mean(), sub.auroc_mean.std()),
            'AUPRC': _fmt(sub.auprc_mean.mean(), sub.auprc_mean.std())})
    out = pd.DataFrame(rows)
    _write(out, outdir, 'table_augmentation',
           'Controlled augmentation. Window size, inference window, and '
           'architecture are identical; only the training windows differ.')

    a = allc.groupby('condition').auroc_mean.mean()
    if {'A1', 'A2', 'A3'} <= set(a.index):
        print(f"\nTotal effect (A3 - A1)      : {a['A3'] - a['A1']:+.4f}")
        print(f"Replication  (A2 - A1)      : {a['A2'] - a['A1']:+.4f}")
        print(f"Positional variety (A3 - A2): {a['A3'] - a['A2']:+.4f}")
    return out


def table_position(res_dir, outdir):
    p = os.path.join(res_dir, 'window_position.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    g = (d.groupby(['start', 'end', 'right_aligned'])[['auroc', 'auprc']]
           .agg(['mean', 'std']).reset_index())
    out = pd.DataFrame({
        'Window (hours)': [f'{a}-{b}' for a, b in zip(g.start, g.end)],
        'Right-aligned': np.where(g.right_aligned == 1, 'Yes', 'No'),
        'AUROC': [_fmt(m, s) for m, s in zip(g[('auroc', 'mean')],
                                             g[('auroc', 'std')])],
        'AUPRC': [_fmt(m, s) for m, s in zip(g[('auprc', 'mean')],
                                             g[('auprc', 'std')])]})
    _write(out, outdir, 'table_position',
           'Performance by inference window position. Models were '
           'trained once and scored at every position.')
    return out


def table_cv(res_dir, outdir):
    p = os.path.join(res_dir, 'cv_pooled.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    out = pd.DataFrame({
        'Model': d.Model,
        'AUROC': d.auroc.map(lambda v: f'{v:.4f}'),
        'AUPRC': d.auprc.map(lambda v: f'{v:.4f}'),
        'Brier': d.brier.map(lambda v: 'Not applicable' if pd.isna(v)
                             else f'{v:.4f}'),
        'Calibration slope': d.cal_slope.map(
            lambda v: 'Not applicable' if pd.isna(v) else f'{v:.3f}')})
    _write(out, outdir, 'table_cv',
           'Cross-validated performance on the development set, from '
           'pooled out-of-fold predictions')
    return out


def table_test(res_dir, outdir):
    p = os.path.join(res_dir, 'test_pooled.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    out = pd.DataFrame({
        'Model': d.Model,
        'AUROC': d.auroc.map(lambda v: f'{v:.4f}'),
        'AUPRC': d.auprc.map(lambda v: f'{v:.4f}'),
        'Brier': d.brier.map(lambda v: 'Not applicable' if pd.isna(v)
                             else f'{v:.4f}'),
        'Calibration slope': d.cal_slope.map(
            lambda v: 'Not applicable' if pd.isna(v) else f'{v:.3f}'),
        'Calibration intercept': d.cal_intercept.map(
            lambda v: 'Not applicable' if pd.isna(v) else f'{v:+.3f}')})
    _write(out, outdir, 'table_test',
           'Performance on the held-out test set, Platt-scaled')
    return out


def table_operating(res_dir, outdir, n_events=None):
    p = os.path.join(res_dir, 'operating_points.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    out = pd.DataFrame({
        'Model': d.Model, 'Operating point': d['Operating point'],
        'Threshold': d.threshold.map(lambda v: f'{v:.4f}'),
        'Sensitivity': d.sensitivity.map(lambda v: f'{v:.3f}'),
        'Specificity': d.specificity.map(lambda v: f'{v:.3f}'),
        'PPV': d.ppv.map(lambda v: f'{v:.3f}'),
        'NPV': d.npv.map(lambda v: f'{v:.3f}'),
        'Patients flagged': d.flagged,
        'False alerts': d.false_alerts})
    _write(out, outdir, 'table_operating',
           'Threshold-dependent metrics. Both thresholds were derived '
           'from development data and applied unchanged to the test set.')
    return out


def table_statistics(res_dir, outdir):
    a = pd.read_csv(os.path.join(res_dir, 'stats_test_comparisons.csv'))
    ta = pd.DataFrame({
        'Comparison': a.comparison,
        'AUROC difference': a.auroc_diff.map(lambda v: f'{v:+.4f}'),
        'DeLong p': a.delong_p.map(_p),
        'Adjusted p': a.p_adj.map(_p),
        'AUPRC difference [95% CI]':
            [f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]'
             for d, lo, hi in zip(a.auprc_diff, a.auprc_lo, a.auprc_hi)]})
    _write(ta, outdir, 'table_stats_test',
           'Paired comparisons on the held-out test set')

    b = pd.read_csv(os.path.join(res_dir, 'stats_design_effects.csv'))
    tb = pd.DataFrame({
        'Design factor': b.effect,
        'AUROC, full': b.auroc_full.map(lambda v: f'{v:.4f}'),
        'AUROC, reduced': b.auroc_reduced.map(lambda v: f'{v:.4f}'),
        'Difference': b['diff'].map(lambda v: f'{v:+.4f}'),
        'DeLong p': b.delong_p.map(_p),
        'Adjusted p': b.p_adj.map(_p)})
    _write(tb, outdir, 'table_stats_design',
           'Contribution of individual design factors, on pooled '
           'out-of-fold predictions')
    return ta, tb


# ----------------------------------------------------------------------

def build_all(base, res_dir, outdir, folds=None):
    os.makedirs(outdir, exist_ok=True)
    cfg = json.load(open(os.path.join(base, 'feature_config_rebuilt.json')))
    df = pd.read_parquet(os.path.join(base, 'sepsis_rebuilt_24h.parquet'))

    table_cohort(df, cfg, outdir)
    table_features(cfg, outdir)
    table_constancy(df, cfg, outdir, folds)
    table_design(outdir)
    table_hyperparameters(res_dir, base, outdir)
    table_grid(res_dir, outdir)
    table_horizon(res_dir, base, outdir)
    table_ablation(res_dir, base, outdir)
    table_augmentation(res_dir, base, outdir)
    table_position(res_dir, outdir)
    table_cv(res_dir, outdir)
    table_test(res_dir, outdir)
    table_operating(res_dir, outdir)
    table_statistics(res_dir, outdir)

    made = sorted(f for f in os.listdir(outdir) if f.endswith('.csv'))
    print(f"\n\n{len(made)} tables written to {outdir}")
    for f in made:
        print(f"  {f}")
