"""
Cross-validated modality experiments
------------------------------------
Does joint US + cytology fusion beat the best single-modality model and late
fusion, and does modality dropout keep performance when a modality is missing?

  E1_us_only      US encoder only                       (CFG mode='us_only')
  E2_cyto_only    cytology patch branch only            (CFG mode='cyto_only')
  E3_late_fusion  mean of E1 and E2 probabilities       (derived, no training)
  E4_joint        joint fusion, no modality dropout     (CFG mode='joint')
  E5_joint_mdrop  joint fusion + modality dropout       (mode='joint', modality_dropout=...)

Every model is evaluated on the held-out fold under the test-time scenarios
both / us_missing / cyto_missing / few_patches (where applicable).

Usage
-----
python experiments.py                         # E1–E5, 5 folds, 1 seed
python experiments.py --n_seeds 3             # seeds 42, 43, 44
python experiments.py --experiments E4 E5 --folds 0 1
python experiments.py --reuse_ckpt            # evaluate existing checkpoints, train only missing ones
python experiments.py --inner_val_frac 0      # no inner split (threshold chosen on the test fold)

Outputs (in --out_dir, default results/)
-------
cv_experiments.csv           one row per experiment × seed × fold × scenario
cv_experiments_summary.csv   mean ± SD over folds (and seeds) per experiment × scenario
oof_<experiment>.csv         out-of-fold per-patient probabilities
delta_auroc.csv              paired-bootstrap ΔAUROC on pooled out-of-fold predictions

TODO: feature cache — precompute frozen-stage (features[:trainable_from_feature])
activations once per image to speed up the 5 × 4 trainings.
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from main import (CFG, device, build_metadata, split_fold, train_fold,
                  evaluate_scenarios, scenario_metrics)
from model import MultimodalTransformer

EXPERIMENTS = {
    'E1_us_only':     dict(mode='us_only',   modality_dropout=None),
    'E2_cyto_only':   dict(mode='cyto_only', modality_dropout=None),
    'E4_joint':       dict(mode='joint',     modality_dropout=None),
    'E5_joint_mdrop': dict(mode='joint',     modality_dropout=dict(p_drop_cyto=0.3, p_drop_us=0.15)),
}
LATE = 'E3_late_fusion'
US_ONLY, CYTO_ONLY = 'E1_us_only', 'E2_cyto_only'

# Late fusion per scenario: (E1 scenario, E2 scenario); None = that model has no input
LATE_FUSION_MAP = {
    'both':         ('both', 'both'),
    'us_missing':   (None,   'both'),
    'cyto_missing': ('both', None),
    'few_patches':  ('both', 'few_patches'),
}

# ΔAUROC comparisons (A − B); 'best_single' = higher pooled AUROC of E1/E2 (scenario 'both')
COMPARISONS = [
    ('E4_joint',       'best_single'),
    ('E4_joint',       LATE),
    ('E5_joint_mdrop', 'E4_joint'),
    ('E5_joint_mdrop', 'best_single'),
    ('E5_joint_mdrop', LATE),
]


def late_fusion(us_res, cyto_res):
    out = {}
    for sc, (s_us, s_cy) in LATE_FUSION_MAP.items():
        parts = [r[s] for r, s in [(us_res, s_us), (cyto_res, s_cy)] if s is not None]
        assert all(p['pids'] == parts[0]['pids'] for p in parts), "patient order mismatch"
        out[sc] = dict(
            p_test=np.mean([p['p_test'] for p in parts], axis=0),
            p_sel=np.mean([p['p_sel'] for p in parts], axis=0),
            y_test=parts[0]['y_test'], y_sel=parts[0]['y_sel'],
            pids=parts[0]['pids'], nested=parts[0]['nested'],
        )
    return out


def paired_bootstrap_delta_auc(y, p_a, p_b, n_boot=2000, seed=42):
    """ΔAUROC = AUROC(a) − AUROC(b) on the same patients, with a paired
    percentile-bootstrap 95% CI and a two-sided bootstrap p-value."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        deltas.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return {
        'AUROC_A': roc_auc_score(y, p_a),
        'AUROC_B': roc_auc_score(y, p_b),
        'delta_AUROC': roc_auc_score(y, p_a) - roc_auc_score(y, p_b),
        'CI_lo': lo, 'CI_hi': hi, 'p_boot': p, 'n': len(y),
    }


def run_model(df, cfg, name, seed, fold, reuse_ckpt):
    ckpt_name = f'{name}_seed{seed}_fold{fold}_best.pth'
    ckpt_path = os.path.join(cfg['checkpoint_dir'], ckpt_name)
    if reuse_ckpt and os.path.exists(ckpt_path):
        print(f"\nReusing {ckpt_path}")
        model = MultimodalTransformer(cfg, pretrained=False).to(device)
    else:
        model, ckpt_path, _ = train_fold(df, cfg, fold, ckpt_name=ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
    _, sel_df, test_df = split_fold(df, cfg, fold)
    res = evaluate_scenarios(model, test_df, sel_df, cfg)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def summarize(rows_df):
    metrics = ['AUROC', 'AUROC_CI_lo', 'AUROC_CI_hi', 'AUPRC', 'Sensitivity', 'Specificity', 'Brier']
    g = rows_df.groupby(['experiment', 'scenario'], sort=False)[metrics]
    mean, sd = g.mean(), g.std(ddof=1)
    summary = pd.DataFrame(index=mean.index)
    summary['n_runs'] = g.size()
    for m in metrics:
        summary[f'{m}_mean'] = mean[m]
        summary[f'{m}_sd']   = sd[m]
    for m in ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Brier']:
        summary[f'{m} (mean ± SD)'] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean[m], sd[m])]
    return summary.reset_index()


def delta_table(oof_df, n_boot):
    rows = []
    for (seed, sc), d in oof_df.groupby(['seed', 'scenario']):
        wide = d.pivot_table(index=['patient_id', 'label'], columns='experiment', values='prob').reset_index()
        singles = [e for e in (US_ONLY, CYTO_ONLY) if e in wide]
        if sc == 'both' and singles:
            best = max(singles, key=lambda e: roc_auc_score(wide['label'], wide[e]))
        for a, b in COMPARISONS:
            if b == 'best_single':
                if sc != 'both' or not singles: continue
                b_name = best
            else:
                b_name = b
            if a not in wide or b_name not in wide: continue
            ok = wide[[a, b_name]].notna().all(axis=1)
            r = paired_bootstrap_delta_auc(wide.loc[ok, 'label'], wide.loc[ok, a], wide.loc[ok, b_name],
                                           n_boot=n_boot, seed=int(seed))
            rows.append({'seed': seed, 'scenario': sc, 'A': a,
                         'B': b_name if b != 'best_single' else f'best_single={b_name}', **r})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Cross-validated modality experiments (E1–E5)')
    ap.add_argument('--experiments', nargs='+', default=['E1', 'E2', 'E4', 'E5'],
                    help='Subset of E1 E2 E4 E5 (E3 late fusion is derived when E1 and E2 run)')
    ap.add_argument('--folds', nargs='+', type=int, default=list(range(CFG['n_folds'])))
    ap.add_argument('--n_seeds', type=int, default=1, help='Seeds CFG seed, seed+1, ...')
    ap.add_argument('--inner_val_frac', type=float, default=0.15,
                    help='Inner stratified val split for checkpoint + threshold selection (0 = off)')
    ap.add_argument('--p_drop_cyto', type=float, default=0.3)
    ap.add_argument('--p_drop_us',   type=float, default=0.15)
    ap.add_argument('--n_boot', type=int, default=2000, help='Bootstrap draws for ΔAUROC')
    ap.add_argument('--reuse_ckpt', action='store_true')
    ap.add_argument('--out_dir', default='results')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    selected = {k: dict(v) for k, v in EXPERIMENTS.items() if k.split('_')[0] in args.experiments}
    if 'E5_joint_mdrop' in selected:
        selected['E5_joint_mdrop']['modality_dropout'] = dict(p_drop_cyto=args.p_drop_cyto,
                                                              p_drop_us=args.p_drop_us)
    seeds = [CFG['seed'] + i for i in range(args.n_seeds)]
    print(f"Experiments: {list(selected)} | folds={args.folds} | seeds={seeds} | "
          f"inner_val_frac={args.inner_val_frac or None}")

    df = build_metadata(CFG)    # outer folds fixed by CFG['seed'] for every run
    rows, oof = [], []
    rows_path = os.path.join(args.out_dir, 'cv_experiments.csv')

    for seed in seeds:
        for fold in args.folds:
            res = {}
            for name, overrides in selected.items():
                cfg = {**CFG, **overrides, 'seed': seed, 'inner_val_frac': args.inner_val_frac or None}
                res[name] = run_model(df, cfg, name, seed, fold, args.reuse_ckpt)
            if US_ONLY in res and CYTO_ONLY in res:
                res[LATE] = late_fusion(res[US_ONLY], res[CYTO_ONLY])

            for name, by_sc in res.items():
                for sc, pred in by_sc.items():
                    m = scenario_metrics(pred, seed=seed)
                    rows.append({'experiment': name, 'seed': seed, 'fold': fold, 'scenario': sc, **m})
                    oof.append(pd.DataFrame({'experiment': name, 'seed': seed, 'fold': fold, 'scenario': sc,
                                             'patient_id': pred['pids'], 'label': pred['y_test'].astype(int),
                                             'prob': pred['p_test']}))
            pd.DataFrame(rows).to_csv(rows_path, index=False)   # write as we go
            print(f"\n[seed {seed} fold {fold}] done → {rows_path}")

    rows_df = pd.DataFrame(rows)
    summary = summarize(rows_df)
    summary_path = os.path.join(args.out_dir, 'cv_experiments_summary.csv')
    summary.to_csv(summary_path, index=False)

    oof_df = pd.concat(oof, ignore_index=True)
    for name, d in oof_df.groupby('experiment'):
        d.to_csv(os.path.join(args.out_dir, f'oof_{name}.csv'), index=False)

    delta = delta_table(oof_df, args.n_boot)
    delta_path = os.path.join(args.out_dir, 'delta_auroc.csv')
    delta.to_csv(delta_path, index=False)

    print("\n=== Summary (mean ± SD over folds × seeds) ===")
    print(summary[['experiment', 'scenario', 'n_runs', 'AUROC (mean ± SD)',
                   'Sensitivity (mean ± SD)', 'Specificity (mean ± SD)']].to_string(index=False))
    if not delta.empty:
        print("\n=== Paired bootstrap ΔAUROC (pooled out-of-fold) ===")
        print(delta[['seed', 'scenario', 'A', 'B', 'delta_AUROC', 'CI_lo', 'CI_hi', 'p_boot']]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved {rows_path}\nSaved {summary_path}\nSaved {delta_path}\n"
          f"Saved out-of-fold predictions → {args.out_dir}/oof_<experiment>.csv")


if __name__ == '__main__':
    main()
