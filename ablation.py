"""
ablation.py — Scientific Research & Communication course, SO1 / RQ1 and SO2 / RQ2.

Overfitting-control ablation over the same patient-level outer folds:

  A0_baseline   original prototype: fully unfrozen ConvNeXt-Tiny, dropout 0.1,
                weight decay 1e-4, lr 1e-4 (constant), 2-layer transformer at
                dim 768, no label smoothing, no gradient clipping, last epoch kept
  A1_freeze     A0 + early backbone stages frozen (features[5:] trainable)
  A2_regularise A1 + dropout 0.4, weight decay 1e-3, label smoothing 0.05,
                1-layer transformer at dim 512
  A3_full       A2 + lr 3e-5 with warm-up + cosine, gradient clipping 1.0,
                early stopping on inner-validation AUROC  (= main.py CFG)

Outputs (results/):
  ablation_runs.csv       one row per configuration × fold
  ablation_summary.csv    mean ± SD per configuration (AUROC, loss gap, …)
  ablation.png            AUROC and loss-gap bars with fold SD

Reproducibility check (SO2): --repro re-runs A3 on the given folds twice with
the same seed and reports |ΔAUROC| per fold (target ≤ 0.005).

Usage:
  python ablation.py --data_root D:/data/thyroid --folds 0 1 2 3 4
  python ablation.py --data_root D:/data/thyroid --configs A3_full --repro --folds 0
"""

import os, json, time, argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score

from main import (CFG, device, build_metadata, split_fold, train_fold, predict_logits)

CONFIGS = {
    'A0_baseline': dict(trainable_from_feature=0, dropout=0.1, weight_decay=1e-4, lr=1e-4,
                        scheduler='constant', depth=2, dim=768, label_smooth=0.0,
                        grad_clip=None, early_stopping=False),
    'A1_freeze': dict(trainable_from_feature=5, dropout=0.1, weight_decay=1e-4, lr=1e-4,
                      scheduler='constant', depth=2, dim=768, label_smooth=0.0,
                      grad_clip=None, early_stopping=False),
    'A2_regularise': dict(trainable_from_feature=5, dropout=0.4, weight_decay=1e-3, lr=1e-4,
                          scheduler='constant', depth=1, dim=512, label_smooth=0.05,
                          grad_clip=None, early_stopping=False),
    'A3_full': dict(),   # main.py CFG as is
}


def run_one(df, cfg, name, fold, tag=''):
    t0 = time.time()
    model, ckpt_path, sel_df = train_fold(df, cfg, fold, ckpt_name=f'ablation_{name}_fold{fold}{tag}.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    _, _, test_df = split_fold(df, cfg, fold)
    logits, y, _ = predict_logits(model, test_df, cfg)
    h = model.history
    b = h['best_epoch']
    row = {
        'config': name, 'fold': fold, 'tag': tag,
        'test_AUROC': roc_auc_score(y, logits),
        'best_epoch': b + 1, 'epochs_run': len(h['train_loss']),
        'train_loss': h['train_loss'][b], 'val_loss': h['val_loss'][b],
        'loss_gap': h['val_loss'][b] - h['train_loss'][b],
        'final_train_loss': h['train_loss'][-1], 'final_val_loss': h['val_loss'][-1],
        'sel_AUROC': h['val_auc'][b],
        'minutes': (time.time() - t0) / 60,
    }
    print(f"[{name} fold {fold}{tag}] test AUROC={row['test_AUROC']:.4f} loss gap={row['loss_gap']:.4f} "
          f"({row['minutes']:.1f} min)")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return row


def summarize(runs, out_dir):
    g = runs.groupby('config', sort=False)
    summary = pd.DataFrame({
        'n_folds': g.size(),
        'AUROC_mean': g['test_AUROC'].mean(), 'AUROC_sd': g['test_AUROC'].std(ddof=1),
        'loss_gap_mean': g['loss_gap'].mean(), 'loss_gap_sd': g['loss_gap'].std(ddof=1),
        'best_epoch_mean': g['best_epoch'].mean(), 'minutes_total': g['minutes'].sum(),
    }).reset_index()
    if 'A0_baseline' in set(summary.config):
        base = summary.loc[summary.config == 'A0_baseline', 'loss_gap_mean'].item()
        summary['loss_gap_change_vs_A0'] = (summary['loss_gap_mean'] - base) / abs(base) if base else np.nan
    summary.to_csv(os.path.join(out_dir, 'ablation_summary.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(summary))
    axes[0].bar(x, summary['AUROC_mean'], yerr=summary['AUROC_sd'].fillna(0), color='#1B2A4A', capsize=4)
    axes[0].set_xticks(x, summary['config'], rotation=15); axes[0].set_ylim(0.5, 1.0)
    axes[0].axhline(0.95, color='#E86B4C', ls='--', lw=1, label='SO1 target 0.95'); axes[0].legend()
    axes[0].set_title('Test AUROC (mean ± SD over folds)')
    axes[1].bar(x, summary['loss_gap_mean'], yerr=summary['loss_gap_sd'].fillna(0), color='#9C3D7E', capsize=4)
    axes[1].set_xticks(x, summary['config'], rotation=15)
    axes[1].set_title('Val − train loss at selected epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ablation.png'), dpi=150)
    plt.close()
    print(summary.to_string(index=False))
    return summary


def main():
    ap = argparse.ArgumentParser(description='Overfitting-control ablation A0–A3 (SO1) and re-run check (SO2)')
    ap.add_argument('--data_root', default=CFG['data_root'])
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    ap.add_argument('--folds', nargs='+', type=int, default=list(range(CFG['n_folds'])))
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    ap.add_argument('--repro', action='store_true', help='Run each fold twice with the same seed (SO2)')
    ap.add_argument('--out_dir', default=CFG['results_dir'])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = {**CFG, 'data_root': args.data_root, 'epochs': args.epochs, 'deletion_test': False}
    df = build_metadata(base)   # identical outer folds for every configuration

    runs_path = os.path.join(args.out_dir, 'repro_runs.csv' if args.repro else 'ablation_runs.csv')
    done = pd.read_csv(runs_path) if os.path.exists(runs_path) else pd.DataFrame()
    rows = done.to_dict('records')
    for name in args.configs:
        cfg = {**base, **CONFIGS[name]}
        for fold in args.folds:
            for tag in (['_runA', '_runB'] if args.repro else ['']):
                if not done.empty and ((done.config == name) & (done.fold == fold) &
                                       (done.tag.fillna('') == tag)).any():
                    print(f"skip {name} fold {fold}{tag} (already in {runs_path})")
                    continue
                rows.append(run_one(df, cfg, name, fold, tag))
                pd.DataFrame(rows).to_csv(runs_path, index=False)   # resume-safe

    runs = pd.DataFrame(rows)
    if args.repro:
        wide = runs.pivot_table(index=['config', 'fold'], columns='tag', values='test_AUROC').reset_index()
        wide['abs_delta_AUROC'] = (wide['_runA'] - wide['_runB']).abs()
        wide.to_csv(os.path.join(args.out_dir, 'repro_summary.csv'), index=False)
        print(wide.to_string(index=False))
        ok = bool((wide['abs_delta_AUROC'] <= 0.005).all())
        print(f"SO2 re-run target (|ΔAUROC| ≤ 0.005): {'MET' if ok else 'NOT MET'}")
    else:
        summarize(runs[runs.config.isin(args.configs)], args.out_dir)


if __name__ == '__main__':
    main()
