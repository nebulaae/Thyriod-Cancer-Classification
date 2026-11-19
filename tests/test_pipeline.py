"""
System tests: the full pipeline on a tiny synthetic dataset (CPU, no downloads).

audit (dedup + group folds) → train_fold → evaluate (calibration, Youden on the
inner split) → deletion test → train_final (deploy block) → reliability helpers.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import main
import model as model_mod
from dedup import find_duplicate_groups, label_conflict_groups, dhash, hamming
from reliability import expected_calibration_error, fit_temperature, sigmoid

IMG = 64
TINY = dict(dim=32, heads=4, depth=1, img_size=IMG, batch_size=2, epochs=1, warmup_epochs=1,
            patience=1, amp=False, num_workers=0, n_folds=3, deletion_test=True, inner_val_frac=0.34)


def _noise_image(path, seed, color=False):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(IMG, IMG, 3 if color else 1), dtype=np.uint8)
    Image.fromarray(arr.squeeze()).convert('RGB').save(path)


def _make_dataset(root, n_per_class=12, duplicate=True, conflict=True):
    """dataset1/4 cytology (.tif patches), dataset2/3 ultrasound, ids 1..n per class."""
    dirs = {k: os.path.join(root, d) for k, d in
            [('b_path', 'dataset1'), ('b_us', 'dataset2'), ('p_us', 'dataset3'), ('p_path', 'dataset4')]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    seed = 0
    for cls, us_dir, p_dir in [('b', dirs['b_us'], dirs['b_path']), ('p', dirs['p_us'], dirs['p_path'])]:
        for pid in range(1, n_per_class + 1):
            seed += 1
            _noise_image(os.path.join(us_dir, f'{pid}.jpg'.replace('.jpg', '.png')), seed)
            for k in range(1, 2 + pid % 3):
                seed += 1
                _noise_image(os.path.join(p_dir, f'{pid}_{k:03d}.tif'), seed, color=True)
    if duplicate:   # benign 2 reuses benign 1's ultrasound → same group, same label
        Image.open(os.path.join(dirs['b_us'], '1.png')).save(os.path.join(dirs['b_us'], '2.png'))
    if conflict:    # PTC 3 reuses benign 3's ultrasound → label conflict, both dropped
        Image.open(os.path.join(dirs['b_us'], '3.png')).save(os.path.join(dirs['p_us'], '3.png'))
    return root


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ('checkpoints', 'graphs', 'results'):
        os.makedirs(d, exist_ok=True)
    # No pretrained weights in tests
    monkeypatch.setattr(main, 'MultimodalTransformer',
                        lambda cfg: model_mod.MultimodalTransformer(cfg, pretrained=False))
    data = _make_dataset(str(tmp_path / 'data'))
    cfg = {**main.CFG, **TINY, 'data_root': data}
    return cfg


def test_dhash_detects_copies_and_separates_noise(tmp_path):
    a, b, c = (str(tmp_path / f'{n}.png') for n in 'abc')
    _noise_image(a, 1); _noise_image(c, 2)
    Image.open(a).save(b)
    assert hamming(dhash(a), dhash(b)) == 0
    assert hamming(dhash(a), dhash(c)) > 8


def test_audit_groups_duplicates_and_drops_label_conflicts(workspace):
    df = main.build_metadata(workspace)
    ids = set(df.patient_id)
    assert 'benign_3' not in ids and 'ptc_3' not in ids           # conflict dropped
    g = df.set_index('patient_id')['group']
    assert g['benign_1'] == g['benign_2']                          # duplicate grouped
    # a group never spans two folds
    assert (df.groupby('group')['fold'].nunique() == 1).all()
    assert os.path.exists(os.path.join('graphs', 'duplicate_pairs.csv'))


def test_inner_split_respects_groups(workspace):
    df = main.build_metadata(workspace)
    for fold in range(workspace['n_folds']):
        tr, sel, te = main.split_fold(df, workspace, fold)
        assert set(tr.group).isdisjoint(sel.group)
        assert set(tr.patient_id).isdisjoint(te.patient_id)


def test_end_to_end_train_evaluate_explain_deploy(workspace):
    cfg = workspace
    df = main.build_metadata(cfg)
    model, ckpt_path, sel_df = main.train_fold(df, cfg, 0)
    assert os.path.exists(ckpt_path) and len(model.history['train_loss']) == 1
    _, _, test_df = main.split_fold(df, cfg, 0)
    metrics, probs_cal, y, pids = main.evaluate(model, test_df, cfg, 0, ckpt_path, sel_df=sel_df)
    assert metrics['Threshold source'] == 'inner_val'
    assert metrics['Temperature'] > 0 and np.isfinite(metrics['ECE (cal)'])
    assert len(probs_cal) == len(test_df) and np.all((probs_cal >= 0) & (probs_cal <= 1))
    summary, rows = main.deletion_test(model, test_df, cfg, threshold=0.0)
    assert 'Deletion win rate' in summary
    final = main.train_final(df, cfg)
    ckpt = torch.load(final, map_location='cpu', weights_only=False)
    assert {'temperature', 'threshold', 'n_train_patients'} <= set(ckpt['deploy'])


def test_cli_runs_one_fold(workspace, monkeypatch):
    monkeypatch.setattr(main, 'CFG', workspace)
    main.run(['--data_root', workspace['data_root'], '--folds', '0', '--epochs', '1'])
    assert os.path.exists(os.path.join('graphs', 'metrics_fold0.csv'))
    assert os.path.exists(os.path.join('results', 'oof_predictions.csv'))


def test_ece_and_temperature():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.02            # calibrated by construction
    z = np.log(p / (1 - p))
    t = fit_temperature(3 * z, y)                              # over-confident logits
    assert 2.5 < t < 3.5
    assert expected_calibration_error(y, sigmoid(3 * z / t)) < 0.02
