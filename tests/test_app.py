"""
MVP web-app tests: load a tiny checkpoint and run the Gradio callback end to end.
"""

import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip('gradio')

from model import MultimodalTransformer

CFG = dict(dim=32, heads=4, depth=1, dropout=0.1, img_size=64, trainable_from_feature=5)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    torch.manual_seed(0)
    m = MultimodalTransformer(CFG, pretrained=False)
    ckpt = tmp_path / 'medai_final.pth'
    torch.save({'cfg': CFG, 'model_state': m.state_dict(), 'epoch': 0, 'val_auc': 0.5,
                'deploy': {'temperature': 1.5, 'threshold': 0.4, 'trained_at': 'test',
                           'n_train_patients': 10, 'n_selection_patients': 2}}, ckpt)
    monkeypatch.setenv('MEDAI_CKPT', str(ckpt))
    sys.modules.pop('app', None)
    return importlib.import_module('app')


def _img(tmp_path, name, seed):
    arr = np.random.default_rng(seed).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    path = tmp_path / name
    Image.fromarray(arr).save(path)
    return str(path)


def test_analyse_returns_calibrated_probability_and_ranked_gallery(app, tmp_path):
    us = Image.open(_img(tmp_path, 'us.png', 1))
    patches = [_img(tmp_path, f'p{i}.tif', 10 + i) for i in range(3)]
    label, md, gallery = app.analyse(us, patches)
    assert set(label) == {'PTC', 'Benign'}
    assert abs(label['PTC'] + label['Benign'] - 1) < 1e-6
    assert 'Decision' in md and 'Research prototype' in md
    assert len(gallery) == 3
    rel = [float(c.rsplit(' ', 1)[-1]) for _, c in gallery]
    assert rel == sorted(rel, reverse=True)
    assert app.THRESHOLD == 0.4 and app.TEMPERATURE == 1.5


def test_analyse_rejects_missing_inputs(app, tmp_path):
    import gradio as gr
    with pytest.raises(gr.Error):
        app.analyse(None, [_img(tmp_path, 'p.tif', 3)])
    with pytest.raises(gr.Error):
        app.analyse(Image.open(_img(tmp_path, 'us.png', 4)), [])


def test_too_many_patches_are_truncated(app, tmp_path):
    us = Image.open(_img(tmp_path, 'us.png', 5))
    patches = [_img(tmp_path, f'p{i}.png', 20 + i) for i in range(9)]
    _, md, gallery = app.analyse(us, patches)
    assert len(gallery) == 7 and 'first 7 patches' in md


def test_ui_builds(app):
    assert app.build_ui() is not None
