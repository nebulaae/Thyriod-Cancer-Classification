"""
CPU smoke tests for model.py (no datasets or pretrained weights needed).
Run from the repo root:  python -m pytest tests/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import MultimodalTransformer, count_parameters

CFG = dict(dim=64, heads=4, depth=1, dropout=0.4, trainable_from_feature=5)
IMG = 64   # ConvNeXt downsamples 32x → 2x2 final map, enough for a smoke test


def _build():
    torch.manual_seed(0)
    return MultimodalTransformer(CFG, pretrained=False)


def _batch():
    us   = torch.randn(2, 3, IMG, IMG)
    path = torch.randn(2, 3, 3, IMG, IMG)
    mask = torch.tensor([[True, True, True],
                         [True, True, False]])   # second patient has one padded patch
    path[1, 2] = 0
    return us, path, mask


def test_forward_with_padded_mask():
    model = _build().eval()
    us, path, mask = _batch()
    with torch.no_grad():
        logit, feats = model(us, path, mask, return_feats=True)
    assert logit.shape == (2, 1)
    assert torch.isfinite(logit).all()
    assert feats['fused'].shape == (2, 3 + 2, CFG['dim'])


def test_padded_patch_does_not_change_output():
    model = _build().eval()
    us, path, mask = _batch()
    path2 = path.clone()
    path2[1, 2] = torch.randn(3, IMG, IMG)
    with torch.no_grad():
        a = model(us, path, mask)
        b = model(us, path2, mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_trainable_parameter_selection():
    model = _build()
    for enc in [model.us_enc, model.path_enc]:
        features = enc.backbone[0]
        for i, stage in enumerate(features):
            for name, p in stage.named_parameters():
                assert p.requires_grad == (i >= CFG['trainable_from_feature']), f"features.{i}.{name}"
        # final LayerNorm after the backbone stays trainable
        assert all(p.requires_grad for p in enc.backbone[2].parameters())

    names = dict(model.us_enc.backbone.named_parameters())
    # Regression: the old substring rule left `block.5` (2nd pointwise Linear) trainable in early stages
    assert not names['0.1.0.block.5.weight'].requires_grad
    assert names['0.5.0.block.5.weight'].requires_grad

    # Fusion / projection / head are always trainable
    for mod in [model.us_proj, model.path_proj, model.path_transformer, model.fusion, model.head]:
        assert all(p.requires_grad for p in mod.parameters())

    n_train, n_total = count_parameters(model)
    assert 0 < n_train < n_total


def test_state_dict_keys_match_checkpoint_layout():
    keys = set(_build().state_dict().keys())
    for k in ['us_enc.backbone.0.0.0.weight', 'path_enc.backbone.0.7.2.block.5.weight',
              'us_enc.backbone.2.weight', 'cls_token', 'head.5.weight']:
        assert k in keys, k
