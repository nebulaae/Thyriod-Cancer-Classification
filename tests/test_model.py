"""
CPU smoke tests for model.py (no datasets or pretrained weights needed).
Run from the repo root:  python -m pytest tests/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import pytest

from model import MultimodalTransformer, count_parameters, sample_modality_dropout

CFG = dict(dim=64, heads=4, depth=1, dropout=0.4, trainable_from_feature=5)
IMG = 64   # ConvNeXt downsamples 32x → 2x2 final map, enough for a smoke test


def _build(mode='joint'):
    torch.manual_seed(0)
    return MultimodalTransformer({**CFG, 'mode': mode}, pretrained=False)


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


# ── Missing-modality support ─────────────────────────────────────
NONE = torch.zeros(2, dtype=torch.bool)
ALL  = torch.ones(2, dtype=torch.bool)


@pytest.mark.parametrize('train', [False, True])
@pytest.mark.parametrize('us_present,cyto_present', [
    (ALL, NONE),                                 # cytology missing for every sample
    (NONE, ALL),                                 # US missing for every sample
    (torch.tensor([True, False]), torch.tensor([False, True])),   # mixed
])
def test_missing_modality_outputs_are_finite(us_present, cyto_present, train):
    model = _build().train(train)
    us, path, mask = _batch()
    with torch.set_grad_enabled(train):
        logit = model(us, path, mask, us_present=us_present, cyto_present=cyto_present)
    assert logit.shape == (2, 1)
    assert torch.isfinite(logit).all()


def test_default_flags_match_all_present():
    model = _build().eval()
    us, path, mask = _batch()
    with torch.no_grad():
        a = model(us, path, mask)
        b = model(us, path, mask, us_present=ALL, cyto_present=ALL)
    assert torch.equal(a, b)


def test_missing_modality_input_is_ignored():
    model = _build().eval()
    us, path, mask = _batch()
    with torch.no_grad():
        a = model(us, path, mask, cyto_present=NONE)
        b = model(us, torch.randn_like(path), mask, cyto_present=NONE)
        c = model(us, path, mask, us_present=NONE)
        d = model(torch.randn_like(us), path, mask, us_present=NONE)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(c, d, atol=1e-5)


def test_both_modalities_missing_is_rejected():
    model = _build().eval()
    us, path, mask = _batch()
    with pytest.raises(AssertionError):
        model(us, path, mask, us_present=torch.tensor([True, False]),
              cyto_present=torch.tensor([True, False]))


def test_modality_dropout_never_drops_both():
    torch.manual_seed(0)
    us_p, cy_p = sample_modality_dropout(20000, p_drop_cyto=0.3, p_drop_us=0.15)
    assert (us_p | cy_p).all()
    assert abs((~cy_p).float().mean().item() - 0.3) < 0.02
    assert abs((~us_p).float().mean().item() - 0.15) < 0.02
    us_p, cy_p = sample_modality_dropout(1000, p_drop_cyto=0.5, p_drop_us=0.5)   # extreme case
    assert (us_p | cy_p).all()


def test_us_only_mode():
    model = _build('us_only').eval()
    assert not hasattr(model, 'path_enc') and not hasattr(model, 'path_transformer')
    assert not any(k.startswith('path_') for k in model.state_dict())
    us, path, mask = _batch()
    with torch.no_grad():
        a = model(us, path, mask)
        b = model(us, None, None)
    assert a.shape == (2, 1) and torch.isfinite(a).all()
    assert torch.allclose(a, b, atol=1e-6)


def test_cyto_only_mode():
    model = _build('cyto_only').eval()
    assert not hasattr(model, 'us_enc') and not hasattr(model, 'us_proj')
    assert not any(k.startswith('us_') for k in model.state_dict())
    us, path, mask = _batch()
    with torch.no_grad():
        a = model(us, path, mask)
        b = model(None, path, mask)
    assert a.shape == (2, 1) and torch.isfinite(a).all()
    assert torch.allclose(a, b, atol=1e-6)
