"""
Shared model definition for training (main.py) and inference (model_inference.py).
The architecture lives here once so both scripts build identical state_dicts.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# convnext_tiny `features` layout: 0 stem, 1 stage1, 2 down, 3 stage2,
# 4 down, 5 stage3, 6 down, 7 stage4
DEFAULT_TRAINABLE_FROM = 5

# joint: US + cytology fusion; us_only / cyto_only: single-modality baselines
# that do not build the other modality's branch.
MODES = ('joint', 'us_only', 'cyto_only')


def get_eval_transform(img_size=224):
    """Deterministic preprocessing used for validation, evaluation and inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def sample_modality_dropout(batch_size, p_drop_cyto=0.0, p_drop_us=0.0, device=None):
    """Per-sample presence flags for training-time modality dropout.
    One uniform draw per sample: u < p_drop_cyto drops cytology,
    p_drop_cyto <= u < p_drop_cyto + p_drop_us drops US, otherwise keep both.
    So at most one modality is ever dropped. Returns (us_present, cyto_present)."""
    assert p_drop_cyto >= 0 and p_drop_us >= 0 and p_drop_cyto + p_drop_us <= 1
    u = torch.rand(batch_size, device=device)
    cyto_present = u >= p_drop_cyto
    us_present   = (u < p_drop_cyto) | (u >= p_drop_cyto + p_drop_us)
    return us_present, cyto_present


class PatchEncoder(nn.Module):
    """ConvNeXt-Tiny backbone, output: [B, feature_dim]"""
    def __init__(self, pretrained=True):
        super().__init__()
        m = models.convnext_tiny(weights='DEFAULT' if pretrained else None)
        self.feat_dim = m.classifier[2].in_features   # 768
        # Keep LayerNorm + Flatten, remove final Linear
        self.backbone = nn.Sequential(m.features, m.avgpool,
                                       m.classifier[0], m.classifier[1])

    def freeze_until(self, trainable_from):
        """Freeze features[:trainable_from]; features[trainable_from:] and the
        final LayerNorm stay trainable."""
        features = self.backbone[0]
        for i, stage in enumerate(features):
            for p in stage.parameters():
                p.requires_grad = i >= trainable_from

    def forward(self, x):
        return self.backbone(x)   # [B, 768]


class MultimodalTransformer(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        dim     = cfg['dim']
        heads   = cfg['heads']
        depth   = cfg['depth']
        dropout = cfg['dropout']

        self.mode = cfg.get('mode', 'joint')
        assert self.mode in MODES, f"mode must be one of {MODES}, got {self.mode!r}"
        self.has_us   = self.mode in ('joint', 'us_only')
        self.has_cyto = self.mode in ('joint', 'cyto_only')
        self.dim = dim

        # Creation order is kept as before so joint-mode init/state_dict are unchanged
        if self.has_us:
            self.us_enc   = PatchEncoder(pretrained)
        if self.has_cyto:
            self.path_enc = PatchEncoder(pretrained)

        # Freeze early backbone stages (default: fine-tune stage 3 onwards)
        trainable_from = cfg.get('trainable_from_feature', DEFAULT_TRAINABLE_FROM)
        for enc in self.encoders():
            enc.freeze_until(trainable_from)

        if self.has_us:
            self.us_proj   = nn.Sequential(nn.Linear(768, dim), nn.LayerNorm(dim), nn.Dropout(dropout))
        if self.has_cyto:
            self.path_proj = nn.Sequential(nn.Linear(768, dim), nn.LayerNorm(dim), nn.Dropout(dropout))

            enc_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim*2,
                dropout=dropout, batch_first=True, norm_first=True)
            self.path_transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        fuse_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(fuse_layer, num_layers=1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def encoders(self):
        return [enc for enc, on in [(getattr(self, 'us_enc', None), self.has_us),
                                    (getattr(self, 'path_enc', None), self.has_cyto)] if on]

    @staticmethod
    def _presence(flags, B, dev, available):
        if not available:
            return torch.zeros(B, dtype=torch.bool, device=dev)
        if flags is None:
            return torch.ones(B, dtype=torch.bool, device=dev)
        return flags.to(device=dev, dtype=torch.bool)

    def forward(self, us, path, mask, return_feats=False, us_present=None, cyto_present=None):
        """
        us:   [B, 3, H, W]      (may be None in cyto_only mode)
        path: [B, N, 3, H, W]   (may be None in us_only mode)
        mask: [B, N] True = real patch
        us_present / cyto_present: optional [B] bool, False = modality missing
            for that sample (default: all present). At least one must be present.
        """
        ref = us if us is not None else path
        B, dev = ref.shape[0], ref.device
        us_present   = self._presence(us_present, B, dev, self.has_us)
        cyto_present = self._presence(cyto_present, B, dev, self.has_cyto)
        assert bool((us_present | cyto_present).all()), "each sample needs at least one modality"

        if self.has_us:
            us_f = self.us_proj(self.us_enc(us)).unsqueeze(1)            # [B,1,dim]
        else:
            us_f = torch.zeros(B, 1, self.dim, device=dev)

        if self.has_cyto:
            _, N, C, H, W = path.shape
            path_f = self.path_enc(path.reshape(B*N, C, H, W))          # [B*N, 768]
            path_f = self.path_proj(path_f).view(B, N, -1)               # [B,N,dim]
            real = mask & cyto_present[:, None]                          # patches the fusion may see
            if bool(cyto_present.all()):
                pt_mask = real
            else:
                # Samples without cytology: zero their tokens and keep one dummy key
                # visible in the patch transformer (an all-masked row gives NaN).
                # The dummy is masked out again in the fusion below.
                path_f = path_f * cyto_present[:, None, None].to(path_f.dtype)
                pt_mask = real.clone()
                pt_mask[:, 0] |= ~cyto_present
            path_f = self.path_transformer(path_f, src_key_padding_mask=~pt_mask)
        else:
            path_f = torch.zeros(B, 1, self.dim, device=dev)             # one dummy token, always masked
            real = torch.zeros(B, 1, dtype=torch.bool, device=dev)

        cls  = self.cls_token.expand(B, -1, -1)
        seq  = torch.cat([cls, us_f, path_f], dim=1)                     # [B, N+2, dim]

        # CLS (index 0) is never masked, so no fusion row is fully masked
        fuse_mask = torch.zeros((B, seq.shape[1]), dtype=torch.bool, device=dev)
        fuse_mask[:, 1]  = ~us_present
        fuse_mask[:, 2:] = ~real
        fused = self.fusion(seq, src_key_padding_mask=fuse_mask)

        logit = self.head(fused[:, 0])                                   # [B,1]

        if return_feats:
            return logit, {'us_f': us_f, 'path_f': path_f, 'fused': fused,
                           'us_present': us_present, 'cyto_present': cyto_present}
        return logit


def count_parameters(model):
    """Return (trainable, total) parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
