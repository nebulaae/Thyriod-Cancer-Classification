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


def get_eval_transform(img_size=224):
    """Deterministic preprocessing used for validation, evaluation and inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


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

        self.us_enc   = PatchEncoder(pretrained)
        self.path_enc = PatchEncoder(pretrained)

        # Freeze early backbone stages (default: fine-tune stage 3 onwards)
        trainable_from = cfg.get('trainable_from_feature', DEFAULT_TRAINABLE_FROM)
        for enc in [self.us_enc, self.path_enc]:
            enc.freeze_until(trainable_from)

        self.us_proj   = nn.Sequential(nn.Linear(768, dim), nn.LayerNorm(dim), nn.Dropout(dropout))
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

    def forward(self, us, path, mask, return_feats=False):
        """
        us:   [B, 3, H, W]
        path: [B, N, 3, H, W]
        mask: [B, N] True = real patch
        """
        B, N, C, H, W = path.shape

        us_f   = self.us_proj(self.us_enc(us)).unsqueeze(1)          # [B,1,dim]
        path_f = self.path_enc(path.view(B*N, C, H, W))              # [B*N, 768]
        path_f = self.path_proj(path_f).view(B, N, -1)               # [B,N,dim]

        path_f = self.path_transformer(path_f, src_key_padding_mask=~mask)

        cls  = self.cls_token.expand(B, -1, -1)
        seq  = torch.cat([cls, us_f, path_f], dim=1)                 # [B, N+2, dim]

        fuse_mask = torch.zeros((B, N+2), dtype=torch.bool, device=us.device)
        fuse_mask[:, 2:] = ~mask
        fused = self.fusion(seq, src_key_padding_mask=fuse_mask)

        logit = self.head(fused[:, 0])                               # [B,1]

        if return_feats:
            return logit, {'us_f': us_f, 'path_f': path_f, 'fused': fused}
        return logit


def count_parameters(model):
    """Return (trainable, total) parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
