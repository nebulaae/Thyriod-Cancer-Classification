"""
model_inference.py
------------------
Manual inference tool. Load a trained checkpoint and predict on
individual patients or a batch CSV.

Usage examples
--------------
# Single patient
python model_inference.py \
    --us_path dataset2/12.jpg \
    --patch_dir dataset1 \
    --patient_id 12 \
    --ckpt checkpoints/model_fold0_best.pth

# Batch mode (CSV with columns: patient_id, us_path, patch_paths)
python model_inference.py \
    --csv metadata.csv \
    --ckpt checkpoints/model_fold0_best.pth \
    --output inference_results.csv
"""

import os, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import pandas as pd
import matplotlib.pyplot as plt


# ── Re-define model (must match main.py exactly) ────────────────
class PatchEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = models.convnext_tiny(weights=None)
        self.feat_dim = m.classifier[2].in_features
        self.backbone = nn.Sequential(m.features, m.avgpool,
                                       m.classifier[0], m.classifier[1])
    def forward(self, x):
        return self.backbone(x)


class MultimodalTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim     = cfg['dim']
        heads   = cfg['heads']
        depth   = cfg['depth']
        dropout = cfg['dropout']

        self.us_enc   = PatchEncoder()
        self.path_enc = PatchEncoder()

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
        B, N, C, H, W = path.shape
        us_f   = self.us_proj(self.us_enc(us)).unsqueeze(1)
        path_f = self.path_enc(path.view(B*N, C, H, W))
        path_f = self.path_proj(path_f).view(B, N, -1)
        path_f = self.path_transformer(path_f, src_key_padding_mask=~mask)
        cls    = self.cls_token.expand(B, -1, -1)
        seq    = torch.cat([cls, us_f, path_f], dim=1)
        fuse_mask = torch.zeros((B, N+2), dtype=torch.bool, device=us.device)
        fuse_mask[:, 2:] = ~mask
        fused  = self.fusion(seq, src_key_padding_mask=fuse_mask)
        logit  = self.head(fused[:, 0])
        if return_feats:
            return logit, {'us_f': us_f, 'path_f': path_f, 'fused': fused}
        return logit


# ── Preprocessing ────────────────────────────────────────────────
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def denorm(tensor):
    t = tensor.clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return torch.clamp(t * std + mean, 0, 1)


# ── Load model from checkpoint ───────────────────────────────────
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('cfg', {
        'dim': 512, 'heads': 8, 'depth': 1, 'dropout': 0.4
    })
    model = MultimodalTransformer(cfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded: {ckpt_path}")
    print(f"  Trained for {ckpt['epoch']+1} epochs | Val AUROC = {ckpt.get('val_auc', 'n/a')}")
    return model, cfg


# ── Single patient inference ─────────────────────────────────────
def predict_single(model, us_path, patch_paths, device, img_size=224, visualize=True):
    """
    Returns: dict with prob, label, patch_similarities
    """
    tf = get_transform(img_size)

    us_img = tf(Image.open(us_path).convert('RGB')).unsqueeze(0).to(device)

    patches = [tf(Image.open(p).convert('RGB')) for p in patch_paths]
    n_p     = len(patches)
    path_t  = torch.stack(patches).unsqueeze(0).to(device)     # [1, N, 3, H, W]
    mask    = torch.ones(1, n_p, dtype=torch.bool).to(device)

    with torch.no_grad():
        logit, feats = model(us_img, path_t, mask, return_feats=True)
        prob  = torch.sigmoid(logit).item()
        cls_f = feats['fused'][0, 0]
        patch_fs = feats['fused'][0, 2:2+n_p]
        sims  = F.cosine_similarity(cls_f.unsqueeze(0), patch_fs).cpu().numpy()

    result = {
        'prob_ptc':   prob,
        'prediction': 'PTC' if prob > 0.5 else 'Benign',
        'confidence': max(prob, 1 - prob),
        'patch_similarities': sims.tolist(),
        'most_relevant_patch': patch_paths[int(np.argmax(sims))],
    }

    print(f"\n{'='*45}")
    print(f"  Ultrasound : {os.path.basename(us_path)}")
    print(f"  Patches    : {[os.path.basename(p) for p in patch_paths]}")
    print(f"  P(PTC)     : {prob:.4f}")
    print(f"  Prediction : {result['prediction']}  (conf={result['confidence']:.2%})")
    print(f"  Patch sims : {[f'{s:.3f}' for s in sims]}")
    print(f"  Most relevant patch: {os.path.basename(result['most_relevant_patch'])}")
    print(f"{'='*45}")

    if visualize:
        _visualize_single(us_path, patch_paths, sims, prob, result['prediction'])

    return result


def _visualize_single(us_path, patch_paths, sims, prob, pred):
    n_p   = len(patch_paths)
    fig, axes = plt.subplots(1, n_p + 1, figsize=(3*(n_p+1), 4))
    if n_p + 1 == 1: axes = [axes]

    axes[0].imshow(Image.open(us_path).convert('RGB'))
    axes[0].set_title(f"Ultrasound\nP(PTC)={prob:.3f}\n→ {pred}", fontsize=9)
    axes[0].axis('off')

    norm_sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
    for j, (p_path, sim, ns) in enumerate(zip(patch_paths, sims, norm_sims)):
        axes[j+1].imshow(Image.open(p_path).convert('RGB'))
        color = plt.cm.RdYlGn(ns)
        axes[j+1].set_title(f"Patch {j+1}\nSim={sim:.3f}", fontsize=9,
                              color=color[:3])
        axes[j+1].axis('off')
        for spine in axes[j+1].spines.values():
            spine.set_edgecolor(color[:3]); spine.set_linewidth(3)

    plt.suptitle(f"Prediction: {pred}  |  P(PTC) = {prob:.4f}", fontsize=12,
                  color='#E86B4C' if pred == 'PTC' else '#4C9BE8')
    plt.tight_layout()
    out = f"inference_{os.path.splitext(os.path.basename(us_path))[0]}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved → {out}")


# ── Batch inference ──────────────────────────────────────────────
def predict_batch(model, csv_path, device, img_size=224, output_csv='inference_results.csv'):
    df = pd.read_csv(csv_path)
    required = {'patient_id', 'us_path', 'patch_paths'}
    assert required.issubset(df.columns), f"CSV must have columns: {required}"

    results = []
    for _, row in df.iterrows():
        patches = row['patch_paths'].split(';')
        r = predict_single(model, row['us_path'], patches, device,
                            img_size=img_size, visualize=False)
        r['patient_id'] = row['patient_id']
        if 'label' in row:
            r['true_label'] = int(row['label'])
        results.append(r)

    out_df = pd.DataFrame(results)

    # If ground truth available → summary metrics
    if 'true_label' in out_df.columns:
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        y_true = out_df['true_label'].values
        y_prob = out_df['prob_ptc'].values
        y_pred = (y_prob > 0.5).astype(int)
        print(f"\n=== Batch Results ===")
        print(f"  AUROC    : {roc_auc_score(y_true, y_prob):.4f}")
        print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"  F1       : {f1_score(y_true, y_pred):.4f}")

    out_df.to_csv(output_csv, index=False)
    print(f"\nSaved batch results → {output_csv}")
    return out_df


# ── CLI ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Thyroid Multimodal Inference')
    parser.add_argument('--ckpt', required=True, help='Checkpoint path (.pth)')
    parser.add_argument('--us_path',    default=None, help='Single: ultrasound image')
    parser.add_argument('--patch_dir',  default=None, help='Single: dir with patient patches')
    parser.add_argument('--patient_id', default=None, help='Single: patient ID (for patch glob)')
    parser.add_argument('--patch_paths',default=None, help='Single: semicolon-separated patch paths')
    parser.add_argument('--csv',        default=None, help='Batch: metadata CSV')
    parser.add_argument('--output',     default='inference_results.csv')
    parser.add_argument('--no_viz',     action='store_true')
    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(args.ckpt, dev)

    if args.csv:
        predict_batch(model, args.csv, dev, output_csv=args.output)

    elif args.us_path:
        if args.patch_paths:
            patches = args.patch_paths.split(';')
        elif args.patch_dir and args.patient_id:
            patches = sorted(glob.glob(os.path.join(args.patch_dir, f"{args.patient_id}_*.tif")))
            if not patches:
                raise FileNotFoundError(f"No patches found for ID={args.patient_id} in {args.patch_dir}")
        else:
            raise ValueError("Provide --patch_paths OR (--patch_dir + --patient_id)")

        predict_single(model, args.us_path, patches, dev, visualize=not args.no_viz)

    else:
        # Interactive demo mode
        print("\n[Demo mode] No --csv or --us_path provided.")
        print("Enter patient ID to predict (or 'q' to quit):")
        df = pd.read_csv('metadata.csv') if os.path.exists('metadata.csv') else None
        if df is None:
            print("metadata.csv not found. Use --us_path or --csv flags.")
            return
        while True:
            pid = input("Patient ID (e.g. benign_5 or ptc_12): ").strip()
            if pid in ('q', 'quit', ''):
                break
            row = df[df.patient_id == pid]
            if row.empty:
                print(f"  Not found: {pid}")
                continue
            row = row.iloc[0]
            patches = row['patch_paths'].split(';')
            predict_single(model, row['us_path'], patches, dev, visualize=True)


if __name__ == '__main__':
    main()