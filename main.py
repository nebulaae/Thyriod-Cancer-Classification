"""
Multimodal Thyroid PTC vs Benign Classification
Unified Pipeline: Audit → Split → Train → Evaluate → Explain
"""

import os, glob, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import (MultimodalTransformer, count_parameters, get_eval_transform,
                   IMAGENET_MEAN, IMAGENET_STD)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.calibration import calibration_curve

# ──────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────
CFG = dict(
    seed        = 42,
    n_folds     = 5,
    fold_id     = 0,          # which fold to run as main experiment
    run_all_folds = False,    # True → train/evaluate every fold + mean ± SD summary
    epochs      = 10,
    warmup_epochs = 5,        # linear warmup, then cosine decay
    lr          = 3e-5,       # lower LR → less overfitting
    weight_decay= 1e-3,
    batch_size  = 2,
    img_size    = 224,
    dim         = 512,        # reduced dim → less overfitting
    depth       = 1,          # 1 transformer layer → less overfitting
    heads       = 8,
    dropout     = 0.4,        # heavy dropout
    label_smooth= 0.05,       # label smoothing
    patience    = 5,          # early stopping patience (must be < epochs to take effect)
    trainable_from_feature = 5,  # convnext features[5:] trainable (stage 3, down, stage 4)
    max_patches = 7,
    checkpoint_dir = 'checkpoints',
    graphs_dir     = 'graphs',
    # Dataset dirs
    benign_us_dir   = 'dataset2',
    benign_path_dir = 'dataset1',
    ptc_us_dir      = 'dataset3',
    ptc_path_dir    = 'dataset4',
)

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
os.makedirs(CFG['checkpoint_dir'], exist_ok=True)
os.makedirs(CFG['graphs_dir'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────
# 1. DATA AUDIT & METADATA
# ──────────────────────────────────────────────
def build_metadata(cfg) -> pd.DataFrame:
    print("\n=== Stage 1: Data Audit & Metadata ===")

    def get_ids(directory, is_pathology=False):
        ids = set()
        for fn in os.listdir(directory):
            ids.add(fn.split('_')[0] if is_pathology else os.path.splitext(fn)[0])
        return ids

    b_us  = get_ids(cfg['benign_us_dir'])
    b_p   = get_ids(cfg['benign_path_dir'], True)
    ptc_us = get_ids(cfg['ptc_us_dir'])
    ptc_p  = get_ids(cfg['ptc_path_dir'], True)

    print(f"Benign US ids: {len(b_us)}, Path ids: {len(b_p)}")
    print(f"PTC    US ids: {len(ptc_us)}, Path ids: {len(ptc_p)}")
    overlap = b_us & ptc_us
    if overlap:
        print(f"WARNING — ID overlap Benign/PTC in US: {overlap}")

    rows = []
    for label, us_dir, p_dir, us_ids, p_ids in [
        (0, cfg['benign_us_dir'], cfg['benign_path_dir'], b_us, b_p),
        (1, cfg['ptc_us_dir'],    cfg['ptc_path_dir'],    ptc_us, ptc_p),
    ]:
        class_name = 'benign' if label == 0 else 'ptc'
        for pid in sorted(us_ids & p_ids, key=int):
            us_files = glob.glob(os.path.join(us_dir, f"{pid}.*"))
            patches  = sorted(glob.glob(os.path.join(p_dir, f"{pid}_*.tif")))
            if not us_files or not patches:
                continue
            rows.append({
                'patient_id': f'{class_name}_{pid}',
                'original_id': pid,
                'label': label,
                'us_path': us_files[0],
                'patch_paths': ';'.join(patches),
                'num_patches': len(patches),
            })

    df = pd.DataFrame(rows)
    print(f"\nPaired patients → Benign: {(df.label==0).sum()}, PTC: {(df.label==1).sum()}, Total: {len(df)}")

    outliers = df[(df.num_patches < 3) | (df.num_patches > 7)]
    if not outliers.empty:
        print(f"WARNING — {len(outliers)} patients outside 3-7 patch range")

    # Stratified K-Fold (patient-level — no leakage)
    skf = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['seed'])
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df['label'])):
        df.loc[val_idx, 'fold'] = fold

    print("Fold distribution:")
    for f in range(cfg['n_folds']):
        fd = df[df.fold == f]
        print(f"  Fold {f}: {len(fd)} pts | Benign={( fd.label==0).sum()} PTC={(fd.label==1).sum()}")

    df.to_csv('metadata.csv', index=False)
    print("Saved metadata.csv")
    _plot_data_audit(df, cfg)
    return df


def _plot_data_audit(df, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Class distribution
    counts = df['label'].value_counts().rename({0: 'Benign', 1: 'PTC'})
    axes[0].bar(counts.index, counts.values, color=['#4C9BE8', '#E86B4C'], edgecolor='k', linewidth=0.5)
    axes[0].set_title('Class Distribution', fontsize=12)
    axes[0].set_ylabel('Patients')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1, str(v), ha='center', fontsize=11)

    # Patch count distribution
    axes[1].hist(df['num_patches'], bins=range(3, 10), color='#4C9BE8', edgecolor='k', linewidth=0.5, align='left')
    axes[1].set_title('Patches per Patient', fontsize=12)
    axes[1].set_xlabel('# patches')
    axes[1].set_ylabel('Patients')
    axes[1].set_xticks(range(3, 8))

    # Fold balance
    fold_data = df.groupby('fold')['label'].value_counts().unstack().fillna(0)
    fold_data.columns = ['Benign', 'PTC']
    fold_data.plot(kind='bar', ax=axes[2], color=['#4C9BE8', '#E86B4C'], edgecolor='k', linewidth=0.5)
    axes[2].set_title('Class Balance per Fold', fontsize=12)
    axes[2].set_xlabel('Fold')
    axes[2].tick_params(axis='x', rotation=0)
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(cfg['graphs_dir'], 'data_audit.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────
# 2. DATASET & COLLATE
# ──────────────────────────────────────────────
class ThyroidDataset(Dataset):
    def __init__(self, df, us_tf=None, path_tf=None):
        self.df = df.reset_index(drop=True)
        self.us_tf   = us_tf
        self.path_tf = path_tf

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        us_img = Image.open(row['us_path']).convert('RGB')
        if self.us_tf:
            us_img = self.us_tf(us_img)

        patches = []
        for p in row['patch_paths'].split(';'):
            img = Image.open(p).convert('RGB')
            if self.path_tf:
                img = self.path_tf(img)
            patches.append(img)

        return {
            'us':       us_img,
            'patches':  torch.stack(patches),
            'n_patches': len(patches),
            'label':    torch.tensor(row['label'], dtype=torch.float32),
            'pid':      row['patient_id'],
        }


def collate_fn(batch):
    us      = torch.stack([x['us'] for x in batch])
    labels  = torch.stack([x['label'] for x in batch])
    max_p   = max(x['n_patches'] for x in batch)

    padded, masks = [], []
    for x in batch:
        p = x['patches']
        n = x['n_patches']
        pad = torch.zeros((max_p - n, *p.shape[1:]), dtype=p.dtype)
        padded.append(torch.cat([p, pad], dim=0))
        mask = torch.zeros(max_p, dtype=torch.bool)
        mask[:n] = True
        masks.append(mask)

    return {
        'us':     us,
        'path':   torch.stack(padded),
        'mask':   torch.stack(masks),
        'labels': labels,
        'pids':   [x['pid'] for x in batch],
    }


def get_transforms(cfg, augment=False):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    sz   = cfg['img_size']
    if augment:
        us_tf = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(), norm,
        ])
        path_tf = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(), norm,
        ])
    else:
        us_tf = path_tf = get_eval_transform(sz)
    return us_tf, path_tf


# ──────────────────────────────────────────────
# 3. MODEL
# ──────────────────────────────────────────────
# PatchEncoder / MultimodalTransformer live in model.py (shared with inference)


# ──────────────────────────────────────────────
# 4. LOSS — Label-smoothed BCE
# ──────────────────────────────────────────────
class LabelSmoothBCE(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.s = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logit, target):
        target_s = target * (1 - self.s) + self.s * 0.5
        return self.bce(logit, target_s.unsqueeze(1))


# ──────────────────────────────────────────────
# 5. TRAINING LOOP (one fold)
# ──────────────────────────────────────────────
def train_fold(df, cfg, fold_id):
    print(f"\n=== Stage 2: Training — Fold {fold_id} ===")
    # Re-seed so each fold is reproducible on its own (same state as a single-fold run)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    train_df = df[df.fold != fold_id]
    val_df   = df[df.fold == fold_id]

    # Compute pos_weight from training set only
    n0 = (train_df.label == 0).sum()
    n1 = (train_df.label == 1).sum()
    pos_w = torch.tensor([n0 / n1], dtype=torch.float32).to(device)
    print(f"Train: {len(train_df)} pts, pos_weight={pos_w.item():.3f}")

    us_tr, path_tr = get_transforms(cfg, augment=True)
    us_vl, path_vl = get_transforms(cfg, augment=False)

    tr_loader = DataLoader(
        ThyroidDataset(train_df, us_tr, path_tr),
        batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=True,
    )
    vl_loader = DataLoader(
        ThyroidDataset(val_df, us_vl, path_vl),
        batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True,
    )

    model     = MultimodalTransformer(cfg).to(device)
    n_train, n_total = count_parameters(model)
    print(f"Params: trainable={n_train:,} / total={n_total:,} ({100*n_train/n_total:.1f}%) "
          f"| backbone features[{cfg['trainable_from_feature']}:] unfrozen")
    criterion = LabelSmoothBCE(smoothing=cfg['label_smooth'], pos_weight=pos_w)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    # Warmup + cosine decay
    warmup_epochs = cfg['warmup_epochs']
    if cfg['patience'] >= cfg['epochs']:
        print(f"WARNING — patience ({cfg['patience']}) >= epochs ({cfg['epochs']}): early stopping can never trigger")
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        prog = (ep - warmup_epochs) / max(1, cfg['epochs'] - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {'train_loss': [], 'val_auc': [], 'val_loss': [], 'lr': []}
    best_auc  = 0.0
    patience_counter = 0
    ckpt_path = os.path.join(cfg['checkpoint_dir'], f'model_fold{fold_id}_best.pth')

    for epoch in range(cfg['epochs']):
        # --- Train ---
        model.train()
        t_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Epoch {epoch+1:02d}/{cfg['epochs']} [Train]", leave=False):
            us    = b['us'].to(device)
            path  = b['path'].to(device)
            mask  = b['mask'].to(device)
            lbls  = b['labels'].to(device)
            optimizer.zero_grad()
            out  = model(us, path, mask)
            loss = criterion(out, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        # --- Validate ---
        model.eval()
        v_loss, probs, targets = 0.0, [], []
        with torch.no_grad():
            for b in vl_loader:
                us   = b['us'].to(device)
                path = b['path'].to(device)
                mask = b['mask'].to(device)
                lbls = b['labels'].to(device)
                out  = model(us, path, mask)
                v_loss += criterion(out, lbls).item()
                probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
                targets.extend(lbls.cpu().numpy())

        probs   = np.array(probs)
        targets = np.array(targets)
        auc     = roc_auc_score(targets, probs)
        avg_tr  = t_loss / len(tr_loader)
        avg_vl  = v_loss / len(vl_loader)
        lr_now  = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_tr)
        history['val_loss'].append(avg_vl)
        history['val_auc'].append(auc)
        history['lr'].append(lr_now)

        scheduler.step()

        print(f"Epoch {epoch+1:02d} | TrainLoss={avg_tr:.4f} ValLoss={avg_vl:.4f} AUROC={auc:.4f} LR={lr_now:.2e}")

        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'cfg': cfg,
                'val_auc': auc,
            }, ckpt_path)
            print(f"  ✓ Saved best model (AUROC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"Early stop at epoch {epoch+1}")
                break

    _plot_training(history, fold_id, cfg)
    return model, ckpt_path, val_df


def _plot_training(history, fold_id, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train', color='#4C9BE8', lw=2)
    axes[0].plot(history['val_loss'],   label='Val',   color='#E86B4C', lw=2)
    axes[0].set_title(f'Loss (Fold {fold_id})', fontsize=12)
    axes[0].set_xlabel('Epoch'); axes[0].legend()

    axes[1].plot(history['val_auc'], color='#2CA02C', lw=2)
    axes[1].set_title(f'Val AUROC (Fold {fold_id})', fontsize=12)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylim(0, 1)
    best_ep = int(np.argmax(history['val_auc']))
    axes[1].axvline(best_ep, color='red', linestyle='--', alpha=0.6, label=f'Best @{best_ep+1}')
    axes[1].legend()

    axes[2].semilogy(history['lr'], color='#9467BD', lw=2)
    axes[2].set_title(f'Learning Rate (Fold {fold_id})', fontsize=12)
    axes[2].set_xlabel('Epoch')

    plt.tight_layout()
    path = os.path.join(cfg['graphs_dir'], f'training_fold{fold_id}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────
# 6. COMPREHENSIVE EVALUATION
# ──────────────────────────────────────────────
def evaluate(model, val_df, cfg, fold_id, ckpt_path=None):
    print(f"\n=== Stage 3: Evaluation — Fold {fold_id} ===")

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded checkpoint (epoch {ckpt['epoch']+1}, AUROC={ckpt['val_auc']:.4f})")

    us_tf, path_tf = get_transforms(cfg, augment=False)
    loader = DataLoader(
        ThyroidDataset(val_df, us_tf, path_tf),
        batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn,
    )

    model.eval()
    probs, targets, pids = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="Evaluating"):
            out = model(b['us'].to(device), b['path'].to(device), b['mask'].to(device))
            probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
            targets.extend(b['labels'].numpy())
            pids.extend(b['pids'])

    probs   = np.array(probs)
    targets = np.array(targets)
    preds   = (probs > 0.5).astype(int)

    # Youden threshold
    fpr_arr, tpr_arr, thresholds = roc_curve(targets, probs)
    youden_idx  = np.argmax(tpr_arr - fpr_arr)
    best_thresh = thresholds[youden_idx]
    preds_y     = (probs > best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    tn_y, fp_y, fn_y, tp_y = confusion_matrix(targets, preds_y).ravel()

    metrics = {
        'AUROC':        roc_auc_score(targets, probs),
        'AUPRC':        average_precision_score(targets, probs),
        'Accuracy':     accuracy_score(targets, preds),
        'Balanced Acc': balanced_accuracy_score(targets, preds),
        'F1':           f1_score(targets, preds),
        'MCC':          matthews_corrcoef(targets, preds),
        'Cohen Kappa':  cohen_kappa_score(targets, preds),
        'Sensitivity':  tp / (tp + fn),
        'Specificity':  tn / (tn + fp),
        'PPV':          tp / (tp + fp) if (tp+fp) > 0 else 0,
        'NPV':          tn / (tn + fn) if (tn+fn) > 0 else 0,
        'Brier Score':  brier_score_loss(targets, probs),
        # Youden threshold
        'F1 (Youden)':  f1_score(targets, preds_y),
        'Sens (Youden)':tp_y / (tp_y + fn_y),
        'Spec (Youden)':tn_y / (tn_y + fp_y),
        'Thresh (Youden)': best_thresh,
    }

    # Bootstrap 95% CI for AUROC
    aucs_boot = []
    rng = np.random.default_rng(cfg['seed'])
    for _ in range(1000):
        idx = rng.choice(len(targets), len(targets), replace=True)
        if len(np.unique(targets[idx])) < 2: continue
        aucs_boot.append(roc_auc_score(targets[idx], probs[idx]))
    ci_lo, ci_hi = np.percentile(aucs_boot, [2.5, 97.5])
    metrics['AUROC 95% CI'] = f"[{ci_lo:.3f}, {ci_hi:.3f}]"

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    _plot_evaluation(targets, probs, preds, pids, metrics, fold_id, cfg)
    return metrics, probs, targets, pids


def _plot_evaluation(targets, probs, preds, pids, metrics, fold_id, cfg):
    # ── Figure 1: ROC + PR + Calibration + Confusion matrix ──
    fig = plt.figure(figsize=(20, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig)

    # ROC
    ax = fig.add_subplot(gs[0])
    fpr, tpr, _ = roc_curve(targets, probs)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC={metrics['AUROC']:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color='darkorange')
    ax.plot([0,1],[0,1],'--',color='navy',lw=1)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve'); ax.legend()

    # PR
    ax = fig.add_subplot(gs[1])
    prec, rec, _ = precision_recall_curve(targets, probs)
    ax.plot(rec, prec, color='green', lw=2, label=f"AP={metrics['AUPRC']:.3f}")
    ax.fill_between(rec, prec, alpha=0.08, color='green')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('PR Curve'); ax.legend()

    # Calibration
    ax = fig.add_subplot(gs[2])
    frac_pos, mean_pred = calibration_curve(targets, probs, n_bins=10)
    ax.plot(mean_pred, frac_pos, 's-', color='#4C9BE8', lw=2, label=f"Brier={metrics['Brier Score']:.3f}")
    ax.plot([0,1],[0,1],'--',color='grey',lw=1)
    ax.set_xlabel('Mean predicted prob'); ax.set_ylabel('Fraction positives')
    ax.set_title('Calibration Curve'); ax.legend()

    # Confusion matrix
    ax = fig.add_subplot(gs[3])
    cm = confusion_matrix(targets, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign','PTC'], yticklabels=['Benign','PTC'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f"Confusion Matrix\nAcc={metrics['Accuracy']:.3f}")

    plt.suptitle(f'Evaluation — Fold {fold_id}', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(cfg['graphs_dir'], f'eval_main_fold{fold_id}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

    # ── Figure 2: Score distributions + per-threshold metrics ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Score distribution by class
    axes[0].hist(probs[targets==0], bins=20, alpha=0.6, color='#4C9BE8', label='Benign', density=True)
    axes[0].hist(probs[targets==1], bins=20, alpha=0.6, color='#E86B4C', label='PTC',    density=True)
    axes[0].axvline(0.5, color='k', linestyle='--', lw=1, label='Thresh=0.5')
    axes[0].axvline(metrics['Thresh (Youden)'], color='red', linestyle='--', lw=1,
                    label=f"Youden={metrics['Thresh (Youden)']:.2f}")
    axes[0].set_xlabel('Predicted probability'); axes[0].set_title('Score Distribution')
    axes[0].legend(fontsize=9)

    # Sensitivity / Specificity vs Threshold
    ths = np.linspace(0, 1, 101)
    sens_list, spec_list, f1_list = [], [], []
    for t in ths:
        p = (probs > t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(targets, p, labels=[0,1]).ravel()
        sens_list.append(tp_t / (tp_t + fn_t + 1e-9))
        spec_list.append(tn_t / (tn_t + fp_t + 1e-9))
        f1_list.append(f1_score(targets, p, zero_division=0))
    axes[1].plot(ths, sens_list, color='#E86B4C', lw=2, label='Sensitivity')
    axes[1].plot(ths, spec_list, color='#4C9BE8', lw=2, label='Specificity')
    axes[1].plot(ths, f1_list,   color='#2CA02C', lw=2, label='F1')
    axes[1].axvline(metrics['Thresh (Youden)'], color='k', linestyle='--', lw=1, label='Youden')
    axes[1].set_xlabel('Threshold'); axes[1].set_title('Metrics vs Threshold'); axes[1].legend()

    # Metrics bar chart
    bar_keys = ['AUROC','AUPRC','Accuracy','Balanced Acc','F1','Sensitivity','Specificity','MCC']
    bar_vals = [metrics[k] for k in bar_keys]
    colors = ['#4C9BE8' if v >= 0.8 else '#E86B4C' if v < 0.6 else '#EFA91A' for v in bar_vals]
    axes[2].barh(bar_keys, bar_vals, color=colors, edgecolor='k', linewidth=0.4)
    axes[2].set_xlim(0, 1.05)
    for i, v in enumerate(bar_vals):
        axes[2].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    axes[2].set_title('Key Metrics Summary')
    axes[2].axvline(0.8, color='grey', linestyle='--', lw=0.8, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(cfg['graphs_dir'], f'eval_detail_fold{fold_id}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

    # ── Figure 3: Patient-level predictions ranked ──
    df_res = pd.DataFrame({'pid': pids, 'true': targets, 'prob': probs})
    df_res = df_res.sort_values('prob', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors_pt = ['#E86B4C' if t == 1 else '#4C9BE8' for t in df_res['true']]
    ax.bar(range(len(df_res)), df_res['prob'], color=colors_pt, edgecolor='none', width=1.0)
    ax.axhline(0.5, color='k',   linestyle='--', lw=1, label='Threshold 0.5')
    ax.axhline(metrics['Thresh (Youden)'], color='red', linestyle=':', lw=1, label='Youden')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#E86B4C', label='PTC'),
                        Patch(color='#4C9BE8', label='Benign')] +
              ax.get_legend_handles_labels()[0], fontsize=9)
    ax.set_xlabel('Patient (ranked by score)'); ax.set_ylabel('Predicted probability')
    ax.set_title(f'Patient-Level Predictions (Fold {fold_id})')
    plt.tight_layout()
    path = os.path.join(cfg['graphs_dir'], f'patient_scores_fold{fold_id}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────
# 7. EXPLAINABILITY — Attention-based
# ──────────────────────────────────────────────
def visualize_attention(model, val_df, cfg, fold_id, n_cases=4):
    print(f"\n=== Stage 4: Explainability ===")
    us_tf, path_tf = get_transforms(cfg, augment=False)
    ds = ThyroidDataset(val_df, us_tf, path_tf)
    model.eval()

    # Sample 2 true positives + 2 true negatives
    sample_indices = []
    probs_all, tgts_all = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            item  = ds[i]
            us    = item['us'].unsqueeze(0).to(device)
            path  = item['patches'].unsqueeze(0).to(device)
            mask  = torch.ones(1, item['n_patches'], dtype=torch.bool).to(device)
            out, feats = model(us, path, mask, return_feats=True)
            p = torch.sigmoid(out).item()
            probs_all.append(p)
            tgts_all.append(item['label'].item())

    probs_all = np.array(probs_all)
    tgts_all  = np.array(tgts_all)
    tp_idx = np.where((tgts_all == 1) & (probs_all > 0.5))[0]
    tn_idx = np.where((tgts_all == 0) & (probs_all < 0.5))[0]
    fp_idx = np.where((tgts_all == 0) & (probs_all > 0.5))[0]
    fn_idx = np.where((tgts_all == 1) & (probs_all < 0.5))[0]

    cases = []
    for group, label in [(tp_idx,'TP'), (tn_idx,'TN'), (fp_idx,'FP'), (fn_idx,'FN')]:
        if len(group) > 0:
            cases.append((group[0], label))
        if len(cases) >= n_cases:
            break

    denorm = transforms.Compose([
        transforms.Normalize(mean=[0,0,0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485,-0.456,-0.406], std=[1,1,1]),
    ])

    for case_idx, case_label in cases:
        item  = ds[case_idx]
        us    = item['us'].unsqueeze(0).to(device)
        path  = item['patches'].unsqueeze(0).to(device)
        n_p   = item['n_patches']
        mask  = torch.zeros(1, n_p, dtype=torch.bool).to(device)
        mask[0, :n_p] = True

        with torch.no_grad():
            out, feats = model(us, path, mask, return_feats=True)
            prob  = torch.sigmoid(out).item()
            cls_f = feats['fused'][0, 0]                      # [dim]
            patch_fs = feats['fused'][0, 2:2+n_p]             # [n_p, dim]
            sims = F.cosine_similarity(cls_f.unsqueeze(0), patch_fs).cpu().numpy()

        n_cols = n_p + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 4))
        if n_cols == 1: axes = [axes]

        us_img = denorm(item['us']).permute(1,2,0).numpy()
        us_img = np.clip(us_img, 0, 1)
        axes[0].imshow(us_img)
        axes[0].set_title(f"Ultrasound\nTrue={int(item['label'].item())} Pred={prob:.2f}", fontsize=9)
        axes[0].axis('off')

        for j in range(n_p):
            p_img = denorm(item['patches'][j]).permute(1,2,0).numpy()
            p_img = np.clip(p_img, 0, 1)
            axes[j+1].imshow(p_img)
            axes[j+1].set_title(f"Patch {j+1}\nSim={sims[j]:.3f}", fontsize=9)
            axes[j+1].axis('off')

        plt.suptitle(f"{case_label} | Patient: {item['pid']} | P(PTC)={prob:.3f}", fontsize=10)
        plt.tight_layout()
        path = os.path.join(cfg['graphs_dir'], f'attention_{case_label}_fold{fold_id}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {path}")


# ──────────────────────────────────────────────
# 8. MAIN ENTRY POINT
# ──────────────────────────────────────────────
def summarize_folds(fold_metrics, cfg):
    """Write per-fold metrics and a mean ± SD summary across folds."""
    all_df = pd.DataFrame(fold_metrics)
    all_path = os.path.join(cfg['graphs_dir'], 'metrics_all_folds.csv')
    all_df.to_csv(all_path, index=False)

    num = all_df.drop(columns=['fold']).select_dtypes(include='number')
    summary = pd.DataFrame({
        'Metric': num.columns,
        'Mean':   num.mean().values,
        'SD':     num.std(ddof=1).values,
    })
    summary['Mean ± SD'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary['Mean'], summary['SD'])]
    summary_path = os.path.join(cfg['graphs_dir'], 'metrics_cv_summary.csv')
    summary.to_csv(summary_path, index=False)

    print(f"\n=== Cross-validation summary ({len(all_df)} folds) ===")
    for _, r in summary.iterrows():
        print(f"  {r['Metric']:20s}: {r['Mean ± SD']}")
    print(f"Saved {all_path}\nSaved {summary_path}")


def run():
    t0 = time.time()

    df    = build_metadata(CFG)
    folds = range(CFG['n_folds']) if CFG['run_all_folds'] else [CFG['fold_id']]

    fold_metrics = []
    for fold_id in folds:
        model, ckpt_path, val_df = train_fold(df, CFG, fold_id)
        metrics, probs, targets, pids = evaluate(model, val_df, CFG, fold_id, ckpt_path)
        visualize_attention(model, val_df, CFG, fold_id)

        # Save metrics table
        out_df = pd.DataFrame([{k: (f"{v:.4f}" if isinstance(v, float) else v)
                                 for k, v in metrics.items()}])
        metrics_path = os.path.join(CFG['graphs_dir'], f"metrics_fold{fold_id}.csv")
        out_df.to_csv(metrics_path, index=False)
        print(f"\nSaved metrics → {metrics_path}")
        fold_metrics.append({'fold': fold_id, **metrics})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(fold_metrics) > 1:
        summarize_folds(fold_metrics, CFG)
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    run()