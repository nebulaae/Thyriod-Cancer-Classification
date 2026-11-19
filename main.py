"""
Multimodal Thyroid PTC vs Benign Classification
Unified Pipeline: Audit → Split → Train → Evaluate → Explain
"""

import os, glob, json, time, random, argparse, warnings
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
                   sample_modality_dropout, IMAGENET_MEAN, IMAGENET_STD)

from dedup import find_duplicate_groups, label_conflict_groups
from reliability import sigmoid, expected_calibration_error, fit_temperature

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
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
    # Modality experiments (defaults reproduce the original joint pipeline)
    mode        = 'joint',    # 'joint' | 'us_only' | 'cyto_only'
    modality_dropout = None,  # e.g. dict(p_drop_cyto=0.3, p_drop_us=0.15); joint mode, training only
    inner_val_frac   = 0.15,  # stratified inner val split of the training folds for checkpoint
                              # selection, Youden threshold and temperature; None → held-out fold
    # Optimisation switches (A3 recipe; ablation.py turns them off for A0–A2)
    scheduler   = 'warmup_cosine',  # 'warmup_cosine' | 'constant'
    grad_clip   = 1.0,        # None → no clipping
    early_stopping = True,
    amp         = True,       # mixed precision on CUDA
    num_workers = 0,
    deterministic = True,     # cuDNN deterministic kernels for reproducible re-runs
    # Data hygiene
    dedup       = True,       # group near-duplicate images into one fold (dHash)
    dedup_max_dist = 8,       # max Hamming distance of 256-bit dHashes
    drop_label_conflicts = True,  # drop groups whose near-duplicates carry both labels
    deletion_test = True,     # relevance faithfulness test on true positives
    max_patches = 7,
    checkpoint_dir = 'checkpoints',
    graphs_dir     = 'graphs',
    results_dir    = 'results',
    # Dataset dirs (relative to data_root)
    data_root       = '.',
    benign_us_dir   = 'dataset2',
    benign_path_dir = 'dataset1',
    ptc_us_dir      = 'dataset3',
    ptc_path_dir    = 'dataset4',
)


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(CFG['seed'])
for _d in (CFG['checkpoint_dir'], CFG['graphs_dir'], CFG['results_dir']):
    os.makedirs(_d, exist_ok=True)

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

    root = cfg.get('data_root', '.')
    dirs = {k: os.path.join(root, cfg[k]) for k in
            ('benign_us_dir', 'benign_path_dir', 'ptc_us_dir', 'ptc_path_dir')}
    for k, d in dirs.items():
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{k} not found: {d} (set --data_root)")

    b_us  = get_ids(dirs['benign_us_dir'])
    b_p   = get_ids(dirs['benign_path_dir'], True)
    ptc_us = get_ids(dirs['ptc_us_dir'])
    ptc_p  = get_ids(dirs['ptc_path_dir'], True)

    print(f"Benign US ids: {len(b_us)}, Path ids: {len(b_p)}")
    print(f"PTC    US ids: {len(ptc_us)}, Path ids: {len(ptc_p)}")
    overlap = b_us & ptc_us
    if overlap:
        print(f"WARNING — ID overlap Benign/PTC in US: {overlap}")

    rows = []
    for label, us_dir, p_dir, us_ids, p_ids in [
        (0, dirs['benign_us_dir'], dirs['benign_path_dir'], b_us, b_p),
        (1, dirs['ptc_us_dir'],    dirs['ptc_path_dir'],    ptc_us, ptc_p),
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

    # Near-duplicate audit: patients sharing (near-)identical images form one group
    if cfg.get('dedup', True):
        groups, pairs = find_duplicate_groups(df, max_dist=cfg.get('dedup_max_dist', 8))
        df['group'] = groups
        pairs_path = os.path.join(cfg['graphs_dir'], 'duplicate_pairs.csv')
        pairs.to_csv(pairs_path, index=False)
        conflicts = label_conflict_groups(df, groups)
        n_grouped = int((df.groupby('group')['patient_id'].transform('size') > 1).sum())
        print(f"Near-duplicates: {len(pairs)} cross-patient image pairs, {n_grouped} patients "
              f"in shared groups, {len(conflicts)} label-conflict groups → {pairs_path}")
        if conflicts and cfg.get('drop_label_conflicts', True):
            drop = df['group'].isin(conflicts)
            print(f"Dropping {int(drop.sum())} patients in label-conflict groups: "
                  f"{df.loc[drop, 'patient_id'].tolist()}")
            df = df[~drop].reset_index(drop=True)
    else:
        df['group'] = np.arange(len(df))

    # Stratified group K-fold (patient level; duplicate groups never split across folds)
    sgkf = StratifiedGroupKFold(n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['seed'])
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df['label'], groups=df['group'])):
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
def _seed_worker(worker_id):
    """DataLoader worker seeding (module level so Windows spawn can pickle it)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def split_fold(df, cfg, fold_id):
    """Return (train_df, sel_df, test_df) for one outer fold.
    sel_df drives checkpoint selection / threshold choice: an inner stratified
    split of the training folds if cfg['inner_val_frac'] is set, else the test fold."""
    train_df = df[df.fold != fold_id]
    test_df  = df[df.fold == fold_id]
    frac = cfg.get('inner_val_frac')
    if frac:
        if 'group' in train_df:
            # Group-aware inner split so near-duplicate patients never straddle train/selection
            k = max(2, int(round(1 / frac)))
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg['seed'] + max(fold_id, 0))
            tr_idx, sel_idx = next(sgkf.split(train_df, train_df['label'], groups=train_df['group']))
            train_df, sel_df = train_df.iloc[tr_idx], train_df.iloc[sel_idx]
        else:
            train_df, sel_df = train_test_split(train_df, test_size=frac, stratify=train_df['label'],
                                                random_state=cfg['seed'] + fold_id)
    else:
        sel_df = test_df
    return train_df, sel_df, test_df


def train_fold(df, cfg, fold_id, ckpt_name=None):
    """Train one fold; returns (model, ckpt_path, selection_df). The selection set
    is the held-out fold unless cfg['inner_val_frac'] creates an inner split."""
    mode = cfg.get('mode', 'joint')
    print(f"\n=== Stage 2: Training — Fold {fold_id} | mode={mode} ===")
    # Re-seed so each fold is reproducible on its own (same state as a single-fold run)
    set_seed(cfg['seed'], cfg.get('deterministic', True))

    train_df, val_df, _ = split_fold(df, cfg, fold_id)   # val_df = selection set
    md = cfg.get('modality_dropout') if mode == 'joint' else None
    if md:
        print(f"Modality dropout: p_drop_cyto={md['p_drop_cyto']} p_drop_us={md['p_drop_us']}")

    # Compute pos_weight from training set only
    n0 = (train_df.label == 0).sum()
    n1 = (train_df.label == 1).sum()
    pos_w = torch.tensor([n0 / n1], dtype=torch.float32).to(device)
    print(f"Train: {len(train_df)} pts, pos_weight={pos_w.item():.3f}")

    us_tr, path_tr = get_transforms(cfg, augment=True)
    us_vl, path_vl = get_transforms(cfg, augment=False)

    nw = cfg.get('num_workers', 0)
    tr_loader = DataLoader(
        ThyroidDataset(train_df, us_tr, path_tr),
        batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
        generator=torch.Generator().manual_seed(cfg['seed']),
        worker_init_fn=_seed_worker,
    )
    vl_loader = DataLoader(
        ThyroidDataset(val_df, us_vl, path_vl),
        batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
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
    # Warmup + cosine decay (A3) or a constant learning rate (baseline)
    warmup_epochs = cfg['warmup_epochs']
    early_stop = cfg.get('early_stopping', True)
    if early_stop and cfg['patience'] >= cfg['epochs']:
        print(f"WARNING — patience ({cfg['patience']}) >= epochs ({cfg['epochs']}): early stopping can never trigger")
    def lr_lambda(ep):
        if cfg.get('scheduler', 'warmup_cosine') == 'constant':
            return 1.0
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        prog = (ep - warmup_epochs) / max(1, cfg['epochs'] - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = bool(cfg.get('amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    grad_clip = cfg.get('grad_clip', 1.0)

    history = {'train_loss': [], 'val_auc': [], 'val_loss': [], 'lr': [], 'epoch_sec': []}
    best_auc  = -1.0
    best_epoch = 0
    patience_counter = 0
    ckpt_name = ckpt_name or f'model_fold{fold_id}_best.pth'
    ckpt_path = os.path.join(cfg['checkpoint_dir'], ckpt_name)

    for epoch in range(cfg['epochs']):
        t_ep = time.time()
        # --- Train ---
        model.train()
        t_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Epoch {epoch+1:02d}/{cfg['epochs']} [Train]", leave=False):
            us    = b['us'].to(device, non_blocking=True)
            path  = b['path'].to(device, non_blocking=True)
            mask  = b['mask'].to(device, non_blocking=True)
            lbls  = b['labels'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                if md:
                    us_present, cyto_present = sample_modality_dropout(
                        us.shape[0], md['p_drop_cyto'], md['p_drop_us'], device=device)
                    out = model(us, path, mask, us_present=us_present, cyto_present=cyto_present)
                else:
                    out = model(us, path, mask)
            loss = criterion(out.float(), lbls)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
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
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    out = model(us, path, mask)
                out = out.float()
                v_loss += criterion(out, lbls).item()
                probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
                targets.extend(lbls.cpu().numpy())

        probs   = np.array(probs)
        targets = np.array(targets)
        auc     = roc_auc_score(targets, probs) if len(np.unique(targets)) == 2 else 0.5
        avg_tr  = t_loss / len(tr_loader)
        avg_vl  = v_loss / len(vl_loader)
        lr_now  = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_tr)
        history['val_loss'].append(avg_vl)
        history['val_auc'].append(auc)
        history['lr'].append(lr_now)
        history['epoch_sec'].append(time.time() - t_ep)

        scheduler.step()

        print(f"Epoch {epoch+1:02d} | TrainLoss={avg_tr:.4f} ValLoss={avg_vl:.4f} AUROC={auc:.4f} "
              f"LR={lr_now:.2e} | {history['epoch_sec'][-1]:.0f}s")

        # Without early stopping (baseline) the last epoch is kept, as in the original prototype
        if (auc > best_auc) if early_stop else True:
            best_auc, best_epoch = auc, epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'cfg': cfg,
                'val_auc': auc,
            }, ckpt_path)
            if early_stop:
                print(f"  ✓ Saved best model (AUROC={best_auc:.4f})")
        elif early_stop:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"Early stop at epoch {epoch+1}")
                break

    history['best_epoch'] = best_epoch
    model.history = history     # read by ablation.py (loss gap, epochs, runtime)
    _plot_training(history, fold_id, cfg, tag=os.path.splitext(ckpt_name)[0].replace('_best', ''))
    return model, ckpt_path, val_df


def _plot_training(history, fold_id, cfg, tag=None):
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
    name = f'training_fold{fold_id}' if tag in (None, f'model_fold{fold_id}') else f'training_{tag}'
    path = os.path.join(cfg['graphs_dir'], f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────
# 6. COMPREHENSIVE EVALUATION
# ──────────────────────────────────────────────
def predict_logits(model, df, cfg, desc=None):
    """Per-patient logits (complete inputs). Returns (logits, targets, pids)."""
    us_tf, path_tf = get_transforms(cfg, augment=False)
    loader = DataLoader(
        ThyroidDataset(df, us_tf, path_tf),
        batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn,
    )
    model.eval()
    logits, targets, pids = [], [], []
    with torch.no_grad():
        for b in (tqdm(loader, desc=desc) if desc else loader):
            out = model(b['us'].to(device), b['path'].to(device), b['mask'].to(device))
            logits.extend(out.float().cpu().numpy().flatten())
            targets.extend(b['labels'].numpy())
            pids.extend(b['pids'])
    return np.array(logits), np.array(targets), pids


def evaluate(model, val_df, cfg, fold_id, ckpt_path=None, sel_df=None):
    """Metric panel on the held-out fold. Threshold (Youden) and temperature are
    fitted on sel_df (inner validation split) when given, otherwise on the
    test fold itself (optimistic; flagged in the output)."""
    print(f"\n=== Stage 3: Evaluation — Fold {fold_id} ===")

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded checkpoint (epoch {ckpt['epoch']+1}, AUROC={ckpt['val_auc']:.4f})")

    logits, targets, pids = predict_logits(model, val_df, cfg, desc="Evaluating")
    probs   = sigmoid(logits)
    preds   = (probs > 0.5).astype(int)

    nested = sel_df is not None and len(sel_df) > 0
    if nested:
        sel_logits, sel_y, _ = predict_logits(model, sel_df, cfg)
    else:
        sel_logits, sel_y = logits, targets
    temperature = fit_temperature(sel_logits, sel_y)
    probs_cal = sigmoid(logits / temperature)

    # Youden threshold chosen on the selection set (calibration does not change ranks,
    # so the threshold is applied to the calibrated probabilities)
    best_thresh = youden_threshold(sel_y, sigmoid(sel_logits / temperature))
    preds_y     = (probs_cal >= best_thresh).astype(int)

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
        'ECE':          expected_calibration_error(targets, probs),
        # Temperature scaling (fitted on the selection set)
        'Temperature':  temperature,
        'Brier (cal)':  brier_score_loss(targets, probs_cal),
        'ECE (cal)':    expected_calibration_error(targets, probs_cal),
        # Youden threshold (selection set) applied to calibrated probabilities
        'F1 (Youden)':  f1_score(targets, preds_y),
        'Sens (Youden)':tp_y / (tp_y + fn_y) if (tp_y + fn_y) > 0 else 0,
        'Spec (Youden)':tn_y / (tn_y + fp_y) if (tn_y + fp_y) > 0 else 0,
        'Thresh (Youden)': float(best_thresh),
        'Threshold source': 'inner_val' if nested else 'test_fold',
    }

    # Bootstrap 95% CI for AUROC
    ci_lo, ci_hi = bootstrap_auc_ci(targets, probs, seed=cfg['seed'])
    metrics['AUROC 95% CI'] = f"[{ci_lo:.3f}, {ci_hi:.3f}]"

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    _plot_evaluation(targets, probs, preds, pids, metrics, fold_id, cfg, probs_cal=probs_cal)
    return metrics, probs_cal, targets, pids


def bootstrap_auc_ci(targets, probs, seed=42, n_boot=1000):
    """Percentile bootstrap 95% CI for AUROC (patient-level resampling)."""
    aucs_boot = []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.choice(len(targets), len(targets), replace=True)
        if len(np.unique(targets[idx])) < 2: continue
        aucs_boot.append(roc_auc_score(targets[idx], probs[idx]))
    return np.percentile(aucs_boot, [2.5, 97.5])


# ──────────────────────────────────────────────
# 6b. MISSING-MODALITY SCENARIOS
# ──────────────────────────────────────────────
SCENARIOS = ('both', 'us_missing', 'cyto_missing', 'few_patches')
# Scenarios that are meaningful for each model mode (a single-modality model
# cannot lose its only modality; us_only ignores cytology entirely)
SCENARIOS_BY_MODE = {
    'joint':     SCENARIOS,
    'us_only':   ('both',),
    'cyto_only': ('both', 'few_patches'),
}
FEW_PATCHES = 2   # 'few_patches' keeps only the first 2 cytology patches


def predict(model, df, cfg, scenario='both'):
    """Per-patient P(PTC) for one test-time scenario. Returns (probs, targets, pids)."""
    us_tf, path_tf = get_transforms(cfg, augment=False)
    loader = DataLoader(
        ThyroidDataset(df, us_tf, path_tf),
        batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn,
    )
    model.eval()
    probs, targets, pids = [], [], []
    with torch.no_grad():
        for b in loader:
            us, path, mask = b['us'].to(device), b['path'].to(device), b['mask'].to(device)
            B = us.shape[0]
            us_present = cyto_present = None
            if scenario == 'us_missing':
                us_present = torch.zeros(B, dtype=torch.bool, device=device)
            elif scenario == 'cyto_missing':
                cyto_present = torch.zeros(B, dtype=torch.bool, device=device)
            elif scenario == 'few_patches':
                path = path[:, :FEW_PATCHES].contiguous()
                mask = mask[:, :FEW_PATCHES]
            elif scenario != 'both':
                raise ValueError(f"Unknown scenario {scenario!r}")
            out = model(us, path, mask, us_present=us_present, cyto_present=cyto_present)
            probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
            targets.extend(b['labels'].numpy())
            pids.extend(b['pids'])
    return np.array(probs), np.array(targets), pids


def evaluate_scenarios(model, test_df, sel_df, cfg, scenarios=None):
    """Predict every applicable scenario on the test fold and on the selection set
    (used to pick the Youden threshold). Returns {scenario: predictions dict}."""
    scenarios = scenarios or SCENARIOS_BY_MODE[cfg.get('mode', 'joint')]
    nested = bool(cfg.get('inner_val_frac'))
    out = {}
    for sc in scenarios:
        p_test, y_test, pids = predict(model, test_df, cfg, sc)
        if nested:
            p_sel, y_sel, _ = predict(model, sel_df, cfg, sc)
        else:
            p_sel, y_sel = p_test, y_test
        out[sc] = dict(p_test=p_test, y_test=y_test, pids=pids,
                       p_sel=p_sel, y_sel=y_sel, nested=nested)
    return out


def youden_threshold(targets, probs):
    fpr, tpr, thresholds = roc_curve(targets, probs)
    return thresholds[np.argmax(tpr - fpr)]


def scenario_metrics(pred, seed=42):
    """AUROC (+bootstrap 95% CI), AUPRC, Brier, and Sens/Spec at the Youden
    threshold chosen on the selection set (inner val if nested, else the test fold)."""
    y, p = pred['y_test'], pred['p_test']
    thresh = youden_threshold(pred['y_sel'], pred['p_sel'])
    preds = (p >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    ci_lo, ci_hi = bootstrap_auc_ci(y, p, seed=seed)
    return {
        'n':            len(y),
        'AUROC':        roc_auc_score(y, p),
        'AUROC_CI_lo':  ci_lo,
        'AUROC_CI_hi':  ci_hi,
        'AUPRC':        average_precision_score(y, p),
        'Sensitivity':  tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'Specificity':  tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'Threshold':    thresh,
        'Brier':        brier_score_loss(y, p),
        'thresh_source': 'inner_val' if pred['nested'] else 'test_fold',
    }


def _plot_evaluation(targets, probs, preds, pids, metrics, fold_id, cfg, probs_cal=None):
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
    ax.plot(mean_pred, frac_pos, 's-', color='#4C9BE8', lw=2,
            label=f"raw: Brier={metrics['Brier Score']:.3f}, ECE={metrics['ECE']:.3f}")
    if probs_cal is not None:
        frac_c, mean_c = calibration_curve(targets, probs_cal, n_bins=10)
        ax.plot(mean_c, frac_c, 'o-', color='#E86B4C', lw=2,
                label=f"T={metrics['Temperature']:.2f}: Brier={metrics['Brier (cal)']:.3f}, ECE={metrics['ECE (cal)']:.3f}")
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
    # (calibrated probabilities, so the Youden threshold is on the same scale)
    if probs_cal is not None:
        probs = probs_cal
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
# 7b. RELEVANCE FAITHFULNESS — Deletion test
# ──────────────────────────────────────────────
def patch_relevance(model, us, path, mask):
    """P(PTC) and cosine relevance of each real patch to the fused [CLS] token
    for one patient (tensors with batch dimension 1)."""
    with torch.no_grad():
        out, feats = model(us, path, mask, return_feats=True)
        n_p = int(mask[0].sum())
        cls_f = feats['fused'][0, 0]
        patch_fs = feats['fused'][0, 2:2 + n_p]
        sims = F.cosine_similarity(cls_f.unsqueeze(0), patch_fs).cpu().numpy()
    return torch.sigmoid(out.float()).item(), sims


def deletion_test(model, df, cfg, threshold=0.5, seed=42):
    """For each true positive with >= 2 patches: remove the top-relevance patch,
    and (separately) each other patch, and compare the drop in P(PTC).
    A case is a 'win' when removing the top-ranked patch lowers P(PTC) more than
    removing a randomly chosen other patch. Returns a summary dict."""
    us_tf, path_tf = get_transforms(cfg, augment=False)
    ds = ThyroidDataset(df, us_tf, path_tf)
    rng = np.random.default_rng(seed)
    model.eval()
    rows = []
    for i in range(len(ds)):
        item = ds[i]
        if item['label'].item() != 1 or item['n_patches'] < 2:
            continue
        us = item['us'].unsqueeze(0).to(device)
        path = item['patches'].unsqueeze(0).to(device)
        mask = torch.ones(1, item['n_patches'], dtype=torch.bool, device=device)
        p_full, sims = patch_relevance(model, us, path, mask)
        if p_full < threshold:
            continue   # only true positives
        top = int(np.argmax(sims))
        others = [j for j in range(item['n_patches']) if j != top]
        rnd = int(rng.choice(others))

        def p_without(j):
            m = mask.clone()
            m[0, j] = False
            with torch.no_grad():
                return torch.sigmoid(model(us, path, m).float()).item()

        drop_top, drop_rnd = p_full - p_without(top), p_full - p_without(rnd)
        rows.append({'pid': item['pid'], 'p_full': p_full, 'drop_top': drop_top,
                     'drop_random': drop_rnd, 'win': drop_top > drop_rnd})
    res = pd.DataFrame(rows)
    if res.empty:
        return {'Deletion n': 0, 'Deletion win rate': np.nan}, res
    return {'Deletion n': len(res), 'Deletion win rate': float(res['win'].mean()),
            'Deletion mean drop top': float(res['drop_top'].mean()),
            'Deletion mean drop random': float(res['drop_random'].mean())}, res


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


def train_final(df, cfg):
    """Deployment model: train on every patient, using a group-aware inner split
    for early stopping, temperature and the Youden threshold. Writes
    checkpoints/medai_final.pth with a 'deploy' block read by app.py."""
    print("\n=== Final model (all patients) ===")
    all_df = df.assign(fold=0)
    model, ckpt_path, sel_df = train_fold(all_df, cfg, fold_id=1, ckpt_name='medai_final.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    sel_logits, sel_y, _ = predict_logits(model, sel_df, cfg)
    temperature = fit_temperature(sel_logits, sel_y)
    threshold = float(youden_threshold(sel_y, sigmoid(sel_logits / temperature)))
    ckpt['deploy'] = {
        'temperature': temperature,
        'threshold': threshold,
        'selection_auroc': float(roc_auc_score(sel_y, sel_logits)),
        'n_train_patients': int(len(all_df) - len(sel_df)),
        'n_selection_patients': int(len(sel_df)),
        'trained_at': time.strftime('%Y-%m-%d %H:%M'),
    }
    torch.save(ckpt, ckpt_path)
    print(f"Final model → {ckpt_path} | T={temperature:.3f} threshold={threshold:.3f} "
          f"(selection AUROC {ckpt['deploy']['selection_auroc']:.3f})")
    return ckpt_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='MedAI thyroid pipeline: audit → train → evaluate → explain')
    ap.add_argument('--data_root', default=CFG['data_root'], help='Folder containing dataset1..4')
    ap.add_argument('--all_folds', action='store_true', help='Run every outer fold (5-fold CV)')
    ap.add_argument('--folds', nargs='+', type=int, help='Explicit list of outer folds to run')
    ap.add_argument('--final', action='store_true', help='Also train the deployment model on all patients')
    ap.add_argument('--final_only', action='store_true', help='Only train the deployment model')
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    ap.add_argument('--num_workers', type=int, default=CFG['num_workers'])
    ap.add_argument('--no_dedup', action='store_true')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--tag', default='', help='Suffix for result files (e.g. a re-run id)')
    return ap.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)
    cfg = {**CFG, 'data_root': args.data_root, 'epochs': args.epochs, 'num_workers': args.num_workers,
           'dedup': not args.no_dedup, 'amp': not args.no_amp}
    t0 = time.time()

    df = build_metadata(cfg)
    if args.final_only:
        train_final(df, cfg)
        print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
        return
    folds = args.folds if args.folds else (range(cfg['n_folds']) if (args.all_folds or cfg['run_all_folds'])
                                            else [cfg['fold_id']])
    tag = f"_{args.tag}" if args.tag else ''

    fold_metrics, oof, deletions = [], [], []
    for fold_id in folds:
        model, ckpt_path, sel_df = train_fold(df, cfg, fold_id, ckpt_name=f'model_fold{fold_id}{tag}_best.pth')
        _, _, test_df = split_fold(df, cfg, fold_id)   # held-out test fold
        nested = bool(cfg.get('inner_val_frac'))
        metrics, probs_cal, targets, pids = evaluate(model, test_df, cfg, fold_id, ckpt_path,
                                                     sel_df=sel_df if nested else None)
        h = model.history
        metrics.update({'Best epoch': h['best_epoch'] + 1, 'Epochs run': len(h['train_loss']),
                        'Train loss (best)': h['train_loss'][h['best_epoch']],
                        'Val loss (best)': h['val_loss'][h['best_epoch']],
                        'Loss gap (best)': h['val_loss'][h['best_epoch']] - h['train_loss'][h['best_epoch']],
                        'Sec per epoch': float(np.mean(h['epoch_sec']))})
        if cfg['mode'] != 'us_only':   # patch-similarity maps need cytology tokens
            visualize_attention(model, test_df, cfg, fold_id)
            if cfg.get('deletion_test', True):
                dsum, drows = deletion_test(model, test_df, cfg, threshold=metrics['Thresh (Youden)'],
                                            seed=cfg['seed'])
                metrics.update(dsum)
                deletions.append(drows.assign(fold=fold_id))
        oof.append(pd.DataFrame({'fold': fold_id, 'pid': pids, 'label': targets, 'p_cal': probs_cal}))

        out_df = pd.DataFrame([{k: (f"{v:.4f}" if isinstance(v, float) else v)
                                 for k, v in metrics.items()}])
        metrics_path = os.path.join(cfg['graphs_dir'], f"metrics_fold{fold_id}{tag}.csv")
        out_df.to_csv(metrics_path, index=False)
        print(f"\nSaved metrics → {metrics_path}")
        fold_metrics.append({'fold': fold_id, **metrics})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.concat(oof).to_csv(os.path.join(cfg['results_dir'], f'oof_predictions{tag}.csv'), index=False)
    if deletions:
        pd.concat(deletions).to_csv(os.path.join(cfg['results_dir'], f'deletion_test{tag}.csv'), index=False)
    if len(fold_metrics) > 1:
        summarize_folds(fold_metrics, cfg)
        oof_df = pd.concat(oof)
        lo, hi = bootstrap_auc_ci(oof_df['label'].values, oof_df['p_cal'].values, seed=cfg['seed'])
        pooled = {'Pooled OOF AUROC': roc_auc_score(oof_df['label'], oof_df['p_cal']),
                  'Pooled OOF AUROC CI': f"[{lo:.3f}, {hi:.3f}]", 'n': len(oof_df)}
        with open(os.path.join(cfg['results_dir'], f'pooled_oof{tag}.json'), 'w') as f:
            json.dump(pooled, f, indent=2)
        print(f"Pooled out-of-fold AUROC: {pooled['Pooled OOF AUROC']:.4f} {pooled['Pooled OOF AUROC CI']}")
    if args.final:
        train_final(df, cfg)
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    run()