import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve

# ==========================================
# 1. Configuration & Device Setup
# ==========================================
SEED = 42
EPOCHS = 20
LR = 1e-4
BATCH_SIZE = 8
IMG_SIZE = 224

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. Data Audit & Preprocessing
# ==========================================
def audit_and_split(n_splits=5):
    print("--- Stage 1: Data Audit & Splitting ---")
    benign_us_dir = 'dataset2'
    benign_path_dir = 'dataset1'
    ptc_us_dir = 'dataset3'
    ptc_path_dir = 'dataset4'

    def get_patient_ids(directory, is_pathology=False):
        ids = set()
        for filename in os.listdir(directory):
            if is_pathology: patient_id = filename.split('_')[0]
            else: patient_id = os.path.splitext(filename)[0]
            ids.add(patient_id)
        return ids

    benign_us_ids = get_patient_ids(benign_us_dir)
    ptc_us_ids = get_patient_ids(ptc_us_dir)
    benign_path_ids = get_patient_ids(benign_path_dir, is_pathology=True)
    ptc_path_ids = get_patient_ids(ptc_path_dir, is_pathology=True)

    data = []
    # Benign
    for pid in sorted(list(benign_us_ids), key=int):
        if pid in benign_path_ids:
            us_path = glob.glob(os.path.join(benign_us_dir, f"{pid}.*"))[0]
            path_patches = sorted(glob.glob(os.path.join(benign_path_dir, f"{pid}_*.tif")))
            data.append({'patient_id': f'benign_{pid}', 'label': 0, 'us_path': us_path, 'path_patches': ';'.join(path_patches), 'num_patches': len(path_patches)})
    
    # PTC
    for pid in sorted(list(ptc_us_ids), key=int):
        if pid in ptc_path_ids:
            us_path = glob.glob(os.path.join(ptc_us_dir, f"{pid}.*"))[0]
            path_patches = sorted(glob.glob(os.path.join(ptc_path_dir, f"{pid}_*.tif")))
            data.append({'patient_id': f'ptc_{pid}', 'label': 1, 'us_path': us_path, 'path_patches': ';'.join(path_patches), 'num_patches': len(path_patches)})

    df = pd.DataFrame(data)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df['label'])):
        df.loc[val_idx, 'fold'] = fold
    
    # Plot distribution
    plt.figure(figsize=(10, 4), dpi=100)
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='label', hue='label', palette='viridis', legend=False)
    plt.title("Class Distribution")
    plt.xticks([0, 1], ['Benign', 'PTC'])
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='num_patches', bins=6, kde=True, color='teal')
    plt.title("Patches per Patient")
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.show()
    
    return df

# ==========================================
# 3. Dataset & Model Definitions
# ==========================================
class MultimodalDataset(Dataset):
    def __init__(self, df, us_transform=None, path_transform=None):
        self.df = df
        self.us_transform = us_transform
        self.path_transform = path_transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        us_img = Image.open(row['us_path']).convert('RGB')
        if self.us_transform: us_img = self.us_transform(us_img)
        
        patch_paths = row['path_patches'].split(';')
        patches = [self.path_transform(Image.open(p).convert('RGB')) for p in patch_paths]
        
        return {'us': us_img, 'path': torch.stack(patches), 'label': torch.tensor(row['label'], dtype=torch.long), 'id': row['patient_id']}

def collate_fn(batch):
    us = torch.stack([x['us'] for x in batch])
    labels = torch.stack([x['label'] for x in batch])
    max_p = max([x['path'].shape[0] for x in batch])
    padded_path, masks = [], []
    for x in batch:
        p = x['path']
        count = p.shape[0]
        mask = torch.zeros(max_p, dtype=torch.bool)
        mask[:count] = 1
        padding = torch.zeros((max_p - count, *p.shape[1:]), dtype=p.dtype)
        padded_path.append(torch.cat([p, padding], dim=0))
        masks.append(mask)
    return {'us': us, 'path': torch.stack(padded_path), 'mask': torch.stack(masks), 'label': labels}

class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.convnext_tiny(weights='DEFAULT')
        self.feat_dim = m.classifier[2].in_features
        self.backbone = nn.Sequential(m.features, m.avgpool, m.classifier[0], m.classifier[1])
    def forward(self, x): return self.backbone(x)

class MultimodalTransformer(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.us_encoder = PatchEncoder()
        self.path_encoder = PatchEncoder()
        self.us_proj = nn.Linear(self.us_encoder.feat_dim, dim)
        self.path_proj = nn.Linear(self.path_encoder.feat_dim, dim)
        self.path_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True), num_layers=2)
        self.fusion_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True), num_layers=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.classifier = nn.Linear(dim, 1)

    def forward(self, us, path, mask, return_attn=False):
        B, N, C, H, W = path.shape
        us_f = self.us_proj(self.us_encoder(us)).unsqueeze(1)
        path_f = self.path_proj(self.path_encoder(path.view(-1, C, H, W))).view(B, N, -1)
        path_seq = self.path_transformer(path_f, src_key_padding_mask=~mask)
        combined = torch.cat([self.cls_token.expand(B, -1, -1), us_f, path_seq], dim=1)
        full_mask = torch.zeros((B, N+2), dtype=torch.bool, device=us.device)
        full_mask[:, 2:] = ~mask
        fused = self.fusion_transformer(combined, src_key_padding_mask=full_mask)
        out = self.classifier(fused[:, 0])
        return (out, fused) if return_attn else out

# ==========================================
# 4. Training & Evaluation Pipeline
# ==========================================
def run_experiment():
    df = audit_and_split()
    
    # Transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_trans = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(), norm])
    val_trans = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])

    fold_id = 0
    train_df = df[df['fold'] != fold_id]
    val_df = df[df['fold'] == fold_id]
    
    train_loader = DataLoader(MultimodalDataset(train_df, train_trans, train_trans), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MultimodalDataset(val_df, val_trans, val_trans), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = MultimodalTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.86]).to(device))
    
    history = {'train_loss': [], 'val_auc': []}
    
    print("\n--- Stage 2: Training ---")
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for b in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            out = model(b['us'].to(device), b['path'].to(device), b['mask'].to(device))
            loss = criterion(out, b['label'].float().to(device).unsqueeze(1))
            loss.backward(); optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b in val_loader:
                out = model(b['us'].to(device), b['path'].to(device), b['mask'].to(device))
                preds.extend(torch.sigmoid(out).cpu().numpy())
                targets.extend(b['label'].numpy())
        
        auc = roc_auc_score(targets, preds)
        history['train_loss'].append(t_loss/len(train_loader))
        history['val_auc'].append(auc)
        print(f"Epoch {epoch+1}: Loss={t_loss/len(train_loader):.4f}, Val AUROC={auc:.4f}")

    # Plot History
    plt.figure(figsize=(12, 5), dpi=100)
    plt.subplot(1, 2, 1); plt.plot(history['train_loss'], color='navy', lw=2); plt.title('Training Loss'); plt.xlabel('Epoch')
    plt.subplot(1, 2, 2); plt.plot(history['val_auc'], color='darkorange', lw=2); plt.title('Validation AUROC'); plt.xlabel('Epoch')
    plt.savefig('training_history.png'); plt.show()

    # Final Evaluation
    print("\n--- Stage 3: Final Evaluation ---")
    probs = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    preds_binary = (probs > 0.5).astype(int)
    
    fpr, tpr, _ = roc_curve(targets, probs)
    prec, rec, _ = precision_recall_curve(targets, probs)
    cm = confusion_matrix(targets, preds_binary)
    
    plt.figure(figsize=(18, 5), dpi=100)
    plt.subplot(1, 3, 1); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc_score(targets, probs):.3f}'); plt.plot([0, 1], [0, 1], '--', color='navy'); plt.title('ROC Curve'); plt.legend()
    plt.subplot(1, 3, 2); plt.plot(rec, prec, color='green', lw=2, label=f'AP={average_precision_score(targets, probs):.3f}'); plt.title('PR Curve'); plt.legend()
    plt.subplot(1, 3, 3); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.title('Confusion Matrix'); plt.xticks([0.5, 1.5], ['Benign', 'PTC']); plt.yticks([0.5, 1.5], ['Benign', 'PTC'])
    plt.savefig('final_metrics.png'); plt.show()
    
    metrics = {
        'AUROC': roc_auc_score(targets, probs),
        'AUPRC': average_precision_score(targets, probs),
        'Accuracy': accuracy_score(targets, preds_binary),
        'F1-Score': f1_score(targets, preds_binary),
        'MCC': matthews_corrcoef(targets, preds_binary)
    }
    print("\nMetrics Summary Table:")
    print(pd.DataFrame([metrics]).T.rename(columns={0: 'Value'}))

    # Stage 4: Explainability
    print("\n--- Stage 4: Explainability (Attention Analysis) ---")
    visualize_attention(model, MultimodalDataset(val_df, val_trans, val_trans))

def visualize_attention(model, dataset, patient_idx=0):
    model.eval()
    item = dataset[patient_idx]
    us = item['us'].unsqueeze(0).to(device)
    path = item['path'].unsqueeze(0).to(device)
    mask = torch.ones((1, path.shape[1]), dtype=torch.bool).to(device)
    
    with torch.no_grad():
        out, fused = model(us, path, mask, return_attn=True)
        # fused: [B, N+2, dim]
        cls_token = fused[0, 0]
        tokens = fused[0, 1:] # US + Path patches
        sim = torch.nn.functional.cosine_similarity(cls_token.unsqueeze(0), tokens).cpu().numpy()
        
    us_sim = sim[0]
    path_sims = sim[1:]
    
    plt.figure(figsize=(15, 6), dpi=100)
    # Ultrasound
    plt.subplot(1, len(path_sims) + 1, 1)
    us_img = item['us'].permute(1, 2, 0).numpy()
    us_img = us_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    plt.imshow(np.clip(us_img, 0, 1))
    plt.title(f"US Score: {us_sim:.3f}"); plt.axis('off')
    
    # Path patches
    for i in range(len(path_sims)):
        plt.subplot(1, len(path_sims) + 1, i + 2)
        p_img = item['path'][i].permute(1, 2, 0).numpy()
        p_img = p_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        plt.imshow(np.clip(p_img, 0, 1))
        plt.title(f"Path {i+1}: {path_sims[i]:.3f}"); plt.axis('off')
        
    plt.suptitle(f"Multimodal Attention Analysis: {item['id']}")
    plt.savefig('explainability.png'); plt.show()

if __name__ == "__main__":
    run_experiment()