import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from dataset import ThyroidMultimodalDataset, collate_multimodal
from model import MultimodalTransformer
import time
import os
from tqdm import tqdm

def train_one_fold(fold_id, metadata_path='metadata.csv', batch_size=8, epochs=30, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    df = pd.read_csv(metadata_path)
    train_df = df[df['fold'] != fold_id].reset_index(drop=True)
    val_df = df[df['fold'] == fold_id].reset_index(drop=True)
    
    # Calculate class weights for weighted BCE
    counts = train_df['label'].value_counts()
    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]
    # Normalize weights
    total = weight_for_0 + weight_for_1
    weight_for_1 /= (total / 2.0)
    pos_weight = torch.tensor([weight_for_1], dtype=torch.float).to(device)
    print(f"Pos weight for label 1: {pos_weight.item():.4f}")

    # Transforms
    # Ultrasound: Resize and normalize
    us_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Pathology: Resize and normalize (Vertical flip too for histology)
    path_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ThyroidMultimodalDataset(train_df, us_transform=us_transform, path_transform=path_transform)
    val_ds = ThyroidMultimodalDataset(val_df, us_transform=val_transform, path_transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal, num_workers=0)
    
    model = MultimodalTransformer(num_classes=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_auc = 0
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            us = batch['us_images'].to(device)
            path = batch['path_patches'].to(device)
            mask = batch['path_masks'].to(device)
            labels = batch['labels'].float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(us, path, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                us = batch['us_images'].to(device)
                path = batch['path_patches'].to(device)
                mask = batch['path_masks'].to(device)
                labels = batch['labels'].numpy()
                
                outputs = model(us, path, mask)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                val_preds.extend(probs)
                val_labels.extend(labels)
        
        val_preds = np.array(val_preds).flatten()
        val_labels = np.array(val_labels).flatten()
        
        auc = roc_auc_score(val_labels, val_preds)
        acc = accuracy_score(val_labels, val_preds > 0.5)
        f1 = f1_score(val_labels, val_preds > 0.5)
        
        print(f"Epoch {epoch+1} Results: AUC: {auc:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f'checkpoints/model_fold_{fold_id}_best.pth')
            print(f"Best model saved with AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_one_fold(0)
