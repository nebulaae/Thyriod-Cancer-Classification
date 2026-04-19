import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from dataset import ThyroidMultimodalDataset, collate_multimodal
from model import MultimodalTransformer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def evaluate_model(fold_id, model_path, metadata_path='metadata.csv', batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(metadata_path)
    val_df = df[df['fold'] == fold_id].reset_index(drop=True)
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_ds = ThyroidMultimodalDataset(val_df, us_transform=val_transform, path_transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)
    
    model = MultimodalTransformer(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            us = batch['us_images'].to(device)
            path = batch['path_patches'].to(device)
            mask = batch['path_masks'].to(device)
            labels = batch['labels'].numpy()
            
            outputs = model(us, path, mask)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
            
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    all_preds = (all_probs > 0.5).astype(int)
    
    metrics = {
        'AUROC': roc_auc_score(all_labels, all_probs),
        'AUPRC': average_precision_score(all_labels, all_probs),
        'Accuracy': accuracy_score(all_labels, all_preds),
        'F1': f1_score(all_labels, all_preds),
        'MCC': matthews_corrcoef(all_labels, all_preds)
    }
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    metrics['Sensitivity'] = tp / (tp + fn)
    metrics['Specificity'] = tn / (tn + fp)
    
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    return metrics

if __name__ == "__main__":
    # This assumes a model exists. Use it after training.
    # evaluate_model(0, 'checkpoints/model_fold_0_best.pth')
    pass
