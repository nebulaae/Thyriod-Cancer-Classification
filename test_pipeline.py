import torch
from dataset import ThyroidMultimodalDataset, collate_multimodal
from model import MultimodalTransformer
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

def test_pipeline():
    print("Testing Pipeline...")
    df = pd.read_csv('metadata.csv').head(2)
    
    us_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    ds = ThyroidMultimodalDataset(df, us_transform=us_transform, path_transform=us_transform)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_multimodal)
    
    batch = next(iter(loader))
    print(f"Batch loaded. US shape: {batch['us_images'].shape}, Path shape: {batch['path_patches'].shape}")
    
    model = MultimodalTransformer(num_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    us = batch['us_images'].to(device)
    path = batch['path_patches'].to(device)
    mask = batch['path_masks'].to(device)
    
    outputs = model(us, path, mask)
    print(f"Forward pass success. Output shape: {outputs.shape}")
    
    loss = outputs.mean()
    loss.backward()
    print("Backward pass success.")

if __name__ == "__main__":
    test_pipeline()
