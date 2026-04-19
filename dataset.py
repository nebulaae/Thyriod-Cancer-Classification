import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class ThyroidMultimodalDataset(Dataset):
    def __init__(self, metadata_df, us_transform=None, path_transform=None, max_patches=7):
        """
        Args:
            metadata_df: Pandas DataFrame with metadata.
            us_transform: Transforms for ultrasound images.
            path_transform: Transforms for pathology patches.
            max_patches: Maximum number of patches per patient (for padding).
        """
        self.df = metadata_df
        self.us_transform = us_transform
        self.path_transform = path_transform
        self.max_patches = max_patches

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Ultrasound
        us_img = Image.open(row['ultrasound_image_path']).convert('RGB')
        if self.us_transform:
            us_img = self.us_transform(us_img)
            
        # Load Pathology Patches
        patch_paths = row['pathology_patch_paths'].split(';')
        patches = []
        for p in patch_paths:
            img = Image.open(p).convert('RGB')
            if self.path_transform:
                img = self.path_transform(img)
            patches.append(img)
            
        num_patches = len(patches)
        
        # Pad patches to max_patches if necessary (or handle variable bag size in model)
        # Here we just return them as a list/stack
        patches_stack = torch.stack(patches)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'us_image': us_img,
            'path_patches': patches_stack,
            'num_patches': num_patches,
            'label': label,
            'patient_id': row['patient_id']
        }

def collate_multimodal(batch):
    """
    Custom collate function to handle variable number of patches.
    """
    us_images = torch.stack([item['us_image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # For path_patches, they have variable count if not padded.
    # We can pad them here to the max count in this batch.
    max_p = max([item['num_patches'] for item in batch])
    padded_patches = []
    masks = []
    
    for item in batch:
        p = item['path_patches']
        count = item['num_patches']
        # Pad with zeros
        pad_size = max_p - count
        if pad_size > 0:
            padding = torch.zeros((pad_size, *p.shape[1:]), dtype=p.dtype)
            p_padded = torch.cat([p, padding], dim=0)
        else:
            p_padded = p
            
        padded_patches.append(p_padded)
        
        # Create mask (1 for real patch, 0 for padding)
        mask = torch.zeros(max_p, dtype=torch.bool)
        mask[:count] = 1
        masks.append(mask)
        
    return {
        'us_images': us_images,
        'path_patches': torch.stack(padded_patches),
        'path_masks': torch.stack(masks),
        'labels': labels,
        'patient_ids': [item['patient_id'] for item in batch]
    }
