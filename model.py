import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class PatchEncoder(nn.Module):
    def __init__(self, backbone='convnext_tiny', pretrained=True):
        super().__init__()
        if backbone == 'convnext_tiny':
            self.model = models.convnext_tiny(weights='DEFAULT' if pretrained else None)
            # Remove only the Linear layer, keep LayerNorm2d and Flatten
            self.feature_dim = self.model.classifier[2].in_features
            self.model.classifier = nn.Sequential(
                self.model.classifier[0],
                self.model.classifier[1]
            )
        else:
            raise NotImplementedError
            
    def forward(self, x):
        # x: [B*N, 3, H, W]
        features = self.model(x)
        return features

class MultimodalTransformer(nn.Module):
    def __init__(self, num_classes=1, dim=768, depth=2, heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        
        # Ultrasound Encoder
        self.us_encoder = PatchEncoder(backbone='convnext_tiny')
        
        # Pathology Patch Encoder (shared or separate, let's use separate or shared?)
        # Sharing might be good if features are similar, but US and Path are very different.
        # Let's use separate encoders.
        self.path_encoder = PatchEncoder(backbone='convnext_tiny')
        
        # Dimension projection if needed
        self.us_proj = nn.Linear(self.us_encoder.feature_dim, dim)
        self.path_proj = nn.Linear(self.path_encoder.feature_dim, dim)
        
        # Pathology MIL Transformer
        # We treat patches as a sequence.
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=depth)
        
        # Cross-modal Fusion Transformer
        # We can use a small transformer where US feature attends to Path features or vice versa.
        # Simple approach: concatenate US token and Path tokens, then another Transformer.
        fusion_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.fusion_transformer = nn.TransformerEncoder(fusion_encoder_layer, num_layers=1)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Final Classifier
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, us_images, path_patches, path_masks, return_attention=False):
        """
        us_images: [B, 3, H, W]
        path_patches: [B, N, 3, H, W]
        path_masks: [B, N] (True for real patches)
        """
        B, N, C, H, W = path_patches.shape
        
        # 1. Encode Ultrasound
        us_feats = self.us_encoder(us_images) # [B, feat_dim]
        us_feats = self.us_proj(us_feats).unsqueeze(1) # [B, 1, dim]
        
        # 2. Encode Pathology Patches
        path_patches_flat = path_patches.view(B * N, C, H, W)
        path_feats_flat = self.path_encoder(path_patches_flat) # [B*N, feat_dim]
        path_feats = path_feats_flat.view(B, N, -1)
        path_feats = self.path_proj(path_feats) # [B, N, dim]
        
        # 3. MIL (Pathology branch)
        path_padding_mask = ~path_masks
        
        # To get attention, we might need to iterate through layers or use a custom transformer
        # For now, let's assume we want to see how US attends to Path
        path_seq = self.path_transformer(path_feats, src_key_padding_mask=path_padding_mask) # [B, N, dim]
        
        # 4. Multimodal Fusion
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, dim]
        combined_seq = torch.cat([cls_tokens, us_feats, path_seq], dim=1) # [B, N+2, dim]
        
        combined_mask = torch.zeros((B, N + 2), dtype=torch.bool, device=us_images.device)
        combined_mask[:, 2:] = path_padding_mask
        
        fused_seq = self.fusion_transformer(combined_seq, src_key_padding_mask=combined_mask) # [B, N+2, dim]
        
        # 5. Classifier
        out = self.mlp_head(fused_seq[:, 0]) # [B, 1]
        
        if return_attention:
            # Note: Standard nn.TransformerEncoder doesn't easily return attention.
            # In a real implementation, we would use a custom layer or hook.
            # For now, we just return the features for qualitative analysis.
            return out, {
                'us_feats': us_feats,
                'path_seq': path_seq,
                'fused_seq': fused_seq
            }
            
        return out
