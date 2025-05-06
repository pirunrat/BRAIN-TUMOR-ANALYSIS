import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class Segmentor(nn.Module):
#     def __init__(self):
#         super(Segmentor, self).__init__()

#         # Load SegFormer model with mismatched size ignored
#         self.backbone_model = SegformerForSemanticSegmentation.from_pretrained(
#             "nvidia/segformer-b0-finetuned-ade-512-512",
#             num_labels=1,  # Single output channel for binary segmentation
#             ignore_mismatched_sizes=True  # Ignore size mismatch for the classifier layer
#         )


#         self.backbone_model.segformer.encoder.patch_embeddings = nn.ModuleList(
#             self.backbone_model.segformer.encoder.patch_embeddings[:3]
#         )

#         self.backbone_model.segformer.encoder.block = nn.ModuleList(
#             self.backbone_model.segformer.encoder.block[:3]
#         )

#         self.backbone_model.segformer.encoder.layer_norm = nn.ModuleList(
#             self.backbone_model.segformer.encoder.layer_norm[:3]
#         )


#         self.backbone_model.decode_head.linear_c = nn.ModuleList(
#             self.backbone_model.decode_head.linear_c[:3]
#         )

#         self.backbone_model.decode_head.linear_fuse = nn.Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
#         # Replace the segmentation head with a binary classifier
#         self.backbone_model.decode_head.classifier = nn.Conv2d(
#             in_channels=self.backbone_model.config.hidden_sizes[-1],  # Last layer's hidden size
#             out_channels=1,  # Single output channel
#             kernel_size=1
#         )

        
#     def forward(self, x):
#         # Get the full model output, including intermediate feature maps
#         outputs = self.backbone_model(x, output_hidden_states=True)

#         # Extract feature maps from different encoder stages
#         patch_embeddings_outputs = outputs.hidden_states  # Feature maps from all encoder layers

#         # Get the last encoder feature map (NOT `outputs.logits` since it's already processed)
#         last_feature_map = patch_embeddings_outputs[-1]  # Shape: (batch_size, 256, H, W)

#         # Pass the last feature map through the classifier
#         logits = outputs.logits

#         # Resize logits to (batch_size, 1, 224, 224)
#         resized_logits = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False)

#         return resized_logits, patch_embeddings_outputs


class Segmentor(nn.Module):
    def __init__(self, pretrained_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()

        # ✅ Now it's safe to assign modules
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(pretrained_name)

        # Add adapter to reduce output from 150 → 1 class
        self.adapter = nn.Sequential(
                                        nn.Conv2d(150, 64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 1, kernel_size=1)
                                    )

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        logits_150 = outputs.logits  # (B, 150, 56, 56)
    
        logits_1 = self.adapter(logits_150)  # (B, 1, 56, 56)
    
        # === Upsample to match input resolution ===
        logits_upsampled = F.interpolate(
            logits_1,
            size=pixel_values.shape[-2:],  # e.g., (224, 224)
            mode='bilinear',
            align_corners=False
        )
    
        return logits_upsampled, logits_150