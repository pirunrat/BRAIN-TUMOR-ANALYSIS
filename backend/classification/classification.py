import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the custom model
class Classifier(nn.Module):
    def __init__(self, num_classes=4, dropout_prob=0.2):
        super(Classifier, self).__init__()
        # Load the pre-trained MobileNetV2 model
        mobilenet = models.mobilenet_v2(pretrained=True)

        # Keep only the first 5 layers of MobileNetV2's feature extractor
        self.features = nn.Sequential(*list(mobilenet.features[:10]))

        # Define Global Average Pooling (GAP) layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # ✅ Converts (C, H, W) → (C, 1, 1)

        # Feature dimension after GAP
        self.feature_dim = 64  # Since GAP reduces (C, H, W) to (C, 1, 1)

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.feature_dim, 512),  # ✅ Uses output from GAP
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)  
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.global_avg_pool(feat)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits
