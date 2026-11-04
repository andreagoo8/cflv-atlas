from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    from trident.slide_encoder_models import ABMILSlideEncoder
except ModuleNotFoundError:
    local_trident_repo = Path(__file__).resolve().parents[1] / "trident"
    if local_trident_repo.exists():
        sys.path.insert(0, str(local_trident_repo))
    from trident.slide_encoder_models import ABMILSlideEncoder


class PatchABMILClassifier(nn.Module):
    """
    Patch-level classifier using ABMIL pooling, then a classification head.
    Input: [batch_size, num_patches, feature_dim]
    Output: [batch_size, num_classes]
    """

    def __init__(
        self,
        input_feature_dim: int,
        num_classes: int,
        n_heads: int = 1,
        head_dim: int = 512,
        dropout: float = 0.1,
        gated: bool = True,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=input_feature_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated,
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_raw_attention: bool = False):
        if return_raw_attention:
            pooled_features, attn = self.feature_encoder({"features": x}, return_raw_attention=True)
            logits = self.classifier(pooled_features)
            return logits, attn

        pooled_features = self.feature_encoder({"features": x})
        logits = self.classifier(pooled_features)
        return logits


class SlideMLPClassifier(nn.Module):
    """
    Slide-level classifier using a light MLP "post-transformer" style head.
    Input: [batch_size, feature_dim]
    Output: [batch_size, num_classes]
    """

    def __init__(
        self,
        input_feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.post_transform = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.post_transform(x)
        logits = self.classifier(x)
        return logits
