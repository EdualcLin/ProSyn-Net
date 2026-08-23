#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fusion_models.py - ProSyn-Net model definition."""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-modal attention block (Eq. 1 in the paper). Operates on
    the globally pooled D-dimensional feature of each modality (sequence length 1)."""

    def __init__(self, in_dim: int, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_dim // num_heads) ** -0.5

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)

        self.to_q = nn.Linear(in_dim, in_dim, bias=False)
        self.to_kv = nn.Linear(in_dim, in_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query, context: [B, N=1, D]
        query_res = query
        query = self.norm1(query)
        context = self.norm2(context)
        b, n, d = query.shape
        h = self.num_heads
        q = self.to_q(query).reshape(b, n, h, d // h).permute(0, 2, 1, 3)       # [B,H,N,D/H]
        kv = self.to_kv(context).reshape(b, n, 2, h, d // h)
        k, v = kv[:, :, 0, :, :].permute(0, 2, 1, 3), kv[:, :, 1, :, :].permute(0, 2, 1, 3)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        return self.to_out(out) + query_res


class PrototypeClassifier(nn.Module):
    """Learnable per-class prototypes + squared-Euclidean distance profile decoded
    by the non-linear mapping Phi (two-layer MLP, hidden 512)."""

    def __init__(self, feature_dim: int, num_classes: int, distance_metric: str = 'euclidean'):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.distance_metric = distance_metric

        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)

        self.distance_to_logits = nn.Sequential(
            nn.Linear(num_classes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """features: [B, D] -> (logits [B, C], distances [B, C])."""
        batch_size = features.size(0)

        if self.distance_metric == 'euclidean':
            expanded_features = features.unsqueeze(1).expand(-1, self.num_classes, -1)  # [B, C, D]
            expanded_prototypes = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C, D]
            distances = torch.sum((expanded_features - expanded_prototypes) ** 2, dim=2)

        elif self.distance_metric == 'cosine':
            features_norm = F.normalize(features, p=2, dim=1)  # [B, D]
            prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)  # [C, D]
            similarities = torch.mm(features_norm, prototypes_norm.t())  # [B, C]
            distances = 1 - similarities

        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        logits = self.distance_to_logits(distances)

        return logits, distances

    def get_prototype_similarities(self, features: torch.Tensor) -> torch.Tensor:
        """Cosine similarities between features and prototypes (interpretability)."""
        features_norm = F.normalize(features, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        similarities = torch.mm(features_norm, prototypes_norm.t())  # [B, C]
        return similarities


class PrototypeEnhancedFusionNetwork(nn.Module):
    """ProSyn-Net: dynamic cross-attention fusion + non-linear prototypical mapping.

    [B-mode] --\
                cross-attention blocks -> dynamic weight fusion -> prototype classifier
    [Doppler] -/
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True,
                 distance_metric: str = 'euclidean'):
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            self.bmode_encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.doppler_encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            self.bmode_encoder = models.convnext_tiny(weights=None)
            self.doppler_encoder = models.convnext_tiny(weights=None)

        feature_dim = 768  # ConvNeXt-Tiny

        # Remove the classification heads
        self.bmode_encoder.classifier = nn.Sequential(
            self.bmode_encoder.classifier[0],  # keep AdaptiveAvgPool2d
            self.bmode_encoder.classifier[1],  # keep LayerNorm
            nn.Flatten(1),
            nn.Identity()  # drop the original linear classifier
        )
        self.doppler_encoder.classifier = nn.Sequential(
            self.doppler_encoder.classifier[0],
            self.doppler_encoder.classifier[1],
            nn.Flatten(1),
            nn.Identity()
        )

        self._freeze_early_layers()

        self.b_on_d_attention = CrossAttentionBlock(feature_dim)
        self.d_on_b_attention = CrossAttentionBlock(feature_dim)

        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),    # 1536 -> 384
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, feature_dim // 4),   # 384 -> 192
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 4, 2),
            nn.Softmax(dim=1)
        )

        self.prototype_classifier = PrototypeClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            distance_metric=distance_metric
        )

    def _freeze_early_layers(self):
        """Freeze the stem and stages 0-1 of both ConvNeXt encoders
        (features[0..3] of the 8-block Sequential)."""
        for i in range(2):
            stage_idx = i * 2  # 0, 2 (stages 0-1)
            for param in self.bmode_encoder.features[stage_idx].parameters():
                param.requires_grad = False
            if stage_idx + 1 < len(self.bmode_encoder.features):
                for param in self.bmode_encoder.features[stage_idx + 1].parameters():
                    param.requires_grad = False

        for i in range(2):
            stage_idx = i * 2
            for param in self.doppler_encoder.features[stage_idx].parameters():
                param.requires_grad = False
            if stage_idx + 1 < len(self.doppler_encoder.features):
                for param in self.doppler_encoder.features[stage_idx + 1].parameters():
                    param.requires_grad = False

    def forward(self, bmode: torch.Tensor, doppler: torch.Tensor, return_details: bool = False):
        """bmode/doppler: [B,3,224,224]. Returns logits [B, num_classes], or a dict
        of intermediate results when return_details=True."""
        bmode_feat = self.bmode_encoder(bmode)      # [B,768]
        doppler_feat = self.doppler_encoder(doppler)

        bmode_feat_seq = bmode_feat.unsqueeze(1)    # [B,1,768]
        doppler_feat_seq = doppler_feat.unsqueeze(1)

        b_enhanced = self.b_on_d_attention(bmode_feat_seq, doppler_feat_seq)  # [B,1,768]
        d_enhanced = self.d_on_b_attention(doppler_feat_seq, bmode_feat_seq)

        combined_feat = torch.cat([b_enhanced.squeeze(1), d_enhanced.squeeze(1)], dim=1)  # [B,1536]
        weights = self.weight_generator(combined_feat)  # [B,2]
        weight_b = weights[:, 0:1].unsqueeze(1)
        weight_d = weights[:, 1:2].unsqueeze(1)

        fused_feat = (weight_b * b_enhanced + weight_d * d_enhanced).squeeze(1)  # [B,768]

        logits, distances = self.prototype_classifier(fused_feat)
        if not return_details:
            return logits

        similarities = self.prototype_classifier.get_prototype_similarities(fused_feat)
        return {
            'logits': logits,
            'fused_features': fused_feat,
            'distances': distances,
            'similarities': similarities,
            'modal_weights': weights,
            'bmode_features': bmode_feat,
            'doppler_features': doppler_feat
        }

    def forward_with_details(self, bmode: torch.Tensor, doppler: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compatibility wrapper around forward(..., return_details=True)."""
        return self.forward(bmode, doppler, return_details=True)

    def get_prototypes(self) -> torch.Tensor:
        """Prototype vectors, kept in the autograd graph so the compactness and
        separation losses can optimize them (detach explicitly for export)."""
        return self.prototype_classifier.prototypes


def get_model(model_key: str, num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """Factory function: returns ProSyn-Net (PrototypeEnhancedFusionNetwork)."""
    key = model_key.lower().strip()

    if key == 'prototype_enhanced':
        return PrototypeEnhancedFusionNetwork(num_classes=num_classes, pretrained=pretrained)

    return PrototypeEnhancedFusionNetwork(num_classes=num_classes, pretrained=pretrained)


__all__ = [
    'CrossAttentionBlock',
    'PrototypeClassifier',
    'PrototypeEnhancedFusionNetwork',
    'get_model',
]
