#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""focal_loss.py - ProSyn-Net losses: Focal Loss + prototype compactness/separation,
combined into PrototypeNetLoss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss: FL(pt) = -alpha * (1-pt)^gamma * log(pt).

    alpha: None / float / per-class list or tensor; gamma: focusing parameter
    (0 = plain cross-entropy, 2 = typical); label_smoothing in [0, 1).
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha])
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        """inputs: logits [B, C]; targets: [B]."""
        p = F.softmax(inputs, dim=1)

        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()

        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                            self.label_smoothing / num_classes

        pt = (p * targets_one_hot).sum(dim=1)

        focal_weight = (1 - pt) ** self.gamma

        ce = -torch.log(pt + 1e-8)

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce
        else:
            focal_loss = focal_weight * ce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PrototypeCompactnessLoss(nn.Module):
    """Pulls each feature towards its class prototype: mean ||f - P_y||^2."""

    def __init__(self, reduction='mean'):
        super(PrototypeCompactnessLoss, self).__init__()
        self.reduction = reduction

    def forward(self, features, prototypes, targets):
        """features: [B, D]; prototypes: [C, D]; targets: [B]."""
        target_prototypes = prototypes[targets]

        distances = torch.norm(features - target_prototypes, p=2, dim=1)

        if self.reduction == 'mean':
            return distances.mean()
        elif self.reduction == 'sum':
            return distances.sum()
        else:
            return distances


class PrototypeSeparationLoss(nn.Module):
    """Pushes class prototypes apart: mean over pairs of exp(-||P_i - P_j||)."""

    def __init__(self, reduction='mean'):
        super(PrototypeSeparationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, prototypes):
        """prototypes: [C, D]."""
        num_classes = prototypes.size(0)

        if num_classes < 2:
            return torch.tensor(0.0, device=prototypes.device)

        total_loss = 0.0
        pair_count = 0

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                dist = torch.norm(prototypes[i] - prototypes[j], p=2)
                pair_loss = torch.exp(-dist)
                total_loss += pair_loss
                pair_count += 1

        if pair_count == 0:
            return torch.tensor(0.0, device=prototypes.device)

        loss = total_loss / pair_count if self.reduction == 'mean' else total_loss
        return loss


class PrototypeNetLoss(nn.Module):
    """total = cls + compact_weight * compact + separation_weight * separation.
    Returns (total_loss, loss_details dict)."""

    def __init__(self, cls_criterion, compact_weight=0.1, separation_weight=0.01, reduction='mean'):
        super(PrototypeNetLoss, self).__init__()
        self.cls_criterion = cls_criterion
        self.compact_weight = compact_weight
        self.separation_weight = separation_weight
        self.reduction = reduction

        self.compact_criterion = PrototypeCompactnessLoss(reduction=reduction)
        self.separation_criterion = PrototypeSeparationLoss(reduction=reduction)

    def forward(self, logits, features, prototypes, targets):
        cls_loss = self.cls_criterion(logits, targets)

        compact_loss = self.compact_criterion(features, prototypes, targets)

        separation_loss = self.separation_criterion(prototypes)

        total_loss = (cls_loss +
                     self.compact_weight * compact_loss +
                     self.separation_weight * separation_loss)

        loss_details = {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'compact_loss': compact_loss.item(),
            'separation_loss': separation_loss.item(),
            'weighted_compact_loss': (self.compact_weight * compact_loss).item(),
            'weighted_separation_loss': (self.separation_weight * separation_loss).item()
        }

        return total_loss, loss_details


__all__ = [
    'FocalLoss',
    'PrototypeCompactnessLoss',
    'PrototypeSeparationLoss',
    'PrototypeNetLoss',
]
