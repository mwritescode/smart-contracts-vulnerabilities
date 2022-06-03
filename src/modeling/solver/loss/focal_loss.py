import torch
from torch import nn
import torch.nn.functional as F

from src.utils.registry import REGISTRY

@REGISTRY.register('focal_loss')
class SigmoidFocalLoss(nn.Module):
    """ Inspired by https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
