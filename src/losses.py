import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        Focal Loss for multi-class classification.

        gamma > 0: focuses more on hard, misclassified examples.
        alpha: optional tensor of shape [num_classes] to weight classes differently.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Gather log_probs and probs for the true class
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        # Focal loss term
        focal_term = (1 - pt) ** self.gamma

        loss = -focal_term * log_pt

        # Optional class weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets.squeeze(1)]
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss