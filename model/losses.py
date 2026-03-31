"""
Multi-task loss for the attribution model.

Three task-specific losses + consistency regularization:
    L = λ_sim * L_sim + λ_ai * L_ai + λ_attr * L_attr + λ_cons * L_consistency

Where:
    L_sim:         BCE on similarity labels
    L_ai:          BCE on AI detection labels (both tracks)
    L_attr:        BCE on attribution labels
    L_consistency: Encourages attribution ≤ similarity
                   (can't attribute if tracks aren't similar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributionLoss(nn.Module):
    """Multi-task loss combining similarity, AI detection, and attribution."""

    def __init__(
        self,
        lambda_sim: float = 1.0,
        lambda_ai: float = 0.5,
        lambda_attr: float = 1.0,
        lambda_consistency: float = 0.1,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_ai = lambda_ai
        self.lambda_attr = lambda_attr
        self.lambda_consistency = lambda_consistency
        self.label_smoothing = label_smoothing

    def _smooth_bce(self, logits: torch.Tensor, targets: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
        """BCE with label smoothing and optional positive class weighting."""
        eps = self.label_smoothing
        targets_smooth = targets * (1 - eps) + (1 - targets) * eps
        loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        weights = torch.where(targets == 1.0, pos_weight, 1.0)
        return (loss * weights).mean()

    def forward(self, model_out: dict, labels: dict) -> dict:
        """
        Args:
            model_out: dict from AttributionModel.forward()
            labels: dict with similarity, is_ai_a, is_ai_b, is_attribution
        Returns:
            dict with total_loss and per-task losses for logging
        """
        sim_logit = model_out["sim_logit"].squeeze(-1)
        ai_logit_a = model_out["ai_logit_a"].squeeze(-1)
        ai_logit_b = model_out["ai_logit_b"].squeeze(-1)
        attr_logit = model_out["attr_logit"].squeeze(-1)

        loss_sim = self._smooth_bce(sim_logit, labels["similarity"])
        loss_ai_a = self._smooth_bce(ai_logit_a, labels["is_ai_a"])
        loss_ai_b = self._smooth_bce(ai_logit_b, labels["is_ai_b"])
        loss_ai = (loss_ai_a + loss_ai_b) / 2
        # Up-weight positives to counteract attribution class imbalance.
        loss_attr = self._smooth_bce(attr_logit, labels["is_attribution"], pos_weight=4.0)

        # Penalise attribution > similarity.
        sim_prob = torch.sigmoid(sim_logit).detach()
        attr_prob = torch.sigmoid(attr_logit)
        consistency = F.relu(attr_prob - sim_prob).mean()

        total = (
            self.lambda_sim * loss_sim
            + self.lambda_ai * loss_ai
            + self.lambda_attr * loss_attr
            + self.lambda_consistency * consistency
        )

        return {
            "total_loss": total,
            "loss_sim": loss_sim.detach(),
            "loss_ai": loss_ai.detach(),
            "loss_attr": loss_attr.detach(),
            "loss_consistency": consistency.detach(),
        }
