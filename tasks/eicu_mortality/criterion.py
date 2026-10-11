"""BCE-with-logits loss for mortality fine-tuning.

Mortality is a minority class, so a fixed pos_weight is applied to the
positive term (set with --pos_weight; 5.0 in the released scripts).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MortalityCriterion(nn.Module):
    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
        labels = batch["labels"].to(logits.device, dtype=logits.dtype)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight
        )
        with torch.no_grad():
            probs = torch.sigmoid(logits.float())
            pos = float(labels.sum().item())
            n = float(labels.numel())
        return loss, {
            "bce_loss": float(loss.item()),
            "total_loss": float(loss.item()),
            "n_pos": pos,
            "batch_size": n,
            "mean_prob": float(probs.mean().item()),
        }
