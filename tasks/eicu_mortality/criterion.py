"""BCE-with-logits loss for mortality fine-tuning.

The ICU demo has ~8.5% positives so we apply a fixed pos_weight to the
loss (default = inverse class frequency, capped at 10x to avoid blowing
up the gradient on the rare class).
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
