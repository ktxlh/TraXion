"""Sampled softmax loss for next-POI recommendation.

At each valid position the loss is an InfoNCE / sampled-softmax objective:
the positive is the actual next POI, K negatives are sampled uniformly
from the full POI vocabulary, and (optionally) all other positives in the
same batch are used as in-batch negatives with duplicate masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class POIRecCriterion(nn.Module):
    """Sampled softmax loss for next-POI ranking.

    Args:
        n_neg_samples: Number of uniformly-sampled negative POIs per position.
        temperature: Scaling factor for dot-product logits.
        label_smoothing: Cross-entropy label smoothing (default 0.0 = none).
        use_in_batch_neg: If True, also treat all other positives in the same
            batch as negatives (with duplicates-of-the-true-positive masked).
    """

    def __init__(
        self,
        n_neg_samples: int = 256,
        temperature: float = 0.1,
        label_smoothing: float = 0.0,
        use_in_batch_neg: bool = False,
    ):
        super().__init__()
        self.n_neg_samples = n_neg_samples
        self.temperature = temperature
        self.label_smoothing = float(label_smoothing)
        self.use_in_batch_neg = bool(use_in_batch_neg)

    def forward(self, query, poi_weight, batch):
        """Compute sampled softmax loss.

        Args:
            query: (B, S, D) query vectors from model.
            poi_weight: (V, D) full POI embedding table.
            batch: dict with ``target_poi_idx`` (B, S) and ``rec_mask`` (B, S).

        Returns:
            (loss, loss_dict) where loss is a scalar tensor.
        """
        target_poi_idx = batch["target_poi_idx"]  # (B, S)
        rec_mask = batch["rec_mask"]               # (B, S)

        valid_queries = query[rec_mask]            # (N, D)
        valid_targets = target_poi_idx[rec_mask]   # (N,)
        n_valid = valid_queries.shape[0]

        if n_valid == 0:
            zero = query.sum() * 0.0
            return zero, {"rec_loss": 0.0}

        n_pois = poi_weight.shape[0]

        pos_emb = poi_weight[valid_targets]  # (N, D)
        pos_scores = (valid_queries * pos_emb).sum(dim=-1, keepdim=True)  # (N, 1)

        neg_idx = torch.randint(
            0, n_pois, (n_valid, self.n_neg_samples), device=query.device
        )
        neg_emb = poi_weight[neg_idx]  # (N, K, D)
        neg_scores = torch.bmm(
            neg_emb, valid_queries.unsqueeze(-1)
        ).squeeze(-1)  # (N, K)

        if self.use_in_batch_neg:
            # Use other valid_targets in the batch as extra negatives.
            # ibn_scores: (N, N) where ibn_scores[i, j] = q_i · e(t_j).
            ibn_scores = valid_queries @ pos_emb.t()  # (N, N)
            # Mask out self (diagonal) and duplicates where target_j == target_i.
            eye = torch.eye(n_valid, dtype=torch.bool, device=query.device)
            same_target = valid_targets.unsqueeze(0) == valid_targets.unsqueeze(1)
            dup_mask = eye | same_target
            ibn_scores = ibn_scores.masked_fill(dup_mask, float("-inf"))
            logits = torch.cat([pos_scores, neg_scores, ibn_scores], dim=-1)
        else:
            logits = torch.cat([pos_scores, neg_scores], dim=-1)

        logits = logits / self.temperature
        labels = torch.zeros(n_valid, dtype=torch.long, device=query.device)

        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        return loss, {"rec_loss": loss.item()}
