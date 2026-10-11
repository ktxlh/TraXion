"""Loss functions for ablation pretraining (A7 causal / A8 STM).

Both objectives share the same prediction heads and target tensors emitted by
``AblationPretrainDataset``; they differ in *where* the loss is applied:

  * **causal** (A7) — at every valid position t, predict (POI, time-bin, cat)
    at position t+1. The model must already have a causal attn_mask in the
    sequence-axis attention so the target at t+1 is not visible to position t.

  * **stm** (A8) — uniformly sample a fraction ``mask_ratio`` of valid
    positions; the encoder receives a [MASK] tile at those positions; predict
    (POI, time-bin, cat) at the masked positions only.

Loss = poi_ce + time_ce + cat_ce. CE skips entries with target == -1
(unknown POI / no category).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_mask_positions(
    valid_mask: torch.Tensor, mask_ratio: float, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a per-position boolean mask uniformly within valid positions.

    Args:
        valid_mask: (B, S) bool, True at non-padded positions.
        mask_ratio: target fraction of valid positions to mask.
    Returns:
        (B, S) bool with True at masked positions (subset of valid_mask).
    """
    rand = torch.rand(valid_mask.shape, device=valid_mask.device, generator=generator)
    chosen = (rand < mask_ratio) & valid_mask
    # Guarantee at least one masked position per row that has any valid ones.
    rows_with_valid = valid_mask.any(dim=1)
    rows_no_mask = rows_with_valid & ~chosen.any(dim=1)
    if rows_no_mask.any():
        # For empty rows pick the first valid position.
        first_valid = valid_mask.float().argmax(dim=1)
        idx = torch.arange(valid_mask.shape[0], device=valid_mask.device)
        chosen[idx[rows_no_mask], first_valid[rows_no_mask]] = True
    return chosen


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged over (mask & target>=0) positions only."""
    valid = mask & (target >= 0)
    n = int(valid.sum().item())
    if n == 0:
        return logits.sum() * 0.0
    logits_v = logits[valid]
    target_v = target[valid]
    return F.cross_entropy(logits_v, target_v)


class AblationCriterion(nn.Module):
    """Compute (poi + time + cat) CE for causal or STM pretraining.

    Args:
        objective: 'causal' or 'stm'.
        mask_ratio: STM masking fraction (ignored for causal).
        poi_weight, time_weight, cat_weight: per-head loss weights (default 1.0).
    """

    def __init__(
        self,
        objective: str,
        mask_ratio: float = 0.3,
        poi_weight: float = 1.0,
        time_weight: float = 1.0,
        cat_weight: float = 1.0,
        contrastive_weight: float = 0.0,
        temperature: float = 0.1,
    ):
        super().__init__()
        if objective not in ("causal", "stm"):
            raise ValueError(f"Unknown objective: {objective!r}")
        self.objective = objective
        self.mask_ratio = float(mask_ratio)
        self.poi_weight = float(poi_weight)
        self.time_weight = float(time_weight)
        self.cat_weight = float(cat_weight)
        # Optional entity-prototype (InfoNCE) term, as in utils/criterion.py.
        self.contrastive_weight = float(contrastive_weight)
        self.temperature = float(temperature)

    def select_eval_positions(self, batch: dict) -> torch.Tensor:
        """Where the loss is scored. For causal: every valid position (target
        will be the *next* visit). For STM: caller provides via batch['mask_positions']."""
        if self.objective == "causal":
            return ~batch["padding_mask"].bool()  # (B, S) valid positions
        return batch["mask_positions"].bool()

    def forward(self, logits_tuple, batch: dict, visit_emb=None, own_agent_emb=None):
        poi_logits, time_logits, cat_logits = logits_tuple
        padding_mask = batch["padding_mask"].bool()
        valid_mask = ~padding_mask  # (B, S)

        if self.objective == "causal":
            # Shift targets left by 1; last position has no successor and is dropped.
            poi_tgt = batch["poi_idx"]
            time_tgt = batch["time_bin"]
            cat_tgt = batch["cat_idx"]
            B, S = valid_mask.shape

            shifted_poi = torch.full_like(poi_tgt, -1)
            shifted_time = torch.zeros_like(time_tgt)
            shifted_cat = torch.full_like(cat_tgt, -1)
            shifted_poi[:, :-1] = poi_tgt[:, 1:]
            shifted_time[:, :-1] = time_tgt[:, 1:]
            shifted_cat[:, :-1] = cat_tgt[:, 1:]

            # A position t is a valid prediction site iff t and t+1 are both
            # non-padded. valid_mask[:, :-1] & valid_mask[:, 1:] over the leading
            # B x (S-1) slice, padded with False at the last position.
            score_mask = torch.zeros_like(valid_mask)
            score_mask[:, :-1] = valid_mask[:, :-1] & valid_mask[:, 1:]

            poi_loss = _masked_ce(poi_logits, shifted_poi, score_mask)
            time_loss = _masked_ce(time_logits, shifted_time, score_mask)
            cat_loss = _masked_ce(cat_logits, shifted_cat, score_mask)
            n_score = int(score_mask.sum().item())

        else:  # 'stm'
            mask_pos = batch["mask_positions"].bool()  # (B, S)
            poi_loss = _masked_ce(poi_logits, batch["poi_idx"], mask_pos)
            time_loss = _masked_ce(time_logits, batch["time_bin"], mask_pos)
            cat_loss = _masked_ce(cat_logits, batch["cat_idx"], mask_pos)
            n_score = int(mask_pos.sum().item())

        loss = (
            self.poi_weight * poi_loss
            + self.time_weight * time_loss
            + self.cat_weight * cat_loss
        )
        # Entity-prototype term (only with --keep_prototype). There is no
        # perturbation on this path, so every valid visit is a normal visit.
        contrastive_val = 0.0
        if self.contrastive_weight > 0.0 and visit_emb is not None and own_agent_emb is not None:
            normal_mask = valid_mask
            valid_agents = normal_mask.any(dim=1)
            if valid_agents.sum().item() >= 2:
                visit_valid = visit_emb[normal_mask]
                batch_indices = torch.arange(
                    normal_mask.shape[0], device=visit_emb.device).unsqueeze(1)
                visit_agent_idx = batch_indices.expand_as(normal_mask)[normal_mask]
                visit_norm = F.normalize(visit_valid, p=2, dim=-1)
                agent_norm = F.normalize(own_agent_emb, p=2, dim=-1)
                sim = visit_norm @ agent_norm.T
                sim[:, ~valid_agents] = -1e9
                sim = sim / max(self.temperature, 1e-6)
                contrastive = F.cross_entropy(sim, visit_agent_idx)
                loss = loss + self.contrastive_weight * contrastive
                contrastive_val = float(contrastive.item())

        loss_dict = {
            "total_loss": loss.item() if isinstance(loss, torch.Tensor) else float(loss),
            "contrastive_loss": contrastive_val,
            "poi_loss": poi_loss.item() if isinstance(poi_loss, torch.Tensor) else float(poi_loss),
            "time_loss": time_loss.item() if isinstance(time_loss, torch.Tensor) else float(time_loss),
            "cat_loss": cat_loss.item() if isinstance(cat_loss, torch.Tensor) else float(cat_loss),
            "n_score": n_score,
        }
        return loss, loss_dict
