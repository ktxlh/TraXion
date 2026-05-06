"""Joint POI + time loss for next-visit prediction.

Total loss = POI sampled-softmax CE + ``time_loss_weight`` * GMM NLL on
the next check-in time delta (in hours).
"""

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


def gmm_nll(weight: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of ``target`` under a 1-D GMM.

    All tensors have the same first dim N. ``target`` is (N,).
    """
    weight = weight / weight.sum(dim=-1, keepdim=True)
    mix = D.Categorical(probs=weight)
    comp = D.Normal(loc, scale)
    gmm = D.MixtureSameFamily(mix, comp)
    return -gmm.log_prob(target)


class NextVisitCriterion(nn.Module):
    """Sampled-softmax POI loss + GMM NLL on time delta.

    With ``time_target='log'`` the GMM models log(Δ_hours + 1) instead of
    Δ_hours.  Heavy right tails of inter-check-in gaps are well-fit by a
    log-normal-like density; this is the same inductive bias as
    Hawkes-style point-process baselines (A-NHP).  We add a Jacobian
    correction ``-log(Δ + 1)`` to the NLL so the loss is the true density
    NLL in raw-hours space (training and eval use the same density).
    """

    def __init__(
        self,
        n_neg_samples: int = 256,
        temperature: float = 0.1,
        label_smoothing: float = 0.0,
        use_in_batch_neg: bool = False,
        time_loss_weight: float = 1.0,
        time_target: str = "raw",
        reg_loss_weight: float = 0.0,
        reg_target: str = "raw",
    ):
        super().__init__()
        self.n_neg_samples = n_neg_samples
        self.temperature = temperature
        self.label_smoothing = float(label_smoothing)
        self.use_in_batch_neg = bool(use_in_batch_neg)
        self.time_loss_weight = float(time_loss_weight)
        self.reg_loss_weight = float(reg_loss_weight)
        if time_target not in {"raw", "log"}:
            raise ValueError(f"time_target must be 'raw' or 'log', got {time_target!r}")
        self.time_target = time_target
        if reg_target not in {"raw", "log"}:
            raise ValueError(f"reg_target must be 'raw' or 'log', got {reg_target!r}")
        self.reg_target = reg_target

    def forward(self, model, query, poi_weight, batch):
        target_poi_idx = batch["target_poi_idx"]
        rec_mask = batch["rec_mask"]
        target_dt = batch["target_time_delta"]

        valid_queries = query[rec_mask]
        valid_targets = target_poi_idx[rec_mask]
        valid_dt = target_dt[rec_mask]
        n_valid = valid_queries.shape[0]

        if n_valid == 0:
            zero = query.sum() * 0.0
            return zero, {"loss": 0.0, "rec_loss": 0.0, "time_nll": 0.0}

        # --- POI sampled-softmax CE ---
        n_pois = poi_weight.shape[0]
        pos_emb = poi_weight[valid_targets]
        pos_scores = (valid_queries * pos_emb).sum(dim=-1, keepdim=True)
        neg_idx = torch.randint(
            0, n_pois, (n_valid, self.n_neg_samples), device=query.device
        )
        neg_emb = poi_weight[neg_idx]
        neg_scores = torch.bmm(neg_emb, valid_queries.unsqueeze(-1)).squeeze(-1)

        if self.use_in_batch_neg:
            ibn_scores = valid_queries @ pos_emb.t()
            eye = torch.eye(n_valid, dtype=torch.bool, device=query.device)
            same_target = valid_targets.unsqueeze(0) == valid_targets.unsqueeze(1)
            dup_mask = eye | same_target
            ibn_scores = ibn_scores.masked_fill(dup_mask, float("-inf"))
            logits = torch.cat([pos_scores, neg_scores, ibn_scores], dim=-1)
        else:
            logits = torch.cat([pos_scores, neg_scores], dim=-1)

        logits = logits / self.temperature
        labels = torch.zeros(n_valid, dtype=torch.long, device=query.device)
        rec_loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        # --- Time GMM NLL ---
        agent_self_masked = None
        if getattr(model, "use_agent_in_time_head", False):
            # batch["agent"] has shape (B, S, K); own agent at K=0, constant along S.
            agent_self_masked = batch["agent"][..., 0][rec_mask]
        anchor_start_masked = None
        if getattr(model, "use_anchor_time_in_head", False):
            # batch["start"] has shape (B, S, K); take own-slice K=0 to align with rec_mask (B, S).
            anchor_start_masked = batch["start"][..., 0][rec_mask]
        gmm_params = model.time_gmm(valid_queries, valid_targets, agent_self_masked,
                                    anchor_start_masked)
        if self.time_target == "log":
            target_for_gmm = torch.log1p(valid_dt.clamp_min(0.0))
            jacobian = torch.log1p(valid_dt.clamp_min(0.0))  # |dY/dΔ| = 1/(Δ+1)
            nll_log = gmm_nll(
                gmm_params["weight"], gmm_params["loc"], gmm_params["scale"], target_for_gmm
            )
            time_nll = (nll_log + jacobian).mean()
        else:
            time_nll = gmm_nll(
                gmm_params["weight"], gmm_params["loc"], gmm_params["scale"], valid_dt
            ).mean()

        loss = rec_loss + self.time_loss_weight * time_nll
        logs = {
            "loss": loss.item(),
            "rec_loss": rec_loss.item(),
            "time_nll": time_nll.item(),
        }

        # --- Mobility-LLM-style L1 regression on Δ (optional) ---
        if self.reg_loss_weight > 0.0 and hasattr(model, "time_reg"):
            reg_pred = model.time_regression_minutes(valid_queries, valid_targets,
                                                    agent_self_masked,
                                                    anchor_start_masked)
            gt_min = valid_dt.clamp_min(0.0) * 60.0
            if self.reg_target == "log":
                # Train on log(min + 1); compresses heavy tail so L1 is bounded
                # and a median estimator is easier to fit at our lr/batch.
                gt_for_reg = torch.log1p(gt_min)
                time_l1 = F.l1_loss(reg_pred, gt_for_reg)
            else:
                time_l1 = F.l1_loss(reg_pred, gt_min)
            loss = loss + self.reg_loss_weight * time_l1
            logs["loss"] = loss.item()
            logs["time_l1"] = time_l1.item()

        return loss, logs
