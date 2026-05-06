import torch
from torch import nn
import torch.nn.functional as F


class CustomCriterion(nn.Module):
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        temperature: float = 0.07,
        agent_level_bce: bool = False,
        no_contrastive: bool = False,
        no_bce: bool = False,
        bce_pos_weight: float | None = None,
    ):
        super().__init__()
        if no_bce and no_contrastive:
            raise ValueError("Cannot disable both BCE and contrastive losses; total loss would be zero.")
        # Optional BCE pos_weight to handle severe class imbalance (e.g. LANL
        # ~0.7% real positives at fine-tune).  Default None = unweighted BCE,
        # preserving prior behaviour for synthetic-perturbation fine-tunes
        # where the perturbation rate is balanced.
        if bce_pos_weight is not None and bce_pos_weight > 0:
            pw = torch.tensor([float(bce_pos_weight)])
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.agent_level_bce = agent_level_bce
        self.no_contrastive = no_contrastive
        self.no_bce = no_bce

    def forward(self, logits, visit_emb, own_agent_emb, batch):
        """
        logits: (batch_size, seq_len)
        visit_emb: (batch_size, seq_len, feature_dim)
        own_agent_emb: (batch_size, feature_dim)
        """
        padding_mask = batch["padding_mask"]  # (batch_size, seq_len)
        anomalous = batch["stay_anomalous"].float()  # (batch_size, seq_len)
        valid_mask = ~padding_mask
        
        # Guard against degenerate all-padding micro-batches.
        if not valid_mask.any():
            zero = logits.sum() * 0.0
            loss_dict = {"total_loss": 0.0}
            if not self.no_bce:
                loss_dict["bce_loss"] = 0.0
            if not self.no_contrastive:
                loss_dict["contrastive_loss"] = 0.0
                loss_dict["contrastive_weight"] = self.contrastive_weight
            return zero, loss_dict

        # Main anomaly detection objective.
        if self.no_bce:
            bce_loss = logits.sum() * 0.0  # zero tensor that preserves graph
        elif self.agent_level_bce:
            # Urban datasets have agent-level labels only: if any stay in a sequence is
            # altered, all visits carry the same anomalous=True label.  Max-pool logits
            # across valid positions per agent and compare against the agent-level label.
            # stay_anomalous padding value is False (0.0), so anomalous.max(dim=1) is safe.
            agent_valid = valid_mask.any(dim=1)                          # (batch_size,)
            logits_filled = logits.masked_fill(~valid_mask, float("-inf"))
            agent_logits = logits_filled[agent_valid].max(dim=1).values  # (n_valid,)
            agent_labels = anomalous[agent_valid].max(dim=1).values      # (n_valid,)
            bce_loss = self.bce_loss(agent_logits, agent_labels)
        else:
            bce_loss = self.bce_loss(logits[valid_mask], anomalous[valid_mask])

        if self.no_contrastive or self.contrastive_weight == 0.0:
            ld = {"total_loss": bce_loss.item()}
            if not self.no_bce:
                ld["bce_loss"] = bce_loss.item()
            return bce_loss, ld

        # Contrastive auxiliary objective — exclude anomalous visits so that agent
        # prototypes reflect normal behaviour only.
        normal_mask = valid_mask & ~anomalous.bool()
        visit_valid = visit_emb[normal_mask]

        # One prototype per batch agent; ignore agents that have no normal visits.
        agent_proto = own_agent_emb
        valid_agents = normal_mask.any(dim=1)

        # If fewer than 2 agents have normal visits, negatives are undefined.
        if valid_agents.sum().item() < 2:
            ld = {"total_loss": bce_loss.item()}
            if not self.no_bce:
                ld["bce_loss"] = bce_loss.item()
            return bce_loss, ld

        # For each normal visit, the positive target is its own batch agent index.
        batch_indices = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
        visit_agent_idx = batch_indices.expand_as(normal_mask)[normal_mask]

        visit_norm = F.normalize(visit_valid, p=2, dim=-1)
        agent_norm = F.normalize(agent_proto, p=2, dim=-1)

        # InfoNCE over batch agents: positives are same-agent, negatives are different agents.
        sim = visit_norm @ agent_norm.T

        # Exclude agents with no valid visits from candidate negatives.
        sim[:, ~valid_agents] = -1e9
        sim = sim / max(self.temperature, 1e-6)
        contrastive_loss = F.cross_entropy(sim, visit_agent_idx)

        loss = bce_loss + self.contrastive_weight * contrastive_loss
        loss_dict = {
            "total_loss": loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "contrastive_weight": self.contrastive_weight,
        }
        if not self.no_bce:
            loss_dict["bce_loss"] = bce_loss.item()
        return loss, loss_dict