"""Per-stay mortality classifier for the eICU-CRD demo task.

Same FeatureEncoder + HumorStack backbone as the anomaly / POI-rec / TUL /
social models (identical attribute names so anomaly checkpoints load with
``strict=False``).

Head: masked mean+max pooling over valid visits → MLP → binary logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.feature_encoder import FeatureEncoder
from modules.humor_layer import HumorStack, HumorStackNoNeighbor, TransformerStack


class MortalityModel(nn.Module):
    def __init__(self, args, tab_feat_dim: int = 0):
        super().__init__()
        self.tab_feat_dim = int(tab_feat_dim)
        self.feature_encoder = FeatureEncoder(args)
        feature_dim = self.feature_encoder.feature_dim
        if feature_dim % args.n_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by n_heads ({args.n_heads})"
            )
        use_transformer = getattr(args, "use_transformer_encoder", False)
        no_neighbor_attn = getattr(args, "no_neighbor_attn", False)
        if use_transformer:
            self.attentions = TransformerStack(
                feature_dim=feature_dim,
                nhead=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                num_humor_layers=args.num_layers,
                agent_only_concat=getattr(args, "agent_only_concat", False),
                model_dim=args.model_dim,
            )
        elif no_neighbor_attn:
            self.attentions = HumorStackNoNeighbor(
                feature_dim=feature_dim,
                nhead=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                num_layers=args.num_layers,
                clique_size=args.clique_size,
            )
        else:
            self.attentions = HumorStack(
                feature_dim=feature_dim,
                nhead=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                num_layers=args.num_layers,
                clique_size=args.clique_size,
            )

        emb_dim = int(getattr(args, "eicu_emb_dim", args.model_dim // 2))
        dropout = float(getattr(args, "eicu_head_dropout", 0.2))
        self.pool_proj = nn.Sequential(
            nn.Linear(2 * args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(args.model_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emb_dim = emb_dim

        # Optional tabular side-feature head: (B, tab_feat_dim) → emb_dim,
        # concatenated with the pooled visit embedding before the binary classifier.
        if self.tab_feat_dim > 0:
            tab_hidden = int(getattr(args, "tab_feat_hidden", 64))
            self.tab_proj = nn.Sequential(
                nn.Linear(self.tab_feat_dim, tab_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(tab_hidden, emb_dim),
            )
            classifier_in = 2 * emb_dim
        else:
            self.tab_proj = None
            classifier_in = emb_dim
        self.classifier = nn.Linear(classifier_in, 1)

    def encode(
        self, location, start, stop, category, agent, padding_mask, neighbor_padding_mask
    ) -> torch.Tensor:
        x, _ = self.feature_encoder(location, start, stop, category, agent)
        batch_size, seq_len = x.shape[:2]
        x, _ = self.attentions(x, padding_mask, neighbor_padding_mask)
        h = x.reshape(batch_size, seq_len, -1)  # (B, S, model_dim)

        valid = (~padding_mask).float().unsqueeze(-1)
        h_sum = (h * valid).sum(dim=1)
        h_count = valid.sum(dim=1).clamp_min(1.0)
        h_mean = h_sum / h_count

        neg_inf = torch.finfo(h.dtype).min
        h_masked = h.masked_fill(padding_mask.unsqueeze(-1), neg_inf)
        h_max = h_masked.max(dim=1).values
        all_padded = padding_mask.all(dim=1, keepdim=True)
        h_max = torch.where(all_padded, torch.zeros_like(h_max), h_max)

        pooled = torch.cat([h_mean, h_max], dim=-1)
        return self.pool_proj(pooled)

    def forward(
        self, location, start, stop, category, agent, padding_mask, neighbor_padding_mask,
        tab_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.encode(location, start, stop, category, agent,
                          padding_mask, neighbor_padding_mask)
        if self.tab_proj is not None:
            if tab_feat is None:
                raise ValueError("tab_feat required when tab_feat_dim > 0")
            t = self.tab_proj(tab_feat)
            emb = torch.cat([emb, t], dim=-1)
        return self.classifier(emb).squeeze(-1)  # (B,)

    def freeze_backbone(self) -> None:
        for p in self.feature_encoder.parameters():
            p.requires_grad = False
        for p in self.attentions.parameters():
            p.requires_grad = False

    def freeze_agent_emb(self) -> None:
        for p in self.feature_encoder.agent_emb.parameters():
            p.requires_grad = False
