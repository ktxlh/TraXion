"""Next-POI recommendation model.

Shares the same backbone (FeatureEncoder + HumorStack) as the anomaly
detection Model.  The attribute names for backbone components are kept
identical so that anomaly-detection checkpoints can be loaded with
``strict=False``.
"""

import torch
import torch.nn as nn

from modules.feature_encoder import FeatureEncoder
from modules.humor_layer import HumorStack, HumorStackNoNeighbor, TransformerStack


class POIRecModel(nn.Module):
    """Next-POI recommendation model.

    Architecture:
        FeatureEncoder + Attention (shared backbone)
        → query_proj (MLP → model_dim)
        → dot-product scoring against learned POI embeddings

    Args:
        args: Configuration namespace (same as anomaly Model).
        n_pois: Size of the POI vocabulary.
    """

    def __init__(self, args, n_pois: int):
        super().__init__()

        # --- Backbone (same attribute names as anomaly Model) ---------------
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

        # --- POI recommendation head ----------------------------------------
        # query_proj Sequential is kept structurally identical (Linear, ReLU, Linear)
        # to preserve backward compatibility with existing POI-rec checkpoints.
        # Head dropout is applied externally in forward() so it does not perturb
        # state_dict keys.
        self.query_proj = nn.Sequential(
            nn.Linear(args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Linear(args.model_dim, args.model_dim),
        )
        self.head_dropout = nn.Dropout(float(getattr(args, "head_dropout", 0.0)))
        self.poi_emb = nn.Embedding(n_pois, args.model_dim)
        nn.init.xavier_uniform_(self.poi_emb.weight)

    def forward(self, location, start, stop, category, agent, padding_mask, neighbor_padding_mask):
        """Encode trajectory and produce query vectors for next-POI ranking.

        Returns:
            query: (batch_size, seq_len, model_dim) — query vectors per position.
            poi_weight: (n_pois, model_dim) — full POI embedding table.
        """
        x, _own_agent_emb = self.feature_encoder(location, start, stop, category, agent)
        batch_size, seq_len = x.shape[:2]
        x, _mean_agent_feat_attn = self.attentions(x, padding_mask, neighbor_padding_mask)
        h = x.reshape(batch_size, seq_len, -1)  # (B, S, model_dim)
        query = self.head_dropout(self.query_proj(h))
        return query, self.poi_emb.weight

    def freeze_backbone(self):
        """Freeze backbone parameters (feature_encoder + attentions)."""
        for param in self.feature_encoder.parameters():
            param.requires_grad = False
        for param in self.attentions.parameters():
            param.requires_grad = False

    def freeze_agent_emb(self):
        for param in self.feature_encoder.agent_emb.parameters():
            param.requires_grad = False
