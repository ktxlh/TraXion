"""Main model architecture for anomaly detection in mobility data.

This module defines the top-level Model class that orchestrates feature encoding,
attention mechanisms, and classification for detecting anomalous mobility patterns.
"""

import torch.nn as nn

from modules.classifier import Classifier
from modules.feature_encoder import FeatureEncoder
from modules.humor_layer import HumorStack, HumorStackNoNeighbor, TransformerStack


class Model(nn.Module):
    """Main anomaly detection model with configurable architecture.

    The model consists of:
    1. FeatureEncoder: Encodes spatial, temporal, agent, and categorical features
    2. Attention/Transform layers: Process encoded features
    3. Classifier: Binary classification for anomaly detection

    Args:
        args: Configuration namespace containing model hyperparameters
    """

    def __init__(self, args):
        super().__init__()
        self.feature_encoder = FeatureEncoder(args)
        self.classifier = Classifier(args.model_dim)
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
        # For contrastive learning between visit embedding and agent embedding
        self.visit_emb_proj = nn.Sequential(
            nn.Linear(args.model_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, feature_dim)
        )

    def forward(self, location, start, stop, category, agent, padding_mask, neighbor_padding_mask):
        # x: (batch_size, seq_len, num_feats, clique_size, feature_dim)
        x, own_agent_emb = self.feature_encoder(location, start, stop, category, agent)
        batch_size, seq_len = x.shape[:2]
        x, mean_agent_feat_attn = self.attentions(x, padding_mask, neighbor_padding_mask)
        # x: own visits with shape (batch_size, seq_len, num_feats, feature_dim)
        h = x.reshape(batch_size, seq_len, -1)
        # h: (batch_size, seq_len, model_dim)
        logits = self.classifier(h)
        visit_emb = self.visit_emb_proj(h)
        return logits, visit_emb, own_agent_emb, mean_agent_feat_attn
