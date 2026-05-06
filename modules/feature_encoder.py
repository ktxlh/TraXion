"""Feature encoding module for mobility data.

This module encodes various features of stay records into dense embeddings:
- Spatial: Location coordinates using Space2Vec with multi-scale encoding
- Temporal: Start/stop times using Time2Vec with periodic functions
- Agent: Agent identity using learned embeddings
- Category: POI categories using linear projection

All features are encoded to the same dimension and stacked for downstream processing.
"""

import torch
import torch.nn as nn

from modules.space2vec import Space2Vec
from modules.time2vec import Time2Vec, LogFreqEmbed
from modules.decomp_emb import DecomposedEmbedding


class FeatureEncoder(nn.Module):
    """Encode multiple feature types into uniform embeddings.

    Creates dense representations for location, time, agent, and category
    features. All features are encoded to the same dimension (model_dim / num_feats)
    and stacked along the feature dimension.

    Args:
        args: Configuration namespace with model parameters:
            - model_dim: Total embedding dimension
            - n_scales: Number of spatial scales for Space2Vec
            - n_agents: Number of unique agents for embedding lookup
            - n_categories: Number of POI category types
            - lambda_min/max: Spatial scale range for Space2Vec
    """

    def __init__(self, args):
        super(FeatureEncoder, self).__init__()
        model_dim = args.model_dim
        n_categories = args.n_categories
        n_agents = args.n_agents
        n_agent_attributes = args.n_agent_attributes

        self.n_scales = args.n_scales
        self.lambda_min = args.lambda_min
        self.lambda_max = args.lambda_max
        self.time_modulo = getattr(args, "time_modulo", "weekly")
        self.time_scale = getattr(args, "time_scale", 1.0)
        self.no_agent_emb = getattr(args, "no_agent_emb", False)
        self.num_feats = 4 if self.no_agent_emb else 5

        assert (
            model_dim % self.num_feats == 0
        ), f"model_dim {model_dim} must be divisible by num_feats {self.num_feats}"
        feat_dim = model_dim // self.num_feats
        self.feature_dim = feat_dim

        temporal_embedding = getattr(args, "temporal_embedding", "time2vec")
        if temporal_embedding == "logfreq":
            self.time2vec = LogFreqEmbed(feat_dim)
        else:
            self.time2vec = Time2Vec(feat_dim)
        self.loc2vec = Space2Vec(feat_dim, self.lambda_min, self.lambda_max, self.n_scales)
        if getattr(args, "standard_agent_emb", False):
            self.agent_emb = nn.Embedding(n_agents, feat_dim)
        else:
            self.agent_emb = DecomposedEmbedding(n_agents, n_agent_attributes, feat_dim)
        self.category_emb = nn.Linear(n_categories, feat_dim, bias=False)

    def forward(self, location, start, stop, category, agent):
        """Input start shape (*); output shape (*, num_feats, model_dim)"""

        # Convert timestamps according to time_modulo setting
        if self.time_modulo == "weekly":
            start = start % (24 * 7)
            stop = stop % (24 * 7)
        elif self.time_modulo == "daily":
            start = start % 24
            stop = stop % 24
        # else: "none" — pass raw timestamps unchanged

        if self.time_scale != 1.0:
            start = start / self.time_scale
            stop = stop / self.time_scale

        # Avoid mutating input tensors in-place; treat empty vectors as uniform.
        category_sums = category.sum(dim=-1, keepdim=True)
        empty_category_mask = category_sums == 0
        safe_category = torch.where(empty_category_mask, torch.ones_like(category), category)
        safe_category = safe_category / safe_category.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        loc_emb = self.loc2vec(location)
        start_emb = self.time2vec(start)
        stop_emb = self.time2vec(stop)
        agent_emb = self.agent_emb(agent)
        cat_emb = self.category_emb(safe_category)

        # own_agent_emb is always computed from the learned embedding for contrastive loss;
        # when no_agent_emb=True we exclude the agent feature slot entirely.
        own_agent_emb = agent_emb[:, 0, 0, :]  # (B, D)

        if self.no_agent_emb:
            x = [loc_emb, start_emb, stop_emb, cat_emb]
        else:
            x = [loc_emb, start_emb, stop_emb, agent_emb, cat_emb]
        
        # Keep feature axis before clique axis: (B, S, num_feats, C, D)
        return torch.stack(x, dim=2), own_agent_emb
