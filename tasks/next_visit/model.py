"""Next-visit prediction model.

Same backbone (FeatureEncoder + HumorStack/Transformer) as anomaly /
POI-rec.  Adds a small GMM time head that predicts the time *delta* (in
hours) to the next check-in conditioned on the encoder query and the
ground-truth next-location embedding.

Backbone attribute names match ``tasks.poi_rec.model.POIRecModel`` so any
anomaly-detection or POI-rec checkpoint loads cleanly with strict=False.
The POI head (``query_proj``, ``poi_emb``) is also kept name-identical so
POI-rec checkpoints could be used as a warm-start if desired.
"""

import torch
import torch.nn as nn

from modules.feature_encoder import FeatureEncoder
from modules.humor_layer import HumorStack, HumorStackNoNeighbor, TransformerStack
from modules.time2vec import Time2Vec, LogFreqEmbed


class GMMHead(nn.Module):
    """Predict (weight, loc, scale) of a 1-D Gaussian mixture from a vector."""

    def __init__(self, d_in: int, num_gaussians: int, hidden: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        hidden = hidden or d_in
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.weight = nn.Linear(hidden, num_gaussians)
        self.loc = nn.Linear(hidden, num_gaussians)
        self.scale = nn.Linear(hidden, num_gaussians)
        self.softplus = nn.Softplus()

    def forward(self, x, eps: float = 1e-6):
        h = self.trunk(x)
        return {
            "weight": self.softplus(self.weight(h)) + eps,
            "loc": self.loc(h),
            "scale": self.softplus(self.scale(h)) + eps,
        }


class NextVisitModel(nn.Module):
    """Joint POI + check-in time prediction model.

    Forward signature mirrors POIRecModel but optionally takes the (masked)
    target POI indices used to condition the time head.  Time predictions are
    only computed at masked positions (rec_mask=True).
    """

    def __init__(self, args, n_pois: int):
        super().__init__()

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

        # POI head (kept name-identical to POIRecModel for ckpt portability).
        self.query_proj = nn.Sequential(
            nn.Linear(args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Linear(args.model_dim, args.model_dim),
        )
        self.head_dropout = nn.Dropout(float(getattr(args, "head_dropout", 0.0)))
        self.poi_emb = nn.Embedding(n_pois, args.model_dim)
        nn.init.xavier_uniform_(self.poi_emb.weight)

        # Time head — takes [query, poi_emb[next_poi] (+optional agent_emb)] -> GMM.
        # When ``use_agent_in_time_head=True`` the time head receives the user's
        # explicit agent embedding (mirrors Mobility-LLM's `user_emb` input to the
        # tpp head; their reported T@60 advantage on next-visit rests partly on
        # this user-conditioning).
        self.num_gaussians = int(getattr(args, "num_gaussians", 3))
        self.use_agent_in_time_head = bool(getattr(args, "use_agent_in_time_head", False))
        agent_emb_dim = self.feature_encoder.feature_dim if self.use_agent_in_time_head else 0

        self.use_anchor_time_in_head = bool(getattr(args, "use_anchor_time_in_head", False))
        self.anchor_time_modulo = getattr(args, "anchor_time_modulo", "daily")
        if self.use_anchor_time_in_head:
            self.anchor_time_emb_dim = int(getattr(args, "anchor_time_emb_dim", 64))
            anchor_temporal = getattr(args, "anchor_temporal_embedding", "time2vec")
            if anchor_temporal == "logfreq":
                self.anchor_time2vec = LogFreqEmbed(self.anchor_time_emb_dim)
            else:
                self.anchor_time2vec = Time2Vec(self.anchor_time_emb_dim)
        else:
            self.anchor_time_emb_dim = 0

        time_head_in = 2 * args.model_dim + agent_emb_dim + self.anchor_time_emb_dim
        self.time_head = GMMHead(
            d_in=time_head_in,
            num_gaussians=self.num_gaussians,
            hidden=args.model_dim,
            dropout=float(getattr(args, "head_dropout", 0.0)),
        )

        # Optional Mobility-LLM-style L1-regression time head: predicts a single
        # scalar Δ (in *minutes*) from the same [query, poi_emb[next_poi] (+optional
        # agent_emb)] input.  Trained jointly with the GMM head; T@t evals can use
        # whichever scores better.
        d_in = time_head_in
        d_hid = args.model_dim
        p = float(getattr(args, "head_dropout", 0.0))
        self.time_reg = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid // 4), nn.ReLU(),
            nn.Linear(d_hid // 4, 1),
        )
        # Bias-initialize the final layer so early predictions land near a typical
        # Δ (in minutes) — without this, raw-min L1 takes O(many epochs) just to
        # drift the bias up from 0 (per-step bias gradient ~ reg_loss_weight at
        # lr=1e-4).  Default keeps the random init for backward compatibility.
        reg_bias_init = float(getattr(args, "reg_bias_init", 0.0))
        if reg_bias_init != 0.0:
            with torch.no_grad():
                self.time_reg[-1].bias.fill_(reg_bias_init)

    def encode(self, location, start, stop, category, agent, padding_mask, neighbor_padding_mask):
        x, _ = self.feature_encoder(location, start, stop, category, agent)
        batch_size, seq_len = x.shape[:2]
        x, _ = self.attentions(x, padding_mask, neighbor_padding_mask)
        h = x.reshape(batch_size, seq_len, -1)
        query = self.head_dropout(self.query_proj(h))
        return query

    def _anchor_time_emb(self, anchor_start_masked: torch.Tensor) -> torch.Tensor:
        """Apply the same modulo as the encoder, then time2vec the anchor start."""
        s = anchor_start_masked
        if self.anchor_time_modulo == "weekly":
            s = s % (24 * 7)
        elif self.anchor_time_modulo == "daily":
            s = s % 24
        # else "none" — pass raw timestamps unchanged
        return self.anchor_time2vec(s)

    def _time_head_input(
        self,
        query_masked: torch.Tensor,
        target_poi_idx_masked: torch.Tensor,
        agent_self_masked: torch.Tensor | None,
        anchor_start_masked: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the input to the time heads at masked positions."""
        next_poi_emb = self.poi_emb(target_poi_idx_masked)  # (N, D)
        parts = [query_masked, next_poi_emb]
        if self.use_agent_in_time_head:
            assert agent_self_masked is not None, (
                "use_agent_in_time_head=True requires agent_self_masked"
            )
            agent_emb_self = self.feature_encoder.agent_emb(agent_self_masked)  # (N, F)
            parts.append(agent_emb_self)
        if self.use_anchor_time_in_head:
            assert anchor_start_masked is not None, (
                "use_anchor_time_in_head=True requires anchor_start_masked"
            )
            parts.append(self._anchor_time_emb(anchor_start_masked))
        return torch.cat(parts, dim=-1)

    def time_gmm(self, query_masked: torch.Tensor, target_poi_idx_masked: torch.Tensor,
                 agent_self_masked: torch.Tensor | None = None,
                 anchor_start_masked: torch.Tensor | None = None):
        """Compute GMM params for masked positions."""
        return self.time_head(self._time_head_input(
            query_masked, target_poi_idx_masked, agent_self_masked, anchor_start_masked))

    def time_regression_minutes(self, query_masked: torch.Tensor,
                                target_poi_idx_masked: torch.Tensor,
                                agent_self_masked: torch.Tensor | None = None,
                                anchor_start_masked: torch.Tensor | None = None):
        """Mobility-LLM-style scalar regressor: predicted Δ at masked positions."""
        return self.time_reg(self._time_head_input(
            query_masked, target_poi_idx_masked, agent_self_masked, anchor_start_masked)).squeeze(-1)

    def forward(self, location, start, stop, category, agent,
                padding_mask, neighbor_padding_mask):
        """Encode trajectory; return per-position query and full POI emb table."""
        query = self.encode(location, start, stop, category, agent,
                            padding_mask, neighbor_padding_mask)
        return query, self.poi_emb.weight

    def freeze_backbone(self):
        for p in self.feature_encoder.parameters():
            p.requires_grad = False
        for p in self.attentions.parameters():
            p.requires_grad = False

    def freeze_agent_emb(self):
        for p in self.feature_encoder.agent_emb.parameters():
            p.requires_grad = False

    def freeze_poi_head(self):
        for p in self.query_proj.parameters():
            p.requires_grad = False
        for p in self.poi_emb.parameters():
            p.requires_grad = False
