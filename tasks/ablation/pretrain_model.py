"""Ablation pretraining model (A7 causal / A8 STM).

Same backbone as `modules.model.Model` (FeatureEncoder + HumorStack), with
identical attribute names (``feature_encoder``, ``attentions``) so that
existing finetune loaders reuse the saved weights via ``strict=False``.

Adds:
  * ``poi_head``   : Linear(model_dim → n_pois)
  * ``time_head``  : Linear(model_dim → n_time_bins)
  * ``cat_head``   : Linear(model_dim → n_categories)
  * ``mask_emb``   : (num_feats, feature_dim) — learned per-feature [MASK] tile
                     used to overwrite a masked visit's agent-slot features in
                     STM mode. Lives on the model so it loads with the weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.feature_encoder import FeatureEncoder
from modules.humor_layer import (
    HumorStack,
    HumorStackNoNeighbor,
    TransformerStack,
)


class AblationPretrainModel(nn.Module):
    def __init__(self, args, n_pois: int, n_time_bins: int = 24):
        super().__init__()

        self.feature_encoder = FeatureEncoder(args)
        feature_dim = self.feature_encoder.feature_dim
        num_feats = self.feature_encoder.num_feats
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

        self.poi_head = nn.Linear(args.model_dim, n_pois)
        self.time_head = nn.Linear(args.model_dim, n_time_bins)
        self.cat_head = nn.Linear(args.model_dim, args.n_categories)

        # Learnable [MASK] embedding tiled across the num_feats axis. Used by
        # STM to overwrite the agent-slot features at masked positions before
        # they enter the attention stack. Causal mode does not touch this.
        self.mask_emb = nn.Parameter(torch.zeros(num_feats, feature_dim))
        nn.init.normal_(self.mask_emb, std=0.02)

    def encode(
        self,
        location: torch.Tensor,
        start: torch.Tensor,
        stop: torch.Tensor,
        category: torch.Tensor,
        agent: torch.Tensor,
        padding_mask: torch.Tensor,
        neighbor_padding_mask: torch.Tensor,
        *,
        mask_positions: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Run the backbone and return per-position hidden states (B, S, model_dim).

        If ``mask_positions`` (B, S, bool) is given, the agent slot's per-feature
        embeddings at those positions are replaced by ``mask_emb`` before
        attention. Neighbor slots are untouched (they still carry context, which
        is fine — we only score the agent stream anyway).

        If ``causal=True``, a triangular attn_mask is sent down to the
        sequence-axis attention so position t sees only positions 0..t.
        """
        x, _own_agent_emb = self.feature_encoder(location, start, stop, category, agent)
        # x: (B, S, F, C, D)

        if mask_positions is not None:
            mp = mask_positions.bool()  # (B, S)
            # mask_emb: (F, D) -> broadcast over (B, S, F, 1, D) at agent slot
            tile = self.mask_emb.view(1, 1, -1, 1, x.shape[-1]).to(x.dtype)
            agent_slot = x[:, :, :, :1, :]
            mp_b = mp[:, :, None, None, None]  # (B, S, 1, 1, 1)
            agent_slot = torch.where(mp_b, tile.expand_as(agent_slot), agent_slot)
            x = torch.cat([agent_slot, x[:, :, :, 1:, :]], dim=3)

        attn_mask: torch.Tensor | None = None
        if causal:
            S = x.shape[1]
            # bool mask: True above the diagonal = "ignore future positions". Same
            # dtype as key_padding_mask (bool) avoids a torch deprecation warning.
            attn_mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1,
            )

        x_out, _ = self.attentions(
            x, padding_mask, neighbor_padding_mask, attn_mask=attn_mask,
        )
        # x_out: (B, S, F, D) — the agent's own embedding stack
        h = x_out.reshape(x_out.shape[0], x_out.shape[1], -1)  # (B, S, model_dim)
        return h

    def predict(self, h: torch.Tensor):
        return self.poi_head(h), self.time_head(h), self.cat_head(h)

    def forward(
        self,
        location, start, stop, category, agent,
        padding_mask, neighbor_padding_mask,
        *,
        mask_positions=None,
        causal: bool = False,
    ):
        h = self.encode(
            location, start, stop, category, agent,
            padding_mask, neighbor_padding_mask,
            mask_positions=mask_positions, causal=causal,
        )
        return self.predict(h)
