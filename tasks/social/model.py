"""Social relationship inference model.

Shares the FeatureEncoder + HumorStack backbone with the anomaly and POI-rec
models (identical attribute names ``feature_encoder`` / ``attentions`` so that
anomaly checkpoints load via ``strict=False`` — POI-rec-style backbone reuse).

Head: masked mean+max pooling over valid visits -> linear -> L2-normalized
per-agent embedding, then a symmetric pair scorer that consumes both the learned
pair features and K precomputed handcrafted co-visitation / graph features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.feature_encoder import FeatureEncoder
from modules.graph_refiner import GraphRefiner
from modules.humor_layer import HumorStack, HumorStackNoNeighbor, TransformerStack


class SocialModel(nn.Module):
    def __init__(self, args, pair_feature_dim: int):
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

        self.emb_dim = int(args.social_emb_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(2 * args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Linear(args.model_dim, self.emb_dim),
        )

        self.pair_feature_dim = int(pair_feature_dim)
        norm_kind = getattr(args, "pair_feat_norm", "layernorm")
        self.pair_feat_norm_kind = norm_kind
        if self.pair_feature_dim > 0:
            if norm_kind == "layernorm":
                self.pair_feat_norm = nn.LayerNorm(self.pair_feature_dim)
            elif norm_kind == "batchnorm":
                self.pair_feat_norm = nn.BatchNorm1d(self.pair_feature_dim)
            elif norm_kind == "zscore":
                self.register_buffer("pair_feat_mean", torch.zeros(self.pair_feature_dim))
                self.register_buffer("pair_feat_std", torch.ones(self.pair_feature_dim))
                self.pair_feat_norm = None  # handled inline in score_pairs
            elif norm_kind == "none":
                self.pair_feat_norm = nn.Identity()
            else:
                raise ValueError(f"Unknown pair_feat_norm: {norm_kind}")
        else:
            self.pair_feat_norm = nn.Identity()
        self.disable_emb_branch = bool(getattr(args, "disable_emb_branch", False))
        self.pair_head_split = bool(getattr(args, "pair_head_split", False))

        # Optional 1-hop graph refinement of per-agent embeddings (fine-tune only).
        # See modules/graph_refiner.py. Wired through `score_pairs_with_ids`,
        # which the trainer/criterion call when --gnn_head is set.
        self.use_gnn_head = bool(getattr(args, "gnn_head", False))
        if self.use_gnn_head:
            gate_init = float(getattr(args, "gnn_gate_init", -2.0))
            self.graph_refiner = GraphRefiner(self.emb_dim, gate_init=gate_init)
        else:
            self.graph_refiner = None
        # Buffers populated by the trainer once per epoch (cache) and once on
        # first use (neighbor index). Registered so checkpoints can persist them.
        self.register_buffer("_emb_cache", torch.zeros(0), persistent=False)
        self.register_buffer("_neighbor_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("_neighbor_offsets", torch.zeros(0, dtype=torch.long), persistent=False)
        ph = int(args.pair_head_hidden)
        if self.pair_head_split and self.pair_feature_dim > 0:
            # Split architecture: emb_head consumes only sym(emb); pf_head is a
            # dedicated linear (LR-like) head on normed pair features. Their
            # logits are summed for the final score.
            emb_in = 3 * self.emb_dim if not self.disable_emb_branch else 0
            # emb_head; only built if we have an embedding branch
            if emb_in > 0:
                self.pair_head = nn.Sequential(
                    nn.Linear(emb_in, ph),
                    nn.ReLU(),
                    nn.Dropout(float(getattr(args, "pair_head_dropout", 0.1))),
                    nn.Linear(ph, ph),
                    nn.ReLU(),
                    nn.Linear(ph, 1),
                )
            else:
                self.pair_head = None
            self.pf_head = nn.Linear(self.pair_feature_dim, 1)
        else:
            emb_in = 0 if self.disable_emb_branch else 3 * self.emb_dim
            pair_in = emb_in + self.pair_feature_dim
            self.pair_head = nn.Sequential(
                nn.Linear(pair_in, ph),
                nn.ReLU(),
                nn.Dropout(float(getattr(args, "pair_head_dropout", 0.1))),
                nn.Linear(ph, ph),
                nn.ReLU(),
                nn.Linear(ph, 1),
            )
            self.pf_head = None

    # ------------------------------------------------------------------ GNN
    def set_neighbor_index(
        self, neighbor_idx: torch.Tensor, neighbor_offsets: torch.Tensor
    ) -> None:
        """Register the train-graph CSR index. Called once by the trainer."""
        self._neighbor_idx = neighbor_idx.to(self._neighbor_idx.device if self._neighbor_idx.numel() > 0 else neighbor_idx.device)
        self._neighbor_offsets = neighbor_offsets.to(self._neighbor_offsets.device if self._neighbor_offsets.numel() > 0 else neighbor_offsets.device)

    def set_emb_cache(self, emb_cache: torch.Tensor) -> None:
        """Register a fresh per-agent embedding cache (detached). Called by the
        trainer at the start of each epoch and once before each eval."""
        self._emb_cache = emb_cache.detach()

    def refine(self, e: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        """Apply GNN refinement if enabled, else identity. Output is re-L2-normalized
        to keep the cosine-similarity scale consistent with the un-refined branch."""
        if self.graph_refiner is None:
            return e
        if self._emb_cache.numel() == 0 or self._neighbor_idx.numel() == 0:
            return e  # cache not yet populated (e.g. first batch before warmup)
        e_ref = self.graph_refiner(
            e,
            agent_ids,
            self._emb_cache,
            self._neighbor_idx,
            self._neighbor_offsets,
        )
        return F.normalize(e_ref, dim=-1)

    def set_pair_feat_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        assert self.pair_feat_norm_kind == "zscore"
        with torch.no_grad():
            self.pair_feat_mean.copy_(mean.to(self.pair_feat_mean))
            self.pair_feat_std.copy_(std.clamp_min(1e-6).to(self.pair_feat_std))

    # ------------------------------------------------------------------
    def encode(
        self, location, start, stop, category, agent, padding_mask, neighbor_padding_mask
    ) -> torch.Tensor:
        """Returns L2-normalized per-agent embedding (B, emb_dim)."""
        x, _ = self.feature_encoder(location, start, stop, category, agent)
        batch_size, seq_len = x.shape[:2]
        x, _ = self.attentions(x, padding_mask, neighbor_padding_mask)
        h = x.reshape(batch_size, seq_len, -1)  # (B, S, model_dim)

        valid = (~padding_mask).float().unsqueeze(-1)  # (B, S, 1)
        h_sum = (h * valid).sum(dim=1)
        h_count = valid.sum(dim=1).clamp_min(1.0)
        h_mean = h_sum / h_count

        neg_inf = torch.finfo(h.dtype).min
        h_masked = h.masked_fill(padding_mask.unsqueeze(-1), neg_inf)
        h_max = h_masked.max(dim=1).values
        # Agents with no valid positions (shouldn't happen) -> zero out maxes
        all_padded = padding_mask.all(dim=1, keepdim=True)  # (B, 1)
        h_max = torch.where(all_padded, torch.zeros_like(h_max), h_max)

        pooled = torch.cat([h_mean, h_max], dim=-1)  # (B, 2*model_dim)
        e = self.pool_proj(pooled)
        return F.normalize(e, dim=-1)

    # ------------------------------------------------------------------
    def score_pairs(
        self, e_i: torch.Tensor, e_j: torch.Tensor, pair_feats: torch.Tensor | None
    ) -> torch.Tensor:
        """Symmetric pair score logits: f(i,j) == f(j,i).

        Args:
            e_i, e_j: (N, emb_dim) L2-normalized embeddings.
            pair_feats: (N, pair_feature_dim) or None.
        Returns: (N,) logits.
        """
        sym = torch.cat([e_i + e_j, (e_i - e_j).abs(), e_i * e_j], dim=-1)
        if self.pair_feature_dim > 0:
            if pair_feats is None:
                raise ValueError("pair_feats required when pair_feature_dim > 0")
            if self.pair_feat_norm_kind == "zscore":
                pf = (pair_feats - self.pair_feat_mean) / self.pair_feat_std
            else:
                pf = self.pair_feat_norm(pair_feats)
        else:
            pf = None

        if self.pair_head_split:
            # Separate emb + pf branches summed into final logit
            logit = 0.0
            if self.pair_head is not None:
                logit = logit + self.pair_head(sym).squeeze(-1)
            if pf is not None:
                logit = logit + self.pf_head(pf).squeeze(-1)
            if isinstance(logit, float):  # both branches empty
                logit = torch.zeros(e_i.shape[0], device=e_i.device, dtype=e_i.dtype)
            return logit
        # Concatenated single-head path (original)
        if self.pair_feature_dim > 0:
            if self.disable_emb_branch:
                feats = pf
            else:
                feats = torch.cat([sym, pf], dim=-1)
        else:
            feats = torch.zeros_like(sym[:, :0]) if self.disable_emb_branch else sym
        return self.pair_head(feats).squeeze(-1)

    def cosine_scores(self, e_i: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        """Raw cosine similarity between two (N, D) batches of L2-normed embeddings."""
        return (e_i * e_j).sum(dim=-1)

    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.feature_encoder.parameters():
            p.requires_grad = False
        for p in self.attentions.parameters():
            p.requires_grad = False
