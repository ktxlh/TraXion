"""Losses for social link prediction.

Given per-batch embeddings and the global train friend graph, construct:
    - Positive pairs:  all in-batch (i, j) with (i, j) in train_edges
    - Negative pairs:  random in-batch (i, k) with (i, k) not in any observed edge
and optimize BPR on the pair-scorer logits + supervised InfoNCE on the raw
embedding cosine similarities. The two terms are complementary: BPR shapes the
pair MLP directly toward the ranking metric; InfoNCE pulls the embedding
geometry so friends cluster, which is what helps the handcrafted-feature-free
branch of the scorer.

If no positives appear in the batch (rare given the edge-centric sampler),
the loss is zero for that step.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SocialCriterion(nn.Module):
    def __init__(
        self,
        friend_graph: dict[int, set[int]],
        observed_edges: set[tuple[int, int]],
        pair_feature_fn=None,
        n_neg_per_pos: int = 8,
        bpr_weight: float = 1.0,
        infonce_weight: float = 0.3,
        infonce_temperature: float = 0.1,
        random_neg_weight: float = 0.0,
        random_neg_batch: int = 256,
        hard_random_neg_weight: float = 0.0,
        hard_random_neg_batch: int = 256,
        n_agents: int | None = None,
        train_edges: np.ndarray | None = None,
    ):
        super().__init__()
        self.friend_graph = friend_graph
        self.observed_edges = observed_edges
        self.pair_feature_fn = pair_feature_fn  # callable: (n, 2) int array -> (n, K) np.float32
        self.n_neg_per_pos = int(n_neg_per_pos)
        self.bpr_weight = float(bpr_weight)
        self.infonce_weight = float(infonce_weight)
        self.infonce_temperature = float(infonce_temperature)
        self.random_neg_weight = float(random_neg_weight)
        self.random_neg_batch = int(random_neg_batch)
        # New: BCE on the *full* pair-scorer using cached agent embeddings, with
        # uniformly-random non-edges as negatives. Closes the train/test
        # distribution gap noted in EXPERIMENTS.md (in-batch BPR negatives are
        # edge-adjacent; random-pair test negatives are not).
        self.hard_random_neg_weight = float(hard_random_neg_weight)
        self.hard_random_neg_batch = int(hard_random_neg_batch)
        self.n_agents = int(n_agents) if n_agents is not None else 0
        self._train_edges = train_edges  # (N, 2) int64, train positives for sampling

    def _mine_pairs(
        self, agent_ids: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pos_idx, neg_idx, pair_agents) — indices into the batch.

        pos_idx: (P, 2) in-batch anchor, positive
        neg_idx: (P, K) in-batch anchor, negative
        pair_agents: (P*(1+K), 2) raw agent IDs for pair feature lookup
        """
        B = len(agent_ids)
        in_batch: dict[int, int] = {int(a): i for i, a in enumerate(agent_ids.tolist())}

        pos_pairs = []
        for i, a in enumerate(agent_ids.tolist()):
            friends_i = self.friend_graph.get(int(a))
            if not friends_i:
                continue
            for f in friends_i:
                j = in_batch.get(int(f))
                if j is not None and j != i:
                    pos_pairs.append((i, j))
        if not pos_pairs:
            return np.empty((0, 2), np.int64), np.empty((0, 0), np.int64), np.empty((0, 2), np.int64)
        pos_pairs = np.array(pos_pairs, dtype=np.int64)

        # For each positive anchor, sample K in-batch negatives: agents not linked to anchor
        # by any observed edge (train/val/test) and not self.
        K = self.n_neg_per_pos
        neg_idx = np.empty((len(pos_pairs), K), dtype=np.int64)
        for row, (i, _) in enumerate(pos_pairs):
            a = int(agent_ids[i])
            pool = []
            for k, b in enumerate(agent_ids.tolist()):
                if k == i:
                    continue
                b = int(b)
                if b == a:
                    continue
                lo, hi = (a, b) if a < b else (b, a)
                if (lo, hi) in self.observed_edges:
                    continue
                pool.append(k)
            if len(pool) == 0:
                neg_idx[row] = i  # degenerate; will be masked out by caller
                continue
            if len(pool) >= K:
                sel = rng.choice(pool, K, replace=False)
            else:
                sel = rng.choice(pool, K, replace=True)
            neg_idx[row] = sel

        # Build pair_agents array: [pos_pairs; each (anchor, neg_j) flat]
        pair_agents = []
        for (i, j), negs in zip(pos_pairs, neg_idx):
            a_i = int(agent_ids[i]); a_j = int(agent_ids[j])
            pair_agents.append((a_i, a_j))
            for n in negs:
                pair_agents.append((a_i, int(agent_ids[int(n)])))
        pair_agents = np.array(pair_agents, dtype=np.int64)
        return pos_pairs, neg_idx, pair_agents

    # ------------------------------------------------------------------
    def forward(
        self,
        model,
        e: torch.Tensor,                # (B, emb_dim) L2-normalized
        agent_ids: torch.Tensor,        # (B,)
    ) -> tuple[torch.Tensor, dict]:
        device = e.device
        agent_np = agent_ids.detach().cpu().numpy()
        rng = np.random.default_rng()

        pos_pairs, neg_idx, pair_agents = self._mine_pairs(agent_np, rng)
        if len(pos_pairs) == 0:
            zero = e.sum() * 0.0
            return zero, {"loss": 0.0, "bpr": 0.0, "infonce": 0.0, "n_pos": 0}

        P, K = neg_idx.shape
        # Pair features (P*(1+K), feat_dim)
        if self.pair_feature_fn is not None and model.pair_feature_dim > 0:
            pf_np = self.pair_feature_fn(pair_agents)
            pf = torch.from_numpy(pf_np).to(device=device, dtype=e.dtype)
        else:
            pf = None

        # Gather pair embeddings — pos first, then flattened negatives (P*K, ...).
        pos_i = torch.from_numpy(pos_pairs[:, 0]).to(device)
        pos_j = torch.from_numpy(pos_pairs[:, 1]).to(device)
        neg_anchors = torch.from_numpy(np.repeat(pos_pairs[:, 0], K)).to(device)
        neg_partners = torch.from_numpy(neg_idx.reshape(-1)).to(device)

        e_pos_i = e[pos_i]; e_pos_j = e[pos_j]
        e_neg_i = e[neg_anchors]; e_neg_j = e[neg_partners]

        if pf is not None:
            pf_pos = pf[: P]
            pf_neg = pf[P :].reshape(P * K, -1)
        else:
            pf_pos = None; pf_neg = None

        pos_logits = model.score_pairs(e_pos_i, e_pos_j, pf_pos)       # (P,)
        neg_logits = model.score_pairs(e_neg_i, e_neg_j, pf_neg)       # (P*K,)

        # BPR: for each positive, compare against its K negatives.
        neg_logits_pk = neg_logits.view(P, K)
        bpr = -F.logsigmoid(pos_logits.unsqueeze(1) - neg_logits_pk).mean()

        # Supervised InfoNCE on raw cosine similarities over the batch.
        # For each anchor i, positives = friends(i) ∩ batch; negatives = everyone else in batch
        # that is not (i, friend-of-i-or-any-observed-edge)-adjacent to the positive definition.
        sim = (e @ e.t()) / self.infonce_temperature  # (B, B)
        # Build per-row positive mask.
        B = e.shape[0]
        pos_mask = torch.zeros(B, B, device=device, dtype=torch.bool)
        in_batch_map = {int(a): i for i, a in enumerate(agent_np.tolist())}
        for i, a in enumerate(agent_np.tolist()):
            for f in self.friend_graph.get(int(a), ()):
                j = in_batch_map.get(int(f))
                if j is not None and j != i:
                    pos_mask[i, j] = True
        # Exclude self-similarity from logits.
        diag = torch.eye(B, device=device, dtype=torch.bool)
        # Use a large-but-finite negative so that (value * pos_mask==0) stays 0
        # after the weighted sum (−inf * 0 → NaN).
        sim_masked = sim.masked_fill(diag, -1e9)
        # For rows with any positive, average -log p(positive) over positives.
        log_prob = F.log_softmax(sim_masked, dim=-1)
        pos_per_row = pos_mask.float().sum(dim=-1).clamp_min(1.0)
        infonce_per_row = -(log_prob * pos_mask.float()).sum(dim=-1) / pos_per_row
        has_pos = pos_mask.any(dim=-1)
        if has_pos.any():
            infonce = infonce_per_row[has_pos].mean()
        else:
            infonce = e.sum() * 0.0

        loss = self.bpr_weight * bpr + self.infonce_weight * infonce

        # --- Auxiliary BCE on random-pair distribution, handcrafted-only ---
        # Positives: random train edges. Negatives: random non-edges.
        # Both scored with ZERO embeddings so only pair_feat_norm+pair_head's
        # handcrafted-features slice is driven here — this calibrates the head
        # to the random-pair distribution used at eval.
        aux_bce_val = 0.0
        if (
            self.random_neg_weight > 0.0
            and self.pair_feature_fn is not None
            and self._train_edges is not None
            and self.n_agents > 0
            and model.pair_feature_dim > 0
            and not getattr(model, "disable_emb_branch", False)
        ):
            R = int(self.random_neg_batch)
            # Sample R positives from train edges
            pos_idx = rng.integers(0, len(self._train_edges), R)
            pos_pairs_g = self._train_edges[pos_idx]
            # Sample R random non-edges
            neg_pairs_g = []
            seen = set()
            while len(neg_pairs_g) < R:
                u = int(rng.integers(0, self.n_agents))
                v = int(rng.integers(0, self.n_agents))
                if u == v:
                    continue
                lo, hi = (u, v) if u < v else (v, u)
                if (lo, hi) in self.observed_edges or (lo, hi) in seen:
                    continue
                seen.add((lo, hi))
                neg_pairs_g.append((lo, hi))
            neg_pairs_g = np.array(neg_pairs_g, dtype=np.int64)
            aux_pairs = np.concatenate([pos_pairs_g, neg_pairs_g], axis=0)
            aux_pf_np = self.pair_feature_fn(aux_pairs)
            aux_pf = torch.from_numpy(aux_pf_np).to(device=device, dtype=e.dtype)
            # zero embeddings: score depends only on pair-feature path of pair_head
            D = e.shape[1]
            e_zero = torch.zeros(aux_pairs.shape[0], D, device=device, dtype=e.dtype)
            aux_logits = model.score_pairs(e_zero, e_zero, aux_pf)
            aux_labels = torch.cat([torch.ones(R), torch.zeros(R)]).to(device=device, dtype=e.dtype)
            aux_bce = F.binary_cross_entropy_with_logits(aux_logits, aux_labels)
            loss = loss + self.random_neg_weight * aux_bce
            aux_bce_val = float(aux_bce.item())

        # --- Hard-random-negative BCE (full model, cached agent embeddings) ---
        # Sample R positive train edges + R uniformly random non-edges. Look up
        # both endpoints in the model's per-agent embedding cache (detached, so
        # backbone isn't backpropped here), apply GNN refinement if enabled,
        # score via the pair head, and apply BCE. Gradient flows into the pair
        # head and (if GNN) the GraphRefiner.
        hard_bce_val = 0.0
        cache = getattr(model, "_emb_cache", None)
        if (
            self.hard_random_neg_weight > 0.0
            and self._train_edges is not None
            and self.n_agents > 0
            and cache is not None and cache.numel() > 0
        ):
            R = int(self.hard_random_neg_batch)
            pos_idx = rng.integers(0, len(self._train_edges), R)
            pos_pairs_g = self._train_edges[pos_idx]
            neg_pairs_g = []
            seen = set()
            attempts = 0
            max_attempts = 50 * R
            while len(neg_pairs_g) < R and attempts < max_attempts:
                u = int(rng.integers(0, self.n_agents))
                v = int(rng.integers(0, self.n_agents))
                attempts += 1
                if u == v:
                    continue
                lo, hi = (u, v) if u < v else (v, u)
                if (lo, hi) in self.observed_edges or (lo, hi) in seen:
                    continue
                seen.add((lo, hi))
                neg_pairs_g.append((lo, hi))
            if len(neg_pairs_g) >= R // 2:
                neg_pairs_g = np.array(neg_pairs_g, dtype=np.int64)
                R_eff = min(len(pos_pairs_g), len(neg_pairs_g))
                pos_pairs_g = pos_pairs_g[:R_eff]
                neg_pairs_g = neg_pairs_g[:R_eff]
                pos_u = torch.from_numpy(pos_pairs_g[:, 0]).long().to(device)
                pos_v = torch.from_numpy(pos_pairs_g[:, 1]).long().to(device)
                neg_u = torch.from_numpy(neg_pairs_g[:, 0]).long().to(device)
                neg_v = torch.from_numpy(neg_pairs_g[:, 1]).long().to(device)
                e_pu = cache[pos_u]
                e_pv = cache[pos_v]
                e_nu = cache[neg_u]
                e_nv = cache[neg_v]
                if getattr(model, "use_gnn_head", False):
                    e_pu = model.refine(e_pu, pos_u)
                    e_pv = model.refine(e_pv, pos_v)
                    e_nu = model.refine(e_nu, neg_u)
                    e_nv = model.refine(e_nv, neg_v)
                if self.pair_feature_fn is not None and model.pair_feature_dim > 0:
                    pf_pos_np = self.pair_feature_fn(pos_pairs_g)
                    pf_neg_np = self.pair_feature_fn(neg_pairs_g)
                    pf_pos = torch.from_numpy(pf_pos_np).to(device=device, dtype=e.dtype)
                    pf_neg = torch.from_numpy(pf_neg_np).to(device=device, dtype=e.dtype)
                else:
                    pf_pos = pf_neg = None
                pos_logits = model.score_pairs(e_pu, e_pv, pf_pos)
                neg_logits = model.score_pairs(e_nu, e_nv, pf_neg)
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_logits, torch.ones_like(pos_logits)
                )
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits)
                )
                hard_bce = 0.5 * (pos_loss + neg_loss)
                loss = loss + self.hard_random_neg_weight * hard_bce
                hard_bce_val = float(hard_bce.item())

        return loss, {
            "loss": float(loss.item()),
            "bpr": float(bpr.item()),
            "infonce": float(infonce.item()) if isinstance(infonce, torch.Tensor) else 0.0,
            "aux_bce": aux_bce_val,
            "hard_bce": hard_bce_val,
            "n_pos": int(P),
        }
