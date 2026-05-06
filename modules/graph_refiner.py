"""1-hop graph refinement of per-agent embeddings (used at fine-tune time only).

Why: pure trajectory backbones cannot recover the train friend graph from
visits alone. Graph-aware baselines (LBSN2Vec, H^3GNN, USRC) consume the train
graph natively. To put TraXion (and any sequence-pretrain backbone, e.g.
UniTraj) on the same access regime *without* late-fusing a separate LR head
on handcrafted features, we attach a 1-hop refinement that aggregates each
agent's train-graph neighbors' embeddings and gates the result back into the
agent's own embedding.

The neighbor pool is fixed (train edges only -> no leakage into val/test). The
neighbor embeddings are read from a frozen *cache* that is refreshed at the
start of each training epoch (and once before each eval). The current batch's
embeddings still carry gradient through the backbone via the anchor's own
embedding; the cache contributes a graph-structural prior with no gradient
into neighbor encoders. This keeps the per-step cost O(B * avg_deg) instead
of O(N).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_neighbor_index(
    friend_graph: dict[int, set[int]],
    n_agents: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten friend_graph into a CSR-style (idx, offsets) pair.

    Returns
    -------
    idx : (E,) long
        Concatenated neighbor agent IDs for agents 0..n_agents-1.
    offsets : (n_agents + 1,) long
        Prefix-sum so agent ``a``'s neighbors are ``idx[offsets[a]:offsets[a+1]]``.
        ``offsets[a+1] - offsets[a]`` is agent ``a``'s degree (0 if isolated).
    """
    counts = np.zeros(n_agents, dtype=np.int64)
    for a, nbrs in friend_graph.items():
        if 0 <= int(a) < n_agents:
            counts[int(a)] = len(nbrs)
    offsets = np.zeros(n_agents + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    idx = np.empty(int(offsets[-1]), dtype=np.int64)
    for a, nbrs in friend_graph.items():
        a = int(a)
        if 0 <= a < n_agents and len(nbrs) > 0:
            start = int(offsets[a])
            idx[start : start + len(nbrs)] = np.fromiter(nbrs, dtype=np.int64, count=len(nbrs))
    return (
        torch.from_numpy(idx).to(device),
        torch.from_numpy(offsets).to(device),
    )


class GraphRefiner(nn.Module):
    """e_u' = e_u + sigmoid(gate) * proj(mean_{v in N(u)} cache[v]).

    For agents with no neighbors (offsets[a] == offsets[a+1]) the refinement is
    zero (mean of empty set defined as zero), so isolated agents reduce to the
    backbone embedding alone.

    Parameters
    ----------
    emb_dim : int
        Embedding dimensionality (same in/out).
    gate_init : float
        Pre-sigmoid initial gate value. ``0.0`` -> 50/50 mix; ``-2.0`` -> small
        neighbor contribution at init (for stable warmup).
    """

    def __init__(self, emb_dim: int, gate_init: float = -2.0):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        # Init proj to identity-ish: zero bias, small weight, so neighbor info
        # starts as a small perturbation rather than dominating the anchor.
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def aggregate(
        self,
        agent_ids: torch.Tensor,        # (B,) long
        emb_cache: torch.Tensor,        # (n_agents, D) — detached, cpu/gpu
        neighbor_idx: torch.Tensor,     # (E,) long, CSR neighbors flat
        neighbor_offsets: torch.Tensor, # (n_agents+1,) long, CSR offsets
    ) -> torch.Tensor:
        """Mean-pool neighbor embeddings for each batch agent. Returns (B, D)."""
        B = agent_ids.shape[0]
        D = emb_cache.shape[1]
        device = emb_cache.device

        starts = neighbor_offsets[agent_ids]      # (B,)
        ends = neighbor_offsets[agent_ids + 1]    # (B,)
        degs = (ends - starts).clamp_min(0)       # (B,)
        total = int(degs.sum().item())
        if total == 0:
            return torch.zeros(B, D, device=device, dtype=emb_cache.dtype)

        # Build flat list of neighbor positions (indices into neighbor_idx),
        # then segment IDs (which batch agent each neighbor belongs to).
        # CSR -> flat: for each i in [0..B), append neighbor_idx[starts[i]:ends[i]]
        # We do this on-device with a vectorized approach:
        seg_ids = torch.repeat_interleave(
            torch.arange(B, device=device), degs
        )                                         # (total,)
        # offsets within each agent's slice: 0,1,..,deg_i-1
        local = torch.arange(total, device=device) - torch.repeat_interleave(
            torch.cumsum(degs, dim=0) - degs, degs
        )
        # absolute positions in neighbor_idx
        abs_pos = torch.repeat_interleave(starts, degs) + local  # (total,)
        flat_neighbor_ids = neighbor_idx[abs_pos]                 # (total,)
        nbr_emb = emb_cache[flat_neighbor_ids]                    # (total, D)

        # Segmented sum
        out = torch.zeros(B, D, device=device, dtype=nbr_emb.dtype)
        out.scatter_add_(0, seg_ids.unsqueeze(-1).expand(-1, D), nbr_emb)
        # Divide by degree (clamp to >=1 for safety; agents with deg=0 are
        # already filtered upstream by the seg_ids broadcasting).
        denom = degs.clamp_min(1).to(out.dtype).unsqueeze(-1)
        return out / denom

    def forward(
        self,
        e: torch.Tensor,                # (B, D) — has grad, anchor embedding
        agent_ids: torch.Tensor,        # (B,) long
        emb_cache: torch.Tensor,        # (n_agents, D) — detached
        neighbor_idx: torch.Tensor,     # (E,) long
        neighbor_offsets: torch.Tensor, # (n_agents+1,) long
    ) -> torch.Tensor:
        nbr = self.aggregate(agent_ids, emb_cache, neighbor_idx, neighbor_offsets)
        gate = torch.sigmoid(self.gate)
        return e + gate * self.proj(nbr)
