"""Dataset + edge-aware batch sampler for social link prediction.

Core idea: edge-centric batching. Every training batch seeds from a set of
positive edges so that positives are *guaranteed* dense in the batch (degree on
these graphs averages ~13, so uniform agent sampling yields <1 positive per
batch of 32 — too sparse for contrastive learning).

A batch is a list of agent indices. For each agent we compute its trajectory
tensor (reusing POIIndex clique lookup). The training step forwards the whole
batch once; the criterion then scores in-batch positive edges vs in-batch
non-edges as negatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from utils.dataset import POIIndex
from utils.misc import make_instance_dict


class SocialTrajectoryDataset(Dataset):
    """Per-agent trajectory dataset. __getitem__(i) returns agent ``i``'s visits.

    Agent indexing is 0..n_agents-1 (from the canonical normalization applied in
    prepare_social_splits.py). Agents with zero training stays are handled by
    returning an empty-visit instance (padding_mask all True), so they can still
    appear in an edge but produce a zero embedding.
    """

    def __init__(
        self,
        stays: pd.DataFrame,
        categories: list,
        clique_size: int,
        n_agents: int,
        max_seq_len: int | None = None,
        overlap_min_frac: float = 0.0,
    ):
        super().__init__()
        self.categories = categories
        self.clique_size = clique_size
        self.n_agents = int(n_agents)
        self.max_seq_len = max_seq_len
        self.poi_index = POIIndex(stays, clique_size, overlap_min_frac=overlap_min_frac)
        self._grouped = self.poi_index.stays.groupby("agent")
        self._groups = dict(self._grouped.groups)  # agent -> Index

    def __len__(self) -> int:
        return self.n_agents

    def __getitem__(self, agent: int) -> dict:
        idx_obj = self._groups.get(int(agent))
        if idx_obj is None or len(idx_obj) == 0:
            # Agent has no training stays — return a length-1 fully-padded instance.
            stay_df = self.poi_index.stays.iloc[[0]].copy()
            stay_df["stay_anomalous"] = False
            seq_len = 1
            stay_df, neighbor_padding_mask = self.poi_index.find_neighbors(stay_df)
            instance = make_instance_dict(stay_df, self.categories)
            instance = self.poi_index.correct_shapes(instance, seq_len, neighbor_padding_mask)
            instance["padding_mask"] = torch.ones(seq_len, dtype=torch.bool)
            instance["agent_id"] = torch.tensor(int(agent), dtype=torch.long)
            return instance

        stay_df = self.poi_index.stays.iloc[idx_obj]
        if self.max_seq_len is not None and len(stay_df) > self.max_seq_len:
            start = np.random.randint(0, len(stay_df) - self.max_seq_len + 1)
            stay_df = stay_df.iloc[start : start + self.max_seq_len]

        stay_df = stay_df.copy()
        stay_df["stay_anomalous"] = False
        seq_len = len(stay_df)
        stay_df, neighbor_padding_mask = self.poi_index.find_neighbors(stay_df)
        instance = make_instance_dict(stay_df, self.categories)
        instance = self.poi_index.correct_shapes(instance, seq_len, neighbor_padding_mask)
        instance["agent_id"] = torch.tensor(int(agent), dtype=torch.long)
        return instance


def social_collate_fn(batch: list[dict]) -> dict:
    """Pad-collate. Exposes ``agent_ids`` (B,) so criterion can index friend graph."""
    collated = {}
    for k in batch[0].keys():
        if k == "agent_id":
            collated["agent_ids"] = torch.stack([b[k] for b in batch], dim=0)
            continue
        pad_val = (k in ("padding_mask", "neighbor_padding_mask"))
        collated[k] = pad_sequence(
            [b[k] for b in batch], batch_first=True, padding_value=pad_val,
        )
    if collated["padding_mask"].dim() == 3:
        collated["padding_mask"] = collated["padding_mask"][..., 0]
        collated["stay_anomalous"] = collated["stay_anomalous"][..., 0]
    return collated


class EdgeCentricBatchSampler(Sampler[list[int]]):
    """Yields batches of agent IDs seeded from positive edges.

    Each batch picks ``n_seed_edges = batch_size // 2`` random positive edges,
    adds both endpoints, then fills the remaining slots with uniform random
    agents (to provide negatives and diversify).

    The DataLoader's ``batch_sampler`` consumes these; each yielded list is a
    single batch.
    """

    def __init__(
        self,
        n_agents: int,
        edges: np.ndarray,
        batch_size: int,
        batches_per_epoch: int,
        seed: int = 0,
    ):
        self.n_agents = int(n_agents)
        self.edges = edges.astype(np.int64)
        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            yield self._build_batch()

    def _build_batch(self) -> list[int]:
        n_seed = max(1, self.batch_size // 2)
        idx = self.rng.integers(0, len(self.edges), size=n_seed)
        seeds = self.edges[idx]
        agents: list[int] = []
        seen: set[int] = set()
        for u, v in seeds:
            for a in (int(u), int(v)):
                if a not in seen:
                    seen.add(a); agents.append(a)
                    if len(agents) >= self.batch_size:
                        break
            if len(agents) >= self.batch_size:
                break
        while len(agents) < self.batch_size:
            a = int(self.rng.integers(0, self.n_agents))
            if a not in seen:
                seen.add(a); agents.append(a)
        return agents
