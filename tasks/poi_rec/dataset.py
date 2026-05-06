"""Dataset for next-POI recommendation.

Wraps the same trajectory data used for anomaly detection but constructs
(context, next-POI) supervision pairs.  Reuses POIIndex for co-location
neighbor lookup.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.dataset import POIIndex
from utils.misc import make_instance_dict


def build_poi_vocab(stays: pd.DataFrame) -> dict:
    """Build a mapping from poi_id → integer index.

    Only POIs that appear in ``stays`` are included.  Unknown POIs map to -1
    at lookup time.

    Returns:
        dict mapping poi_id to int in [0, n_pois).
    """
    unique_pois = stays["poi_id"].dropna().unique()
    return {pid: idx for idx, pid in enumerate(sorted(unique_pois))}


class POIRecDataset(Dataset):
    """Per-agent trajectory dataset for next-POI prediction.

    Each sample is one agent's trajectory.  For a sequence of length T the
    model receives visits [0 .. T-1] and the target at position t is the
    ``poi_id`` at position t+1 (shifted by one).  The last position has no
    target and is masked out via ``rec_mask``.

    Args:
        stays: DataFrame of stay records (with poi_id column).
        categories: list of category column names.
        clique_size: number of neighbors per visit.
        poi_to_idx: mapping from poi_id → int index.
        max_seq_len: optional cap on sequence length.
        overlap_min_frac: temporal overlap filter for neighbors.
    """

    def __init__(
        self,
        stays: pd.DataFrame,
        categories: list,
        clique_size: int,
        poi_to_idx: dict,
        max_seq_len: int | None = None,
        overlap_min_frac: float = 0.0,
    ):
        super().__init__()
        self.categories = categories
        self.poi_to_idx = poi_to_idx
        self.max_seq_len = max_seq_len
        self.poi_index = POIIndex(stays, clique_size, overlap_min_frac=overlap_min_frac)
        self.grouped_stays = self.poi_index.stays.groupby("agent")
        self.agent_list = list(self.grouped_stays.groups.keys())

    @property
    def n_pois(self) -> int:
        return len(self.poi_to_idx)

    def __len__(self):
        return len(self.agent_list)

    def __getitem__(self, idx):
        agent = self.agent_list[idx]
        stay_df = self.grouped_stays.get_group(agent)

        if self.max_seq_len is not None and len(stay_df) > self.max_seq_len:
            start = np.random.randint(0, len(stay_df) - self.max_seq_len + 1)
            stay_df = stay_df.iloc[start : start + self.max_seq_len]

        stay_df = stay_df.copy()
        # Dummy anomaly column so make_instance_dict works unchanged
        stay_df["stay_anomalous"] = False

        seq_len = len(stay_df)

        # Build shifted target: position t predicts POI at t+1
        poi_ids = stay_df["poi_id"].values
        target_poi_idx = np.full(seq_len, -1, dtype=np.int64)
        for t in range(seq_len - 1):
            next_poi = poi_ids[t + 1]
            if next_poi is not None and not (isinstance(next_poi, float) and np.isnan(next_poi)):
                target_poi_idx[t] = self.poi_to_idx.get(next_poi, -1)
        rec_mask = target_poi_idx >= 0

        # Clique neighbor lookup (same as CliqueDataset)
        stay_df_with_neighbors, neighbor_padding_mask = self.poi_index.find_neighbors(stay_df)
        instance = make_instance_dict(stay_df_with_neighbors, self.categories)
        instance = self.poi_index.correct_shapes(instance, seq_len, neighbor_padding_mask)

        # Add POI-rec fields
        instance["target_poi_idx"] = torch.tensor(target_poi_idx, dtype=torch.long)
        instance["rec_mask"] = torch.tensor(rec_mask, dtype=torch.bool)
        return instance


def poi_rec_collate_fn(batch):
    """Collate function for POIRecDataset batches.

    Pads all tensors to the longest sequence in the batch.  ``target_poi_idx``
    is padded with -1 and ``rec_mask`` with False.
    """
    collated = {}
    for k in batch[0].keys():
        if k == "target_poi_idx":
            pad_val = -1
        elif k in ("padding_mask", "neighbor_padding_mask", "rec_mask"):
            pad_val = True if k != "rec_mask" else False
        else:
            pad_val = 0
        collated[k] = pad_sequence(
            [item[k] for item in batch],
            batch_first=True,
            padding_value=pad_val,
        )

    if len(collated["padding_mask"].shape) == 3:
        collated["padding_mask"] = collated["padding_mask"][..., 0]
        collated["stay_anomalous"] = collated["stay_anomalous"][..., 0]

    return collated
