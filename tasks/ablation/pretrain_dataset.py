"""Dataset wrapper for ablation pretraining (causal / STM).

Reuses POIIndex co-location lookup but, instead of synthetic anomaly labels,
emits per-visit token targets:
    - poi_idx (int): vocabulary index of the visited POI (or -1 if unknown)
    - time_bin (int): hour-of-day index in [0, 24) computed from `start`
    - cat_idx (int): argmax of the category one-hot (or -1 if all zero)

The dataset stays compatible with `make_instance_dict` and `correct_shapes`,
so the standard model `forward(...)` signature is unchanged. Only the
training script consumes the extra target tensors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.dataset import POIIndex
from utils.misc import make_instance_dict


N_TIME_BINS = 24  # hour-of-day bins (start is in hours since min_time)


def build_poi_vocab(stays: pd.DataFrame) -> dict:
    unique_pois = stays["poi_id"].dropna().unique()
    return {pid: idx for idx, pid in enumerate(sorted(unique_pois))}


class AblationPretrainDataset(Dataset):
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
        stay_df["stay_anomalous"] = False  # required by make_instance_dict
        seq_len = len(stay_df)

        # POI vocab indices (or -1 for unknown / NaN)
        poi_ids = stay_df["poi_id"].values
        poi_idx = np.full(seq_len, -1, dtype=np.int64)
        for t in range(seq_len):
            pid = poi_ids[t]
            if pid is None or (isinstance(pid, float) and np.isnan(pid)):
                continue
            poi_idx[t] = self.poi_to_idx.get(pid, -1)

        # Time bin (hour of day). `start` is hours since min_time → mod 24.
        starts = stay_df["start"].values.astype(np.float64)
        time_bin = (np.floor(starts).astype(np.int64) % N_TIME_BINS).astype(np.int64)

        # Category index = argmax of one-hot. -1 if every column is zero.
        cat_arr = stay_df[self.categories].values.astype(np.int64)
        cat_sum = cat_arr.sum(axis=1)
        cat_idx = np.where(cat_sum > 0, cat_arr.argmax(axis=1), -1).astype(np.int64)

        stay_df_with_neighbors, neighbor_padding_mask = self.poi_index.find_neighbors(stay_df)
        instance = make_instance_dict(stay_df_with_neighbors, self.categories)
        instance = self.poi_index.correct_shapes(instance, seq_len, neighbor_padding_mask)

        instance["poi_idx"] = torch.tensor(poi_idx, dtype=torch.long)
        instance["time_bin"] = torch.tensor(time_bin, dtype=torch.long)
        instance["cat_idx"] = torch.tensor(cat_idx, dtype=torch.long)
        return instance


def ablation_collate_fn(batch):
    collated = {}
    for k in batch[0].keys():
        if k in ("poi_idx", "cat_idx"):
            pad_val = -1
        elif k in ("padding_mask", "neighbor_padding_mask"):
            pad_val = True
        elif k == "time_bin":
            pad_val = 0  # time_bin uses padding_mask to suppress; value is irrelevant
        else:
            pad_val = 0
        collated[k] = pad_sequence(
            [item[k] for item in batch],
            batch_first=True,
            padding_value=pad_val,
        )

    if len(collated["padding_mask"].shape) == 3:
        collated["padding_mask"] = collated["padding_mask"][..., 0]
        if "stay_anomalous" in collated and len(collated["stay_anomalous"].shape) == 3:
            collated["stay_anomalous"] = collated["stay_anomalous"][..., 0]
    return collated
