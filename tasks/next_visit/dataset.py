"""Dataset for next-visit (location + check-in time) prediction.

Builds on the same per-agent trajectory layout as ``tasks.poi_rec.dataset``
but additionally returns the *time delta* (in hours) between the current
visit's check-in time and the next visit's check-in time as the temporal
target for the GMM time head.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from tasks.poi_rec.dataset import build_poi_vocab  # re-export for symmetry
from utils.dataset import POIIndex
from utils.misc import make_instance_dict


class NextVisitDataset(Dataset):
    """Per-agent trajectory dataset for next-visit (POI + time) prediction.

    Each sample is one agent's trajectory.  Position ``t`` predicts both the
    POI and the check-in time of position ``t+1``.  The last position has no
    target and is masked out via ``rec_mask``.

    Targets:
        target_poi_idx: int index into the POI vocabulary, -1 if invalid.
        target_time_delta: float hours between ``stay[t].start`` and
            ``stay[t+1].start``.  Strictly >= 0 because stays are time-ordered.
        target_time_abs: ``stay[t+1].start`` in raw hours (kept for diagnostics
            and for converting predicted deltas back to absolute timestamps).

    Args mirror ``POIRecDataset``.
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
        stay_df["stay_anomalous"] = False  # keep make_instance_dict happy

        seq_len = len(stay_df)

        poi_ids = stay_df["poi_id"].values
        starts = stay_df["start"].values.astype(np.float64)  # hours from min_time

        target_poi_idx = np.full(seq_len, -1, dtype=np.int64)
        target_delta = np.zeros(seq_len, dtype=np.float32)
        target_abs = np.zeros(seq_len, dtype=np.float32)
        for t in range(seq_len - 1):
            next_poi = poi_ids[t + 1]
            if next_poi is None or (isinstance(next_poi, float) and np.isnan(next_poi)):
                continue
            poi_idx = self.poi_to_idx.get(next_poi, -1)
            target_poi_idx[t] = poi_idx
            target_delta[t] = max(0.0, float(starts[t + 1] - starts[t]))
            target_abs[t] = float(starts[t + 1])

        rec_mask = target_poi_idx >= 0

        stay_df_with_neighbors, neighbor_padding_mask = self.poi_index.find_neighbors(stay_df)
        instance = make_instance_dict(stay_df_with_neighbors, self.categories)
        instance = self.poi_index.correct_shapes(instance, seq_len, neighbor_padding_mask)

        instance["target_poi_idx"] = torch.tensor(target_poi_idx, dtype=torch.long)
        instance["target_time_delta"] = torch.tensor(target_delta, dtype=torch.float32)
        instance["target_time_abs"] = torch.tensor(target_abs, dtype=torch.float32)
        instance["rec_mask"] = torch.tensor(rec_mask, dtype=torch.bool)
        return instance


def next_visit_collate_fn(batch):
    collated = {}
    for k in batch[0].keys():
        if k == "target_poi_idx":
            pad_val = -1
        elif k == "rec_mask":
            pad_val = False
        elif k in ("padding_mask", "neighbor_padding_mask"):
            pad_val = True
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
