import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.misc import make_instance_dict


class POIIndex:
    """Indexes visits by POI for efficient co-location queries.

    For each query visit (with a known poi_id), find_neighbors returns up to
    clique_size - 1 other visits at the same POI, ranked by temporal proximity
    (combined |Δstart| + |Δstop|). Slots beyond the actual number of co-located
    visits are padded and flagged in the returned neighbor_padding_mask.

    If overlap_min_frac > 0, only co-visitors whose temporal overlap with the
    query visit meets the threshold are considered as candidates before ranking.
    Overlap fraction = intersection_duration / min(query_duration, candidate_duration).
    """

    def __init__(self, stays: pd.DataFrame, clique_size: int, overlap_min_frac: float = 0.0) -> None:
        self.clique_size = clique_size
        self.overlap_min_frac = overlap_min_frac
        stays = stays.reset_index(drop=True)
        self.stays = stays
        # Map poi_id -> int32 array of row positions in self.stays
        valid = stays["poi_id"].notna()
        self._poi_to_indices: dict = (
            stays[valid]
            .groupby("poi_id")
            .apply(lambda g: g.index.values.astype(np.int32), include_groups=False)
            .to_dict()
        )

    def find_neighbors(
        self, self_stay_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the combined agent+neighbor DataFrame and neighbor padding mask.

        The returned stay_df has clique_size * seq_len rows ordered so that
        correct_shapes() can reshape it to (seq_len, clique_size, ...).

        neighbor_padding_mask has shape (seq_len, clique_size):
            - index 0 is the agent (always False / valid)
            - indices 1..clique_size-1 are neighbor slots (True = padded)
        """
        seq_len = len(self_stay_df)
        n_neighbors = self.clique_size - 1

        neighbor_padding_mask = np.zeros((seq_len, self.clique_size), dtype=bool)

        # per_slot[j] accumulates exactly seq_len single-row DataFrames
        per_slot: list[list[pd.DataFrame]] = [[] for _ in range(n_neighbors)]

        # Dummy row for padded slots (values are masked out by neighbor_padding_mask)
        dummy = self.stays.iloc[[0]].copy()
        dummy["stay_anomalous"] = False

        for i, (_, visit) in enumerate(self_stay_df.iterrows()):
            poi_id = visit.get("poi_id", None)
            if poi_id is None or (isinstance(poi_id, float) and np.isnan(poi_id)):
                # No POI association (raw stay with no matching POI in the join) — all slots padded
                neighbor_padding_mask[i, 1:] = True
                for j in range(n_neighbors):
                    per_slot[j].append(dummy)
                continue

            candidate_pos = self._poi_to_indices.get(poi_id, None)
            if candidate_pos is None or len(candidate_pos) == 0:
                neighbor_padding_mask[i, 1:] = True
                for j in range(n_neighbors):
                    per_slot[j].append(dummy)
                continue

            # Exclude the visit by the agent themselves
            cand = self.stays.iloc[candidate_pos]
            not_self = ~(cand["agent"] == visit["agent"])
            cand = cand[not_self]

            # Optionally restrict to co-visitors with sufficient temporal overlap
            if self.overlap_min_frac > 0.0:
                v_start = pd.Timestamp(visit["start"]).value  # nanoseconds
                v_stop  = pd.Timestamp(visit["stop"]).value
                c_start = pd.to_datetime(cand["start"]).values.astype(np.int64)
                c_stop  = pd.to_datetime(cand["stop"]).values.astype(np.int64)
                intersect = np.maximum(0, np.minimum(c_stop, v_stop) - np.maximum(c_start, v_start))
                min_dur   = np.minimum(v_stop - v_start, c_stop - c_start)
                frac = np.divide(intersect, min_dur, out=np.zeros(len(min_dur), dtype=np.float64), where=min_dur > 0)
                cand = cand[frac >= self.overlap_min_frac]

            n_actual = len(cand)
            if n_actual == 0:
                neighbor_padding_mask[i, 1:] = True
                for j in range(n_neighbors):
                    per_slot[j].append(dummy)
                continue

            # Rank by temporal proximity; keep at most n_neighbors
            diffs = (cand["start"] - visit["start"]).abs() + (
                cand["stop"] - visit["stop"]
            ).abs()
            order = np.argsort(diffs.values)[:n_neighbors]
            selected = cand.iloc[order].copy()
            selected["stay_anomalous"] = False

            n_selected = len(selected)
            for j in range(n_neighbors):
                if j < n_selected:
                    per_slot[j].append(selected.iloc[[j]])
                else:
                    neighbor_padding_mask[i, j + 1] = True
                    per_slot[j].append(dummy)

        # Concatenate per slot: each slot block has seq_len rows.
        # When clique_size=1 there are no neighbor slots — return the agent rows only.
        if n_neighbors == 0:
            return self_stay_df.reset_index(drop=True), neighbor_padding_mask
        neighbor_stay_df = pd.concat(
            [pd.concat(rows, ignore_index=True) for rows in per_slot],
            ignore_index=True,
        )
        stay_df = pd.concat(
            [self_stay_df.reset_index(drop=True), neighbor_stay_df],
            ignore_index=True,
        )
        return stay_df, neighbor_padding_mask

    def correct_shapes(
        self,
        instance: dict,
        seq_len: int,
        neighbor_padding_mask: np.ndarray,
    ) -> dict:
        for k, v in instance.items():
            if len(v.shape) == 1:  # (clique_size * seq_len,)
                instance[k] = v.view(self.clique_size, seq_len).T
            elif len(v.shape) == 2:  # (clique_size * seq_len, feature_dim)
                instance[k] = v.view(self.clique_size, seq_len, -1).transpose(0, 1)
        instance["neighbor_padding_mask"] = torch.tensor(
            neighbor_padding_mask, dtype=torch.bool
        )
        return instance


class CliqueDataset(Dataset):
    def __init__(
        self,
        stays,
        categories,
        clique_size,
        noisifier=None,
        anomaly_col=None,
        max_seq_len=None,
        overlap_min_frac=0.0,
    ):
        if noisifier is not None and anomaly_col is not None:
            raise ValueError("Only one of noisifier and anomaly_col can be specified.")
        super().__init__()
        self.categories = categories
        self.noisifier = noisifier
        self.anomaly_col = anomaly_col
        self.max_seq_len = max_seq_len
        self.poi_index = POIIndex(stays, clique_size, overlap_min_frac=overlap_min_frac)
        self.grouped_stays = self.poi_index.stays.groupby("agent")
        self.agent_list = list(self.grouped_stays.groups.keys())

    def __len__(self):
        return len(self.agent_list)

    def __getitem__(self, idx):
        agent = self.agent_list[idx]
        stay_df = self.grouped_stays.get_group(agent)

        if self.max_seq_len is not None and len(stay_df) > self.max_seq_len:
            start = np.random.randint(0, len(stay_df) - self.max_seq_len + 1)
            stay_df = stay_df.iloc[start : start + self.max_seq_len]

        if self.noisifier:
            stay_df = self.noisifier(stay_df)
        elif self.anomaly_col is not None:
            stay_df = stay_df.copy()
            stay_df["stay_anomalous"] = stay_df[self.anomaly_col]

        seq_len = len(stay_df)
        stay_df, neighbor_padding_mask = self.poi_index.find_neighbors(stay_df)
        instance = make_instance_dict(stay_df, self.categories)
        return self.poi_index.correct_shapes(instance, seq_len, neighbor_padding_mask)


def collate_fn(batch):
    collated = {
        k: pad_sequence(
            [item[k] for item in batch],
            batch_first=True,
            padding_value=(k in ("padding_mask", "neighbor_padding_mask")),
        )
        for k in batch[0].keys()
    }
    if len(collated["padding_mask"].shape) == 3:  # (batch_size, seq_len, clique_size)
        collated["padding_mask"] = collated["padding_mask"][..., 0]
        collated["stay_anomalous"] = collated["stay_anomalous"][..., 0]

    for k, v in collated.items():
        assert not torch.isnan(v).any(), f"Collated batch contains NaNs in {k}"
    return collated
