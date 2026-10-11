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

    neighbor_search selects how candidates are ranked:
      - "scan": rank every same-POI candidate.
      - "sorted": exact top-k over a start-time-sorted index with an expanding
        window. Returns the same top-k distances as "scan" and is much faster when
        each POI has many visits (e.g. the EHR corpora, where a context is an item
        type shared by every stay). Requires overlap_min_frac == 0.
    """

    def __init__(self, stays: pd.DataFrame, clique_size: int, overlap_min_frac: float = 0.0,
                 neighbor_search: str = "scan") -> None:
        self.clique_size = clique_size
        self.overlap_min_frac = overlap_min_frac
        if neighbor_search not in ("scan", "sorted"):
            raise ValueError(f"neighbor_search must be 'scan' or 'sorted', got {neighbor_search!r}")
        if neighbor_search == "sorted" and overlap_min_frac > 0.0:
            raise NotImplementedError("neighbor_search='sorted' requires overlap_min_frac == 0")
        self.neighbor_search = neighbor_search
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
        self._start = stays["start"].to_numpy()
        self._stop = stays["stop"].to_numpy()
        self._agent = stays["agent"].to_numpy()

        # Start-sorted candidates per POI, built once from the unperturbed visits.
        self._sorted_starts: dict = {}
        self._sorted_rows: dict = {}
        if neighbor_search == "sorted":
            for poi, pos in self._poi_to_indices.items():
                sv = self._start[pos]
                order = np.argsort(sv, kind="stable")
                self._sorted_rows[poi] = pos[order]
                self._sorted_starts[poi] = sv[order]

    @staticmethod
    def _poi_missing_mask(poi_arr: np.ndarray) -> np.ndarray:
        if poi_arr.dtype.kind == "f":
            return np.isnan(poi_arr)
        if poi_arr.dtype.kind in "iub":
            return np.zeros(len(poi_arr), dtype=bool)
        out = np.empty(len(poi_arr), dtype=bool)
        for i, v in enumerate(poi_arr):
            out[i] = v is None or (isinstance(v, float) and np.isnan(v))
        return out

    _EMPTY_SEL = np.empty(0, dtype=np.int64)

    def _topk_window(self, poi, q_start, q_stop, q_agent, k):
        """Exact k nearest same-POI rows by |Δstart| + |Δstop|, excluding q's own agent.

        Since |Δstart| + |Δstop| >= |Δstart|, any candidate outside a start-time
        window of radius r has distance > r. The window grows until the k-th best
        distance inside it is <= r, at which point no outside candidate can enter
        the top k. Returns row positions into self.stays, nearest first.
        """
        starts = self._sorted_starts.get(poi)
        if starts is None or len(starts) == 0:
            return self._EMPTY_SEL
        rows = self._sorted_rows[poi]
        n = len(starts)
        j = int(np.searchsorted(starts, q_start))
        agent_all, start_all, stop_all = self._agent, self._start, self._stop
        half = 8
        while True:
            lo = max(0, j - half)
            hi = min(n, j + half)
            exhausted = lo == 0 and hi == n
            win_rows = rows[lo:hi]
            keep = agent_all[win_rows] != q_agent
            if keep.any():
                cand = win_rows[keep]
                d = np.abs(start_all[cand] - q_start) + np.abs(stop_all[cand] - q_stop)
                kk = min(k, len(d))
                part = np.argpartition(d, kk - 1)[:kk] if len(d) > kk else np.arange(len(d))
                part = part[np.argsort(d[part], kind="stable")]
                if exhausted:
                    return cand[part]
                r = None
                if lo > 0:
                    r = q_start - start_all[rows[lo - 1]]
                if hi < n:
                    rr = start_all[rows[hi]] - q_start
                    r = rr if r is None else min(r, rr)
                if len(d) >= k and r is not None and d[part[-1]] <= r:
                    return cand[part]
            elif exhausted:
                return self._EMPTY_SEL
            half *= 4

    def find_neighbors(
        self, self_stay_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the clique-expanded stay dataframe and neighbor padding mask.

        The returned DataFrame has (clique_size * seq_len) rows: the first seq_len
        rows are the agent's own visits; each subsequent block of seq_len rows is
        one neighbor slot.
        neighbor_padding_mask has shape (seq_len, clique_size):
            - index 0 is the agent (always False / valid)
            - indices 1..clique_size-1 are neighbor slots (True = padded)
        """
        seq_len = len(self_stay_df)
        n_neighbors = self.clique_size - 1
        neighbor_padding_mask = np.zeros((seq_len, self.clique_size), dtype=bool)

        # When clique_size=1 there are no neighbor slots — return the agent rows only.
        if n_neighbors == 0:
            return self_stay_df.reset_index(drop=True), neighbor_padding_mask

        if "poi_id" in self_stay_df.columns:
            poi_arr = self_stay_df["poi_id"].to_numpy()
            missing = self._poi_missing_mask(poi_arr)
        else:
            poi_arr = np.empty(seq_len, dtype=object)
            missing = np.ones(seq_len, dtype=bool)
        start_arr = self_stay_df["start"].to_numpy()
        stop_arr = self_stay_df["stop"].to_numpy()
        agent_arr = self_stay_df["agent"].to_numpy()

        # gather[j, i] = row in self.stays for neighbor slot j of visit i
        # (row 0 serves as the dummy for padded slots, which are masked out).
        gather = np.zeros((n_neighbors, seq_len), dtype=np.int64)

        for i in range(seq_len):
            if missing[i]:
                # No POI association (raw stay with no matching POI in the join) — all slots padded
                neighbor_padding_mask[i, 1:] = True
                continue

            if self.neighbor_search == "sorted":
                selected = self._topk_window(poi_arr[i], start_arr[i], stop_arr[i], agent_arr[i], n_neighbors)
            else:
                candidate_pos = self._poi_to_indices.get(poi_arr[i])
                if candidate_pos is None or len(candidate_pos) == 0:
                    neighbor_padding_mask[i, 1:] = True
                    continue
                # Exclude the visit by the agent themselves
                cpos = candidate_pos[self._agent[candidate_pos] != agent_arr[i]]

                # Optionally restrict to co-visitors with sufficient temporal overlap
                if self.overlap_min_frac > 0.0 and len(cpos) > 0:
                    v_start, v_stop = np.int64(start_arr[i]), np.int64(stop_arr[i])
                    c_start = self._start[cpos].astype(np.int64)
                    c_stop = self._stop[cpos].astype(np.int64)
                    intersect = np.maximum(0, np.minimum(c_stop, v_stop) - np.maximum(c_start, v_start))
                    min_dur = np.minimum(v_stop - v_start, c_stop - c_start)
                    frac = np.divide(intersect, min_dur, out=np.zeros(len(min_dur), dtype=np.float64),
                                     where=min_dur > 0)
                    cpos = cpos[frac >= self.overlap_min_frac]

                # Rank by temporal proximity; keep at most n_neighbors
                diffs = np.abs(self._start[cpos] - start_arr[i]) + np.abs(self._stop[cpos] - stop_arr[i])
                selected = cpos[np.argsort(diffs)[:n_neighbors]]

            if len(selected) == 0:
                neighbor_padding_mask[i, 1:] = True
                continue
            for j in range(n_neighbors):
                if j < len(selected):
                    gather[j, i] = selected[j]
                else:
                    neighbor_padding_mask[i, j + 1] = True

        neighbor_rows = self.stays.iloc[gather.reshape(-1)].reset_index(drop=True)
        neighbor_rows["stay_anomalous"] = False
        stay_df = pd.concat(
            [self_stay_df.reset_index(drop=True), neighbor_rows],
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
        neighbor_search="scan",
    ):
        if noisifier is not None and anomaly_col is not None:
            raise ValueError("Only one of noisifier and anomaly_col can be specified.")
        super().__init__()
        self.categories = categories
        self.noisifier = noisifier
        self.anomaly_col = anomaly_col
        self.max_seq_len = max_seq_len
        self.poi_index = POIIndex(stays, clique_size, overlap_min_frac=overlap_min_frac,
                                  neighbor_search=neighbor_search)
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
