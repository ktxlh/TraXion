"""Per-stay dataset for eICU mortality fine-tuning.

Each ``__getitem__`` returns one stay's events (with neighbor padding from
the shared POIIndex) plus a binary mortality label. Long sequences are
randomly trimmed at training time (``max_seq_len``) and centred on a
representative window at eval time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.dataset import POIIndex
from utils.misc import make_instance_dict


class StayMortalityDataset(Dataset):
    """Per-stay mortality dataset.

    At training time (`train=True`, `n_eval_windows=1`) randomly samples one
    window per stay. At eval time set `n_eval_windows>1` to enumerate multiple
    deterministic, evenly-spaced windows per stay (their logits are then
    averaged in `evaluate_split`).
    """

    def __init__(
        self,
        stays: pd.DataFrame,
        agent_to_label: dict[int, int],
        categories: list[str],
        clique_size: int,
        poi_index: POIIndex,
        max_seq_len: int | None = None,
        zero_agent_input: bool = True,
        train: bool = True,
        tab_features: dict[int, np.ndarray] | None = None,
        agent_to_stay_id: dict[int, int] | None = None,
        n_eval_windows: int = 1,
    ) -> None:
        super().__init__()
        self.categories = categories
        self.clique_size = clique_size
        self.max_seq_len = max_seq_len
        self.poi_index = poi_index
        self.train = train
        self.zero_agent_input = zero_agent_input
        self.tab_features = tab_features
        self.agent_to_stay_id = agent_to_stay_id or {}
        self.n_eval_windows = max(1, int(n_eval_windows))

        # Drop stays without events; preserve the order in agent_to_label.
        self._grouped = stays.groupby("agent")
        valid_agents = [int(a) for a in agent_to_label if int(a) in self._grouped.groups]
        self.agents = valid_agents
        self.labels = np.asarray(
            [int(agent_to_label[int(a)]) for a in self.agents], dtype=np.int64
        )

        # Build the (agent, window_idx) item list.  At training time we collapse
        # to one item per agent; at eval time we expand to n_eval_windows items
        # per agent so identical-agent logits can be averaged downstream.
        windows = 1 if self.train else self.n_eval_windows
        self._items = [(a, w) for a in self.agents for w in range(windows)]

    def __len__(self) -> int:
        return len(self._items)

    def _window_start(self, n_events: int, w: int) -> int:
        if self.max_seq_len is None or n_events <= self.max_seq_len:
            return 0
        slack = n_events - self.max_seq_len
        if self.train:
            return int(np.random.randint(0, slack + 1))
        if self.n_eval_windows == 1:
            return slack // 2
        # Evenly-spaced deterministic windows (covers head, middle and tail).
        denom = max(1, self.n_eval_windows - 1)
        return int(round(w * slack / denom))

    def __getitem__(self, idx: int) -> dict:
        agent, w = self._items[idx]
        label_idx = self.agents.index(agent) if w > 0 else None  # cheap when w=0
        if w == 0:
            label = int(self.labels[idx if self.train else idx // self.n_eval_windows])
        else:
            label = int(self.labels[idx // self.n_eval_windows])
        stay_df = self._grouped.get_group(agent)

        if self.max_seq_len is not None and len(stay_df) > self.max_seq_len:
            start = self._window_start(len(stay_df), w)
            stay_df = stay_df.iloc[start : start + self.max_seq_len]

        stay_df = stay_df.copy()
        stay_df["stay_anomalous"] = False
        seq_len = len(stay_df)
        stay_df, neighbor_padding_mask = self.poi_index.find_neighbors(stay_df)
        instance = make_instance_dict(stay_df, self.categories)
        instance = self.poi_index.correct_shapes(instance, seq_len, neighbor_padding_mask)

        if self.zero_agent_input:
            instance["agent"] = torch.zeros_like(instance["agent"])

        instance["label"] = torch.tensor(label, dtype=torch.float32)
        instance["agent_id"] = torch.tensor(int(agent), dtype=torch.long)

        if self.tab_features is not None:
            stay_id = int(self.agent_to_stay_id.get(int(agent), -1))
            vec = self.tab_features.get(stay_id, None)
            if vec is None:
                vec = np.zeros(next(iter(self.tab_features.values())).shape, dtype=np.float32)
            instance["tab_feat"] = torch.from_numpy(np.asarray(vec, dtype=np.float32))
        return instance


def stay_mortality_collate_fn(batch: list[dict]) -> dict:
    out: dict[str, torch.Tensor] = {}
    for k in batch[0].keys():
        if k == "label":
            out["labels"] = torch.stack([b[k] for b in batch], dim=0)
            continue
        if k == "agent_id":
            out["agent_ids"] = torch.stack([b[k] for b in batch], dim=0)
            continue
        if k == "tab_feat":
            out["tab_feat"] = torch.stack([b[k] for b in batch], dim=0)
            continue
        pad_val = (k in ("padding_mask", "neighbor_padding_mask"))
        out[k] = pad_sequence(
            [b[k] for b in batch], batch_first=True, padding_value=pad_val,
        )
    if out["padding_mask"].dim() == 3:
        out["padding_mask"] = out["padding_mask"][..., 0]
        out["stay_anomalous"] = out["stay_anomalous"][..., 0]
    return out
