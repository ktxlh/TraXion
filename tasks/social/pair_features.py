"""Handcrafted co-visitation + graph-structural features for agent pairs.

These features are well-known to dominate friendship prediction on Foursquare /
Gowalla. Built once from the *training* half of the stay parquet and the
*training* half of the edge split (no leakage). Used as auxiliary inputs to the
pair scorer alongside the learned pooled embeddings.

Feature vector (dim = 8), in order:
    [0] log1p(n_shared_pois)          — count of POIs visited by both agents
    [1] rarity_weighted_overlap       — sum_p log(N / popularity_p)   (p in shared)
    [2] jaccard_poi                   — |A ∩ B| / |A ∪ B|
    [3] adamic_adar_poi               — sum_p 1 / log(popularity_p)   (p in shared)
    [4] log1p(n_temporal_cooccur)     — count of (p, visit_u, visit_v) within τ hrs
    [5] log1p(common_neighbors)       — on train friend graph
    [6] adamic_adar_graph             — sum_w 1 / log(deg_w)           (w ∈ common)
    [7] resource_allocation_graph     — sum_w 1 / deg_w

All features are *symmetric*: f(u, v) == f(v, u).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "log_n_shared_pois",
    "rarity_weighted_overlap",
    "jaccard_poi",
    "adamic_adar_poi",
    "log_n_temporal_cooccur",
    "log_common_neighbors",
    "adamic_adar_graph",
    "resource_allocation_graph",
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class PairFeatureComputer:
    """Precomputes per-agent structures; serves pair features on demand.

    Parameters
    ----------
    train_stays
        Normalized stay dataframe (agent is int index). Must contain columns
        ``agent``, ``poi_id``, ``start``, ``stop``. ``start``/``stop`` are
        float hours from the dataset origin.
    train_edges
        DataFrame with columns ``u``, ``v`` (int, undirected, u < v) — train
        split only, to avoid leakage into val/test structural features.
    tau_hours
        Temporal co-occurrence window. Two visits to the same POI with
        |Δstart| <= τ count as a co-occurrence.
    mega_poi_threshold
        POIs with > this many unique visitors are dropped from the temporal
        co-occurrence count (tourist/transit hubs; noisy signal).
    """

    train_stays: pd.DataFrame
    train_edges: pd.DataFrame
    tau_hours: float = 3.0
    mega_poi_threshold: int = 500

    def __post_init__(self) -> None:
        df = self.train_stays
        if df["poi_id"].isna().any():
            df = df.dropna(subset=["poi_id"])
        df = df.copy()
        df["agent"] = df["agent"].astype(np.int64)
        # poi_id stays as-is (may be int for Gowalla, str for Foursquare)

        # --- per-POI popularity (unique visitors) ---
        poi_visitors = df.groupby("poi_id")["agent"].nunique()
        self._poi_pop = poi_visitors.to_dict()
        n_total_visitors = int(df["agent"].nunique())
        self._log_N = math.log(max(2, n_total_visitors))

        # --- per-agent POI sets ---
        agent_pois = df.groupby("agent")["poi_id"].apply(lambda s: frozenset(s.tolist()))
        self._agent_poi_set: dict[int, frozenset] = agent_pois.to_dict()

        # --- per-(agent, poi) visit times (sorted starts) ---
        # Drop mega-POIs to keep the temporal join cheap.
        df_small = df[df["poi_id"].map(poi_visitors) <= self.mega_poi_threshold]
        tmp: dict[int, dict] = defaultdict(dict)
        for (agent, poi), grp in df_small.groupby(["agent", "poi_id"], sort=False):
            tmp[int(agent)][poi] = np.sort(grp["start"].to_numpy(dtype=np.float64))
        self._agent_poi_times: dict[int, dict] = dict(tmp)

        # --- friend graph (train edges only) ---
        friends: dict[int, set[int]] = defaultdict(set)
        for u, v in self.train_edges[["u", "v"]].itertuples(index=False):
            u, v = int(u), int(v)
            friends[u].add(v)
            friends[v].add(u)
        self._friends: dict[int, frozenset[int]] = {a: frozenset(fs) for a, fs in friends.items()}
        self._deg: dict[int, int] = {a: len(fs) for a, fs in self._friends.items()}

    # ------------------------------------------------------------------
    def _temporal_cooccur(self, u: int, v: int, shared: frozenset) -> int:
        """Count (p, visit_u, visit_v) with |Δstart| <= tau over shared POIs."""
        times_u = self._agent_poi_times.get(u, {})
        times_v = self._agent_poi_times.get(v, {})
        if not times_u or not times_v:
            return 0
        tau = self.tau_hours
        total = 0
        for p in shared:
            tu = times_u.get(p)
            tv = times_v.get(p)
            if tu is None or tv is None:
                continue
            # Two-pointer count of pairs within tau.
            i = j = 0
            nu, nv = len(tu), len(tv)
            while i < nu and j < nv:
                diff = tu[i] - tv[j]
                if diff > tau:
                    j += 1
                elif diff < -tau:
                    i += 1
                else:
                    # tv[j] is within tau of tu[i]; count all j' with tv[j'] in [tu[i]-tau, tu[i]+tau].
                    k = j
                    while k < nv and abs(tu[i] - tv[k]) <= tau:
                        total += 1
                        k += 1
                    i += 1
        return total

    def _pair(self, u: int, v: int) -> np.ndarray:
        if u == v:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        pu = self._agent_poi_set.get(u, frozenset())
        pv = self._agent_poi_set.get(v, frozenset())
        shared = pu & pv
        union = pu | pv
        n_shared = len(shared)
        n_union = len(union)

        rarity = 0.0
        adamic_poi = 0.0
        for p in shared:
            pop = self._poi_pop.get(p, 1)
            rarity += self._log_N - math.log(max(1, pop))
            if pop > 1:
                adamic_poi += 1.0 / math.log(pop)

        jaccard = n_shared / n_union if n_union > 0 else 0.0
        tco = self._temporal_cooccur(u, v, shared) if n_shared > 0 else 0

        fu = self._friends.get(u, frozenset())
        fv = self._friends.get(v, frozenset())
        common = fu & fv
        cn = len(common)
        aa_g = 0.0
        ra_g = 0.0
        for w in common:
            dw = self._deg.get(w, 0)
            if dw > 1:
                aa_g += 1.0 / math.log(dw)
            if dw > 0:
                ra_g += 1.0 / dw

        return np.array(
            [
                math.log1p(n_shared),
                rarity,
                jaccard,
                adamic_poi,
                math.log1p(tco),
                math.log1p(cn),
                aa_g,
                ra_g,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def features_batch(self, pairs: np.ndarray) -> np.ndarray:
        """Compute features for an (N, 2) int array of pairs.

        Returns (N, FEATURE_DIM) float32.
        """
        out = np.empty((len(pairs), FEATURE_DIM), dtype=np.float32)
        for i, (u, v) in enumerate(pairs):
            out[i] = self._pair(int(u), int(v))
        return out

    __call__ = features_batch
