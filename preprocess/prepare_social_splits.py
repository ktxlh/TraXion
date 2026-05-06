"""Prepare social-link-prediction splits for a per-city Foursquare/Gowalla dataset.

Reproduces the canonical agent-ID normalization used by ``train.py`` so that
edge endpoints align with the agent embedding indices the pretrained checkpoint
already knows::

    prepare_stays -> split_train_to_train_val(seed=42) -> normalize_and_validate_agents

Emits (under ``$TRAXION_DATA_DIR/social_splits/<dataset>/``):
  agent_to_idx.parquet   raw_agent -> int idx  (the canonical mapping)
  train_edges.parquet    u,v                   (undirected; u < v)
  val_edges.parquet      u,v
  test_edges.parquet     u,v
  test_nonedges.parquet  u,v                   same size as test_edges, for AUC/AP
  meta.json              n_agents, counts, seed

Negative non-edges are drawn from the full pair space minus *all* observed edges
(train+val+test positives) to avoid leakage. Training-time negatives are sampled
on-the-fly during fit and are NOT materialized here.

Usage::

    python -m preprocess.prepare_social_splits --dataset foursquare-tokyo
    python -m preprocess.prepare_social_splits --dataset gowalla-stockholm-v1
    python -m preprocess.prepare_social_splits --dataset gowalla-austin-v1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATASET_PATHS, get_dataset_config
from utils.file_io import load_dataset_metadata, load_stays
from utils.misc import split_train_to_train_val
from utils.training_data import normalize_and_validate_agents, prepare_stays


_DATA_ROOT = Path(os.environ.get("TRAXION_DATA_DIR", "./data"))


def _social_edges_path(dataset: str) -> Path:
    """Standard layout: ``<data>/<dataset>/<dataset>_social_edges.csv``."""
    return _DATA_ROOT / dataset / f"{dataset}_social_edges.csv"


SOCIAL_EDGE_PATHS = {
    name: str(_social_edges_path(name))
    for name in (
        "foursquare-tokyo",
        "gowalla-stockholm-v1",
        "gowalla-austin-v1",
    )
}

OUT_ROOT = _DATA_ROOT / "social_splits"


def _build_agent_mapping(dataset: str) -> tuple[pd.Series, int]:
    """Reproduce train.py's canonical agent_id -> int mapping."""
    cfg = get_dataset_config(dataset)
    meta = load_dataset_metadata(cfg)
    min_time = meta["min_time"]

    raw_train = load_stays("train", cfg)
    train_stays = prepare_stays(raw_train, min_time=min_time)
    train_stays, val_stays = split_train_to_train_val(train_stays)
    train_norm, val_norm = normalize_and_validate_agents(train_stays, val_stays)

    raw_train_ids = pd.concat([train_stays["agent"], val_stays["agent"]], ignore_index=True)
    int_ids = pd.concat([train_norm["agent"], val_norm["agent"]], ignore_index=True)
    agent_to_idx = (
        pd.DataFrame({"raw": raw_train_ids, "idx": int_ids})
        .drop_duplicates("raw")
        .set_index("raw")["idx"]
        .sort_values()
    )
    n_agents = int(agent_to_idx.max()) + 1
    assert len(agent_to_idx) == n_agents, "agent_to_idx must be contiguous"
    return agent_to_idx, n_agents


def _load_and_map_edges(edges_path: str, agent_to_idx: pd.Series) -> pd.DataFrame:
    raw = pd.read_csv(edges_path)
    assert set(raw.columns) >= {"u", "v"}, f"expected columns u,v in {edges_path}"

    before = len(raw)
    # Some endpoints may be absent from train (edges between city-users whose
    # stays were dropped by prepare_stays for NaNs, or users present only in
    # test). Keep only edges where both endpoints exist in the canonical map.
    mapped = raw.assign(
        u_idx=raw["u"].map(agent_to_idx),
        v_idx=raw["v"].map(agent_to_idx),
    ).dropna(subset=["u_idx", "v_idx"])
    dropped = before - len(mapped)

    edges = mapped[["u_idx", "v_idx"]].astype(np.int64).rename(columns={"u_idx": "u", "v_idx": "v"})
    # Undirected: canonicalize so u < v; drop self-loops + duplicates.
    edges = edges[edges["u"] != edges["v"]].copy()
    lo = np.minimum(edges["u"].values, edges["v"].values)
    hi = np.maximum(edges["u"].values, edges["v"].values)
    edges["u"], edges["v"] = lo, hi
    edges = edges.drop_duplicates(["u", "v"]).reset_index(drop=True)
    print(f"  loaded {before:,} raw edges -> {len(edges):,} undirected unique (dropped {dropped:,} unmapped)")
    return edges


def _split_edges(edges: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 random split (undirected edges treated atomically)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edges))
    n = len(edges)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train = edges.iloc[perm[:n_train]].reset_index(drop=True)
    val = edges.iloc[perm[n_train : n_train + n_val]].reset_index(drop=True)
    test = edges.iloc[perm[n_train + n_val :]].reset_index(drop=True)
    return train, val, test


def _sample_nonedges(n_agents: int, forbidden: set[tuple[int, int]], n_sample: int, seed: int) -> pd.DataFrame:
    """Uniform random non-edges over all unordered agent pairs."""
    rng = np.random.default_rng(seed)
    out = []
    seen: set[tuple[int, int]] = set()
    # Oversample to account for collisions; total pair space is O(n_agents^2).
    target = n_sample
    attempts = 0
    max_attempts = max(20 * target, 200_000)
    while len(out) < target and attempts < max_attempts:
        batch = max(2 * (target - len(out)), 1024)
        u = rng.integers(0, n_agents, size=batch)
        v = rng.integers(0, n_agents, size=batch)
        mask = u != v
        u, v = u[mask], v[mask]
        lo = np.minimum(u, v); hi = np.maximum(u, v)
        for a, b in zip(lo.tolist(), hi.tolist()):
            key = (a, b)
            if key in forbidden or key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= target:
                break
        attempts += batch
    if len(out) < target:
        raise RuntimeError(f"Could only sample {len(out):,} / {target:,} non-edges; graph may be too dense")
    return pd.DataFrame(out, columns=["u", "v"], dtype=np.int64)


def run(dataset: str, seed: int = 42) -> None:
    if dataset not in SOCIAL_EDGE_PATHS:
        raise ValueError(f"No social edges configured for {dataset!r}")
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset {dataset!r}")

    print(f"[{dataset}] building canonical agent mapping from train stays ...")
    agent_to_idx, n_agents = _build_agent_mapping(dataset)
    print(f"  n_agents (train+val) = {n_agents:,}")

    print(f"[{dataset}] loading + mapping social edges ...")
    edges = _load_and_map_edges(SOCIAL_EDGE_PATHS[dataset], agent_to_idx)

    print(f"[{dataset}] splitting 70/15/15 with seed={seed} ...")
    train_e, val_e, test_e = _split_edges(edges, seed=seed)
    print(f"  train={len(train_e):,}  val={len(val_e):,}  test={len(test_e):,}")

    print(f"[{dataset}] sampling test non-edges (|test_nonedges| = |test_edges|) ...")
    forbidden = set(map(tuple, edges[["u", "v"]].values.tolist()))
    test_nonedges = _sample_nonedges(n_agents, forbidden, len(test_e), seed=seed + 1)

    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_to_idx.rename_axis("raw").reset_index().rename(columns={0: "idx"}).to_parquet(
        out_dir / "agent_to_idx.parquet", index=False
    )
    train_e.to_parquet(out_dir / "train_edges.parquet", index=False)
    val_e.to_parquet(out_dir / "val_edges.parquet", index=False)
    test_e.to_parquet(out_dir / "test_edges.parquet", index=False)
    test_nonedges.to_parquet(out_dir / "test_nonedges.parquet", index=False)

    meta = {
        "dataset": dataset,
        "n_agents": n_agents,
        "n_edges_total": int(len(edges)),
        "n_train_edges": int(len(train_e)),
        "n_val_edges": int(len(val_e)),
        "n_test_edges": int(len(test_e)),
        "n_test_nonedges": int(len(test_nonedges)),
        "seed": seed,
        "split_ratios": [0.70, 0.15, 0.15],
        "source_edges_csv": SOCIAL_EDGE_PATHS[dataset],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{dataset}] wrote outputs to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SOCIAL_EDGE_PATHS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.dataset, seed=args.seed)


if __name__ == "__main__":
    main()
