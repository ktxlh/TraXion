"""Evaluate a saved social-link-prediction checkpoint on the test split.

Prints AUC, AP, Hits@10, MRR with 4 decimals.

Example::

    python -m cli.evaluate_social --dataset foursquare-tokyo \\
        --resume_from_checkpoint runs/social_<id>.pt --device cuda:0
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from config import get_dataset_config
from tasks.social.dataset import SocialTrajectoryDataset
from tasks.social.metrics import classification_metrics, ranking_metrics
from tasks.social.model import SocialModel
from tasks.social.pair_features import FEATURE_DIM as PAIR_FEATURE_DIM
from tasks.social.pair_features import PairFeatureComputer
from cli.train_social import (
    SPLITS_ROOT,
    _compute_agent_embeddings,
    _load_social_splits,
    _score_pairs_with_model,
)
from utils.argparse import parse_social_args
from utils.file_io import load_dataset_metadata, load_pois, load_stays
from utils.misc import split_train_to_train_val
from utils.training_data import (
    ensure_category_columns,
    normalize_and_validate_agents,
    prepare_stays,
)
from utils.wandb_logger import WandbLogger


def main(args) -> None:
    if not args.resume_from_checkpoint:
        raise ValueError("--resume_from_checkpoint is required for evaluate_social.py")

    cfg = get_dataset_config(args.dataset)
    metadata = load_dataset_metadata(cfg)
    _, categories = load_pois(cfg)

    raw_train = load_stays("train", cfg)
    stays = prepare_stays(raw_train, min_time=metadata["min_time"])
    stays = ensure_category_columns(stays, categories)
    train_stays, val_stays = split_train_to_train_val(stays)
    train_stays, val_stays = normalize_and_validate_agents(train_stays, val_stays)
    all_stays = pd.concat([train_stays, val_stays], ignore_index=True)

    splits = _load_social_splits(args.dataset)
    n_agents = int(splits["meta"]["n_agents"])
    test_edges = splits["test_edges"]
    test_nonedges = splits["test_nonedges"]
    train_edges = splits["train_edges"]
    all_observed = set(
        map(tuple, pd.concat([train_edges, splits["val_edges"], test_edges])[["u", "v"]].values.tolist())
    )

    # Checkpoint carries its own args — use them to build a matching model.
    ckpt = torch.load(args.resume_from_checkpoint, map_location=args.device, weights_only=False)
    ck_args = SimpleNamespace(**ckpt["args"])
    ck_args.device = args.device
    ck_args.num_workers = args.num_workers
    ck_args.val_batch_size = args.val_batch_size or 64

    pfc = PairFeatureComputer(train_stays=train_stays, train_edges=train_edges)
    pair_feature_dim = 0 if getattr(ck_args, "disable_pair_features", False) else PAIR_FEATURE_DIM
    pair_feature_fn = None if pair_feature_dim == 0 else pfc.features_batch

    traj_dataset = SocialTrajectoryDataset(
        stays=all_stays, categories=categories, clique_size=ck_args.clique_size,
        n_agents=n_agents, max_seq_len=ck_args.max_seq_len,
        overlap_min_frac=getattr(ck_args, "overlap_min_frac", 0.0),
    )

    model = SocialModel(ck_args, pair_feature_dim=pair_feature_dim).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Re-build the GNN train-graph CSR if this checkpoint was trained with --gnn_head.
    # The neighbor index and per-agent embedding cache are not persisted in the
    # checkpoint (intentional: they are derived state), so re-derive both here.
    if getattr(model, "use_gnn_head", False):
        from collections import defaultdict
        from modules.graph_refiner import build_neighbor_index
        friend_graph: dict[int, set[int]] = defaultdict(set)
        for u, v in train_edges[["u", "v"]].itertuples(index=False):
            u, v = int(u), int(v)
            friend_graph[u].add(v); friend_graph[v].add(u)
        nbr_idx, nbr_off = build_neighbor_index(friend_graph, n_agents, device=args.device)
        model.set_neighbor_index(nbr_idx, nbr_off)
        print(f"[eval gnn] rebuilt CSR: {int(nbr_idx.numel())} edges, avg deg "
              f"{nbr_idx.numel() / max(1, n_agents):.2f}")

    embs = _compute_agent_embeddings(traj_dataset, model, args.device, ck_args.val_batch_size, args.num_workers)
    if getattr(model, "use_gnn_head", False):
        model.set_emb_cache(embs)
    def score_fn(pairs: np.ndarray) -> np.ndarray:
        return _score_pairs_with_model(pairs, model, embs, pair_feature_fn, args.device)

    cls = classification_metrics(
        test_edges[["u", "v"]].values.astype(np.int64),
        test_nonedges[["u", "v"]].values.astype(np.int64),
        score_fn,
    )
    rnk = ranking_metrics(
        test_edges[["u", "v"]].values.astype(np.int64),
        all_observed, n_agents, score_fn, n_negatives=100, k=10, seed=7,
    )
    m = {**cls, **rnk}
    print(f"[test] AUC={m['auc']:.4f}  AP={m['ap']:.4f}  Hits@10={m['hits_at_10']:.4f}  MRR={m['mrr']:.4f}")
    print(f"AUC & AP & Hits@10 & MRR:  {m['auc']:.4f} & {m['ap']:.4f} & {m['hits_at_10']:.4f} & {m['mrr']:.4f}")

    if args.use_wandb:
        logger = WandbLogger("eval", args, project=args.wandb_project)
        logger.log({f"test/{k}": v for k, v in m.items()}, split="test")
        logger.finish()


if __name__ == "__main__":
    main(parse_social_args())
