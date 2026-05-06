"""LANL test-time scorer that uses the pretrained checkpoint as-is.

Dumps per-event (agent_idx, label, BCE_logit, visit_emb) for the train+test
splits and the full agent prototype bank. Used for unsupervised post-hoc
scoring on LANL: BCE noise score, prototype self-similarity, prototype
impostor signal, and per-user score calibration.

Usage:
    python -m cli.evaluate_lanl_unsup --checkpoint /path/to.pt \
        --out_dir logs/lanl_unsup_dumps/<tag> \
        --device cuda:0 [--batch_size 32]
"""

import argparse
import os
import types
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _build_loader(stays_df, categories, args, batch_size: int, num_workers: int):
    from utils.dataset import CliqueDataset, collate_fn

    has_anomaly = "anomaly" in stays_df.columns
    dataset = CliqueDataset(
        stays_df,
        categories,
        args.clique_size,
        noisifier=None,
        max_seq_len=None,
        anomaly_col="anomaly" if has_anomaly else None,
        overlap_min_frac=getattr(args, "overlap_min_frac", 0.0),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return loader


def _run_split(model, loader, device) -> Tuple[np.ndarray, ...]:
    """Iterate the loader and collect per-event arrays."""
    model.eval()
    agents = []
    labels = []
    bce_logits = []
    visit_embs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Dump"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, visit_emb, _own_proto, _ = model(
                batch["location"],
                batch["start"],
                batch.get("stop", None),
                batch["category"],
                batch["agent"],
                batch["padding_mask"],
                batch["neighbor_padding_mask"],
            )
            valid = ~batch["padding_mask"]
            bce_logits.append(logits[valid].detach().cpu().numpy())
            visit_embs.append(visit_emb[valid].detach().cpu().numpy())
            labels.append(batch["stay_anomalous"][valid].detach().cpu().numpy())
            agent_ids = batch["agent"][..., 0]
            agents.append(agent_ids[valid].detach().cpu().numpy())

    agents = np.concatenate(agents).astype(np.int64)
    labels = np.concatenate(labels).astype(np.float32)
    bce_logits = np.concatenate(bce_logits).astype(np.float32)
    visit_embs = np.concatenate(visit_embs).astype(np.float32)
    return agents, labels, bce_logits, visit_embs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--skip_train", action="store_true",
                   help="If set, skip dumping the train split (saves time when only test is needed).")
    eval_args = p.parse_args()

    os.makedirs(eval_args.out_dir, exist_ok=True)

    raw_ckpt = torch.load(
        eval_args.checkpoint, map_location=eval_args.device, weights_only=False
    )
    args = types.SimpleNamespace(**raw_ckpt["args"])
    args.device = eval_args.device
    args.num_workers = eval_args.num_workers
    args.max_seq_len = None

    from config import get_dataset_config
    from modules.model import Model
    from utils.file_io import load_dataset_metadata, load_pois, load_stays
    from utils.training_data import ensure_category_columns, prepare_stays

    dataset_name = getattr(args, "dataset", None)
    print(f"Dataset: {dataset_name}")
    dataset_cfg = get_dataset_config(dataset_name)
    metadata = load_dataset_metadata(dataset_cfg)
    args.training_log_dir = dataset_cfg.training_log_dir
    args.n_agents = int(metadata["n_agents"])
    pois, categories = load_pois(dataset_cfg)
    args.n_categories = len(categories)
    min_time = metadata["min_time"]

    # Rebuild train agent map (same as evaluate.py does).
    print("Loading train stays for agent map ...")
    train_stays_raw = prepare_stays(load_stays("train", dataset_cfg), min_time=min_time)
    train_stays_raw = ensure_category_columns(train_stays_raw, categories)
    unique_agents = pd.Index(train_stays_raw["agent"].unique())
    agent_to_idx = pd.Series(
        np.arange(len(unique_agents), dtype=np.int64), index=unique_agents
    )

    train_stays_raw["agent"] = train_stays_raw["agent"].map(agent_to_idx).astype(np.int64)
    print(f"Train stays: {len(train_stays_raw):,}, train agents: {len(unique_agents):,}")

    print("Loading test stays ...")
    test_stays = prepare_stays(load_stays("test", dataset_cfg), min_time=min_time)
    test_stays = ensure_category_columns(test_stays, categories)
    n_unknown = test_stays["agent"].isin(unique_agents).eq(False).sum()
    if n_unknown > 0:
        print(f"Warning: {n_unknown} test stays have agents unseen in training -> mapped to 0")
    test_stays["agent"] = test_stays["agent"].map(agent_to_idx).fillna(0).astype(np.int64)
    print(f"Test stays: {len(test_stays):,}")

    model = Model(args)
    model.load_state_dict(raw_ckpt["model_state_dict"])
    model.to(eval_args.device)

    # --- prototype bank ---
    n_agents = int(args.n_agents)
    with torch.no_grad():
        all_idx = torch.arange(n_agents, device=eval_args.device)
        protos = model.feature_encoder.agent_emb(all_idx).detach().cpu().numpy().astype(np.float32)
    print(f"Prototype bank: {protos.shape}")

    # --- test split ---
    print("Dumping TEST ...")
    test_loader = _build_loader(
        test_stays, categories, args, eval_args.batch_size, eval_args.num_workers
    )
    te_agents, te_labels, te_bce, te_visit = _run_split(
        model, test_loader, eval_args.device
    )
    print(f"  test events: {len(te_agents):,}, anomalies: {int(te_labels.sum())}")
    np.savez_compressed(
        os.path.join(eval_args.out_dir, "test.npz"),
        agents=te_agents, labels=te_labels, bce_logits=te_bce, visit_embs=te_visit,
    )

    if not eval_args.skip_train:
        # Train split: keep anomaly column so CliqueDataset populates stay_anomalous.
        # Per-user z stats are fit over all train events; the small number of train-period
        # anomalous rows (~0.7%) barely shifts per-user means, and we never use the label.
        train_dump = train_stays_raw.copy()
        print("Dumping TRAIN ...")
        train_loader = _build_loader(
            train_dump, categories, args, eval_args.batch_size, eval_args.num_workers
        )
        tr_agents, tr_labels, tr_bce, tr_visit = _run_split(
            model, train_loader, eval_args.device
        )
        print(f"  train events: {len(tr_agents):,}")
        np.savez_compressed(
            os.path.join(eval_args.out_dir, "train.npz"),
            agents=tr_agents, labels=tr_labels, bce_logits=tr_bce, visit_embs=tr_visit,
        )

    np.savez_compressed(
        os.path.join(eval_args.out_dir, "protos.npz"),
        protos=protos,
    )
    print(f"Done. Outputs at {eval_args.out_dir}")


if __name__ == "__main__":
    main()
