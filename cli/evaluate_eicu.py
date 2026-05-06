"""Evaluate a saved eICU mortality checkpoint on the held-out test split.

Prints AUROC, AUPRC, Max-F1, and Sensitivity at 90% Specificity (4 decimals).

Example::

    python -m cli.evaluate_eicu --dataset eicu-demo \\
        --resume_from_checkpoint runs/eicu_<id>.pt --device cuda:0
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import get_dataset_config
from tasks.eicu_mortality.dataset import (
    StayMortalityDataset,
    stay_mortality_collate_fn,
)
from tasks.eicu_mortality.metrics import mortality_scores
from tasks.eicu_mortality.model import MortalityModel
from tasks.eicu_mortality.tabular_features import build_tabular_features
from cli.train_eicu import _build_label_map, evaluate_split
from utils.argparse import parse_eicu_args
from utils.dataset import POIIndex
from utils.file_io import load_dataset_metadata, load_pois, load_stays
from utils.training_data import ensure_category_columns, prepare_stays
from utils.wandb_logger import WandbLogger


def main(args) -> None:
    if not args.resume_from_checkpoint:
        raise ValueError("--resume_from_checkpoint is required for evaluate_eicu.py")

    cfg = get_dataset_config(args.dataset)
    metadata = load_dataset_metadata(cfg)
    _, categories = load_pois(cfg)

    # Use the checkpoint's saved args so the model architecture matches exactly.
    ckpt = torch.load(args.resume_from_checkpoint, map_location=args.device, weights_only=False)
    ck_args = SimpleNamespace(**ckpt["args"])
    ck_args.device = args.device
    ck_args.num_workers = args.num_workers
    ck_args.val_batch_size = args.val_batch_size or 64
    ck_args.n_categories = len(categories)
    ck_args.n_agents = int(metadata["n_agents"])

    # Load both splits' events.
    train_stays = prepare_stays(load_stays("train", cfg), min_time=metadata["min_time"])
    test_stays = prepare_stays(load_stays("test", cfg), min_time=metadata["min_time"])
    train_stays = ensure_category_columns(train_stays, categories)
    test_stays = ensure_category_columns(test_stays, categories)

    labels_path = os.path.join(
        os.path.dirname(cfg.train_data_path), "eicu_stay_labels.parquet"
    )
    sl = pd.read_parquet(labels_path)
    sl["mortality"] = sl["mortality"].astype(np.int64)
    sl["patientunitstayid"] = sl["patientunitstayid"].astype(np.int64)
    test_stay_ids = set(test_stays["patientunitstayid"].astype(np.int64).unique())
    sl_test = sl[(sl["split"] == "test") & sl["patientunitstayid"].isin(test_stay_ids)]
    test_agent_label = _build_label_map(sl_test, test_stays)

    all_stays = pd.concat([train_stays, test_stays], ignore_index=True)
    poi_index = POIIndex(all_stays, ck_args.clique_size,
                        overlap_min_frac=getattr(ck_args, "overlap_min_frac", 0.0))

    tab_features = None
    tab_feat_dim = 0
    if getattr(ck_args, "use_tabular_features", False):
        bundle = build_tabular_features(sl, train_stays, test_stays)
        tab_features = bundle.by_stay
        tab_feat_dim = bundle.feature_dim
    agent_to_stay_id_test = (
        test_stays.groupby("agent")["patientunitstayid"]
        .agg(lambda s: int(s.iloc[0])).to_dict()
    )

    n_eval_windows = max(1, int(getattr(ck_args, "n_eval_windows", 1)))
    test_data = StayMortalityDataset(
        stays=test_stays, agent_to_label=test_agent_label, categories=categories,
        clique_size=ck_args.clique_size, poi_index=poi_index,
        max_seq_len=ck_args.max_seq_len,
        zero_agent_input=getattr(ck_args, "zero_agent_input", True),
        train=False, n_eval_windows=n_eval_windows,
        tab_features=tab_features, agent_to_stay_id=agent_to_stay_id_test,
    )
    test_loader = DataLoader(
        test_data, ck_args.val_batch_size, shuffle=False,
        collate_fn=stay_mortality_collate_fn, num_workers=args.num_workers,
    )
    print(f"Test dataset: {len(test_data):,} stays, "
          f"mortality rate={float(test_data.labels.mean()):.4f}")

    model = MortalityModel(ck_args, tab_feat_dim=tab_feat_dim).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics, _, _ = evaluate_split(model, test_loader, args.device)
    print(f"[test] AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
          f"MaxF1={metrics['max_f1']:.4f}  "
          f"Sens@Spec0.9={metrics.get('sens_at_spec_90', float('nan')):.4f}")
    print(f"AUROC & AUPRC & MaxF1 & Sens@Spec0.9:  "
          f"{metrics['auroc']:.4f} & {metrics['auprc']:.4f} & "
          f"{metrics['max_f1']:.4f} & {metrics.get('sens_at_spec_90', float('nan')):.4f}")

    if args.use_wandb:
        logger = WandbLogger("eval", args, project=args.wandb_project)
        logger.log({f"test/{k}": v for k, v in metrics.items()}, split="test")
        logger.finish()


if __name__ == "__main__":
    main(parse_eicu_args())
