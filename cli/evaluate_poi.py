"""Evaluation script for next-POI recommendation.

Loads a trained POIRecModel checkpoint and evaluates on the test set using
full ranking against all POIs.

Usage:
    python -m cli.evaluate_poi --run_id <wandb-run-id> [--device cuda:0]
    python -m cli.evaluate_poi --checkpoint /path/to/poi_rec_ckpt.pt --no_wandb
"""

import argparse
import os
import tempfile
import types

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_dataset_config
from tasks.poi_rec.metrics import compute_rec_scores


def evaluate_poi(
    model,
    loader: DataLoader,
    device: str,
    n_pois: int,
    ks: list[int] = (1, 5, 10, 20),
    print_scores: bool = True,
) -> dict[str, float]:
    """Evaluate a POI rec model on a test set using full ranking.

    For each valid (non-padded, non-last) position, rank all POIs by
    dot-product score and record where the ground-truth POI falls.

    Returns:
        dict of metric_name → float.
    """
    model.eval()
    all_ranks = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate (POI rec)"):
            batch = {k: v.to(device) for k, v in batch.items()}

            query, poi_weight = model(
                batch["location"],
                batch["start"],
                batch["stop"],
                batch["category"],
                batch["agent"],
                batch["padding_mask"],
                batch["neighbor_padding_mask"],
            )

            rec_mask = batch["rec_mask"]                    # (B, S)
            target_poi_idx = batch["target_poi_idx"]        # (B, S)

            valid_queries = query[rec_mask]                 # (N, D)
            valid_targets = target_poi_idx[rec_mask]        # (N,)

            if valid_queries.shape[0] == 0:
                continue

            # Full ranking: scores against all POIs
            # Process in chunks to avoid OOM on large vocab
            chunk_size = 4096
            n_queries = valid_queries.shape[0]
            for i in range(0, n_queries, chunk_size):
                q = valid_queries[i : i + chunk_size]       # (chunk, D)
                t = valid_targets[i : i + chunk_size]       # (chunk,)

                scores = q @ poi_weight.T                   # (chunk, V)
                # Rank of the target POI (0-indexed; lower is better)
                target_scores = scores.gather(1, t.unsqueeze(1)).squeeze(1)  # (chunk,)
                ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1)     # (chunk,)
                all_ranks.append(ranks.cpu())

    if not all_ranks:
        print("No valid predictions found in test set.")
        return {}

    all_ranks = torch.cat(all_ranks).numpy()
    print(f"Total predictions: {len(all_ranks):,}")

    result = compute_rec_scores(all_ranks, ks=ks, print_result=print_scores)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_eval_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained POI rec checkpoint"
    )
    parser.add_argument("--run_id", type=str, default=None, help="W&B run ID")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (default: $WANDB_ENTITY)")
    parser.add_argument("--project", type=str, default="traxion-poi", help="W&B project")
    parser.add_argument("--no_wandb", action="store_true", help="Skip W&B logging")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local .pt checkpoint path")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Max sequence length (None = full trajectory)")
    parser.add_argument("--dataset_override", type=str, default=None, help="Override dataset name (for runs whose wandb config and ckpt args disagree)")
    return parser.parse_args()


def _download_latest_checkpoint(api_run, dest_dir: str) -> str:
    artifacts = [a for a in api_run.logged_artifacts() if a.type == "model"]
    if not artifacts:
        raise RuntimeError(f"No model artifacts found in W&B run {api_run.id}")
    latest = sorted(artifacts, key=lambda a: a.version)[-1]
    print(f"Downloading artifact {latest.name} (v{latest.version}) ...")
    artifact_dir = latest.download(root=dest_dir)
    pt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pt")]
    if not pt_files:
        raise RuntimeError(f"No .pt file in artifact {latest.name}")
    return os.path.join(artifact_dir, pt_files[0])


def main() -> None:
    eval_args = _parse_eval_args()

    # --- Load checkpoint ---
    if eval_args.checkpoint:
        print(f"Loading local checkpoint: {eval_args.checkpoint}")
        raw_ckpt = torch.load(eval_args.checkpoint, map_location=eval_args.device, weights_only=False)
        api_run = None
    else:
        api = wandb.Api()
        run_path = f"{eval_args.entity}/{eval_args.project}/{eval_args.run_id}"
        print(f"Fetching W&B run: {run_path}")
        api_run = api.run(run_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = _download_latest_checkpoint(api_run, tmpdir)
            raw_ckpt = torch.load(ckpt_path, map_location=eval_args.device, weights_only=False)

    print(
        f"Checkpoint: {raw_ckpt['epoch']:.1f} epoch(s), "
        f"{raw_ckpt['iteration']} iters, best val loss {raw_ckpt['best_val_loss']:.6f}"
    )

    # --- Reconstruct model from saved args ---
    args = types.SimpleNamespace(**raw_ckpt["args"])
    args.device = eval_args.device
    args.num_workers = eval_args.num_workers
    # Use full trajectory at eval time when possible; fall back to training
    # max_seq_len if sequences are too long for GPU memory.
    args.max_seq_len = getattr(eval_args, "max_seq_len", None)

    from tasks.poi_rec.model import POIRecModel
    from tasks.poi_rec.dataset import POIRecDataset, poi_rec_collate_fn, build_poi_vocab
    from utils.file_io import load_dataset_metadata, load_pois, load_stays
    from utils.training_data import ensure_category_columns, normalize_and_validate_agents, prepare_stays
    from utils.misc import split_train_to_train_val

    dataset_name = eval_args.dataset_override or getattr(args, "dataset", None) or (api_run.config.get("dataset") if api_run else None)
    if not dataset_name:
        raise RuntimeError("Cannot infer dataset from checkpoint or W&B run.")
    args.dataset = dataset_name
    print(f"Dataset: {dataset_name}")

    dataset_cfg = get_dataset_config(dataset_name)
    metadata = load_dataset_metadata(dataset_cfg)
    args.n_agents = int(metadata["n_agents"])
    args.training_log_dir = dataset_cfg.training_log_dir

    pois, categories = load_pois(dataset_cfg)
    args.n_categories = len(categories)
    min_time = metadata["min_time"]

    # Rebuild POI vocab and agent mapping from training data
    # Must mirror the train/val split used in train_poi.py so vocab matches
    print("Loading training data for vocab + agent mapping ...")
    train_stays = prepare_stays(load_stays("train", dataset_cfg), min_time=min_time)
    train_stays = ensure_category_columns(train_stays, categories)
    train_stays, _val_stays = split_train_to_train_val(
        train_stays, from_tail=getattr(args, "val_from_tail", False),
    )
    poi_to_idx = build_poi_vocab(train_stays)
    n_pois = len(poi_to_idx)
    print(f"POI vocab: {n_pois:,} POIs")

    # Build agent mapping from original IDs before normalization
    all_agents = pd.concat([train_stays["agent"], _val_stays["agent"]], ignore_index=True)
    unique_agents = pd.Index(all_agents.unique())
    agent_to_idx = pd.Series(np.arange(len(unique_agents), dtype=np.int64), index=unique_agents)

    # Prepare test data
    print("Loading test data ...")
    test_stays = prepare_stays(load_stays("test", dataset_cfg), min_time=min_time)
    test_stays = ensure_category_columns(test_stays, categories)
    n_unknown = test_stays["agent"].isin(unique_agents).eq(False).sum()
    if n_unknown > 0:
        print(f"Warning: {n_unknown} test stays with unseen agents — mapped to index 0")
    test_stays["agent"] = test_stays["agent"].map(agent_to_idx).fillna(0).astype(np.int64)

    test_data = POIRecDataset(
        test_stays, categories, args.clique_size, poi_to_idx,
        max_seq_len=args.max_seq_len,
        overlap_min_frac=getattr(args, "overlap_min_frac", 0.0),
    )

    batch_size = eval_args.batch_size or getattr(args, "val_batch_size", None) or 256
    test_loader = DataLoader(
        test_data, batch_size, shuffle=False,
        collate_fn=poi_rec_collate_fn, num_workers=eval_args.num_workers,
    )

    # --- Build model and load weights ---
    model = POIRecModel(args, n_pois)
    model.load_state_dict(raw_ckpt["model_state_dict"])
    model.to(eval_args.device)

    # --- Run evaluation ---
    print("Evaluating ...")
    result = evaluate_poi(model, test_loader, eval_args.device, n_pois)

    print("\n=== Test Results ===")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")

    if eval_args.no_wandb:
        print("Skipping W&B logging (--no_wandb).")
        return

    run = wandb.init(
        id=eval_args.run_id, resume="must",
        entity=eval_args.entity, project=eval_args.project,
        dir=args.training_log_dir,
    )
    run.log({f"test/{k}": v for k, v in result.items()})
    run.finish()
    print("Test metrics logged to W&B run", eval_args.run_id)


if __name__ == "__main__":
    main()
