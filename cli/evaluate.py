"""Evaluation module for trained anomaly detection models.

This module provides:
  - evaluate(): importable function used during training
  - __main__ entry point for evaluating a checkpoint on the NUMOSIM test set

Example command:
    python -m cli.evaluate --run_id <wandb-run-id> [--device cuda:0]

The dataset is inferred automatically from the checkpoint (or W&B run config for
older checkpoints that predate dataset-key saving).
"""

import argparse
import os
import tempfile
import types
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_dataset_config

from utils.metrics import compute_scores

_AGENT_LEVEL_MAX_SIZE = None  # agents are far fewer than visits; no sampling needed


def evaluate(
    model: nn.Module, loader: DataLoader, device: str, max_size: int, print_scores: bool = True
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a trained model on test data.

    Args:
        model: Trained neural network model
        loader: DataLoader containing test data batches
        device: Device to run evaluation on (cpu/cuda)
        print_scores: Whether to print computed metrics (default: True)
        max_size: Maximum number of samples to use for score computation
    Returns:
        tuple: (result_dict, stay_probs, stay_targets, stay_agents)
            - result_dict: Dictionary containing evaluation metrics (AUC, etc.)
            - stay_probs: Array of predicted probabilities for each stay
            - stay_targets: Array of true anomaly labels for each stay
            - stay_agents: Array of agent indices for each stay
    """
    model.eval()
    stay_probs, stay_targets, stay_agents = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate"):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits, _, _, _ = model(
                batch["location"],
                batch["start"],
                batch.get("stop", None),
                batch["category"],
                batch["agent"],
                batch["padding_mask"],
                batch["neighbor_padding_mask"],
            )

            assert not torch.isnan(logits).any(), "NaN detected in stay scores"

            stay_prob = torch.sigmoid(logits)
            padding_mask = batch["padding_mask"].to(device)
            stay_probs.extend(stay_prob[~padding_mask].cpu().detach().tolist())
            stay_targets.extend(
                batch["stay_anomalous"][~padding_mask].cpu().detach().tolist()
            )
            # agent dim: (batch_size, seq_len, clique_size); index 0 is the self-agent
            agent_ids = batch["agent"][..., 0]
            stay_agents.extend(agent_ids[~padding_mask].cpu().detach().tolist())

    stay_probs = np.array(stay_probs)
    stay_targets = np.array(stay_targets).astype(float)
    stay_agents = np.array(stay_agents)

    # Visit-level metrics
    visit_result = compute_scores(stay_targets, stay_probs, print_result=print_scores, max_size=max_size)

    # Agent-level metrics: aggregate per agent using max pooling
    agent_df = pd.DataFrame({"agent": stay_agents, "prob": stay_probs, "target": stay_targets})
    agent_agg = agent_df.groupby("agent").agg(
        agent_prob=("prob", "max"),
        agent_target=("target", "max"),
    )
    agent_result = compute_scores(
        agent_agg["agent_target"].values,
        agent_agg["agent_prob"].values,
        print_result=print_scores,
        max_size=_AGENT_LEVEL_MAX_SIZE,
    )

    result = {f"visit/{k}": v for k, v in visit_result.items()}
    result.update({f"combined/agent/{k}": v for k, v in agent_result.items()})
    return result, stay_probs, stay_targets, stay_agents



def _parse_eval_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Clique checkpoint on the NUMOSIM test set"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="W&B run ID whose latest checkpoint will be evaluated (required unless --checkpoint is set)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3",
        help="Device to run evaluation on (default: cuda:3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (defaults to val_batch_size saved in checkpoint)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker processes (default: 8)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (default: pulled from $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="traxion",
        help="W&B project (default: traxion)",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=None,
        help="Maximum number of samples to use for score computation (default: 20 million)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Skip logging results to W&B (still uses W&B API to download checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a local .pt checkpoint file (skips W&B artifact download; requires --no_wandb)",
    )
    return parser.parse_args()


def _download_latest_checkpoint(api_run, dest_dir: str) -> str:
    """Download the latest model artifact from a W&B run and return its local path."""
    artifacts = [a for a in api_run.logged_artifacts() if a.type == "model"]
    if not artifacts:
        raise RuntimeError(f"No model artifacts found in W&B run {api_run.id}")

    # Pick the highest version (v0 < v1 < v2 …)
    latest = sorted(artifacts, key=lambda a: a.version)[-1]
    print(f"Downloading artifact {latest.name} (version v{latest.version}) ...")
    artifact_dir = latest.download(root=dest_dir)

    pt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pt")]
    if not pt_files:
        raise RuntimeError(f"No .pt file found in artifact {latest.name}")
    return os.path.join(artifact_dir, pt_files[0])


def main() -> None:
    eval_args = _parse_eval_args()

    # --- Fetch run metadata and download checkpoint via W&B API ---
    if eval_args.checkpoint:
        print(f"Loading local checkpoint: {eval_args.checkpoint}")
        raw_ckpt = torch.load(eval_args.checkpoint, map_location=eval_args.device, weights_only=False)
        api_run = None
    else:
        api = wandb.Api()
        run_path = f"{eval_args.entity}/{eval_args.project}/{eval_args.run_id}"
        print(f"Fetching W&B run: {run_path}")
        api_run = api.run(run_path)

        # Download into a temp dir; torch.load pulls everything into RAM so the
        # files can be deleted immediately afterwards.
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = _download_latest_checkpoint(api_run, tmpdir)
            raw_ckpt = torch.load(ckpt_path, map_location=eval_args.device, weights_only=False)

    print(
        f"Checkpoint loaded — trained for {raw_ckpt['epoch']:.1f} epoch(s), "
        f"{raw_ckpt['iteration']} iterations, "
        f"best val loss {raw_ckpt['best_val_loss']:.6f}"
    )

    # --- Reconstruct model args from the saved checkpoint ---
    args = types.SimpleNamespace(**raw_ckpt["args"])
    # Override runtime-specific settings with eval-time values.
    args.device = eval_args.device
    args.num_workers = eval_args.num_workers
    # Never trim sequences during evaluation: we want the full trajectory so
    # that no anomalous stays are accidentally cut out and results are
    # deterministic and complete.
    args.max_seq_len = None

    # --- Lazy imports (keep module importable without all deps at load time) ---
    from modules.model import Model
    from utils.dataset import CliqueDataset, collate_fn
    from utils.file_io import load_dataset_metadata, load_pois, load_stays
    from utils.training_data import ensure_category_columns, prepare_stays

    # Infer dataset from checkpoint args; fall back to W&B run config for
    # checkpoints saved before the dataset key was included.
    dataset_name = getattr(args, "dataset", None) or (api_run.config.get("dataset") if api_run else None)
    if not dataset_name:
        raise RuntimeError(
            "Could not determine dataset from checkpoint or W&B run config. "
            "The checkpoint predates dataset-key saving and the W&B run has no 'dataset' config entry."
        )
    print(f"Dataset inferred from checkpoint: {dataset_name}")
    dataset_cfg = get_dataset_config(dataset_name)
    metadata = load_dataset_metadata(dataset_cfg)
    args.training_log_dir = dataset_cfg.training_log_dir

    args.n_agents = int(metadata["n_agents"])  # embedding vocab size

    # --- Load POI categories (must match training) ---
    pois, categories = load_pois(dataset_cfg)
    args.n_categories = len(categories)
    min_time = metadata["min_time"]

    # --- Prepare test stays (no noisifier — labels are ground-truth NUMOSIM) ---
    print("Loading test data ...")
    test_stays = prepare_stays(load_stays("test", dataset_cfg), min_time=min_time)
    test_stays = ensure_category_columns(test_stays, categories)

    # Re-build the agent→index mapping from training data so that agent
    # embeddings refer to the same entries as during training.
    print("Loading training data to rebuild agent mapping ...")
    train_stays_raw = prepare_stays(load_stays("train", dataset_cfg), min_time=min_time)
    unique_agents = pd.Index(train_stays_raw["agent"].unique())
    agent_to_idx = pd.Series(
        np.arange(len(unique_agents), dtype=np.int64), index=unique_agents
    )
    n_unknown = test_stays["agent"].isin(unique_agents).eq(False).sum()
    if n_unknown > 0:
        print(f"Warning: {n_unknown} test stays have agents unseen in training — mapped to index 0")
    test_stays["agent"] = test_stays["agent"].map(agent_to_idx).fillna(0).astype(np.int64)

    print(f"Test stays: {len(test_stays):,}, categories: {categories}")

    test_data = CliqueDataset(
        test_stays,
        categories,
        args.clique_size,
        noisifier=None,
        max_seq_len=None,
        anomaly_col="anomaly", # use ground-truth labels from dataset
        overlap_min_frac=getattr(args, "overlap_min_frac", 0.0),
    )

    # Determine batch size: CLI flag > checkpoint val_batch_size > safe default
    batch_size = eval_args.batch_size
    if batch_size is None:
        batch_size = getattr(args, "val_batch_size", None) or 256
        print(f"Using batch size: {batch_size} (from checkpoint val_batch_size)")

    test_loader = DataLoader(
        test_data,
        batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=eval_args.num_workers,
    )

    # --- Build model and load weights ---
    model = Model(args)
    model.load_state_dict(raw_ckpt["model_state_dict"])
    model.to(eval_args.device)

    # --- Run evaluation ---
    print("Running evaluation on test set ...")
    result, stay_probs, stay_targets, stay_agents = evaluate(
        model, test_loader, eval_args.device, eval_args.max_size, print_scores=True
    )

    # Print results with 4 decimal places
    print("\n=== Test Results (4 decimal places) ===")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")

    if eval_args.no_wandb:
        print("Skipping W&B logging (--no_wandb).")
        return

    # --- Resume the original W&B run and log test metrics ---
    run = wandb.init(
        id=eval_args.run_id,
        resume="must",
        entity=eval_args.entity,
        project=eval_args.project,
        dir=args.training_log_dir,
    )
    run.log({f"test/{k}": v for k, v in result.items()})

    # Save raw predictions alongside the W&B run files for error analysis.
    # wandb.run.dir points to the local run folder (e.g. .../wandb/run-*/files/).
    preds_path = os.path.join(run.dir, "test_predictions.npz")
    np.savez(preds_path, stay_probs=stay_probs, stay_targets=stay_targets)
    print(f"Predictions saved to {preds_path}")

    run.finish()
    print("Test metrics logged to W&B run", eval_args.run_id)


if __name__ == "__main__":
    main()
