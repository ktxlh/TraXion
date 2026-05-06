from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import CliqueDataset
from utils.misc import determine_batch_size


def _batch_size_cache_path(training_log_dir: str) -> str:
    return os.path.join(training_log_dir, "batch_size_cache.json")


def _batch_size_cache_key(args: Any) -> str:
    params = {
        "model_dim": args.model_dim,
        "dim_feedforward": args.dim_feedforward,
        "n_scales": args.n_scales,
        "n_heads": args.n_heads,
        "num_layers": args.num_layers,
        "n_agent_attributes": args.n_agent_attributes,
        "hidden_dim": args.hidden_dim,
        "n_categories": args.n_categories,
        "n_agents": args.n_agents,
        "clique_size": args.clique_size,
        "dataset": args.dataset,
    }
    key_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _load_batch_size_cache(training_log_dir: str) -> dict:
    cache_path = _batch_size_cache_path(training_log_dir)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_batch_size_cache(cache: dict, training_log_dir: str) -> None:
    cache_path = _batch_size_cache_path(training_log_dir)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    logger: "WandbLogger",
    best_val_loss: float,
    epochs_without_improvement: int,
    args: Any,
) -> str:
    """Save a complete training checkpoint with all state needed for resumption.

    Args:
        path: Path to save checkpoint to
        model: Model to checkpoint
        optimizer: Optimizer with current state
        scheduler: LR scheduler (can be None)
        logger: Training logger with epoch/iteration info
        best_val_loss: Best validation loss seen so far
        epochs_without_improvement: Early stopping counter
        args: Training arguments for reference

    Returns:
        Path to the saved checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': logger.epoch,
        'iteration': logger.iteration,
        'best_val_loss': best_val_loss,
        'epochs_without_improvement': epochs_without_improvement,
        'rng_states': {
            'torch': torch.get_rng_state().cpu().numpy().tolist(),
            'cuda': torch.cuda.get_rng_state().cpu().numpy().tolist() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        },
        'args': vars(args),
        'wandb_run_id': logger.run.id if logger.use_wandb else None,
    }

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: str,
    weights_only_model: bool = False,
) -> dict:
    """Load a complete training checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: LR scheduler to load state into (can be None)
        device: Device to load tensors to
        weights_only_model: If True, load only model weights and skip
            optimizer/scheduler state (fine-tune mode)

    Returns:
        Complete checkpoint dictionary with all saved state
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if weights_only_model:
        # Fine-tune: allow architectural deltas between pretrain and finetune
        # (e.g. dropping the neighbor/co-location sub-layer via --no_neighbor_attn).
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False,
        )
        if missing:
            print(f"Newly initialized (missing in ckpt): {missing}")
        if unexpected:
            print(f"Skipped (not in current model): {unexpected}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def restore_rng_states(rng_states: dict) -> None:
    """Restore random number generator states for reproducibility.

    Args:
        rng_states: Dictionary with 'torch', 'cuda', 'numpy', 'random' RNG states
    """
    torch_state = torch.ByteTensor(rng_states['torch'])
    torch.set_rng_state(torch_state)

    if rng_states['cuda'] is not None and torch.cuda.is_available():
        cuda_state = torch.ByteTensor(rng_states['cuda'])
        torch.cuda.set_rng_state(cuda_state)

    np.random.set_state(rng_states['numpy'])
    random.setstate(rng_states['random'])


def safe_determine_batch_size(
    dataset: CliqueDataset,
    model: nn.Module,
    device: str,
    no_grad: bool,
    fallback: int = 1,
) -> int:
    """Best-effort batch-size detection with a conservative fallback."""
    try:
        return determine_batch_size(dataset, model, device, no_grad=no_grad)
    except Exception as exc:
        print(
            "Automatic batch-size detection failed "
            f"({type(exc).__name__}: {exc}). Falling back to batch_size={fallback}."
        )
        return fallback


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True if an exception corresponds to CUDA out-of-memory."""
    return (
        isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
    )


def _clear_cuda_cache_if_needed(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    wandb_logger: WandbLogger,
    device: str,
    split: str = "val",
):
    """Compute mean validation loss for one epoch."""
    model.eval()
    val_losses = []
    val_loss_dicts = []

    val_agent_attns = []
    with torch.no_grad():
        for batch in val_loader:
            if batch["location"].numel() == 0:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                logits, visit_emb, own_agent_emb, mean_agent_feat_attn = model(
                    batch["location"],
                    batch["start"],
                    batch["stop"],
                    batch["category"],
                    batch["agent"],
                    batch["padding_mask"],
                    batch["neighbor_padding_mask"],
                )
                loss, loss_dict = criterion(logits, visit_emb, own_agent_emb, batch)
                val_losses.append(loss.item())
                val_loss_dicts.append(loss_dict)
                val_agent_attns.append(mean_agent_feat_attn.item())
            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise

                _clear_cuda_cache_if_needed(device)
                raise RuntimeError(
                    "Validation CUDA OOM. Please reduce --val_batch_size or use the "
                    "batch-size selected by determine_batch_size in utils/misc.py."
                ) from exc

    agg_loss = float(np.mean(val_losses)) if val_losses else float("inf")
    agg_loss_dict = (
        {
            k: float(np.mean([d[k] for d in val_loss_dicts]))
            for k in val_loss_dicts[0].keys()
        }
        if val_loss_dicts
        else {}
    )
    wandb_logger.log(agg_loss_dict, split=split)
    if val_agent_attns:
        wandb_logger.log({"agent_feat_attn": float(np.mean(val_agent_attns))}, split=split)
    if agg_loss_dict:
        bce = agg_loss_dict.get("bce_loss", agg_loss)
        print(f"[{split}] bce_loss={bce:.6f}", flush=True)
    return agg_loss, agg_loss_dict


def train_epoch(
    train_loader,
    model,
    optimizer,
    criterion,
    logger,
    pbar,
    device,
    scheduler=None,
    holdout_loader=None,
    holdout_eval_every: int = 0,
):
    """Run one training epoch and log per-iteration loss."""
    model.train()
    total_batches = len(train_loader)
    epoch_index = int(logger.epoch)

    def _should_step_scheduler() -> bool:
        if scheduler is None:
            return False

        # For CosineAnnealingLR, keep LR fixed at eta_min once the floor is reached.
        # This prevents the post-T_max cosine upswing when continuing training.
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            eta_min = float(getattr(scheduler, "eta_min", 0.0))
            tol = 1e-12
            return any(group["lr"] > eta_min + tol for group in optimizer.param_groups)

        return True

    for batch_index, batch in enumerate(train_loader, start=1):
        logger.set_epoch_progress(epoch_index, batch_index, total_batches)

        if batch["location"].numel() == 0:
            continue

        batch = {k: v.to(device) for k, v in batch.items()}

        try:
            optimizer.zero_grad()
            logits, visit_emb, own_agent_emb, mean_agent_feat_attn = model(
                batch["location"],
                batch["start"],
                batch["stop"],
                batch["category"],
                batch["agent"],
                batch["padding_mask"],
                batch["neighbor_padding_mask"],
            )

            loss, loss_dict = criterion(logits, visit_emb, own_agent_emb, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if _should_step_scheduler():
                scheduler.step()

        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise

            _clear_cuda_cache_if_needed(device)
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                "Training CUDA OOM. Please reduce --train_batch_size or use the "
                "batch-size selected by determine_batch_size in utils/misc.py."
            ) from exc

        logger.iteration += 1
        logger.log(loss_dict, split="train")
        logger.log({"lr": optimizer.param_groups[0]["lr"]}, split="train")
        logger.log({"agent_feat_attn": mean_agent_feat_attn.item()}, split="train")

        if (
            holdout_loader is not None
            and holdout_eval_every > 0
            and logger.iteration % holdout_eval_every == 0
        ):
            validate_epoch(
                model,
                holdout_loader,
                criterion,
                logger,
                device,
                split="holdout",
            )
            model.train()

        pbar.update(1)

    logger.complete_epoch()
