"""Pre-training and anomaly-detection fine-tuning entry point for TraXion.

Pre-trains a TraXion backbone with the noise-detection BCE plus prototype-contrastive
objective on any of the supported datasets, or fine-tunes for anomaly detection from
an existing checkpoint via ``--finetune --resume_from_checkpoint``.

Example (pre-train Urban Anomalies-Berlin)::

    python -m cli.train --dataset urban-berlin --device cuda:0 \\
        --time_modulo daily --clique_size 4 --num_epochs 1000 \\
        --early_stopping_patience 100 --ema_early_stopping
"""
import os
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import get_dataset_config
from modules.model import Model
from utils.argparse import parse_args
from utils.dataset import CliqueDataset, collate_fn
from utils.file_io import load_dataset_metadata, load_pois, load_stays
from utils.info import print_parameter_distribution
from utils.wandb_logger import WandbLogger
from utils.noisifier import Noisifier
from utils.criterion import CustomCriterion
from utils.misc import split_train_to_train_val
from utils.training_data import (
    ensure_category_columns,
    normalize_and_validate_agents,
    prepare_stays,
)
from utils.training_runtime import (
    safe_determine_batch_size,
    train_epoch,
    validate_epoch,
    _batch_size_cache_key,
    _load_batch_size_cache,
    _save_batch_size_cache,
    save_checkpoint,
    load_checkpoint,
    restore_rng_states,
)


def main(args: Any) -> None:
    """Main execution function for training and evaluation.

    Args:
        args: Parsed command line arguments containing all configuration
    """
    # Deterministic-ish seeding for multi-seed runs (LANL cross-domain stats).
    # Affects Python/numpy/torch RNGs at entry; dataset splits use their own
    # chunk_split_seed argument for reproducibility.
    seed = int(getattr(args, "seed", 0) or 0)
    if seed > 0:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        perturb_seed = int(getattr(args, "perturb_seed", 0) or 0)
        if perturb_seed > 0:
            np.random.seed(perturb_seed)
        else:
            np.random.seed(None)  # OS entropy so Noisifier perturbations vary across runs
        print(f"[train] seeded RNGs with seed={seed}, perturb_seed={perturb_seed or 'OS-entropy'}")

    dataset_cfg = get_dataset_config(args.dataset)
    args.training_log_dir = dataset_cfg.training_log_dir

    metadata = load_dataset_metadata(dataset_cfg)
    args.n_agents = int(metadata["n_agents"])

    pois, categories = load_pois(dataset_cfg)
    use_real_labels = getattr(args, "use_real_anomaly_labels", False)
    if use_real_labels:
        print(
            "[train] --use_real_anomaly_labels: bypassing Noisifier; "
            "reading per-visit anomaly labels from the `anomaly` column."
        )
    args.n_categories = len(categories)
    # Noisifier is constructed below, after train_stays is loaded — needed
    # when user_host_swap / replace_agent require the full stays DataFrame.

    factor = (4 if args.no_agent_emb else 5) * args.n_heads
    if args.model_dim % factor != 0:
        adjusted = ((args.model_dim + factor - 1) // factor) * factor
        print(
            f"Adjusting model_dim from {args.model_dim} to {adjusted} "
            "to satisfy feature encoder shape constraints and multi-head attention requirements"
        )
        args.model_dim = adjusted

    min_time = metadata["min_time"]
    train_stays = prepare_stays(load_stays("train", dataset_cfg), min_time=min_time)
    train_stays = ensure_category_columns(train_stays, categories)
    train_stays, val_stays = split_train_to_train_val(train_stays)

    train_stays, val_stays = normalize_and_validate_agents(train_stays, val_stays)

    print(
        f"Train stays: {len(train_stays):,}, Val stays: {len(val_stays):,}, Categories: {categories}"
    )

    overlap_min_frac = getattr(args, "overlap_min_frac", 0.0)
    anomaly_col = "anomaly" if use_real_labels else None
    if use_real_labels:
        for split_name, df in (("train", train_stays), ("val", val_stays)):
            if "anomaly" not in df.columns:
                raise KeyError(
                    f"--use_real_anomaly_labels set but the {split_name} stays "
                    "dataframe has no `anomaly` column. Make sure the dataset "
                    "preprocessor emits it."
                )

    # Build Noisifier here (after train_stays exists) so user_host_swap /
    # replace_agent can sample from the full stays DataFrame.
    if use_real_labels:
        noisifier = None
    else:
        _need_all_stays = (
            getattr(args, "user_host_swap", False)
            or getattr(args, "replace_agent", False)
        )
        _all_agent_stays = train_stays if _need_all_stays else None
        noisifier = Noisifier(
            pois,
            categories,
            args.p_normal,
            dataset_cfg=dataset_cfg,
            timestamp_step_size=float(metadata["timestamp_step_size"]),
            insert_visits=args.insert_visits,
            perturb_locations=args.perturb_locations,
            perturb_timestamps=args.perturb_timestamps,
            behavioral_anomaly=args.behavioral_anomaly,
            legacy_perturb_timestamps=getattr(args, "legacy_perturb_timestamps", False),
            insert_covisits=getattr(args, "insert_covisits", False),
            insert_recurring_visits=getattr(args, "insert_recurring_visits", False),
            insert_chained_visits=getattr(args, "insert_chained_visits", False),
            user_host_swap=getattr(args, "user_host_swap", False),
            user_host_swap_frac=getattr(args, "user_host_swap_frac", 0.3),
            all_agent_stays=_all_agent_stays,
        )

    train_data = CliqueDataset(train_stays, categories, args.clique_size,
                               noisifier=noisifier, anomaly_col=anomaly_col,
                               max_seq_len=args.max_seq_len,
                               overlap_min_frac=overlap_min_frac)
    val_data = CliqueDataset(val_stays, categories, args.clique_size,
                             noisifier=noisifier, anomaly_col=anomaly_col,
                             max_seq_len=args.max_seq_len,
                             overlap_min_frac=overlap_min_frac)

    model = Model(args)
    model.to(args.device)
    print_parameter_distribution(model, max_depth=1)

    if args.train_batch_size is None or args.val_batch_size is None:
        cache = _load_batch_size_cache(args.training_log_dir)
        cache_key = _batch_size_cache_key(args)
        cached = cache.get(cache_key, {})

        if args.train_batch_size is None:
            if "train" in cached:
                args.train_batch_size = cached["train"]
                print(f"Loaded cached train batch size: {args.train_batch_size}")
            else:
                args.train_batch_size = safe_determine_batch_size(
                    train_data, model, args.device, no_grad=False
                )
                print(f"Determined train batch size: {args.train_batch_size}")
                cached["train"] = args.train_batch_size

        if args.val_batch_size is None:
            if "val" in cached:
                args.val_batch_size = cached["val"]
                print(f"Loaded cached val batch size: {args.val_batch_size}")
            else:
                args.val_batch_size = safe_determine_batch_size(
                    val_data, model, args.device, no_grad=True
                )
                print(f"Determined val batch size: {args.val_batch_size}")
                cached["val"] = args.val_batch_size

        cache[cache_key] = cached
        _save_batch_size_cache(cache, args.training_log_dir)

    perturb_seed_cfg = int(getattr(args, "perturb_seed", 0) or 0)
    def _train_worker_init(worker_id):
        import numpy as _np, random as _rand, os as _os
        if perturb_seed_cfg > 0:
            s = perturb_seed_cfg + worker_id
        else:
            s = int.from_bytes(_os.urandom(4), "little")
        _np.random.seed(s)
        _rand.seed(s + 1)

    _dl_extra = {}
    if args.pin_memory:
        _dl_extra["pin_memory"] = True
    if args.persistent_workers and args.num_workers > 0:
        _dl_extra["persistent_workers"] = True
    if args.prefetch_factor is not None and args.num_workers > 0:
        _dl_extra["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_data,
        args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        worker_init_fn=_train_worker_init,
        **_dl_extra,
    )
    val_loader = DataLoader(
        val_data,
        args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        **_dl_extra,
    )

    holdout_size = min(len(val_data), max(1, args.val_batch_size)) if len(val_data) > 0 else 0
    holdout_loader = None
    if holdout_size > 0 and args.holdout_eval_every > 0:
        generator = torch.Generator().manual_seed(42)
        holdout_indices = torch.randperm(len(val_data), generator=generator)[:holdout_size].tolist()
        holdout_data = Subset(val_data, holdout_indices)
        holdout_loader = DataLoader(
            holdout_data,
            args.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            **_dl_extra,
        )
        print(
            f"Using holdout validation every {args.holdout_eval_every} iterations "
            f"on {holdout_size:,} / {len(val_data):,} val samples"
        )

    print("Starting training...")
    # If --bce_pos_weight=auto, compute n_neg/n_pos from the train labels.
    # Only relevant when --use_real_anomaly_labels is on; with synthetic
    # perturbations (default) the rate is already balanced so we skip auto.
    bce_pos_weight = getattr(args, "bce_pos_weight", None)
    if bce_pos_weight == "auto":
        if not getattr(args, "use_real_anomaly_labels", False):
            print("[train] --bce_pos_weight=auto ignored: only auto-computes when --use_real_anomaly_labels is set")
            bce_pos_weight = None
        else:
            n_pos = int((train_stays["anomaly"] == 1).sum())
            n_neg = int((train_stays["anomaly"] == 0).sum())
            bce_pos_weight = max(1.0, n_neg / max(1, n_pos))
            print(f"[train] auto-computed bce_pos_weight={bce_pos_weight:.2f} (n_pos={n_pos}, n_neg={n_neg})")
    elif bce_pos_weight is not None:
        bce_pos_weight = float(bce_pos_weight)
        print(f"[train] using bce_pos_weight={bce_pos_weight:.2f}")

    criterion = CustomCriterion(
        contrastive_weight=0.0 if getattr(args, "no_contrastive", False) else 0.5,
        agent_level_bce=dataset_cfg.name.startswith("urban-"),
        no_contrastive=getattr(args, "no_contrastive", False),
        no_bce=getattr(args, "no_bce", False),
        bce_pos_weight=bce_pos_weight,
    )
    # Move pos_weight tensor (if any) onto the model device.
    if hasattr(criterion.bce_loss, "pos_weight") and criterion.bce_loss.pos_weight is not None:
        criterion.bce_loss.pos_weight = criterion.bce_loss.pos_weight.to(args.device)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_steps = args.num_epochs * len(train_loader)
    scheduler = None
    if total_steps > 0:
        cosine_eta_min = max(0.0, float(args.cosine_eta_min))
        if cosine_eta_min > args.lr:
            print(
                f"cosine_eta_min ({cosine_eta_min}) is larger than lr ({args.lr}); "
                f"clamping to {args.lr}"
            )
            cosine_eta_min = float(args.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=cosine_eta_min,
        )
        print(
            "Using cosine annealing LR decay "
            f"(steps={total_steps}, eta_min={cosine_eta_min:g})"
        )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = max(0, int(args.early_stopping_patience))
    early_stopping_min_delta = max(0.0, float(args.early_stopping_min_delta))
    ema_loss = None  # initialized on first epoch

    if early_stopping_patience > 0:
        ema_tag = (
            f", EMA alpha={args.ema_alpha:g}" if getattr(args, "ema_early_stopping", False) else ""
        )
        print(
            "Early stopping enabled "
            f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta:g}{ema_tag})"
        )

    pbar = tqdm(
        total=args.num_epochs * len(train_loader),
        desc="Training",
        dynamic_ncols=True,
        bar_format="{desc}: {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    wandb_logger = WandbLogger("train", args)

    # Load checkpoint if provided
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint file not found: {args.resume_from_checkpoint}"
            )
        print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = load_checkpoint(
            args.resume_from_checkpoint, model, optimizer, scheduler, args.device,
            weights_only_model=args.finetune,
        )

        if args.finetune:
            # Fine-tune: start fresh training state, only model weights are carried over
            print("Fine-tune mode: optimizer/scheduler/epoch state reset.")
            # On the first finetune run (args.wandb_run_id is None → brand-new W&B run),
            # record which pre-training run this checkpoint came from.  On subsequent
            # resumes of the fine-tune run, args.wandb_run_id is already set so we skip
            # this — the key is already in the run's config and must not be overwritten.
            if wandb_logger.use_wandb and not args.wandb_run_id:
                pretrain_run_id = checkpoint.get("wandb_run_id") or checkpoint.get("args", {}).get("wandb_run_id")
                if pretrain_run_id:
                    wandb_logger.run.config.update({"pretrained_from": pretrain_run_id})
                    print(f"Logged pretrained_from={pretrain_run_id} to W&B config")
        else:
            # Restore training state
            best_val_loss = checkpoint["best_val_loss"]
            epochs_without_improvement = checkpoint["epochs_without_improvement"]
            wandb_logger.epoch = checkpoint["epoch"]
            wandb_logger.iteration = checkpoint["iteration"]

            # Restore RNG states for reproducibility
            restore_rng_states(checkpoint["rng_states"])

        # Validate configuration matches
        checkpoint_args = checkpoint["args"]
        critical_keys = [
            "model_dim",
            "dim_feedforward",
            "n_scales",
            "n_heads",
            "num_layers",
            "n_agent_attributes",
            "hidden_dim",
            "n_categories",
            "clique_size",
            "dataset",
            "standard_agent_emb",
        ]
        mismatches = []
        for key in critical_keys:
            if key in checkpoint_args and checkpoint_args[key] != getattr(args, key):
                mismatches.append(
                    f"{key}: checkpoint={checkpoint_args[key]} vs current={getattr(args, key)}"
                )

        if mismatches:
            raise ValueError(
                f"Checkpoint configuration mismatch:\n  " + "\n  ".join(mismatches)
            )

        if args.finetune or args.reset_best_val_loss:
            best_val_loss = float("inf")
            epochs_without_improvement = 0
            print("Reset best_val_loss to inf (--reset_best_val_loss)")

        print(
            f"Resumed from checkpoint: epoch={checkpoint['epoch']:.2f}, "
            f"iteration={checkpoint['iteration']}, best_val_loss={best_val_loss:.6f}"
        )

        # Adjust progress bar
        pbar.total = args.num_epochs * len(train_loader)
        pbar.n = int(checkpoint["iteration"])

    for _ in range(args.num_epochs):
        train_epoch(
            train_loader,
            model,
            optimizer,
            criterion,
            wandb_logger,
            pbar,
            args.device,
            scheduler=scheduler,
            holdout_loader=holdout_loader,
            holdout_eval_every=max(0, int(args.holdout_eval_every)),
        )

        val_result = validate_epoch(
            model, val_loader, criterion, wandb_logger, args.device
        )
        if isinstance(val_result, tuple):
            val_loss = float(val_result[0])
            if len(val_result) > 1:
                val_bce_loss = float(val_result[1].get("bce_loss", val_loss))
                print(f"Val BCE loss: {val_bce_loss:.6f}")
            else:
                val_bce_loss = val_loss
        else:
            val_loss = float(val_result)
            val_bce_loss = val_loss

        early_stopping_criterion = getattr(args, "early_stopping_criterion", "total_loss")
        monitored_loss = val_bce_loss if early_stopping_criterion == "bce_loss" else val_loss

        if getattr(args, "ema_early_stopping", False):
            alpha = float(args.ema_alpha)
            ema_loss = monitored_loss if ema_loss is None else alpha * monitored_loss + (1 - alpha) * ema_loss
            effective_loss = ema_loss
            print(f"EMA loss: {ema_loss:.6f}")
        else:
            effective_loss = monitored_loss

        # Save checkpoint only if monitored validation loss improved.
        improved = effective_loss < (best_val_loss - early_stopping_min_delta)
        if improved:
            best_val_loss = effective_loss
            epochs_without_improvement = 0

            if not os.path.exists(args.training_log_dir):
                os.makedirs(args.training_log_dir)

            if args.use_wandb:
                model_name = f"{wandb_logger.run.id}"
            else:
                model_name = wandb_logger.run_name
            checkpoint_path = f"{args.training_log_dir}/{model_name}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                wandb_logger,
                best_val_loss,
                epochs_without_improvement,
                args,
            )
            wandb_logger.log_artifact(wandb_logger.epoch, checkpoint_path)

            print(f"New best validation loss {best_val_loss:.6f} - checkpoint saved")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0:
                print(
                    "No validation improvement for "
                    f"{epochs_without_improvement} epoch(s)"
                )
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        "Early stopping triggered: "
                        f"no val loss improvement for {early_stopping_patience} epoch(s)"
                    )
                    break

    pbar.close()
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    wandb_logger.finish()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    main(args)
