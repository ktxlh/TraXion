"""Training script for next-visit (POI + check-in time) prediction.

Mirrors ``train_poi.py`` but uses ``NextVisitModel`` and the joint
POI-CE + GMM-NLL criterion.
"""

import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import get_dataset_config
from tasks.next_visit.model import NextVisitModel
from tasks.next_visit.criterion import NextVisitCriterion
from tasks.next_visit.dataset import (
    NextVisitDataset,
    next_visit_collate_fn,
    build_poi_vocab,
)
from utils.argparse import parse_next_visit_args
from utils.file_io import load_dataset_metadata, load_pois, load_stays
from utils.info import print_parameter_distribution
from utils.wandb_logger import WandbLogger
from utils.misc import split_train_to_train_val
from utils.training_data import (
    ensure_category_columns,
    normalize_and_validate_agents,
    prepare_stays,
)
from utils.training_runtime import save_checkpoint, restore_rng_states


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _clear_cuda(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def train_epoch(train_loader, model, optimizer, criterion, logger, pbar, device,
                scheduler=None, holdout_loader=None, holdout_eval_every: int = 0):
    model.train()
    total_batches = len(train_loader)
    epoch_index = int(logger.epoch)

    def _should_step_scheduler() -> bool:
        if scheduler is None:
            return False
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            eta_min = float(getattr(scheduler, "eta_min", 0.0))
            return any(g["lr"] > eta_min + 1e-12 for g in optimizer.param_groups)
        return True

    for batch_index, batch in enumerate(train_loader, start=1):
        logger.set_epoch_progress(epoch_index, batch_index, total_batches)

        if batch["location"].numel() == 0:
            continue
        batch = {k: v.to(device) for k, v in batch.items()}

        try:
            optimizer.zero_grad()
            query, poi_weight = model(
                batch["location"], batch["start"], batch["stop"],
                batch["category"], batch["agent"],
                batch["padding_mask"], batch["neighbor_padding_mask"],
            )
            loss, loss_dict = criterion(model, query, poi_weight, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if _should_step_scheduler():
                scheduler.step()
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            _clear_cuda(device)
            optimizer.zero_grad(set_to_none=True)
            raise

        logger.iteration += 1
        logger.log(loss_dict, split="train")
        logger.log({"lr": optimizer.param_groups[0]["lr"]}, split="train")

        if (holdout_loader is not None and holdout_eval_every > 0
                and logger.iteration % holdout_eval_every == 0):
            validate_epoch(model, holdout_loader, criterion, logger, device, split="holdout")
            model.train()

        pbar.update(1)

    logger.complete_epoch()


def validate_epoch(model, val_loader, criterion, logger, device, split="val"):
    model.eval()
    val_losses, val_loss_dicts = [], []
    with torch.no_grad():
        for batch in val_loader:
            if batch["location"].numel() == 0:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                query, poi_weight = model(
                    batch["location"], batch["start"], batch["stop"],
                    batch["category"], batch["agent"],
                    batch["padding_mask"], batch["neighbor_padding_mask"],
                )
                loss, loss_dict = criterion(model, query, poi_weight, batch)
                val_losses.append(loss.item())
                val_loss_dicts.append(loss_dict)
            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise
                _clear_cuda(device)
                raise RuntimeError(
                    "Validation CUDA OOM. Please reduce --val_batch_size."
                ) from exc

    agg_loss = float(np.mean(val_losses)) if val_losses else float("inf")
    agg_loss_dict = (
        {k: float(np.mean([d[k] for d in val_loss_dicts])) for k in val_loss_dicts[0].keys()}
        if val_loss_dicts else {}
    )
    logger.log(agg_loss_dict, split=split)
    if agg_loss_dict:
        line = (
            f"[{split}] loss={agg_loss_dict.get('loss', agg_loss):.6f} "
            f"rec={agg_loss_dict.get('rec_loss', float('nan')):.6f} "
            f"time_nll={agg_loss_dict.get('time_nll', float('nan')):.6f}"
        )
        if "time_l1" in agg_loss_dict:
            line += f" time_l1={agg_loss_dict['time_l1']:.4f}"
        print(line, flush=True)
    return agg_loss, agg_loss_dict


def load_backbone_from_anomaly_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    own = model.state_dict()
    skipped_shape = []
    filtered = {}
    for k, v in state.items():
        if k in own and own[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(own[k].shape)))
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Backbone loaded from anomaly checkpoint: {path}")
    if skipped_shape:
        print(f"  Skipped (shape mismatch — re-initialized): {skipped_shape}")
    if missing:
        print(f"  Newly initialized (missing in ckpt): {missing}")
    if unexpected:
        print(f"  Skipped (anomaly-specific): {unexpected}")
    return checkpoint


def load_next_visit_checkpoint(path, model, optimizer, scheduler, device,
                               weights_only_model: bool = False):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if not weights_only_model:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def main(args: Any) -> None:
    seed = int(getattr(args, "seed", 0) or 0)
    if seed > 0:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        perturb_seed = int(getattr(args, "perturb_seed", 0) or 0)
        if perturb_seed > 0:
            np.random.seed(perturb_seed)
        else:
            np.random.seed(None)
        print(f"[train_next_visit] seeded RNGs with seed={seed}, "
              f"perturb_seed={perturb_seed or 'OS-entropy'}")

    dataset_cfg = get_dataset_config(args.dataset)
    args.training_log_dir = dataset_cfg.training_log_dir

    metadata = load_dataset_metadata(dataset_cfg)
    args.n_agents = int(metadata["n_agents"])

    pois, categories = load_pois(dataset_cfg)
    args.n_categories = len(categories)

    factor = (4 if getattr(args, "no_agent_emb", False) else 5) * args.n_heads
    if args.model_dim % factor != 0:
        adjusted = ((args.model_dim + factor - 1) // factor) * factor
        print(f"Adjusting model_dim from {args.model_dim} to {adjusted}")
        args.model_dim = adjusted

    min_time = metadata["min_time"]
    train_stays = prepare_stays(load_stays("train", dataset_cfg), min_time=min_time)
    train_stays = ensure_category_columns(train_stays, categories)
    train_stays, val_stays = split_train_to_train_val(
        train_stays, from_tail=getattr(args, "val_from_tail", False),
    )
    train_stays, val_stays = normalize_and_validate_agents(train_stays, val_stays)

    poi_to_idx = build_poi_vocab(train_stays)
    n_pois = len(poi_to_idx)
    args.n_pois = n_pois
    print(f"POI vocabulary: {n_pois:,} unique POIs")
    print(f"Train stays: {len(train_stays):,}, Val stays: {len(val_stays):,}")

    overlap_min_frac = getattr(args, "overlap_min_frac", 0.0)
    train_data = NextVisitDataset(
        train_stays, categories, args.clique_size, poi_to_idx,
        max_seq_len=args.max_seq_len, overlap_min_frac=overlap_min_frac,
    )
    val_data = NextVisitDataset(
        val_stays, categories, args.clique_size, poi_to_idx,
        max_seq_len=args.max_seq_len, overlap_min_frac=overlap_min_frac,
    )

    model = NextVisitModel(args, n_pois)
    model.to(args.device)
    print_parameter_distribution(model, max_depth=1)

    if args.train_batch_size is None:
        args.train_batch_size = 64
    if args.val_batch_size is None:
        args.val_batch_size = 256

    perturb_seed_cfg = int(getattr(args, "perturb_seed", 0) or 0)
    def _train_worker_init(worker_id):
        import numpy as _np, random as _rand, os as _os
        if perturb_seed_cfg > 0:
            s = perturb_seed_cfg + worker_id
        else:
            s = int.from_bytes(_os.urandom(4), "little")
        _np.random.seed(s)
        _rand.seed(s + 1)

    train_loader = DataLoader(
        train_data, args.train_batch_size, shuffle=True,
        collate_fn=next_visit_collate_fn, num_workers=args.num_workers,
        worker_init_fn=_train_worker_init,
    )
    val_loader = DataLoader(
        val_data, args.val_batch_size, shuffle=False,
        collate_fn=next_visit_collate_fn, num_workers=args.num_workers,
    )

    holdout_loader = None
    if args.holdout_eval_every > 0 and len(val_data) > 0:
        holdout_size = min(len(val_data), max(1, args.val_batch_size))
        generator = torch.Generator().manual_seed(42)
        holdout_indices = torch.randperm(len(val_data), generator=generator)[:holdout_size].tolist()
        holdout_data = Subset(val_data, holdout_indices)
        holdout_loader = DataLoader(
            holdout_data, args.val_batch_size, shuffle=False,
            collate_fn=next_visit_collate_fn, num_workers=args.num_workers,
        )

    criterion = NextVisitCriterion(
        n_neg_samples=args.n_neg_samples,
        temperature=args.temperature,
        label_smoothing=getattr(args, "label_smoothing", 0.0),
        use_in_batch_neg=getattr(args, "use_in_batch_neg", False),
        time_loss_weight=getattr(args, "time_loss_weight", 1.0),
        time_target=getattr(args, "time_target", "raw"),
        reg_loss_weight=getattr(args, "reg_loss_weight", 0.0),
        reg_target=getattr(args, "reg_target", "raw"),
    )

    if getattr(args, "freeze_agent_emb", False) and getattr(args, "finetune", False):
        model.freeze_agent_emb()
        print("Agent embedding frozen (excluded from optimizer).")

    head_lr_mult = float(getattr(args, "head_lr_mult", 1.0))
    reg_lr_mult = float(getattr(args, "reg_lr_mult", 1.0))
    head_modules = (model.query_proj, model.poi_emb, model.time_head)
    head_param_ids = {id(p) for m in head_modules for p in m.parameters()}
    reg_param_ids = {id(p) for p in model.time_reg.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_param_ids]
    reg_params = [p for p in model.parameters() if p.requires_grad and id(p) in reg_param_ids]
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_param_ids and id(p) not in reg_param_ids
    ]
    if (head_lr_mult != 1.0 or reg_lr_mult != 1.0) and (head_params or reg_params):
        param_groups = [
            {"params": backbone_params, "lr": args.lr, "name": "backbone"},
            {"params": head_params, "lr": args.lr * head_lr_mult, "name": "head"},
            {"params": reg_params, "lr": args.lr * reg_lr_mult, "name": "reg"},
        ]
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=args.lr)

    total_steps = args.num_epochs * len(train_loader)
    warmup_steps = max(0, int(getattr(args, "warmup_steps", 0)))
    scheduler = None
    if total_steps > 0:
        cosine_eta_min = min(float(args.cosine_eta_min), args.lr)
        if warmup_steps > 0:
            decay_steps = max(1, total_steps - warmup_steps)
            base_lrs = [pg["lr"] for pg in optimizer.param_groups]

            def _lr_lambda(step, base_lr):
                if step < warmup_steps:
                    return max(1e-8, (step + 1) / max(1, warmup_steps))
                progress = (step - warmup_steps) / decay_steps
                import math
                cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
                floor = cosine_eta_min / max(base_lr, 1e-12)
                return floor + (1.0 - floor) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=[(lambda step, blr=blr: _lr_lambda(step, blr)) for blr in base_lrs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps), eta_min=cosine_eta_min,
            )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = max(0, int(args.early_stopping_patience))
    early_stopping_min_delta = max(0.0, float(args.early_stopping_min_delta))
    ema_loss = None

    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from_checkpoint}")
        if args.finetune:
            ckpt = load_backbone_from_anomaly_checkpoint(
                args.resume_from_checkpoint, model, args.device,
            )
            if getattr(args, "freeze_backbone", False):
                model.freeze_backbone()
                print("Backbone frozen — only the head will be trained.")
            if getattr(args, "freeze_poi_head", False):
                model.freeze_poi_head()
                print("POI head (query_proj + poi_emb) frozen — locks loaded POI quality.")
        else:
            ckpt = load_next_visit_checkpoint(
                args.resume_from_checkpoint, model, optimizer, scheduler, args.device,
            )
            best_val_loss = ckpt["best_val_loss"]
            epochs_without_improvement = ckpt["epochs_without_improvement"]
            restore_rng_states(ckpt["rng_states"])
        if args.finetune or args.reset_best_val_loss:
            best_val_loss = float("inf")
            epochs_without_improvement = 0

    pbar = tqdm(
        total=args.num_epochs * len(train_loader),
        desc="Training (next-visit)",
        dynamic_ncols=True,
        bar_format="{desc}: {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    wandb_logger = WandbLogger("train", args, project=getattr(args, "wandb_project", "traxion-next-visit"))

    if args.resume_from_checkpoint and not args.finetune:
        pbar.n = int(ckpt.get("iteration", 0))
        wandb_logger.epoch = ckpt.get("epoch", 0.0)
        wandb_logger.iteration = ckpt.get("iteration", 0)

    if args.use_wandb:
        wandb_logger.run.config.update({"n_pois": n_pois}, allow_val_change=True)

    print("Starting next-visit training...")
    monitor_key = getattr(args, "early_stopping_criterion", "loss")
    for _ in range(args.num_epochs):
        train_epoch(
            train_loader, model, optimizer, criterion, wandb_logger, pbar, args.device,
            scheduler=scheduler, holdout_loader=holdout_loader,
            holdout_eval_every=max(0, int(args.holdout_eval_every)),
        )
        val_loss, val_loss_dict = validate_epoch(model, val_loader, criterion, wandb_logger, args.device)
        monitored_loss = val_loss_dict.get(monitor_key, val_loss)
        if getattr(args, "ema_early_stopping", False):
            alpha = float(args.ema_alpha)
            ema_loss = monitored_loss if ema_loss is None else alpha * monitored_loss + (1 - alpha) * ema_loss
            effective_loss = ema_loss
        else:
            effective_loss = monitored_loss

        improved = effective_loss < (best_val_loss - early_stopping_min_delta)
        if improved:
            best_val_loss = effective_loss
            epochs_without_improvement = 0
            os.makedirs(args.training_log_dir, exist_ok=True)
            model_name = wandb_logger.run.id if args.use_wandb else wandb_logger.run_name
            checkpoint_path = f"{args.training_log_dir}/next_visit_{model_name}.pt"
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler,
                wandb_logger, best_val_loss, epochs_without_improvement, args,
            )
            wandb_logger.log_artifact(wandb_logger.epoch, checkpoint_path)
            print(f"New best val {monitor_key}={best_val_loss:.6f} — checkpoint saved")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0:
                print(f"No improvement for {epochs_without_improvement} epoch(s)")
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping after {early_stopping_patience} epoch(s) without improvement")
                    break

    pbar.close()
    print(f"Training completed. Best val {monitor_key}: {best_val_loss:.6f}")
    wandb_logger.finish()


if __name__ == "__main__":
    args = parse_next_visit_args()
    main(args)
