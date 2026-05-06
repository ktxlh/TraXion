"""Ablation pretraining: causal LM (A7) or STM masked modeling (A8).

The saved checkpoint shares attribute names with `modules.model.Model`'s
backbone (`feature_encoder.*`, `attentions.*`), so finetune scripts pick up
the pretrained weights via `strict=False` and randomly initialize their own
task heads — exactly as they do today for full-recipe checkpoints.

Example (UA-Berlin, causal):
    python -m cli.train_ablation_pretrain --dataset urban-berlin --device cuda:1 \\
        --use_wandb --pretrain_objective causal \\
        --num_epochs 300 --early_stopping_patience 80 --time_modulo daily \\
        --clique_size 4 --lr 0.0002 --no-no_agent_emb --ema_early_stopping

Example (Gowalla-Austin, STM):
    python -m cli.train_ablation_pretrain --dataset gowalla-austin-core --device cuda:2 \\
        --use_wandb --pretrain_objective stm --mask_ratio 0.3 \\
        --num_epochs 500 --early_stopping_patience 80 --time_modulo daily \\
        --clique_size 4 --lr 0.0002 --lambda_min 1e-3 --lambda_max 360 \\
        --no-no_agent_emb --ema_early_stopping
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_dataset_config
from tasks.ablation.pretrain_criterion import AblationCriterion, sample_mask_positions
from tasks.ablation.pretrain_dataset import (
    AblationPretrainDataset,
    ablation_collate_fn,
    build_poi_vocab,
)
from tasks.ablation.pretrain_model import AblationPretrainModel
from utils.argparse import parse_ablation_args
from utils.file_io import load_dataset_metadata, load_pois, load_stays
from utils.info import print_parameter_distribution
from utils.misc import split_train_to_train_val
from utils.training_data import (
    ensure_category_columns,
    normalize_and_validate_agents,
    prepare_stays,
)
from utils.training_runtime import save_checkpoint
from utils.wandb_logger import WandbLogger


# Keys used by the existing critical-key check in train.py when finetuning
# anomaly detection from this checkpoint. Must align so finetune doesn't fail.
_CKPT_ARG_KEYS = (
    "model_dim", "dim_feedforward", "n_scales", "n_heads", "num_layers",
    "n_agent_attributes", "hidden_dim", "n_categories", "clique_size",
    "dataset", "standard_agent_emb",
)


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _seed_everything(args):
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
            np.random.seed(None)
        print(f"[ablation_pretrain] seeded RNGs with seed={seed}, "
              f"perturb_seed={perturb_seed or 'OS-entropy'}")


def _train_epoch(loader, model, optimizer, criterion, logger, pbar, device,
                 scheduler=None, mask_ratio: float = 0.0, objective: str = "causal"):
    model.train()
    total_batches = len(loader)
    epoch_index = int(logger.epoch)
    causal = objective == "causal"

    for batch_idx, batch in enumerate(loader, start=1):
        logger.set_epoch_progress(epoch_index, batch_idx, total_batches)
        if batch["location"].numel() == 0:
            continue
        batch = _to_device(batch, device)

        valid = ~batch["padding_mask"].bool()
        if objective == "stm":
            mp = sample_mask_positions(valid, mask_ratio)
            batch["mask_positions"] = mp
        else:
            mp = None

        try:
            optimizer.zero_grad()
            poi_logits, time_logits, cat_logits = model(
                batch["location"], batch["start"], batch["stop"],
                batch["category"], batch["agent"],
                batch["padding_mask"], batch["neighbor_padding_mask"],
                mask_positions=mp, causal=causal,
            )
            loss, loss_dict = criterion((poi_logits, time_logits, cat_logits), batch)
            if loss_dict["n_score"] > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            raise

        logger.iteration += 1
        logger.log(loss_dict, split="train")
        logger.log({"lr": optimizer.param_groups[0]["lr"]}, split="train")
        pbar.update(1)
    logger.complete_epoch()


def _validate_epoch(loader, model, criterion, logger, device,
                    mask_ratio: float = 0.0, objective: str = "causal"):
    model.eval()
    losses = []
    dicts = []
    causal = objective == "causal"
    with torch.no_grad():
        for batch in loader:
            if batch["location"].numel() == 0:
                continue
            batch = _to_device(batch, device)
            valid = ~batch["padding_mask"].bool()
            if objective == "stm":
                mp = sample_mask_positions(valid, mask_ratio)
                batch["mask_positions"] = mp
            else:
                mp = None
            poi_logits, time_logits, cat_logits = model(
                batch["location"], batch["start"], batch["stop"],
                batch["category"], batch["agent"],
                batch["padding_mask"], batch["neighbor_padding_mask"],
                mask_positions=mp, causal=causal,
            )
            loss, loss_dict = criterion((poi_logits, time_logits, cat_logits), batch)
            losses.append(float(loss.item()))
            dicts.append(loss_dict)
    agg_loss = float(np.mean(losses)) if losses else float("inf")
    agg_dict = (
        {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0].keys()}
        if dicts else {}
    )
    logger.log(agg_dict, split="val")
    if agg_dict:
        print(f"[val] total={agg_dict.get('total_loss', agg_loss):.4f} "
              f"poi={agg_dict.get('poi_loss', 0):.4f} "
              f"time={agg_dict.get('time_loss', 0):.4f} "
              f"cat={agg_dict.get('cat_loss', 0):.4f}", flush=True)
    return agg_loss, agg_dict


def main(args: Any) -> None:
    _seed_everything(args)

    cfg = get_dataset_config(args.dataset)
    args.training_log_dir = cfg.training_log_dir
    metadata = load_dataset_metadata(cfg)
    args.n_agents = int(metadata["n_agents"])
    _, categories = load_pois(cfg)
    args.n_categories = len(categories)

    factor = (4 if getattr(args, "no_agent_emb", False) else 5) * args.n_heads
    if args.model_dim % factor != 0:
        adjusted = ((args.model_dim + factor - 1) // factor) * factor
        print(f"Adjusting model_dim from {args.model_dim} to {adjusted}")
        args.model_dim = adjusted

    train_stays = prepare_stays(load_stays("train", cfg), min_time=metadata["min_time"])
    train_stays = ensure_category_columns(train_stays, categories)
    train_stays, val_stays = split_train_to_train_val(train_stays)
    train_stays, val_stays = normalize_and_validate_agents(train_stays, val_stays)

    poi_to_idx = build_poi_vocab(train_stays)
    n_pois = len(poi_to_idx)
    args.n_pois = n_pois
    print(f"POI vocabulary: {n_pois:,} unique POIs")
    print(f"Train stays: {len(train_stays):,}, Val stays: {len(val_stays):,}")

    overlap_min_frac = getattr(args, "overlap_min_frac", 0.0)
    train_data = AblationPretrainDataset(
        train_stays, categories, args.clique_size, poi_to_idx,
        max_seq_len=args.max_seq_len, overlap_min_frac=overlap_min_frac,
    )
    val_data = AblationPretrainDataset(
        val_stays, categories, args.clique_size, poi_to_idx,
        max_seq_len=args.max_seq_len, overlap_min_frac=overlap_min_frac,
    )

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
        collate_fn=ablation_collate_fn, num_workers=args.num_workers,
        worker_init_fn=_train_worker_init,
    )
    val_loader = DataLoader(
        val_data, args.val_batch_size, shuffle=False,
        collate_fn=ablation_collate_fn, num_workers=args.num_workers,
    )

    model = AblationPretrainModel(args, n_pois=n_pois,
                                  n_time_bins=int(args.n_time_bins)).to(args.device)
    print_parameter_distribution(model, max_depth=1)

    criterion = AblationCriterion(
        objective=args.pretrain_objective,
        mask_ratio=float(args.mask_ratio),
        poi_weight=float(args.poi_loss_weight),
        time_weight=float(args.time_loss_weight),
        cat_weight=float(args.cat_loss_weight),
    )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_steps = args.num_epochs * len(train_loader)
    scheduler = None
    if total_steps > 0:
        cosine_eta_min = min(float(args.cosine_eta_min), args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps), eta_min=cosine_eta_min,
        )
        print(f"Cosine annealing LR (steps={total_steps}, eta_min={cosine_eta_min:g})")

    pbar = tqdm(
        total=args.num_epochs * len(train_loader),
        desc=f"Pretrain ({args.pretrain_objective})",
        dynamic_ncols=True,
        bar_format="{desc}: {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    wandb_logger = WandbLogger("train", args,
                               project=getattr(args, "wandb_project", "traxion-ablation"))

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = max(0, int(args.early_stopping_patience))
    early_stopping_min_delta = max(0.0, float(args.early_stopping_min_delta))
    ema_loss = None

    print(f"Starting ablation pretraining (objective={args.pretrain_objective})...")
    for epoch in range(args.num_epochs):
        _train_epoch(
            train_loader, model, optimizer, criterion, wandb_logger, pbar,
            args.device, scheduler=scheduler,
            mask_ratio=float(args.mask_ratio), objective=args.pretrain_objective,
        )
        val_loss, val_dict = _validate_epoch(
            val_loader, model, criterion, wandb_logger, args.device,
            mask_ratio=float(args.mask_ratio), objective=args.pretrain_objective,
        )

        monitored = val_dict.get("total_loss", val_loss)
        if getattr(args, "ema_early_stopping", False):
            alpha = float(args.ema_alpha)
            ema_loss = monitored if ema_loss is None else alpha * monitored + (1 - alpha) * ema_loss
            effective = ema_loss
        else:
            effective = monitored

        improved = effective < (best_val_loss - early_stopping_min_delta)
        if improved:
            best_val_loss = effective
            epochs_without_improvement = 0
            os.makedirs(args.training_log_dir, exist_ok=True)
            model_name = wandb_logger.run.id if args.use_wandb else wandb_logger.run_name
            ckpt_path = f"{args.training_log_dir}/{model_name}.pt"
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler,
                wandb_logger, best_val_loss, epochs_without_improvement, args,
            )
            wandb_logger.log_artifact(wandb_logger.epoch, ckpt_path)
            print(f"New best val loss {best_val_loss:.6f} — checkpoint saved to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0:
                print(f"No improvement for {epochs_without_improvement} epoch(s)")
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping after {early_stopping_patience} epoch(s)")
                    break

    pbar.close()
    print(f"Pretraining done. Best val loss: {best_val_loss:.6f}")
    wandb_logger.finish()


if __name__ == "__main__":
    args = parse_ablation_args()
    main(args)
