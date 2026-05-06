"""Fine-tune the TraXion backbone on per-stay ICU mortality (eICU-CRD demo).

Loads a pretrained checkpoint via ``--resume_from_checkpoint --finetune``,
swaps in a binary classification head pooled over each stay's event sequence,
and trains it on the train + val pid splits prepared by ``prepare_eicu.py``.
Final test-set evaluation runs on the held-out test pid split with the best
checkpoint by validation AUROC.

Example::

    python -m cli.train_eicu --dataset eicu-demo --device cuda:0 \\
        --finetune --resume_from_checkpoint runs/<pretrain_id>.pt \\
        --lr 1e-4 --num_epochs 25 --early_stopping_patience 8 \\
        --max_seq_len 128 --pos_weight 5.0
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import get_dataset_config
from tasks.eicu_mortality.criterion import MortalityCriterion
from tasks.eicu_mortality.dataset import (
    StayMortalityDataset,
    stay_mortality_collate_fn,
)
from tasks.eicu_mortality.metrics import mortality_scores
from tasks.eicu_mortality.model import MortalityModel
from tasks.eicu_mortality.tabular_features import build_tabular_features
from utils.argparse import parse_eicu_args
from utils.dataset import POIIndex
from utils.file_io import load_dataset_metadata, load_pois, load_stays
from utils.info import print_parameter_distribution
from utils.training_data import (
    ensure_category_columns,
    normalize_and_validate_agents,
    prepare_stays,
)
from utils.training_runtime import restore_rng_states, save_checkpoint
from utils.wandb_logger import WandbLogger


# ---------------------------------------------------------------------------
# Arch sync from pretrained checkpoint
# ---------------------------------------------------------------------------

_ARCH_KEYS_FROM_CKPT = (
    "model_dim", "dim_feedforward", "n_scales", "n_heads", "num_layers",
    "n_agent_attributes", "hidden_dim", "clique_size", "lambda_min", "lambda_max",
    "time_modulo", "time_scale", "standard_agent_emb", "no_agent_emb",
    "use_transformer_encoder", "no_neighbor_attn", "agent_only_concat",
    "overlap_min_frac", "temporal_embedding",
)


def _sync_arch_from_checkpoint(args) -> None:
    if not (args.finetune and args.resume_from_checkpoint
            and os.path.exists(args.resume_from_checkpoint)):
        return
    ck = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    for k in _ARCH_KEYS_FROM_CKPT:
        if k not in ck_args:
            continue
        new = ck_args[k]
        old = getattr(args, k, None)
        if new != old:
            print(f"  [ckpt-sync] {k}: {old!r} -> {new!r}")
            setattr(args, k, new)


def _load_backbone_from_anomaly_checkpoint(path: str, model: MortalityModel, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    own = model.state_dict()
    filtered = {}
    shape_mismatch = []
    for k, v in state.items():
        if k in own and tuple(own[k].shape) != tuple(v.shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(own[k].shape)))
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Backbone loaded from anomaly checkpoint: {path}")
    if shape_mismatch:
        print(f"  Skipped (shape mismatch): {shape_mismatch[:5]}")
    if missing:
        print(f"  Newly initialized (missing in ckpt): "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"  Skipped (anomaly-specific): "
              f"{unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    return ckpt


def _load_eicu_checkpoint(path, model, optimizer, scheduler, device, weights_only_model=False):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if not weights_only_model:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(model, loader: DataLoader, device: str) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Forward every batch, then average per-stay logits across multi-window items.

    The dataset enumerates `n_eval_windows` items per stay (sharing `agent_ids`),
    so we group probabilities by `agent_ids` and take the mean before scoring.
    """
    model.eval()
    logits_all, labels_all, agent_ids_all = [], [], []
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        logits = model(
            batch["location"], batch["start"], batch["stop"],
            batch["category"], batch["agent"],
            batch["padding_mask"], batch["neighbor_padding_mask"],
            tab_feat=batch.get("tab_feat", None),
        )
        logits_all.append(logits.detach().cpu())
        labels_all.append(batch["labels"].detach().cpu())
        agent_ids_all.append(batch["agent_ids"].detach().cpu())
    if not logits_all:
        return {}, np.empty(0), np.empty(0)
    logits_arr = torch.cat(logits_all, dim=0).numpy()
    labels_arr = torch.cat(labels_all, dim=0).numpy()
    agents_arr = torch.cat(agent_ids_all, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_arr))

    # Average the probabilities across windows of the same stay.
    df = pd.DataFrame({"agent": agents_arr, "prob": probs, "label": labels_arr})
    g = df.groupby("agent", as_index=False).agg({"prob": "mean", "label": "first"})
    return (
        mortality_scores(g["prob"].values, g["label"].values),
        g["prob"].values, g["label"].values,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_label_map(stay_labels: pd.DataFrame, stays: pd.DataFrame) -> dict[int, int]:
    """Map agent_id -> mortality label, restricted to agents present in `stays`."""
    label_by_stay = dict(zip(
        stay_labels["patientunitstayid"].astype(np.int64),
        stay_labels["mortality"].astype(np.int64),
    ))
    out: dict[int, int] = {}
    by_agent_stay = (
        stays.groupby("agent")["patientunitstayid"]
        .agg(lambda s: int(s.iloc[0])).to_dict()
    )
    for agent, stay_id in by_agent_stay.items():
        if int(stay_id) in label_by_stay:
            out[int(agent)] = int(label_by_stay[int(stay_id)])
    return out


def main(args: Any) -> None:
    _sync_arch_from_checkpoint(args)

    seed = int(getattr(args, "seed", 0) or 0)
    if seed > 0:
        import random
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        print(f"[train_eicu] seeded RNGs with seed={seed}")

    cfg = get_dataset_config(args.dataset)
    args.training_log_dir = cfg.training_log_dir

    metadata = load_dataset_metadata(cfg)
    args.n_agents = int(metadata["n_agents"])
    _, categories = load_pois(cfg)
    args.n_categories = len(categories)

    # Load both train and test parquets up front so the POIIndex sees the full
    # event universe (so "neighbor" co-visit lookups don't drop out at eval).
    raw_train = load_stays("train", cfg)
    raw_test = load_stays("test", cfg)
    train_stays = prepare_stays(raw_train, min_time=metadata["min_time"])
    test_stays = prepare_stays(raw_test, min_time=metadata["min_time"])
    train_stays = ensure_category_columns(train_stays, categories)
    test_stays = ensure_category_columns(test_stays, categories)
    # IMPORTANT: keep agent ids stable across splits (DON'T renumber). The
    # preprocessor already assigned contiguous global agent ids; renumbering
    # would break the parity with the pretrained agent embedding table.
    print(
        f"Loaded {len(train_stays):,} train events ({train_stays['agent'].nunique():,} stays), "
        f"{len(test_stays):,} test events ({test_stays['agent'].nunique():,} stays)"
    )

    # Read per-stay labels. The preprocessor saved a sidecar parquet with one
    # row per stay tagged by split (train/val/test) — use it to derive the
    # held-out validation slice from the pretrain bundle (= train + val pids).
    labels_path = os.path.join(
        os.path.dirname(cfg.train_data_path), "eicu_stay_labels.parquet"
    )
    stay_labels = pd.read_parquet(labels_path)
    stay_labels["mortality"] = stay_labels["mortality"].astype(np.int64)

    # Train / val split inside the pretrain bundle (these are the agents in
    # train_stays). Test uses test_stays.
    pretrain_stay_ids = set(train_stays["patientunitstayid"].astype(np.int64).unique())
    test_stay_ids = set(test_stays["patientunitstayid"].astype(np.int64).unique())

    sl = stay_labels.copy()
    sl["patientunitstayid"] = sl["patientunitstayid"].astype(np.int64)
    sl_train = sl[(sl["split"] == "train") & sl["patientunitstayid"].isin(pretrain_stay_ids)]
    sl_val = sl[(sl["split"] == "val") & sl["patientunitstayid"].isin(pretrain_stay_ids)]
    sl_test = sl[(sl["split"] == "test") & sl["patientunitstayid"].isin(test_stay_ids)]

    print(f"Stay labels: train={len(sl_train):,} (mortality={sl_train['mortality'].mean():.4f}), "
          f"val={len(sl_val):,} (mortality={sl_val['mortality'].mean():.4f}), "
          f"test={len(sl_test):,} (mortality={sl_test['mortality'].mean():.4f})")

    # Map per-stay labels to agent ids using the train/test event tables.
    train_agent_label = _build_label_map(sl_train, train_stays)
    val_agent_label = _build_label_map(sl_val, train_stays)
    test_agent_label = _build_label_map(sl_test, test_stays)

    # Build per-stay tabular features (z-scored on train+val) and an
    # agent_id -> stay_id lookup so the dataset can fetch them.
    tab_features = None
    tab_feat_dim = 0
    if getattr(args, "use_tabular_features", True):
        bundle = build_tabular_features(stay_labels, train_stays, test_stays)
        tab_features = bundle.by_stay
        tab_feat_dim = bundle.feature_dim
        print(f"[tabular] feature_dim={tab_feat_dim}  features={bundle.feature_names[:6]}... "
              f"(+{max(0, len(bundle.feature_names)-6)} more)")
    agent_to_stay_id_train = (
        train_stays.groupby("agent")["patientunitstayid"]
        .agg(lambda s: int(s.iloc[0])).to_dict()
    )
    agent_to_stay_id_test = (
        test_stays.groupby("agent")["patientunitstayid"]
        .agg(lambda s: int(s.iloc[0])).to_dict()
    )

    # model_dim divisibility check.
    factor = (4 if getattr(args, "no_agent_emb", False) else 5) * args.n_heads
    if args.model_dim % factor != 0:
        adjusted = ((args.model_dim + factor - 1) // factor) * factor
        print(f"Adjusting model_dim from {args.model_dim} to {adjusted}")
        args.model_dim = adjusted

    # Shared POIIndex over the union of train + test events (so eval-time
    # co-visit neighbor lookup matches what the model trains with).
    all_stays = pd.concat([train_stays, test_stays], ignore_index=True)
    poi_index = POIIndex(
        all_stays, args.clique_size,
        overlap_min_frac=getattr(args, "overlap_min_frac", 0.0),
    )

    n_eval_windows = max(1, int(getattr(args, "n_eval_windows", 1)))
    train_data = StayMortalityDataset(
        stays=train_stays, agent_to_label=train_agent_label, categories=categories,
        clique_size=args.clique_size, poi_index=poi_index,
        max_seq_len=args.max_seq_len, zero_agent_input=args.zero_agent_input,
        train=True,
        tab_features=tab_features, agent_to_stay_id=agent_to_stay_id_train,
    )
    val_data = StayMortalityDataset(
        stays=train_stays, agent_to_label=val_agent_label, categories=categories,
        clique_size=args.clique_size, poi_index=poi_index,
        max_seq_len=args.max_seq_len, zero_agent_input=args.zero_agent_input,
        train=False, n_eval_windows=n_eval_windows,
        tab_features=tab_features, agent_to_stay_id=agent_to_stay_id_train,
    )
    test_data = StayMortalityDataset(
        stays=test_stays, agent_to_label=test_agent_label, categories=categories,
        clique_size=args.clique_size, poi_index=poi_index,
        max_seq_len=args.max_seq_len, zero_agent_input=args.zero_agent_input,
        train=False, n_eval_windows=n_eval_windows,
        tab_features=tab_features, agent_to_stay_id=agent_to_stay_id_test,
    )
    print(f"Datasets: train={len(train_data):,}  val={len(val_data):,} "
          f"(unique stays={len(val_data.agents)})  "
          f"test={len(test_data):,} (unique stays={len(test_data.agents)})  "
          f"n_eval_windows={n_eval_windows}")

    if args.train_batch_size is None:
        args.train_batch_size = 32
    if args.val_batch_size is None:
        args.val_batch_size = 64

    if getattr(args, "oversample_positives", False):
        n = len(train_data)
        labels = train_data.labels.astype(np.int64)
        pos_w = 0.5 / max(1, labels.sum())
        neg_w = 0.5 / max(1, (labels == 0).sum())
        weights = np.where(labels == 1, pos_w, neg_w)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=n, replacement=True,
        )
        train_loader = DataLoader(
            train_data, args.train_batch_size, sampler=sampler,
            collate_fn=stay_mortality_collate_fn, num_workers=args.num_workers,
        )
        print(f"[sampler] WeightedRandomSampler enabled "
              f"(pos={int(labels.sum())}, neg={int((labels==0).sum())}, "
              f"pos rate per batch ≈ 0.5)")
    else:
        train_loader = DataLoader(
            train_data, args.train_batch_size, shuffle=True,
            collate_fn=stay_mortality_collate_fn, num_workers=args.num_workers,
        )
    val_loader = DataLoader(
        val_data, args.val_batch_size, shuffle=False,
        collate_fn=stay_mortality_collate_fn, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_data, args.val_batch_size, shuffle=False,
        collate_fn=stay_mortality_collate_fn, num_workers=args.num_workers,
    )

    model = MortalityModel(args, tab_feat_dim=tab_feat_dim).to(args.device)
    print_parameter_distribution(model, max_depth=1)

    criterion = MortalityCriterion(pos_weight=args.pos_weight)
    criterion.to(args.device)

    head_modules = [model.pool_proj, model.classifier]
    if model.tab_proj is not None:
        head_modules.append(model.tab_proj)
    head_param_ids = {id(p) for m in head_modules for p in m.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_param_ids]
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
    head_lr_mult = float(getattr(args, "head_lr_mult", 1.0))
    if head_lr_mult != 1.0 and head_params:
        param_groups = [
            {"params": backbone_params, "lr": args.lr, "name": "backbone"},
            {"params": head_params, "lr": args.lr * head_lr_mult, "name": "head"},
        ]
        print(f"Param groups: backbone lr={args.lr:g}, head lr={args.lr * head_lr_mult:g}")
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=args.lr)

    total_steps = args.num_epochs * max(1, len(train_loader))
    warmup_steps = max(0, int(getattr(args, "warmup_steps", 0)))
    scheduler = None
    if total_steps > 0:
        cosine_eta_min = min(float(args.cosine_eta_min), args.lr)
        if warmup_steps > 0:
            decay_steps = max(1, total_steps - warmup_steps)
            base_lrs = [pg.get("lr", args.lr) if isinstance(pg, dict) else args.lr
                        for pg in optimizer.param_groups]

            def _lr_lambda(step, base_lr):
                if step < warmup_steps:
                    return max(1e-8, (step + 1) / max(1, warmup_steps))
                prog = (step - warmup_steps) / decay_steps
                cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
                floor = cosine_eta_min / max(base_lr, 1e-12)
                return floor + (1.0 - floor) * cos

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=[(lambda s, blr=blr: _lr_lambda(s, blr)) for blr in base_lrs],
            )
            print(f"Warmup ({warmup_steps}) + cosine decay (total={total_steps}, eta_min={cosine_eta_min:g})")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps), eta_min=cosine_eta_min,
            )
            print(f"Cosine annealing LR (steps={total_steps}, eta_min={cosine_eta_min:g})")

    best_val_metric = -float("inf")  # maximize AUROC
    epochs_without_improvement = 0
    early_stopping_patience = max(0, int(args.early_stopping_patience))
    early_stopping_min_delta = max(0.0, float(args.early_stopping_min_delta))
    ema_metric = None

    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from_checkpoint}")
        if args.finetune:
            _load_backbone_from_anomaly_checkpoint(args.resume_from_checkpoint, model, args.device)
            print("Fine-tune mode: optimizer/scheduler reset; mortality head randomly initialized.")
            if args.freeze_backbone:
                model.freeze_backbone()
                print("Backbone frozen — training pool_proj + classifier only.")
            if args.freeze_agent_emb:
                model.freeze_agent_emb()
                print("Input agent embedding frozen.")
        else:
            ckpt = _load_eicu_checkpoint(
                args.resume_from_checkpoint, model, optimizer, scheduler, args.device,
            )
            stored = ckpt.get("best_val_loss", float("inf"))
            if stored != float("inf"):
                best_val_metric = -stored
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            if "rng_states" in ckpt:
                restore_rng_states(ckpt["rng_states"])

    wandb_logger = WandbLogger("train", args, project=args.wandb_project)
    pbar = tqdm(total=args.num_epochs * len(train_loader),
                desc="Training (eICU)",
                dynamic_ncols=True,
                bar_format="{desc}: {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader, start=1):
            wandb_logger.set_epoch_progress(epoch, batch_idx, len(train_loader))
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            optimizer.zero_grad()
            logits = model(
                batch["location"], batch["start"], batch["stop"],
                batch["category"], batch["agent"],
                batch["padding_mask"], batch["neighbor_padding_mask"],
                tab_feat=batch.get("tab_feat", None),
            )
            loss, loss_dict = criterion(logits, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            wandb_logger.iteration += 1
            wandb_logger.log(loss_dict, split="train")
            wandb_logger.log({"lr": optimizer.param_groups[0]["lr"]}, split="train")
            pbar.update(1)
        wandb_logger.complete_epoch()

        val_metrics, _, _ = evaluate_split(model, val_loader, args.device)
        wandb_logger.log({f"val/{k}": v for k, v in val_metrics.items()}, split="val")
        print(f"[val] epoch={epoch}  "
              f"AUROC={val_metrics.get('auroc', float('nan')):.4f}  "
              f"AUPRC={val_metrics.get('auprc', float('nan')):.4f}  "
              f"MaxF1={val_metrics.get('max_f1', float('nan')):.4f}", flush=True)

        monitored = val_metrics.get("auroc", -float("inf"))
        if args.ema_early_stopping:
            alpha = float(args.ema_alpha)
            ema_metric = monitored if ema_metric is None else alpha * monitored + (1 - alpha) * ema_metric
            effective = ema_metric
        else:
            effective = monitored

        improved = effective > (best_val_metric + early_stopping_min_delta)
        if improved:
            best_val_metric = effective
            epochs_without_improvement = 0
            os.makedirs(args.training_log_dir, exist_ok=True)
            model_name = wandb_logger.run.id if args.use_wandb else wandb_logger.run_name
            ckpt_path = f"{args.training_log_dir}/eicu_{model_name}.pt"
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler,
                wandb_logger, -best_val_metric, epochs_without_improvement, args,
            )
            wandb_logger.log_artifact(wandb_logger.epoch, ckpt_path)
            print(f"New best val AUROC {best_val_metric:.4f} — checkpoint saved to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0:
                print(f"No improvement for {epochs_without_improvement} epoch(s)")
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping after {early_stopping_patience} epoch(s)")
                    break

    pbar.close()

    # Final test eval with best checkpoint.
    print("Final test-set evaluation (loading best checkpoint) ...")
    model_name = wandb_logger.run.id if args.use_wandb else wandb_logger.run_name
    ckpt_path = f"{args.training_log_dir}/eicu_{model_name}.pt"
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])

    test_metrics, _, _ = evaluate_split(model, test_loader, args.device)
    wandb_logger.log({f"test/{k}": v for k, v in test_metrics.items()}, split="test")
    print(f"[test] AUROC={test_metrics.get('auroc', float('nan')):.4f}  "
          f"AUPRC={test_metrics.get('auprc', float('nan')):.4f}  "
          f"MaxF1={test_metrics.get('max_f1', float('nan')):.4f}  "
          f"Sens@Spec0.9={test_metrics.get('sens_at_spec_90', float('nan')):.4f}")
    print(f"AUROC & AUPRC & MaxF1 & Sens@Spec0.9:  "
          f"{test_metrics.get('auroc', float('nan')):.4f} & "
          f"{test_metrics.get('auprc', float('nan')):.4f} & "
          f"{test_metrics.get('max_f1', float('nan')):.4f} & "
          f"{test_metrics.get('sens_at_spec_90', float('nan')):.4f}")
    wandb_logger.finish()


if __name__ == "__main__":
    args = parse_eicu_args()
    main(args)
