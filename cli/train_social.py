"""Training script for social link prediction.

Finetunes the FeatureEncoder + HumorStack backbone from an existing anomaly
pre-training checkpoint (``--resume_from_checkpoint --finetune``) and trains a
pair-scoring head that consumes pooled per-agent embeddings + handcrafted
co-visitation/graph features.

Example::

    python -m cli.train_social --dataset foursquare-tokyo --device cuda:0 \\
        --finetune --resume_from_checkpoint runs/<pretrain_id>.pt \\
        --lr 1e-4 --ema_early_stopping
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_dataset_config
from tasks.social.criterion import SocialCriterion
from tasks.social.dataset import (
    EdgeCentricBatchSampler,
    SocialTrajectoryDataset,
    social_collate_fn,
)
from tasks.social.metrics import classification_metrics, ranking_metrics
from tasks.social.model import SocialModel
from tasks.social.pair_features import FEATURE_DIM as PAIR_FEATURE_DIM
from tasks.social.pair_features import PairFeatureComputer
from utils.argparse import parse_social_args
from utils.file_io import load_dataset_metadata, load_pois, load_stays
from utils.info import print_parameter_distribution
from utils.misc import split_train_to_train_val
from utils.training_data import (
    ensure_category_columns,
    normalize_and_validate_agents,
    prepare_stays,
)
from utils.training_runtime import restore_rng_states, save_checkpoint
from utils.wandb_logger import WandbLogger


SPLITS_ROOT = Path(os.environ.get("TRAXION_DATA_DIR", "./data")) / "social_splits"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_social_splits(dataset: str):
    d = SPLITS_ROOT / dataset
    if not d.exists():
        raise FileNotFoundError(
            f"Missing social splits for {dataset}. Run: "
            f"python -m preprocess.prepare_social_splits --dataset {dataset}"
        )
    return {
        "train_edges": pd.read_parquet(d / "train_edges.parquet"),
        "val_edges": pd.read_parquet(d / "val_edges.parquet"),
        "test_edges": pd.read_parquet(d / "test_edges.parquet"),
        "test_nonedges": pd.read_parquet(d / "test_nonedges.parquet"),
        "meta": pd.read_json(d / "meta.json", typ="series").to_dict(),
    }


def _sample_nonedges(n_agents: int, forbidden: set[tuple[int, int]], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(out) < n:
        batch = max(2 * (n - len(out)), 1024)
        u = rng.integers(0, n_agents, batch)
        v = rng.integers(0, n_agents, batch)
        m = u != v; u, v = u[m], v[m]
        lo = np.minimum(u, v); hi = np.maximum(u, v)
        for a, b in zip(lo.tolist(), hi.tolist()):
            key = (a, b)
            if key in forbidden or key in seen:
                continue
            seen.add(key); out.append(key)
            if len(out) >= n:
                break
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Scoring helpers (used for validation/test metrics)
# ---------------------------------------------------------------------------

def _score_pairs_with_model(
    pairs: np.ndarray,
    model: SocialModel,
    agent_embeddings: torch.Tensor,   # (n_agents, emb_dim) — RAW backbone embeddings
    pair_feature_fn,
    device: str,
    chunk: int = 16384,
) -> np.ndarray:
    """Score arbitrary (u, v) pairs using a cached per-agent embedding table.

    The table is populated by forwarding every agent once at the start of eval
    and holds RAW (un-refined) embeddings. If GNN refinement is enabled, the
    refinement is applied here on-the-fly per pair, using the same cache as
    the neighbor source.
    """
    model.eval()
    out = np.empty(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(pairs), chunk):
            p = pairs[s : s + chunk]
            i = torch.from_numpy(p[:, 0]).long().to(device)
            j = torch.from_numpy(p[:, 1]).long().to(device)
            e_i = agent_embeddings[i]
            e_j = agent_embeddings[j]
            if model.use_gnn_head:
                e_i = model.refine(e_i, i)
                e_j = model.refine(e_j, j)
            if pair_feature_fn is not None and model.pair_feature_dim > 0:
                pf_np = pair_feature_fn(p)
                pf = torch.from_numpy(pf_np).to(device=device, dtype=e_i.dtype)
            else:
                pf = None
            logits = model.score_pairs(e_i, e_j, pf)
            out[s : s + chunk] = logits.detach().cpu().numpy()
    return out


def _compute_agent_embeddings(
    dataset: SocialTrajectoryDataset,
    model: SocialModel,
    device: str,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    """Forward every agent once; return (n_agents, emb_dim) on `device`."""
    model.eval()

    class _SeqSampler(torch.utils.data.Sampler):
        def __init__(self, n): self.n = n
        def __iter__(self): return iter(range(self.n))
        def __len__(self): return self.n

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=_SeqSampler(len(dataset)),
        collate_fn=social_collate_fn,
        num_workers=num_workers,
    )
    embs = torch.zeros(len(dataset), model.emb_dim, device=device)
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            e = model.encode(
                batch["location"], batch["start"], batch["stop"],
                batch["category"], batch["agent"],
                batch["padding_mask"], batch["neighbor_padding_mask"],
            )
            ids = batch["agent_ids"].to(device)
            embs[ids] = e
    return embs


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(
    model: SocialModel,
    dataset: SocialTrajectoryDataset,
    pair_feature_fn,
    all_edges: set[tuple[int, int]],
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    n_agents: int,
    device: str,
    val_batch_size: int,
    num_workers: int,
    seed: int = 7,
) -> dict[str, float]:
    embs = _compute_agent_embeddings(dataset, model, device, val_batch_size, num_workers)
    if model.use_gnn_head:
        # Re-use the same RAW embedding table as the GraphRefiner cache; the
        # scorer applies refinement per-pair on the fly.
        model.set_emb_cache(embs)
    def score_fn(pairs: np.ndarray) -> np.ndarray:
        return _score_pairs_with_model(pairs, model, embs, pair_feature_fn, device)
    cls = classification_metrics(pos_pairs, neg_pairs, score_fn)
    rnk = ranking_metrics(pos_pairs, all_edges, n_agents, score_fn,
                          n_negatives=100, k=10, seed=seed)
    return {**cls, **rnk}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_backbone_from_anomaly_checkpoint(path: str, model: SocialModel, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    # Drop any state-dict entries whose shape does not match the current model,
    # so partial backbones saved before a category-embedding refactor still load
    # via strict=False without raising on shape mismatch.
    model_state = model.state_dict()
    dropped = []
    for k in list(state.keys()):
        if k in model_state and state[k].shape != model_state[k].shape:
            dropped.append((k, tuple(state[k].shape), tuple(model_state[k].shape)))
            del state[k]
    if dropped:
        print(f"  Dropped {len(dropped)} shape-mismatched keys:")
        for k, s_ckpt, s_model in dropped[:10]:
            print(f"    {k}: ckpt={s_ckpt} model={s_model}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Backbone loaded from anomaly checkpoint: {path}")
    if missing:
        print(f"  Newly initialized (missing in ckpt): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"  Skipped (anomaly-specific): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    return ckpt


def _load_social_checkpoint(path, model, optimizer, scheduler, device, weights_only_model=False):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if not weights_only_model:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ARCH_KEYS_FROM_CKPT = (
    "model_dim", "dim_feedforward", "n_scales", "n_heads", "num_layers",
    "n_agent_attributes", "hidden_dim", "clique_size", "lambda_min", "lambda_max",
    "time_modulo", "time_scale", "standard_agent_emb", "no_agent_emb",
    "use_transformer_encoder", "no_neighbor_attn", "agent_only_concat", "max_seq_len",
    "overlap_min_frac",
)


def _sync_arch_from_checkpoint(args) -> None:
    """In --finetune mode, copy architectural hyperparams from the pretrained ckpt
    so the user doesn't have to re-pass them on the command line."""
    if not (args.finetune and args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint)):
        return
    ck = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    for k in _ARCH_KEYS_FROM_CKPT:
        if k not in ck_args:
            continue
        new = ck_args[k]; old = getattr(args, k, None)
        if new != old:
            print(f"  [ckpt-sync] {k}: {old!r} -> {new!r}")
            setattr(args, k, new)


def main(args: Any) -> None:
    seed = int(getattr(args, "seed", 0) or 0)
    if seed > 0:
        # NOTE: do NOT alias numpy here — the module-level `np` import is already
        # in scope, and rebinding it inside this function would make `np` a local
        # name everywhere in main(), so the seed=0 branch would later
        # UnboundLocalError on references like `np.int64`.
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
        print(f"[train_social] seeded RNGs with seed={seed}, perturb_seed={perturb_seed or 'OS-entropy'}")

    _sync_arch_from_checkpoint(args)
    cfg = get_dataset_config(args.dataset)
    args.training_log_dir = cfg.training_log_dir

    metadata = load_dataset_metadata(cfg)
    args.n_agents = int(metadata["n_agents"])
    _, categories = load_pois(cfg)
    args.n_categories = len(categories)

    # Load canonical stays (matches pretraining normalization exactly)
    raw_train = load_stays("train", cfg)
    stays = prepare_stays(raw_train, min_time=metadata["min_time"])
    stays = ensure_category_columns(stays, categories)
    train_stays, val_stays = split_train_to_train_val(stays)
    train_stays, val_stays = normalize_and_validate_agents(train_stays, val_stays)
    # For social: use the UNION of train+val stays as the trajectory source, so every
    # agent with data contributes to its embedding (we don't need visit-level held-out).
    all_stays = pd.concat([train_stays, val_stays], ignore_index=True)

    splits = _load_social_splits(args.dataset)
    train_edges = splits["train_edges"]
    val_edges = splits["val_edges"]
    test_edges = splits["test_edges"]
    test_nonedges = splits["test_nonedges"]
    n_agents = int(splits["meta"]["n_agents"])

    # Observed-edge set (train+val+test) — negatives must avoid all of these.
    all_observed = set(
        map(tuple, pd.concat([train_edges, val_edges, test_edges])[["u", "v"]].values.tolist())
    )
    # Val non-edges (for AUC/AP on val)
    val_nonedges = _sample_nonedges(n_agents, all_observed, len(val_edges), seed=321)

    # Train-only friend graph (leakage-safe structural features + loss positives)
    friend_graph: dict[int, set[int]] = {}
    for u, v in train_edges[["u", "v"]].itertuples(index=False):
        friend_graph.setdefault(int(u), set()).add(int(v))
        friend_graph.setdefault(int(v), set()).add(int(u))

    # Pair feature computer (train stays + train edges only)
    print("Building PairFeatureComputer ...")
    pfc = PairFeatureComputer(train_stays=train_stays, train_edges=train_edges)
    pair_feature_dim = 0 if args.disable_pair_features else PAIR_FEATURE_DIM
    pair_feature_fn = None if args.disable_pair_features else pfc.features_batch

    # Adjust model_dim for divisibility (same rule the other trainers use)
    factor = (4 if getattr(args, "no_agent_emb", False) else 5) * args.n_heads
    if args.model_dim % factor != 0:
        adjusted = ((args.model_dim + factor - 1) // factor) * factor
        print(f"Adjusting model_dim from {args.model_dim} to {adjusted}")
        args.model_dim = adjusted

    traj_dataset = SocialTrajectoryDataset(
        stays=all_stays,
        categories=categories,
        clique_size=args.clique_size,
        n_agents=n_agents,
        max_seq_len=args.max_seq_len,
        overlap_min_frac=getattr(args, "overlap_min_frac", 0.0),
    )

    # Sensible default batch sizes
    if args.train_batch_size is None:
        args.train_batch_size = 64
    if args.val_batch_size is None:
        args.val_batch_size = 64

    train_edges_np = train_edges[["u", "v"]].values.astype(np.int64)
    batch_sampler = EdgeCentricBatchSampler(
        n_agents=n_agents, edges=train_edges_np,
        batch_size=args.train_batch_size, batches_per_epoch=args.batches_per_epoch,
        seed=int(getattr(args, "seed", 0) or 0),
    )
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
        traj_dataset,
        batch_sampler=batch_sampler,
        collate_fn=social_collate_fn,
        num_workers=args.num_workers,
        worker_init_fn=_train_worker_init,
    )

    model = SocialModel(args, pair_feature_dim=pair_feature_dim).to(args.device)

    # For --pair_feat_norm zscore: compute per-feature mean/std from train pos+neg
    # pairs and freeze as buffers. Mirrors the StandardScaler used by the baseline.
    if pair_feature_dim > 0 and getattr(args, "pair_feat_norm", "layernorm") == "zscore":
        sample_neg = _sample_nonedges(n_agents, all_observed, len(train_edges), seed=9999)
        sample_pairs = np.vstack([train_edges[["u", "v"]].values.astype(np.int64), sample_neg])
        feats = pfc.features_batch(sample_pairs)
        fmean = torch.from_numpy(feats.mean(axis=0)).to(args.device, dtype=torch.float32)
        fstd = torch.from_numpy(feats.std(axis=0)).to(args.device, dtype=torch.float32)
        model.set_pair_feat_stats(fmean, fstd)
        print(f"[pair_feat_norm=zscore] mean={fmean.detach().cpu().numpy().round(3).tolist()}")
        print(f"[pair_feat_norm=zscore] std ={fstd.detach().cpu().numpy().round(3).tolist()}")

    print_parameter_distribution(model, max_depth=1)

    criterion = SocialCriterion(
        friend_graph=friend_graph,
        observed_edges=all_observed,
        pair_feature_fn=pair_feature_fn,
        n_neg_per_pos=args.n_neg_per_pos,
        bpr_weight=args.bpr_weight,
        infonce_weight=args.infonce_weight,
        infonce_temperature=args.infonce_temperature,
        random_neg_weight=float(getattr(args, "random_neg_weight", 0.0)),
        random_neg_batch=int(getattr(args, "random_neg_batch", 256)),
        hard_random_neg_weight=float(getattr(args, "hard_random_neg_weight", 0.0)),
        hard_random_neg_batch=int(getattr(args, "hard_random_neg_batch", 256)),
        n_agents=n_agents,
        train_edges=train_edges[["u", "v"]].values.astype(np.int64),
    )

    head_mult = float(getattr(args, "head_lr_mult", 1.0))
    if head_mult != 1.0:
        backbone_params, head_params = [], []
        for n, p in model.named_parameters():
            (head_params if (n.startswith("pool_proj") or n.startswith("pair_head")
                             or n.startswith("pair_feat_norm")
                             or n.startswith("pf_head")
                             or n.startswith("graph_refiner")) else backbone_params).append(p)
        param_groups = [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params,     "lr": args.lr * head_mult},
        ]
        print(f"[head_lr_mult={head_mult}] backbone lr={args.lr:g}, head lr={args.lr*head_mult:g}")
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(param_groups)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    best_val_metric = -float("inf")  # we maximize AUC
    epochs_without_improvement = 0
    early_stopping_patience = max(0, int(args.early_stopping_patience))
    early_stopping_min_delta = max(0.0, float(args.early_stopping_min_delta))
    ema_metric = None

    # --- Load checkpoint ---
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from_checkpoint}")
        if args.finetune:
            _load_backbone_from_anomaly_checkpoint(args.resume_from_checkpoint, model, args.device)
            print("Fine-tune mode: optimizer/scheduler reset; social head randomly initialized.")
            if args.freeze_backbone:
                model.freeze_backbone()
                print("Backbone frozen — training pool_proj + pair_head only.")
        else:
            ckpt = _load_social_checkpoint(
                args.resume_from_checkpoint, model, optimizer, scheduler, args.device
            )
            best_val_metric = ckpt.get("best_val_loss", -float("inf"))
            # Convention in save_checkpoint: best_val_loss stores negative AUC for social.
            if best_val_metric != -float("inf"):
                best_val_metric = -best_val_metric
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            if "rng_states" in ckpt:
                restore_rng_states(ckpt["rng_states"])

    wandb_logger = WandbLogger("train", args, project=args.wandb_project)
    pbar = tqdm(total=args.num_epochs * len(train_loader), desc="Training (social)",
                dynamic_ncols=True, bar_format="{desc}: {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    val_pos = val_edges[["u", "v"]].values.astype(np.int64)
    val_neg = val_nonedges

    # ----------------------------------------------------------- GNN setup
    if model.use_gnn_head:
        from modules.graph_refiner import build_neighbor_index
        nbr_idx, nbr_off = build_neighbor_index(friend_graph, n_agents, device=args.device)
        model.set_neighbor_index(nbr_idx, nbr_off)
        print(f"[gnn_head] built train-graph CSR: {int(nbr_idx.numel())} edges over "
              f"{n_agents} agents (avg deg {nbr_idx.numel() / max(1, n_agents):.2f})")

    refresh_every = max(1, int(getattr(args, "gnn_refresh_every_n_epochs", 1)))
    # Cache is needed by --gnn_head OR --hard_random_neg_weight>0 (full-pair-head BCE
    # over uniform-random pairs uses cached agent embeddings as the lookup table).
    needs_cache = (
        model.use_gnn_head
        or float(getattr(args, "hard_random_neg_weight", 0.0)) > 0.0
    )

    for epoch in range(args.num_epochs):
        # Refresh per-agent embedding cache. Must run before model.train() so we
        # can use no_grad encoder forwards.
        if needs_cache and (epoch % refresh_every == 0):
            cache = _compute_agent_embeddings(
                traj_dataset, model, args.device, args.val_batch_size, args.num_workers,
            )
            model.set_emb_cache(cache)
        model.train()
        for batch_idx, batch in enumerate(train_loader, start=1):
            wandb_logger.set_epoch_progress(epoch, batch_idx, len(train_loader))
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            optimizer.zero_grad()
            e = model.encode(
                batch["location"], batch["start"], batch["stop"],
                batch["category"], batch["agent"],
                batch["padding_mask"], batch["neighbor_padding_mask"],
            )
            if model.use_gnn_head:
                e = model.refine(e, batch["agent_ids"])
            loss, loss_dict = criterion(model, e, batch["agent_ids"])
            if loss_dict["n_pos"] > 0:
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

        val_metrics = evaluate(
            model=model, dataset=traj_dataset, pair_feature_fn=pair_feature_fn,
            all_edges=all_observed, pos_pairs=val_pos, neg_pairs=val_neg,
            n_agents=n_agents, device=args.device, val_batch_size=args.val_batch_size,
            num_workers=args.num_workers, seed=7,
        )
        wandb_logger.log({f"val/{k}": v for k, v in val_metrics.items()}, split="val")
        print(f"[val] epoch={epoch}  AUC={val_metrics['auc']:.4f}  AP={val_metrics['ap']:.4f}  "
              f"Hits@10={val_metrics['hits_at_10']:.4f}  MRR={val_metrics['mrr']:.4f}", flush=True)

        monitored = val_metrics["auc"]
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
            ckpt_path = f"{args.training_log_dir}/social_{model_name}.pt"
            # save_checkpoint expects a "best_val_loss" concept; store -AUC so lower-is-better holds.
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler,
                wandb_logger, -best_val_metric, epochs_without_improvement, args,
            )
            wandb_logger.log_artifact(wandb_logger.epoch, ckpt_path)
            print(f"New best val AUC {best_val_metric:.4f} — checkpoint saved to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0:
                print(f"No improvement for {epochs_without_improvement} epoch(s)")
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping after {early_stopping_patience} epoch(s)")
                    break

    pbar.close()

    # Final test-set evaluation with the best checkpoint loaded
    print("Final test-set evaluation (loading best checkpoint) ...")
    ckpt_path = f"{args.training_log_dir}/social_{wandb_logger.run.id if args.use_wandb else wandb_logger.run_name}.pt"
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])

    test_metrics = evaluate(
        model=model, dataset=traj_dataset, pair_feature_fn=pair_feature_fn,
        all_edges=all_observed,
        pos_pairs=test_edges[["u", "v"]].values.astype(np.int64),
        neg_pairs=test_nonedges[["u", "v"]].values.astype(np.int64),
        n_agents=n_agents, device=args.device, val_batch_size=args.val_batch_size,
        num_workers=args.num_workers, seed=7,
    )
    wandb_logger.log({f"test/{k}": v for k, v in test_metrics.items()}, split="test")
    print(f"[test] AUC={test_metrics['auc']:.4f}  AP={test_metrics['ap']:.4f}  "
          f"Hits@10={test_metrics['hits_at_10']:.4f}  MRR={test_metrics['mrr']:.4f}")
    print(f"AUC & AP & Hits@10 & MRR:  "
          f"{test_metrics['auc']:.4f} & {test_metrics['ap']:.4f} & "
          f"{test_metrics['hits_at_10']:.4f} & {test_metrics['mrr']:.4f}")
    wandb_logger.finish()


if __name__ == "__main__":
    args = parse_social_args()
    main(args)
