"""Evaluation script for next-visit (POI + check-in time) prediction.

Conditions the time prediction on the *true* next location (the
ground-truth POI index), as the paper specifies.  Reports H@k for several
k and T@t for several t (minutes) — pick a t after the fact.
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
from tasks.next_visit.metrics import (
    compute_time_scores,
    gmm_expected_value,
    gmm_mode_loc,
    gmm_mass_within,
    T_VALUES_DEFAULT_MIN,
)


def _parse_t_values(s: str | None) -> tuple[int, ...]:
    if not s:
        return T_VALUES_DEFAULT_MIN
    return tuple(int(x) for x in str(s).split(",") if x.strip())


def evaluate_next_visit(
    model,
    loader: DataLoader,
    device: str,
    n_pois: int,
    ks: tuple[int, ...] = (1, 5, 10, 20),
    t_values_min: tuple[int, ...] = T_VALUES_DEFAULT_MIN,
) -> dict[str, float]:
    model.eval()
    all_ranks = []
    pred_mean, pred_mode, pred_reg, gt_dt = [], [], [], []
    mass_per_t: dict[int, list] = {t: [] for t in t_values_min}
    # Only report time_reg-based predictions if the checkpoint was actually
    # trained with the L1 regression head (signaled by ``model._reg_trained``).
    has_reg = hasattr(model, "time_reg") and bool(getattr(model, "_reg_trained", False))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate (next-visit)"):
            batch = {k: v.to(device) for k, v in batch.items()}
            query, poi_weight = model(
                batch["location"], batch["start"], batch["stop"],
                batch["category"], batch["agent"],
                batch["padding_mask"], batch["neighbor_padding_mask"],
            )
            rec_mask = batch["rec_mask"]
            target_poi_idx = batch["target_poi_idx"]
            target_dt = batch["target_time_delta"]

            valid_queries = query[rec_mask]
            valid_targets = target_poi_idx[rec_mask]
            valid_dt = target_dt[rec_mask]
            if valid_queries.shape[0] == 0:
                continue

            agent_self_full = None
            if getattr(model, "use_agent_in_time_head", False):
                agent_self_full = batch["agent"][..., 0][rec_mask]
            anchor_start_full = None
            if getattr(model, "use_anchor_time_in_head", False):
                # batch["start"] is (B, S, K); take own-slice K=0 like agent.
                anchor_start_full = batch["start"][..., 0][rec_mask]
            chunk = 4096
            n = valid_queries.shape[0]
            for i in range(0, n, chunk):
                q = valid_queries[i:i + chunk]
                t = valid_targets[i:i + chunk]
                gt = valid_dt[i:i + chunk]
                a = None if agent_self_full is None else agent_self_full[i:i + chunk]
                an = None if anchor_start_full is None else anchor_start_full[i:i + chunk]
                scores = q @ poi_weight.T
                target_scores = scores.gather(1, t.unsqueeze(1)).squeeze(1)
                ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1)
                all_ranks.append(ranks.cpu())

                gmm_params = model.time_gmm(q, t, a, an)
                # The GMM's natural space depends on how it was trained.
                target_space = getattr(model, "_time_target", "raw")
                if target_space == "log":
                    # Mode/mean are still in the GMM's own (log) space; convert
                    # back to hours via Δ = exp(Y) - 1.
                    pred_mean_log = gmm_expected_value(gmm_params["weight"], gmm_params["loc"])
                    pred_mode_log = gmm_mode_loc(gmm_params["weight"], gmm_params["loc"])
                    pred_mean.append((pred_mean_log.exp() - 1.0).clamp_min(0.0).cpu())
                    pred_mode.append((pred_mode_log.exp() - 1.0).clamp_min(0.0).cpu())
                else:
                    pred_mean.append(gmm_expected_value(gmm_params["weight"], gmm_params["loc"]).cpu())
                    pred_mode.append(gmm_mode_loc(gmm_params["weight"], gmm_params["loc"]).cpu())
                if has_reg:
                    reg_pred = model.time_regression_minutes(q, t, a, an)
                    if getattr(model, "_reg_target", "raw") == "log":
                        reg_min = (reg_pred.exp() - 1.0).clamp_min(0.0)
                    else:
                        reg_min = reg_pred.clamp_min(0.0)
                    pred_reg.append((reg_min / 60.0).cpu())  # → hours
                for tm in t_values_min:
                    half = torch.full_like(gt, tm / 60.0)
                    mass = gmm_mass_within(
                        gmm_params["weight"], gmm_params["loc"], gmm_params["scale"],
                        gt, half, target_space=target_space,
                    )
                    mass_per_t[tm].append(mass.cpu())

            gt_dt.append(valid_dt.cpu())

    if not all_ranks:
        print("No valid predictions found in test set.")
        return {}

    all_ranks = torch.cat(all_ranks).numpy()
    pred_mean = torch.cat(pred_mean).numpy()
    pred_mode = torch.cat(pred_mode).numpy()
    gt_dt = torch.cat(gt_dt).numpy()
    mass_arrays = {tm: torch.cat(v).numpy() for tm, v in mass_per_t.items()}
    print(f"Total predictions: {len(all_ranks):,}")

    rec_scores = compute_rec_scores(all_ranks, ks=list(ks), print_result=False)
    point_estimators = {"mean": pred_mean, "mode": pred_mode}
    if pred_reg:
        point_estimators["reg"] = torch.cat(pred_reg).numpy()
    time_scores = compute_time_scores(
        point_estimators,
        gt_dt,
        t_values_min=t_values_min,
        mass_within=mass_arrays,
        print_result=False,
    )

    out: dict[str, float] = {}
    for k, v in rec_scores.items():
        out[f"loc/{k}"] = v
    for k, v in time_scores.items():
        out[f"time/{k}"] = v
    return out


def _parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained next-visit checkpoint")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (default: $WANDB_ENTITY)")
    parser.add_argument("--project", type=str, default="traxion-next-visit")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--dataset_override", type=str, default=None)
    parser.add_argument("--t_values_min", type=str, default=None,
                        help="Comma-separated list of t (minutes) for T@t.")
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
    t_values = _parse_t_values(eval_args.t_values_min)

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
        f"{raw_ckpt['iteration']} iters, best val {raw_ckpt['best_val_loss']:.6f}"
    )

    args = types.SimpleNamespace(**raw_ckpt["args"])
    args.device = eval_args.device
    args.num_workers = eval_args.num_workers
    args.max_seq_len = getattr(eval_args, "max_seq_len", None)

    from tasks.next_visit.model import NextVisitModel
    from tasks.next_visit.dataset import (
        NextVisitDataset, next_visit_collate_fn, build_poi_vocab,
    )
    from utils.file_io import load_dataset_metadata, load_pois, load_stays
    from utils.training_data import (
        ensure_category_columns, normalize_and_validate_agents, prepare_stays,
    )
    from utils.misc import split_train_to_train_val

    dataset_name = (
        eval_args.dataset_override or getattr(args, "dataset", None)
        or (api_run.config.get("dataset") if api_run else None)
    )
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

    print("Loading training data for vocab + agent mapping ...")
    train_stays = prepare_stays(load_stays("train", dataset_cfg), min_time=min_time)
    train_stays = ensure_category_columns(train_stays, categories)
    train_stays, _val_stays = split_train_to_train_val(
        train_stays, from_tail=getattr(args, "val_from_tail", False),
    )
    poi_to_idx = build_poi_vocab(train_stays)
    n_pois = len(poi_to_idx)
    print(f"POI vocab: {n_pois:,} POIs")

    all_agents = pd.concat([train_stays["agent"], _val_stays["agent"]], ignore_index=True)
    unique_agents = pd.Index(all_agents.unique())
    agent_to_idx = pd.Series(np.arange(len(unique_agents), dtype=np.int64), index=unique_agents)

    print("Loading test data ...")
    test_stays = prepare_stays(load_stays("test", dataset_cfg), min_time=min_time)
    test_stays = ensure_category_columns(test_stays, categories)
    n_unknown = test_stays["agent"].isin(unique_agents).eq(False).sum()
    if n_unknown > 0:
        print(f"Warning: {n_unknown} test stays with unseen agents — mapped to index 0")
    test_stays["agent"] = test_stays["agent"].map(agent_to_idx).fillna(0).astype(np.int64)

    test_data = NextVisitDataset(
        test_stays, categories, args.clique_size, poi_to_idx,
        max_seq_len=args.max_seq_len,
        overlap_min_frac=getattr(args, "overlap_min_frac", 0.0),
    )

    batch_size = eval_args.batch_size or getattr(args, "val_batch_size", None) or 256
    test_loader = DataLoader(
        test_data, batch_size, shuffle=False,
        collate_fn=next_visit_collate_fn, num_workers=eval_args.num_workers,
    )

    model = NextVisitModel(args, n_pois)
    # strict=False so checkpoints predating the time_reg head still load (the
    # newly-added head will be randomly initialized and just not used).
    missing, unexpected = model.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"Unexpected ckpt keys (ignored): {unexpected[:6]}")
    reg_missing = [k for k in missing if k.startswith("time_reg.")]
    if reg_missing:
        print(f"time_reg head not in ckpt — random init ({len(reg_missing)} tensors)")
    # Stash the target space so evaluate_next_visit can convert / pick the
    # right CDF integration.
    model._time_target = getattr(args, "time_target", "raw")
    # Was the time_reg head actually trained?  A non-zero reg_loss_weight in
    # the saved args is the signal.  (Even if the ckpt holds time_reg weights
    # because the architecture always has the head, we don't want to report
    # randomly-trained predictions.)
    model._reg_trained = (
        float(getattr(args, "reg_loss_weight", 0.0)) > 0.0
        and not reg_missing  # time_reg weights actually present in ckpt
    )
    model._reg_target = getattr(args, "reg_target", "raw")
    model.to(eval_args.device)

    print(f"Evaluating with t values (min) = {t_values}")
    result = evaluate_next_visit(model, test_loader, eval_args.device, n_pois,
                                 t_values_min=t_values)

    print("\n=== Test Results ===")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")

    if eval_args.no_wandb:
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
