"""Command-line argument parsing for Clique.

Shared arguments are defined in _add_shared_args.  Task-specific parsers
(parse_args for anomaly detection, parse_poi_args for next-POI recommendation)
add their own arguments on top.
"""

import argparse

from config import SUPPORTED_DATASETS


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across all tasks."""

    parser.add_argument(
        "--seed", type=int, default=0,
        help="Global RNG seed (0 = don't reseed; >0 seeds python/numpy/torch at entry for multi-seed runs).",
    )
    parser.add_argument(
        "--perturb_seed", type=int, default=0,
        help=(
            "Numpy RNG seed for Noisifier perturbations only. "
            "0 (default) → re-seed numpy with OS entropy after --seed is applied, so perturbations "
            "are non-deterministic across runs (avoids the all-anomalous batch collapse that nooped "
            "the contrastive loss in seed=1 reruns). >0 → deterministic perturbations."
        ),
    )

    # Model architecture parameters
    parser.add_argument("--model_dim", type=int, default=1024, help="Model dimension")
    parser.add_argument(
        "--dim_feedforward", type=int, default=1024, help="Feedforward dimension"
    )
    parser.add_argument("--n_scales", type=int, default=32, help="Number of scales")
    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_agent_attributes", type=int, default=32, help="Number of agent attributes")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for projections")

    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Batch size for validation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Stop training after this many epochs without val loss improvement (0 disables)",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum val loss decrease required to count as an improvement",
    )
    parser.add_argument(
        "--early_stopping_criterion",
        type=str,
        default="total_loss",
        help="Which validation loss to monitor for early stopping and checkpointing",
    )
    parser.add_argument(
        "--ema_early_stopping",
        action="store_true",
        help="Use EMA-smoothed monitored loss for early stopping instead of the raw value (default: False)",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.1,
        help="EMA smoothing factor (0 < alpha <= 1); higher = less smoothing (default: 0.1)",
    )
    parser.add_argument(
        "--holdout_eval_every",
        type=int,
        default=0,
        help="Run validation on a small holdout subset every N training iterations (0 disables)",
    )
    parser.add_argument(
        "--cosine_eta_min",
        type=float,
        default=1e-6,
        help="Minimum LR for cosine annealing decay",
    )

    # Data parameters
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--pin_memory", action="store_true", help="DataLoader pin_memory=True (faster H2D copies)")
    parser.add_argument("--persistent_workers", action="store_true", help="Keep DataLoader workers alive between epochs")
    parser.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch_factor (samples prefetched per worker); None = torch default")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=32,
        help="If set, sequences longer than this are randomly trimmed to this length",
    )

    # Model configuration
    parser.add_argument("--clique_size", type=int, default=4, help="Clique size")
    parser.add_argument(
        "--temporal_embedding",
        type=str,
        default="time2vec",
        choices=["time2vec", "logfreq"],
        help=(
            "Temporal embedding method. 'time2vec' (default): learnable linear + sinusoidal "
            "projection (Kazemi et al. 2019). 'logfreq': learns log-scale frequencies (exp(log_freq) "
            "as multipliers) so the model can capture human periodicities spanning many orders of "
            "magnitude (10s to weekly) in a well-conditioned way."
        ),
    )
    parser.add_argument(
        "--time_modulo",
        type=str,
        default="weekly",
        choices=["weekly", "daily", "none"],
        help=(
            "How to wrap timestamps before temporal embedding encoding. "
            "'weekly' = t %% (24*7) [default], 'daily' = t %% 24, 'none' = raw timestamps."
        ),
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        default=1.0,
        help=(
            "Divide timestamps by this factor after --time_modulo is applied. "
            "Use 24.0 to convert hours → days (reduces gradient magnitude ~7x vs. weekly modulo "
            "in raw hours). Default: 1.0 (no scaling, backward compatible)."
        ),
    )

    # Technical parameters
    parser.add_argument("--lambda_min", type=float, default=1e-6, help="Lambda min")
    parser.add_argument("--lambda_max", type=float, default=2.0, help="Lambda max")
    parser.add_argument("--device", type=str, default='cuda:3', help="Device to use")

    # Ablation flags
    parser.add_argument(
        "--standard_agent_emb",
        action="store_true",
        help=(
            "Replace the decomposed agent embedding (vocab × hidden → hidden × feat) "
            "with a standard nn.Embedding(num_agents, feat_dim) lookup table."
        ),
    )
    parser.add_argument(
        "--no_agent_emb",
        action="store_true",
        help="Zero out agent embeddings in feature encoder (removes agent identity from input features)",
    )
    parser.add_argument(
        "--use_transformer_encoder",
        action="store_true",
        help=(
            "Replace HumorStack with a flat TransformerEncoder. "
            "Uses 3 * num_layers standard TransformerEncoderLayers to match HumorStack depth "
            "(1 HumorLayer ≡ 3 TransformerEncoderLayers: seq + feat + neighbor)."
        ),
    )
    parser.add_argument(
        "--no_neighbor_attn",
        action="store_true",
        help=(
            "Remove neighbor attention sub-layer from each HumorLayer. "
            "Use 1.5 * num_layers layers to preserve total TransformerEncoder-equivalent depth "
            "(2 sub-layers per layer instead of 3)."
        ),
    )
    parser.add_argument(
        "--agent_only_concat",
        action="store_true",
        help=(
            "For TransformerEncoder (--use_transformer_encoder): instead of mean-pooling over "
            "the clique dimension, use only the agent of interest (clique index 0) and concatenate "
            "all feature embeddings into a single vector per timestep (standard practice). "
            "The transformer operates on a sequence of S tokens each of dimension model_dim."
        ),
    )
    parser.add_argument(
        "--overlap_min_frac",
        type=float,
        default=0.0,
        help=(
            "Minimum temporal overlap fraction for a co-visitor to be considered a neighbor. "
            "Overlap fraction = intersection_duration / min(query_duration, candidate_duration). "
            "0.0 (default) disables filtering; 0.5 requires at least 50%% overlap of the shorter visit."
        ),
    )

    # Paths and output
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Existing W&B run ID to resume (sets resume='must' in wandb.init)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--reset_best_val_loss",
        action="store_true",
        help="Reset best_val_loss to inf after loading checkpoint (useful when fine-tuning on a different task/distribution)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help=(
            "Fine-tune mode: load only model weights from --resume_from_checkpoint, "
            "skipping optimizer/scheduler state, epoch/iteration counters, and RNG states. "
            "Implies --reset_best_val_loss. Use with --use_wandb (without --wandb_run_id) "
            "to start a fresh W&B run."
        ),
    )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def parse_args(args_list=None):
    """Parse command line arguments for anomaly detection training.

    Args:
        args_list: Optional list of argument strings (for testing). If None, uses sys.argv.

    Returns:
        Namespace: Parsed arguments with all configuration parameters
    """
    parser = argparse.ArgumentParser(
        description="Anomer: Anomaly Detection in Mobility Data"
    )
    _add_shared_args(parser)

    # Anomaly-specific data parameters
    parser.add_argument(
        "--p_normal", type=float, default=0.7, help="Probability normal"
    )
    parser.add_argument(
        "--insert_visits",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Insert synthetic anomalous visits (add_random_stays / add_recurring_stays)",
    )
    parser.add_argument(
        "--perturb_locations",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply location perturbations to flagged stays",
    )
    parser.add_argument(
        "--perturb_timestamps",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply timestamp perturbations to flagged stays",
    )
    parser.add_argument(
        "--behavioral_anomaly",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable behavioral anomaly regimes (hunger, interest, social, work).",
    )
    parser.add_argument(
        "--user_host_swap",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable per-event user-host swap regime (LANL lateral-movement-style synthesis). "
            "For each event, with probability --user_host_swap_frac, replace the host fields "
            "(longitude, latitude, poi_id, act_types, venue_type, cat_*) with another user's "
            "event while keeping the original src_user and timestamps. Requires "
            "Noisifier.all_agent_stays."
        ),
    )
    parser.add_argument(
        "--user_host_swap_frac",
        default=0.3,
        type=float,
        help="Per-event swap probability for --user_host_swap (default 0.3).",
    )
    parser.add_argument(
        "--insert_covisits",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable the co-visit (rendezvous) anomaly regime. "
            "Inserts 1–3 short visits (0.2–2 h) to POIs NOT in the agent's habitual "
            "top-10 locations, timed during local business hours (08:00–22:00). "
            "Designed to simulate clandestine group meetings as observed in HAYSTAC. "
            "anomaly_type='covisit'. Backward compatible: default False."
        ),
    )
    parser.add_argument(
        "--insert_recurring_visits",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable the recurring-visit anomaly regime. "
            "Inserts 4–10 visits to the SAME novel POI (not in the agent's top-10 habitual "
            "locations), spaced uniformly across the sequence window with ≥12 h gaps. "
            "Designed to simulate HAYSTAC schema types 1b (recurring area surveillance), "
            "1c (recurring fixed-POI visits), and 2c (recurring two-location route). "
            "anomaly_type='recurring_visit'. Backward compatible: default False."
        ),
    )
    parser.add_argument(
        "--insert_chained_visits",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable the chained two-stop visit anomaly regime. "
            "Inserts a stay at novel POI-A followed by a stay at a different novel POI-B "
            "starting 1–4 h after POI-A ends, simulating a two-location clandestine trip. "
            "Designed to simulate HAYSTAC schema types 2a and 2b (stay→trip→stay patterns). "
            "anomaly_type='chained_visit'. Backward compatible: default False."
        ),
    )

    # Real-label fine-tuning (for datasets that ship ground-truth anomaly
    # labels per visit, e.g. LANL red-team events)
    parser.add_argument(
        "--use_real_anomaly_labels",
        action="store_true",
        help=(
            "Fine-tune on real per-visit anomaly labels from the train parquet's "
            "`anomaly` column (1 = anomalous) instead of synthetic perturbations. "
            "When set, the Noisifier is bypassed and --insert_visits / "
            "--perturb_locations / --perturb_timestamps / --behavioral_anomaly are "
            "ignored. Typical use: LANL lateral-movement fine-tuning."
        ),
    )
    parser.add_argument(
        "--bce_pos_weight",
        type=str,
        default=None,
        help=(
            "Optional BCE class-imbalance correction. Pass a float to set "
            "pos_weight directly, or 'auto' to compute n_neg/n_pos from the "
            "training labels (only valid alongside --use_real_anomaly_labels; "
            "synthetic perturbations are already balanced). Recommended for "
            "real-label datasets with <1%% positives, e.g. LANL."
        ),
    )

    # Anomaly-specific ablation
    parser.add_argument(
        "--no_contrastive",
        action="store_true",
        help="Disable contrastive loss (sets contrastive_weight=0); only BCE is used.",
    )
    parser.add_argument(
        "--no_noise_loss", "--no_bce",
        dest="no_bce",
        action="store_true",
        help=(
            "Disable the noise-detection BCE loss L_noise (paper Appendix C2 ablation). "
            "Only the prototype-contrastive loss trains the backbone. Pair with "
            "--no-perturb_locations --no-perturb_timestamps --no-insert_visits to keep "
            "compute from being wasted producing labels nothing reads."
        ),
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.5,
        help="Weight for the contrastive loss term (default: 0.5)",
    )
    parser.add_argument(
        "--legacy_perturb_timestamps",
        action="store_true",
        help=(
            "Use the pre-5549b53 timestamp perturbation: randomly resample start/stop for "
            "each anomalous stay independently, without adjusting or flagging adjacent stays. "
            "Default (False) uses the current behavior that preserves travel-gap continuity."
        ),
    )

    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args


# ---------------------------------------------------------------------------
# Next-POI recommendation
# ---------------------------------------------------------------------------

def parse_poi_args(args_list=None):
    """Parse command line arguments for next-POI recommendation training.

    Args:
        args_list: Optional list of argument strings (for testing). If None, uses sys.argv.

    Returns:
        Namespace: Parsed arguments with all configuration parameters
    """
    parser = argparse.ArgumentParser(
        description="Clique: Next POI Recommendation"
    )
    _add_shared_args(parser)

    parser.add_argument(
        "--n_neg_samples",
        type=int,
        default=256,
        help="Number of negative POI samples per position during training (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampled softmax scoring (default: 0.1)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="traxion-poi",
        help="W&B project name for POI recommendation runs (default: traxion-poi)",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the backbone (feature_encoder + attentions) and only train the POI rec head.",
    )
    parser.add_argument(
        "--freeze_agent_emb",
        action="store_true",
        help="Freeze the per-agent embedding (keeps pretrained agent emb fixed during finetune).",
    )
    parser.add_argument(
        "--val_from_tail",
        action="store_true",
        help=(
            "Use the latest val_ratio of train stays as validation (instead of the earliest). "
            "Recommended for next-POI rec so val is temporally adjacent to the held-out test window."
        ),
    )
    parser.add_argument(
        "--head_lr_mult",
        type=float,
        default=1.0,
        help=(
            "LR multiplier for the POI recommendation head (query_proj + poi_emb). "
            "Default 1.0 keeps the current behavior. Values >1 (e.g. 3.0) speed up "
            "learning of the randomly-initialized head during fine-tuning while "
            "preserving the pretrained backbone."
        ),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help=(
            "Number of linear-warmup steps before cosine decay kicks in. "
            "Default 0 keeps the current behavior (cosine from step 0). "
            "Useful when the POI head is randomly initialized and the backbone is pretrained."
        ),
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.0,
        help="Dropout probability inside the query_proj head (default: 0.0).",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for the sampled-softmax cross-entropy (default: 0.0).",
    )
    parser.add_argument(
        "--use_in_batch_neg",
        action="store_true",
        help=(
            "Also use all other positive targets in the batch as negatives "
            "(in-batch negatives). Deduplicates against the actual positive "
            "of each query. Adds ~B*S strong negatives on top of "
            "--n_neg_samples uniform negatives."
        ),
    )

    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args


def parse_next_visit_args(args_list=None):
    """Parse command line arguments for next-visit (POI + time) prediction.

    Adds GMM/time-loss arguments on top of the same args used for POI rec.
    """
    parser = argparse.ArgumentParser(
        description="Clique: Next-Visit (POI + check-in time) Prediction"
    )
    _add_shared_args(parser)

    parser.add_argument("--n_neg_samples", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--wandb_project", type=str, default="traxion-next-visit",
    )
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_agent_emb", action="store_true")
    parser.add_argument("--val_from_tail", action="store_true")
    parser.add_argument("--head_lr_mult", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_in_batch_neg", action="store_true")

    parser.add_argument(
        "--num_gaussians", type=int, default=3,
        help="Number of mixture components for the time GMM head (default: 3).",
    )
    parser.add_argument(
        "--time_loss_weight", type=float, default=1.0,
        help="Weight applied to the GMM NLL term (default: 1.0).",
    )
    parser.add_argument(
        "--time_target", type=str, default="raw", choices=["raw", "log"],
        help=(
            "Whether the GMM models the raw delta in hours ('raw', default, "
            "backward-compatible) or log(delta+1) ('log'; matches the "
            "log-normal-style heavy-tailed density used by TPP baselines like A-NHP)."
        ),
    )
    parser.add_argument(
        "--reg_loss_weight", type=float, default=0.0,
        help=(
            "Weight on a Mobility-LLM-style scalar L1 regression loss for "
            "time prediction (target = Δ in *minutes*).  Default 0.0 disables "
            "the auxiliary head; >0 trains the L1 head jointly with the GMM "
            "head and lets evaluation report a T@t variant from its scalar."
        ),
    )
    parser.add_argument(
        "--reg_lr_mult", type=float, default=1.0,
        help=(
            "Per-parameter-group LR multiplier for the time_reg L1 head.  L1 "
            "loss has small gradients in raw-minute scale; raise this (e.g. 50) "
            "so the regression bias converges in a feasible number of steps "
            "without amplifying gradients elsewhere."
        ),
    )
    parser.add_argument(
        "--reg_bias_init", type=float, default=0.0,
        help=(
            "Initial bias for the final Linear of the L1 regression head.  "
            "Units depend on --reg_target: 'raw' -> minutes, 'log' -> log(min+1)."
        ),
    )
    parser.add_argument(
        "--reg_target", type=str, default="raw", choices=["raw", "log"],
        help=(
            "Target space for the L1 regression head.  'raw' (default) uses Δ "
            "in minutes (matches Mobility-LLM).  'log' uses log(Δ_min + 1), so "
            "L1 is bounded and the median of |log Δ| is learnable in O(epochs); "
            "at eval the prediction is exp(yhat) - 1 minutes."
        ),
    )
    parser.add_argument(
        "--use_agent_in_time_head", action="store_true",
        help=(
            "Concat the user's explicit agent_emb to the time head input "
            "(mirrors Mobility-LLM's user_emb routing to the tpp head).  "
            "When set, both the GMM head and the L1 reg head receive "
            "[query, poi_emb[next_poi], agent_emb_self]."
        ),
    )
    parser.add_argument(
        "--t_values_min", type=str, default="5,10,30,60,120,240",
        help=(
            "Comma-separated list of t (minutes) for the T@t metric. "
            "Defaults span small-to-large windows so a single t can be "
            "picked afterwards."
        ),
    )
    parser.add_argument(
        "--use_anchor_time_in_head", action="store_true",
        help=(
            "Concat a fresh time2vec(anchor_start) to the time head input. "
            "Anchor = the masked position's own start time (i.e. the most "
            "recent observed check-in). Lets the time head condition directly "
            "on hour-of-day / day-of-week without relying on encoder pooling."
        ),
    )
    parser.add_argument(
        "--anchor_time_emb_dim", type=int, default=64,
        help="Output dim of the anchor time2vec (default: 64).",
    )
    parser.add_argument(
        "--anchor_time_modulo", type=str, default="daily",
        choices=["weekly", "daily", "none"],
        help="Modulo applied to anchor_start before time2vec (default: daily).",
    )
    parser.add_argument(
        "--anchor_temporal_embedding", type=str, default="time2vec",
        choices=["time2vec", "logfreq"],
        help="Embedding family used for the anchor time (default: time2vec).",
    )
    parser.add_argument(
        "--freeze_poi_head", action="store_true",
        help=(
            "Freeze query_proj + poi_emb so the loaded POI head is locked. "
            "Use together with --freeze_backbone --resume_from_checkpoint <next-POI ckpt> "
            "to two-stage train: POI part inherits next-POI quality, time head is fitted "
            "fresh on the frozen encoder."
        ),
    )

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


def parse_tul_args(args_list=None):
    """Parse command line arguments for Trajectory User Linking training."""
    parser = argparse.ArgumentParser(description="Clique: Trajectory User Linking")
    _add_shared_args(parser)

    parser.add_argument(
        "--sub_traj_len", type=int, default=32,
        help="Number of visits per sub-trajectory chunk (default: 32).",
    )
    parser.add_argument(
        "--min_chunks", type=int, default=3,
        help="Agents with fewer than this many full chunks are excluded (default: 3).",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.7,
        help="Per-agent fraction of chunks routed to train (default: 0.7).",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1,
        help="Per-agent fraction of chunks routed to val (default: 0.1).",
    )
    parser.add_argument(
        "--chunk_split_seed", type=int, default=42,
        help="Seed for the per-agent chunk split (default: 42).",
    )
    parser.add_argument(
        "--tul_emb_dim", type=int, default=256,
        help="Pooled chunk-embedding dim before the classifier (default: 256).",
    )
    parser.add_argument(
        "--tul_head_dropout", type=float, default=0.1,
        help="Dropout inside the TUL pooling MLP (default: 0.1).",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0,
        help="Cross-entropy label smoothing (default: 0.0).",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="traxion-tul",
        help="W&B project name for TUL runs (default: traxion-tul).",
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true",
        help="Freeze backbone (feature_encoder + attentions) and only train the TUL head.",
    )
    parser.add_argument(
        "--freeze_agent_emb", action="store_true",
        help=(
            "Freeze the input agent embedding weights. Combined with the dataset-level "
            "agent zeroing this guarantees the backbone sees no agent-identity signal "
            "through the feature encoder during fine-tuning."
        ),
    )

    # --- SupCon / contrastive fine-tuning ------------------------------------
    parser.add_argument(
        "--supcon_weight", type=float, default=0.0,
        help=(
            "Weight for supervised contrastive loss (Khosla et al. 2020) applied "
            "to the ℓ2-normalized projection-head output. 0.0 (default) disables it "
            "and reproduces the pure CE baseline."
        ),
    )
    parser.add_argument(
        "--supcon_temperature", type=float, default=0.1,
        help="Temperature for the supervised contrastive loss (default: 0.1).",
    )
    parser.add_argument(
        "--tul_proj_dim", type=int, default=128,
        help="Dimensionality of the contrastive projection head (default: 128).",
    )
    parser.add_argument(
        "--tul_mask_ratio", type=float, default=0.15,
        help=(
            "Fraction of valid visits randomly masked per augmented view when "
            "supcon_weight>0 (default: 0.15). Applied by flipping padding_mask bits."
        ),
    )
    parser.add_argument(
        "--tul_head_lr_mult", type=float, default=1.0,
        help=(
            "LR multiplier applied to the TUL head (pool_proj + classifier + "
            "proj_head). >1.0 lets the randomly-initialized head catch up to a "
            "pretrained backbone. Default 1.0 reproduces single-group optimizer."
        ),
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0,
        help="Linear LR warmup steps before cosine decay (default: 0 disables).",
    )
    parser.add_argument(
        "--tul_cosine_classifier", action="store_true",
        help=(
            "Use a cosine-similarity classifier head (normalize both emb and "
            "classifier weight, scale by tul_cosine_scale). Standard trick for "
            "fine-grained identity classification."
        ),
    )
    parser.add_argument(
        "--tul_cosine_scale", type=float, default=20.0,
        help="Scale for cosine-similarity logits when tul_cosine_classifier is set (default 20).",
    )
    parser.add_argument(
        "--pk_P", type=int, default=0,
        help=(
            "SupCon PK-sampler: classes per batch (P). 0 disables PK and falls "
            "back to uniform shuffling."
        ),
    )
    parser.add_argument(
        "--pk_K", type=int, default=0,
        help="SupCon PK-sampler: chunks per class per batch (K). Batch = P*K.",
    )
    parser.add_argument(
        "--pk_batches_per_epoch", type=int, default=0,
        help="Override batches/epoch under PK sampler (0 = n_train / (P*K)).",
    )

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


def parse_eicu_args(args_list=None):
    """Parse command line arguments for eICU mortality fine-tuning."""
    parser = argparse.ArgumentParser(description="Clique: ICU Mortality Prediction (eICU demo)")
    _add_shared_args(parser)

    parser.add_argument("--eicu_emb_dim", type=int, default=128,
                        help="Pooled per-stay embedding dim before the binary classifier (default: 128)")
    parser.add_argument("--eicu_head_dropout", type=float, default=0.2,
                        help="Dropout inside the per-stay pooling MLP (default: 0.2)")
    parser.add_argument("--pos_weight", type=float, default=10.0,
                        help="BCE pos_weight for the rare positive class (default: 10.0; ~1/0.085).")
    parser.add_argument("--head_lr_mult", type=float, default=3.0,
                        help="LR multiplier on pool_proj + classifier (default: 3.0).")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Linear LR warmup steps before cosine decay (default: 0 disables).")
    parser.add_argument("--zero_agent_input", action="store_true", default=True,
                        help="Zero out the agent input id at fine-tune time (default: True).")
    parser.add_argument("--no-zero_agent_input", action="store_false", dest="zero_agent_input",
                        help="Allow the per-stay agent_id to flow through the agent embedding.")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze feature_encoder + attentions; train only the head.")
    parser.add_argument("--freeze_agent_emb", action="store_true",
                        help="Freeze the input agent embedding (defensive when --zero_agent_input is off).")
    parser.add_argument("--wandb_project", type=str, default="traxion-eicu",
                        help="W&B project name for eICU runs (default: traxion-eicu)")

    parser.add_argument("--use_tabular_features", action="store_true", default=True,
                        help="Concatenate per-stay demographic + admit + event-aggregate features into the head (default True).")
    parser.add_argument("--no-use_tabular_features", action="store_false", dest="use_tabular_features",
                        help="Disable tabular side-features (events-only ablation).")
    parser.add_argument("--tab_feat_hidden", type=int, default=64,
                        help="Hidden width for the tabular-feature projection MLP (default 64).")
    parser.add_argument("--oversample_positives", action="store_true", default=True,
                        help="Use a class-balanced WeightedRandomSampler so each batch is ~50%% positives (default True).")
    parser.add_argument("--no-oversample_positives", action="store_false", dest="oversample_positives",
                        help="Disable positive oversampling (relies on pos_weight only).")
    parser.add_argument("--n_eval_windows", type=int, default=4,
                        help="Number of evenly-spaced windows averaged at eval time (default 4).")

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


def parse_social_args(args_list=None):
    """Parse command line arguments for social relationship inference training."""
    parser = argparse.ArgumentParser(description="Clique: Social Link Prediction")
    _add_shared_args(parser)

    parser.add_argument("--social_emb_dim", type=int, default=128,
                        help="Pooled per-agent embedding dimension (default: 128)")
    parser.add_argument("--pair_head_hidden", type=int, default=128,
                        help="Hidden width of pair scorer MLP (default: 128)")
    parser.add_argument("--pair_head_dropout", type=float, default=0.1,
                        help="Dropout inside pair scorer MLP (default: 0.1)")
    parser.add_argument("--n_neg_per_pos", type=int, default=8,
                        help="In-batch negatives per positive edge for BPR (default: 8)")
    parser.add_argument("--bpr_weight", type=float, default=1.0)
    parser.add_argument("--infonce_weight", type=float, default=0.3)
    parser.add_argument("--infonce_temperature", type=float, default=0.1)
    parser.add_argument("--batches_per_epoch", type=int, default=200,
                        help="Number of edge-centric batches per training epoch")
    parser.add_argument("--disable_pair_features", action="store_true",
                        help="Do not use handcrafted pair features (learned-only ablation)")
    parser.add_argument("--wandb_project", type=str, default="traxion-social",
                        help="W&B project name for social runs (default: traxion-social)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the backbone (feature_encoder + attentions) and train only the social head.")
    parser.add_argument("--pair_feat_norm", type=str, default="layernorm",
                        choices=["layernorm", "batchnorm", "zscore", "none"],
                        help="How to normalize handcrafted pair features before the pair head.")
    parser.add_argument("--disable_emb_branch", action="store_true",
                        help="Drop the embedding-derived features; head sees handcrafted pair features only.")
    parser.add_argument("--head_lr_mult", type=float, default=1.0,
                        help="LR multiplier applied to pool_proj + pair_head (vs backbone lr).")
    parser.add_argument("--random_neg_weight", type=float, default=0.0,
                        help="Weight of auxiliary BCE loss on random-pair non-edges (handcrafted-only slice).")
    parser.add_argument("--random_neg_batch", type=int, default=256,
                        help="Random-negative BCE batch size per step (half positives, half negatives).")
    parser.add_argument("--pair_head_split", action="store_true",
                        help="Split the pair scorer into emb_head(sym) + pf_head(pair_features, linear). Summed logits.")
    parser.add_argument("--gnn_head", action="store_true",
                        help="Add 1-hop GraphRefiner over the train friend graph between encode() and score_pairs() (fine-tune only).")
    parser.add_argument("--gnn_gate_init", type=float, default=-2.0,
                        help="Pre-sigmoid initial gate value for GraphRefiner (default -2.0 for small neighbor contribution).")
    parser.add_argument("--gnn_refresh_every_n_epochs", type=int, default=1,
                        help="Refresh the per-agent embedding cache every N epochs (default 1).")
    parser.add_argument("--hard_random_neg_weight", type=float, default=0.0,
                        help="Weight of full-model BCE on uniform-random pairs (closes the in-batch BPR / random-test-pair distribution gap).")
    parser.add_argument("--hard_random_neg_batch", type=int, default=256,
                        help="Number of pos+neg pairs per step for the hard-random-negative BCE term.")

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ablation pretraining (A7 causal / A8 STM)
# ---------------------------------------------------------------------------

def parse_ablation_args(args_list=None):
    """Parse command line arguments for ablation pretraining objectives.

    Adds the same shared backbone args used elsewhere, plus the ablation
    objective selector and STM mask ratio. The resulting checkpoint is
    loadable by the standard finetune scripts via strict=False.
    """
    parser = argparse.ArgumentParser(description="Clique: Ablation Pretraining (causal / STM)")
    _add_shared_args(parser)

    parser.add_argument(
        "--pretrain_objective",
        type=str,
        required=True,
        choices=["causal", "stm"],
        help="Pretraining objective. 'causal' = next-(POI,time-bin,cat) prediction "
             "with causal seq attention; 'stm' = masked-token prediction at masked "
             "positions only (UniTraj Self-supervised Trajectory Masking).",
    )
    parser.add_argument(
        "--mask_ratio", type=float, default=0.3,
        help="STM masking ratio (default: 0.3). Ignored for --pretrain_objective causal.",
    )
    parser.add_argument(
        "--n_time_bins", type=int, default=24,
        help="Number of hour-of-day bins for the time-bin head (default: 24).",
    )
    parser.add_argument(
        "--poi_loss_weight", type=float, default=1.0,
    )
    parser.add_argument(
        "--time_loss_weight", type=float, default=1.0,
    )
    parser.add_argument(
        "--cat_loss_weight", type=float, default=1.0,
    )
    parser.add_argument(
        "--wandb_project", type=str, default="traxion-ablation",
        help="W&B project name for ablation pretraining (default: traxion-ablation).",
    )

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()
