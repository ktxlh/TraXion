#!/usr/bin/env bash
# Fine-tune the pre-trained TraXion backbone for social-link inference.
#
# Usage:
#   bash scripts/finetune_social.sh <dataset> <pretrain_ckpt> [route] [extra args ...]
#
#   route ∈ {active, bypass} (default: active). Run both routes for each seed and
#   keep the one selected by scripts/select_route.py.
#
# Evaluate the selected checkpoint with late fusion; alpha is picked on
# validation AUC and the "TEST:" line is the reported result:
#   python -m tasks.social.ensemble_eval --dataset <dataset> \
#       --social_checkpoint runs/social_<id>.pt --device cuda:0
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <dataset> <pretrain_ckpt> [route=active|bypass] [extra args ...]" >&2
    exit 1
fi

DATASET="$1"; CKPT="$2"; shift 2
ROUTE="active"
if [[ $# -ge 1 && ( "$1" == "active" || "$1" == "bypass" ) ]]; then
    ROUTE="$1"; shift
fi

COMMON=(
    --dataset "$DATASET"
    --finetune --resume_from_checkpoint "$CKPT"
    --lr 1e-4
    --num_epochs 50 --early_stopping_patience 15
    --train_batch_size 64 --val_batch_size 64
    --early_stopping_criterion total_loss
    --time_modulo daily --lambda_min 1e-3 --lambda_max 360
)

[[ "$ROUTE" == "bypass" ]] && COMMON+=(--no_neighbor_attn)

case "$DATASET" in
    foursquare-tokyo)
        EXTRA=(--clique_size 8 --pair_head_split --ema_early_stopping) ;;
    gowalla-stockholm-v1)
        EXTRA=(--clique_size 4 --pair_feat_norm layernorm --random_neg_weight 0.0 --ema_early_stopping) ;;
    gowalla-austin-v1)
        # Early stopping on the raw validation AUC.
        EXTRA=(--clique_size 4 --pair_feat_norm zscore --random_neg_weight 1.0) ;;
    *) echo "social-link fine-tune is not defined for $DATASET" >&2; exit 2 ;;
esac

set -x
exec python -m cli.train_social "${COMMON[@]}" "${EXTRA[@]}" "$@"
