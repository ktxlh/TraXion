#!/usr/bin/env bash
# Fine-tune the pre-trained TraXion backbone for next-visit (joint POI + time) prediction.
#
# Usage:
#   bash scripts/finetune_next_visit.sh <dataset> <pretrain_ckpt> [variant] [extra args ...]
#   variant ∈ {active (TraXion), bypass (TraXion-NoColoc, default for next-visit)}
#
# Recipe matches Appendix B (anchor-time head, time_loss_weight=0.3).
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <dataset> <pretrain_ckpt> [variant=active|bypass] [extra args ...]" >&2
    exit 1
fi

DATASET="$1"; CKPT="$2"; shift 2
VARIANT="bypass"
if [[ $# -ge 1 && ( "$1" == "active" || "$1" == "bypass" ) ]]; then
    VARIANT="$1"; shift
fi

COMMON=(
    --dataset "$DATASET"
    --finetune --resume_from_checkpoint "$CKPT"
    --lr 1e-4 --weight_decay 0.01
    --train_batch_size 64 --val_batch_size 256
    --num_epochs 200 --early_stopping_patience 40
    --early_stopping_criterion loss --ema_early_stopping
    --time_modulo daily --lambda_min 1e-3 --lambda_max 360
    --max_seq_len 32
    --time_target log --num_gaussians 8 --time_loss_weight 0.3
    --use_anchor_time_in_head --anchor_time_emb_dim 64 --anchor_time_modulo daily
)

[[ "$VARIANT" == "bypass" ]] && COMMON+=(--no_neighbor_attn)

case "$DATASET" in
    foursquare-tokyo)        EXTRA=(--clique_size 8) ;;
    gowalla-stockholm-v1)    EXTRA=(--clique_size 4) ;;
    gowalla-austin-v1)       EXTRA=(--clique_size 4) ;;
    *) echo "next-visit fine-tune is not defined for $DATASET" >&2; exit 2 ;;
esac

set -x
exec python -m cli.train_next_visit "${COMMON[@]}" "${EXTRA[@]}" "$@"
