#!/usr/bin/env bash
# Fine-tune the pre-trained TraXion backbone for next-POI recommendation.
#
# Usage:
#   bash scripts/finetune_poi.sh <dataset> <pretrain_ckpt> [variant] [extra args ...]
#   variant ∈ {active (TraXion), bypass (TraXion-NoColoc, default for POI)}
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
    --lr 1e-4
    --train_batch_size 64 --val_batch_size 256
    --early_stopping_criterion total_loss --ema_early_stopping
    --time_modulo daily --lambda_min 1e-3 --lambda_max 360
)

[[ "$VARIANT" == "bypass" ]] && COMMON+=(--no_neighbor_attn)

case "$DATASET" in
    foursquare-tokyo)
        EXTRA=(--clique_size 8 --num_epochs 100 --early_stopping_patience 15) ;;
    gowalla-stockholm-v1)
        EXTRA=(--clique_size 4 --num_epochs 100 --early_stopping_patience 15) ;;
    gowalla-austin-v1)
        EXTRA=(--clique_size 4 --num_epochs 200 --early_stopping_patience 80) ;;
    *)
        echo "POI fine-tune is not defined for $DATASET" >&2; exit 2 ;;
esac

set -x
exec python -m cli.train_poi "${COMMON[@]}" "${EXTRA[@]}" "$@"
