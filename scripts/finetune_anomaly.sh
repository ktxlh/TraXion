#!/usr/bin/env bash
# Fine-tune the pre-trained TraXion backbone for mobility anomaly detection.
#
# Usage:
#   bash scripts/finetune_anomaly.sh <dataset> <pretrain_ckpt> [route] [extra args ...]
#
#   route ∈ {active, bypass} (default: active). bypass skips the pre-trained
#   co-occurrence sub-layer (--no_neighbor_attn). Run both routes for each seed
#   and keep the one selected by scripts/select_route.py.
#
# Per-task fine-tune settings follow the fine-tuning table in Appendix B.
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
    --train_batch_size 64 --val_batch_size 256
    --no_contrastive
    --early_stopping_criterion bce_loss --ema_early_stopping
)

[[ "$ROUTE" == "bypass" ]] && COMMON+=(--no_neighbor_attn)

case "$DATASET" in
    numosim)
        EXTRA=(--num_epochs 100 --early_stopping_patience 10
               --no-perturb_timestamps --no-perturb_locations
               --insert_visits --no-behavioral_anomaly
               --time_modulo weekly --clique_size 8
               --lambda_min 1e-6 --lambda_max 2)
        ;;
    urban-atlanta)
        EXTRA=(--num_epochs 250 --early_stopping_patience 50
               --no-perturb_timestamps --no-perturb_locations
               --no-insert_visits --behavioral_anomaly
               --time_modulo daily --clique_size 8)
        ;;
    urban-berlin)
        EXTRA=(--num_epochs 500 --early_stopping_patience 50
               --no-perturb_timestamps --no-perturb_locations
               --no-insert_visits --behavioral_anomaly
               --time_modulo daily --clique_size 4)
        ;;
    *)
        echo "anomaly fine-tune is not defined for $DATASET" >&2; exit 2 ;;
esac

set -x
exec python -m cli.train "${COMMON[@]}" "${EXTRA[@]}" "$@"
