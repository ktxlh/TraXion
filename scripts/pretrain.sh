#!/usr/bin/env bash
# Pre-train a TraXion backbone on a single dataset.
#
# Usage:
#   bash scripts/pretrain.sh <dataset> [extra args ...]
# Example:
#   bash scripts/pretrain.sh urban-berlin --device cuda:0 --seed 0
#
# Per-dataset settings follow Appendix Table 5 of the paper.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <dataset> [extra args ...]" >&2
    exit 1
fi

DATASET="$1"; shift

COMMON=(
    --dataset "$DATASET"
    --lr 2e-4
    --weight_decay 1e-3
    --perturb_locations --perturb_timestamps
    --no-insert_visits --no-behavioral_anomaly
    --early_stopping_criterion total_loss --ema_early_stopping
)

case "$DATASET" in
    numosim)
        EXTRA=(--time_modulo weekly --clique_size 8
               --num_epochs 200 --early_stopping_patience 20)
        ;;
    urban-atlanta)
        EXTRA=(--time_modulo daily --clique_size 8
               --num_epochs 200 --early_stopping_patience 40)
        ;;
    urban-berlin)
        EXTRA=(--time_modulo daily --clique_size 4
               --num_epochs 1000 --early_stopping_patience 100)
        ;;
    foursquare-tokyo)
        EXTRA=(--time_modulo daily --clique_size 8
               --lambda_min 1e-3 --lambda_max 360
               --num_epochs 200 --early_stopping_patience 40)
        ;;
    gowalla-stockholm-v1)
        EXTRA=(--time_modulo daily --clique_size 4
               --lambda_min 1e-3 --lambda_max 360
               --num_epochs 200 --early_stopping_patience 80)
        ;;
    gowalla-austin-v1)
        EXTRA=(--time_modulo daily --clique_size 4
               --lambda_min 1e-3 --lambda_max 360
               --num_epochs 500 --early_stopping_patience 80)
        ;;
    lanl-core)
        EXTRA=(--time_modulo daily --clique_size 8
               --lambda_min 1e-3 --lambda_max 2
               --num_epochs 200 --early_stopping_patience 40)
        ;;
    eicu-demo)
        EXTRA=(--time_modulo daily --clique_size 4 --no_agent_emb
               --model_dim 128 --num_layers 2 --n_heads 4 --max_seq_len 32
               --num_epochs 200 --early_stopping_patience 40)
        ;;
    *)
        echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

set -x
exec python -m cli.train "${COMMON[@]}" "${EXTRA[@]}" "$@"
