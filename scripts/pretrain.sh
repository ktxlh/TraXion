#!/usr/bin/env bash
# Pre-train a TraXion backbone on a single dataset.
#
# Usage:
#   bash scripts/pretrain.sh <dataset> [extra args ...]
# Example:
#   bash scripts/pretrain.sh urban-berlin --device cuda:0 --seed 0
#
# Per-dataset settings follow the per-dataset pre-training table in Appendix B.
# The two EHR corpora are pre-trained on their patient-keyed versions
# (eicu-full-pid, mimiciv-full-pid; see preprocess/rekey_patient.py).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <dataset> [extra args ...]" >&2
    exit 1
fi

DATASET="$1"; shift

COMMON=(
    --dataset "$DATASET"
    --lr 2e-4
    --perturb_locations --perturb_timestamps
    --no-insert_visits --no-behavioral_anomaly
    --early_stopping_criterion total_loss --ema_early_stopping
)

case "$DATASET" in
    numosim)
        EXTRA=(--time_modulo weekly --clique_size 8
               --weight_decay 1e-3
               --num_epochs 200 --early_stopping_patience 20
               --train_batch_size 64 --val_batch_size 512)
        ;;
    urban-atlanta)
        EXTRA=(--time_modulo daily --clique_size 8
               --weight_decay 1e-3
               --num_epochs 200 --early_stopping_patience 40
               --train_batch_size 64 --val_batch_size 512)
        ;;
    urban-berlin)
        EXTRA=(--time_modulo daily --clique_size 4
               --weight_decay 1e-2 --no-perturb_locations
               --num_epochs 800 --early_stopping_patience 80
               --train_batch_size 64 --val_batch_size 256)
        ;;
    foursquare-tokyo)
        EXTRA=(--time_modulo daily --clique_size 8
               --lambda_min 1e-3 --lambda_max 360
               --weight_decay 1e-2
               --num_epochs 400 --early_stopping_patience 40
               --train_batch_size 128 --val_batch_size 256)
        ;;
    gowalla-stockholm-v1)
        EXTRA=(--time_modulo daily --clique_size 4
               --lambda_min 1e-3 --lambda_max 360
               --weight_decay 1e-2
               --num_epochs 500 --early_stopping_patience 80
               --train_batch_size 64 --val_batch_size 512)
        ;;
    gowalla-austin-v1)
        EXTRA=(--time_modulo daily --clique_size 4
               --lambda_min 1e-3 --lambda_max 360
               --weight_decay 1e-2 --no-perturb_timestamps
               --num_epochs 500 --early_stopping_patience 80
               --train_batch_size 64 --val_batch_size 256)
        ;;
    lanl-core)
        EXTRA=(--time_modulo daily --clique_size 8
               --lambda_min 1e-3 --lambda_max 2
               --weight_decay 1e-2
               --num_epochs 200 --early_stopping_patience 40
               --train_batch_size 64 --val_batch_size 64)
        ;;
    eicu-full-pid|mimiciv-full-pid)
        EXTRA=(--time_modulo daily --clique_size 4
               --weight_decay 1e-3
               --model_dim 128 --num_layers 2 --n_heads 4 --max_seq_len 32 --no_agent_emb
               --neighbor_search sorted
               --num_epochs 80 --early_stopping_patience 15
               --train_batch_size 32 --val_batch_size 64)
        ;;
    *)
        echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

set -x
exec python -m cli.train "${COMMON[@]}" "${EXTRA[@]}" "$@"
