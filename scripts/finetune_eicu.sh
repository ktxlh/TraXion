#!/usr/bin/env bash
# Fine-tune the pre-trained TraXion backbone for ICU mortality prediction
# on the eICU-CRD demo cohort.
#
# Usage:
#   bash scripts/finetune_eicu.sh <pretrain_ckpt> [extra args ...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <pretrain_ckpt> [extra args ...]" >&2
    exit 1
fi

CKPT="$1"; shift

set -x
exec python -m cli.train_eicu \
    --dataset eicu-demo \
    --finetune --resume_from_checkpoint "$CKPT" \
    --lr 1e-4 --num_epochs 25 --early_stopping_patience 8 \
    --train_batch_size 32 --val_batch_size 64 \
    --max_seq_len 128 --pos_weight 5.0 --head_lr_mult 3.0 --warmup_steps 50 \
    --use_tabular_features --oversample_positives --n_eval_windows 4 \
    "$@"
