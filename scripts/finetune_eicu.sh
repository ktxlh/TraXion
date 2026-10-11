#!/usr/bin/env bash
# Fine-tune the pre-trained TraXion backbone for ICU mortality prediction.
#
# Usage:
#   bash scripts/finetune_eicu.sh <dataset> <pretrain_ckpt> [route] [extra args ...]
#
#   dataset ∈ {eicu-full, mimiciv-full}: the stay-keyed corpus (one label per ICU
#   stay). The backbone is the one pre-trained on the matching *-pid corpus.
#   route ∈ {active, bypass} (default: active). Run both routes for each seed and
#   keep the one selected by scripts/select_route.py.
#
# The test metrics of the best-validation checkpoint are printed at the end of
# training; cli/evaluate_eicu.py recomputes them from a saved checkpoint.
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
    --num_epochs 25 --early_stopping_patience 8
    --train_batch_size 32 --val_batch_size 64
    --max_seq_len 128 --pos_weight 5.0 --warmup_steps 50
    --use_tabular_features --oversample_positives --n_eval_windows 4
    --zero_agent_input --neighbor_search sorted
)

[[ "$ROUTE" == "bypass" ]] && COMMON+=(--no_neighbor_attn)

case "$DATASET" in
    eicu-full)    EXTRA=(--lr 5e-5 --head_lr_mult 1.0) ;;
    mimiciv-full) EXTRA=(--lr 2e-4 --head_lr_mult 3.0) ;;
    *) echo "mortality fine-tune is not defined for $DATASET" >&2; exit 2 ;;
esac

set -x
exec python -m cli.train_eicu "${COMMON[@]}" "${EXTRA[@]}" "$@"
