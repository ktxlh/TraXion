#!/usr/bin/env bash
# Fine-tune the pre-trained TraXion backbone for anomaly detection.
#
# Usage:
#   bash scripts/finetune_anomaly.sh <dataset> <pretrain_ckpt> [variant] [extra args ...]
#
#   variant ∈ {active, bypass}    (default: active = TraXion; bypass = TraXion-NoColoc)
#
# Per-task fine-tune settings follow Appendix Table 6 of the paper.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <dataset> <pretrain_ckpt> [variant=active|bypass] [extra args ...]" >&2
    exit 1
fi

DATASET="$1"; CKPT="$2"; shift 2
VARIANT="active"
if [[ $# -ge 1 && ( "$1" == "active" || "$1" == "bypass" ) ]]; then
    VARIANT="$1"; shift
fi

COMMON=(
    --dataset "$DATASET"
    --finetune --resume_from_checkpoint "$CKPT"
    --lr 1e-4
    --no_contrastive
    --early_stopping_criterion bce_loss --ema_early_stopping
)

[[ "$VARIANT" == "bypass" ]] && COMMON+=(--no_neighbor_attn)

case "$DATASET" in
    numosim)
        EXTRA=(--num_epochs 100 --early_stopping_patience 10
               --no-perturb_timestamps --no-perturb_locations
               --insert_visits --no-behavioral_anomaly
               --time_modulo weekly --clique_size 8
               --lambda_min 1e-6 --lambda_max 2)
        ;;
    urban-atlanta|urban-berlin)
        if [[ "$DATASET" == "urban-berlin" ]]; then
            CSZ=4
        else
            CSZ=8
        fi
        EXTRA=(--num_epochs 250 --early_stopping_patience 50
               --no-perturb_timestamps --no-perturb_locations
               --no-insert_visits --behavioral_anomaly
               --time_modulo daily --clique_size "$CSZ")
        ;;
    lanl-core)
        # Use the LANL-specific lateral-movement perturbation; see Appendix C.6.
        EXTRA=(--num_epochs 100 --early_stopping_patience 15
               --no-perturb_timestamps --no-perturb_locations
               --no-insert_visits --no-behavioral_anomaly
               --user_host_swap --user_host_swap_frac 0.3
               --time_modulo daily --clique_size 8
               --lambda_min 1e-3 --lambda_max 2 --lr 5e-5)
        ;;
    *)
        echo "anomaly fine-tune is not defined for $DATASET" >&2; exit 2 ;;
esac

set -x
exec python -m cli.train "${COMMON[@]}" "${EXTRA[@]}" "$@"
