#!/usr/bin/env bash
# LANL anomaly fine-tune wrapper. Equivalent to:
#   bash scripts/finetune_anomaly.sh lanl-core <pretrain_ckpt> [variant]
# kept as a separate script because LANL evaluation involves a post-hoc
# rank-fusion step (see usage below).
#
# Usage:
#   bash scripts/finetune_lanl.sh <pretrain_ckpt> [variant=active|bypass] [extra args ...]
#
# After fine-tuning, dump per-event scores and apply rank fusion of the noise
# and prototype-contrastive heads:
#   python -m cli.evaluate_lanl_unsup --checkpoint runs/<finetune_id>.pt \
#       --out_dir logs/lanl_unsup_dumps/<tag> --no_wandb --device cuda:0
#   python -m cli.score_lanl_unsup --dump_dir logs/lanl_unsup_dumps/<tag>
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <pretrain_ckpt> [variant=active|bypass] [extra args ...]" >&2
    exit 1
fi

CKPT="$1"; shift
VARIANT="active"
if [[ $# -ge 1 && ( "$1" == "active" || "$1" == "bypass" ) ]]; then
    VARIANT="$1"; shift
fi

exec bash "$(dirname "$0")/finetune_anomaly.sh" lanl-core "$CKPT" "$VARIANT" "$@"
