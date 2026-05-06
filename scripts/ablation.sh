#!/usr/bin/env bash
# Ablation harness for variants C1..C7 (Section 4.6 of the paper).
#
# Each variant has its own pre-training run, and each task is then fine-tuned
# from that variant's own backbone.
#
# Usage:
#   bash scripts/ablation.sh pretrain   <dataset> <variant>
#   bash scripts/ablation.sh finetune   <task> <dataset> <variant_pretrain_ckpt>
#
#   variant ∈ {C1, C2, C3, C4, C5, C6, C7}
#   task    ∈ {anomaly, poi, next_visit, social, eicu}
#   dataset ∈ {urban-berlin, gowalla-austin-v1, lanl-core}  (for pretrain)
#
# Variant definitions (paper Table 4):
#   C1: w/o L_prototype          (--no_contrastive)
#   C2: w/o L_noise              (--no_noise_loss)
#   C3: w/o prototype-as-input   (--no_agent_emb --model_dim 1024)
#   C4: causal next-token pretrain   (train_ablation_pretrain.py --variant C4)
#   C5: masked-token pretrain        (train_ablation_pretrain.py --variant C5)
#   C6: no pre-training              (no pre-train; fine-tune from scratch)
#   C7: flat transformer encoder (--use_transformer_encoder)
set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat >&2 <<EOF
usage:
  $0 pretrain  <dataset> <variant>
  $0 finetune  <task> <dataset> <variant_pretrain_ckpt>
EOF
    exit 1
fi

MODE="$1"; shift

case "$MODE" in
    pretrain)
        if [[ $# -lt 2 ]]; then
            echo "usage: $0 pretrain <dataset> <variant>" >&2; exit 1
        fi
        DATASET="$1"; VARIANT="$2"; shift 2
        case "$VARIANT" in
            C1) exec bash "$(dirname "$0")/pretrain.sh" "$DATASET" --no_contrastive "$@" ;;
            C2) exec bash "$(dirname "$0")/pretrain.sh" "$DATASET" --no_noise_loss "$@" ;;
            C3) exec bash "$(dirname "$0")/pretrain.sh" "$DATASET" --no_agent_emb --model_dim 1024 "$@" ;;
            C4) exec python -m cli.train_ablation_pretrain --dataset "$DATASET" --variant C4 "$@" ;;
            C5) exec python -m cli.train_ablation_pretrain --dataset "$DATASET" --variant C5 "$@" ;;
            C6) echo "C6 has no pre-training; skip to finetune from scratch."; exit 0 ;;
            C7) exec bash "$(dirname "$0")/pretrain.sh" "$DATASET" --use_transformer_encoder "$@" ;;
            *)  echo "unknown variant $VARIANT" >&2; exit 2 ;;
        esac ;;

    finetune)
        if [[ $# -lt 3 ]]; then
            echo "usage: $0 finetune <task> <dataset> <ckpt>" >&2; exit 1
        fi
        TASK="$1"; DATASET="$2"; CKPT="$3"; shift 3
        case "$TASK" in
            anomaly)    exec bash "$(dirname "$0")/finetune_anomaly.sh"    "$DATASET" "$CKPT" bypass "$@" ;;
            poi)        exec bash "$(dirname "$0")/finetune_poi.sh"        "$DATASET" "$CKPT" bypass "$@" ;;
            next_visit) exec bash "$(dirname "$0")/finetune_next_visit.sh" "$DATASET" "$CKPT" bypass "$@" ;;
            social)     exec bash "$(dirname "$0")/finetune_social.sh"     "$DATASET" "$CKPT"        "$@" ;;
            eicu)       exec bash "$(dirname "$0")/finetune_eicu.sh"                  "$CKPT"        "$@" ;;
            *)          echo "unknown task $TASK" >&2; exit 2 ;;
        esac ;;

    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
