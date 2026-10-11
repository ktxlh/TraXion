#!/usr/bin/env bash
# Ablation study: rows Full and C1-C7 of the ablation table, four tasks.
#
# Usage:
#   bash scripts/ablation.sh pretrain   <arm> <task> [extra args ...]
#   bash scripts/ablation.sh finetune   <arm> <task> <pretrain_ckpt> [extra args ...]
#   bash scripts/ablation.sh finetune   C6    <task> [extra args ...]
#   bash scripts/ablation.sh evaluate   <task> <finetuned_ckpt> [extra args ...]
#   bash scripts/ablation.sh score-lanl <ckpt> <out_dir> [extra args ...]
#
#   arm  ∈ {Full, C1, C2, C3, C4, C5, C6, C7}
#   task ∈ {mobility_ad, next_visit, social, auth_log}
#
# Tasks:
#   mobility_ad  UA-Berlin anomaly detection, active route; AP / Max F1 = combined/agent/{AP,Max F1}
#   next_visit   Gowalla-Austin next visit, bypass route; H@10 = loc/recall@10, T±60 = time/t@60min/mass
#   social       Gowalla-Austin social link, active route; AP / Hits@10 of the model score
#   auth_log     LANL authentication log, active route; user/AP and user/MaxF1 from score-lanl
#
# Arms:
#   Full  denoising objective with L_noise and L_prototype
#   C1    w/o L_prototype           --no_contrastive (pre-training; LANL fine-tuning too)
#   C2    w/o L_noise               --no_bce (pre-training; LANL fine-tuning too)
#   C3    w/o prototype-as-input    --no_agent_emb --model_dim 1024 (pre-training and fine-tuning)
#   C4    autoregressive next-token pre-training   cli.train_ablation_pretrain --pretrain_objective causal
#   C5    masked-token pre-training cli.train_ablation_pretrain --pretrain_objective stm
#   C6    no pre-training           fine-tuning starts from random weights
#   C7    flat transformer encoder  --use_transformer_encoder (pre-training and fine-tuning)
#
# Seeds (pass --seed N as an extra argument; cells with ± average five runs):
#   mobility_ad  Full, C5: pre-training seeds 0 101 202 303 404. Other arms: one run, seed 0.
#                Fine-tuning seed 0.
#   next_visit   Full: pre-training seeds 101 202 303 404 505. C4, C5: 0 101 202 303 404.
#                Fine-tuning seed 1 for Full, C4, C5; seed 0 for C1, C2, C3, C6, C7 (one run).
#   social       Pre-training seeds 0 101 202 303 404, fine-tuning seed 0.
#                C6: fine-tuning seeds 0 101 202 303 404.
#   auth_log     Full: pre-training seeds 0 1 2 3 4, scored without fine-tuning (score-lanl).
#                C6: seeds 0 101 202 303 404. Other arms: one run, seed 0.
#
# Arguments after the fixed ones are appended to the python command and take
# precedence (e.g. --seed 101 --device cuda:0).
set -euo pipefail

usage() {
    sed -n '4,12p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
}

# Value of the last --seed in the arguments (default 0).
seed_from() {
    local s=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --seed)   s="${2:-0}"; shift; if [[ $# -gt 0 ]]; then shift; fi ;;
            --seed=*) s="${1#--seed=}"; shift ;;
            *)        shift ;;
        esac
    done
    echo "$s"
}

check_arm() {
    case "$1" in
        Full|C1|C2|C3|C4|C5|C6|C7) ;;
        *) echo "unknown arm: $1" >&2; exit 2 ;;
    esac
}

# Dataset, co-occurrence size C, Space2Vec scale range and time wrap per task.
set_data() {
    case "$1" in
        mobility_ad)
            DATA=(--dataset urban-berlin --clique_size 4 --time_modulo daily
                  --lambda_min 1e-6 --lambda_max 2) ;;
        next_visit|social)
            DATA=(--dataset gowalla-austin-v1 --clique_size 4 --time_modulo daily
                  --lambda_min 1e-3 --lambda_max 360) ;;
        auth_log)
            DATA=(--dataset lanl-core --clique_size 8 --time_modulo daily
                  --lambda_min 1e-3 --lambda_max 2) ;;
        *) echo "unknown task: $1" >&2; exit 2 ;;
    esac
}

# Backbone shape. C3 and C7 change the architecture, so their fine-tuning
# commands carry the same flags as their pre-training.
set_arch() {
    ARCH=(--dim_feedforward 1024 --n_heads 4 --num_layers 6 --hidden_dim 256
          --n_scales 32 --n_agent_attributes 32 --max_seq_len 32
          --temporal_embedding time2vec)
    case "$1" in
        C3) ARCH+=(--no_agent_emb --model_dim 1024) ;;
        C7) ARCH+=(--use_transformer_encoder --model_dim 1040) ;;
        *)  ARCH+=(--model_dim 1040) ;;
    esac
}

# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------
pretrain() {
    [[ $# -lt 2 ]] && usage
    local ARM="$1" TASK="$2"; shift 2
    check_arm "$ARM"; set_data "$TASK"; set_arch "$ARM"
    local SEED; SEED="$(seed_from "$@")"

    if [[ "$ARM" == "C6" ]]; then
        echo "C6 has no pre-training: bash $0 finetune C6 $TASK [--seed N]" >&2
        exit 0
    fi

    local COMMON=(--lr 2e-4 --weight_decay 0.01
                  --early_stopping_criterion total_loss --ema_early_stopping)

    # C4, C5: the denoising objective is replaced by next-token (causal) or
    # masked-token (stm) prediction. --keep_prototype adds L_prototype.
    if [[ "$ARM" == "C4" || "$ARM" == "C5" ]]; then
        local OBJ=causal
        if [[ "$ARM" == "C5" ]]; then OBJ=stm; fi
        local KEEP=() BUDGET=()
        case "$TASK" in
            mobility_ad)       BUDGET=(--num_epochs 300 --early_stopping_patience 80)
                               if [[ "$ARM" == "C5" ]]; then KEEP=(--keep_prototype); fi ;;
            next_visit|social) BUDGET=(--num_epochs 500 --early_stopping_patience 80)
                               KEEP=(--keep_prototype) ;;
            auth_log)          BUDGET=(--num_epochs 200 --early_stopping_patience 40) ;;
        esac
        set -x
        exec python -m cli.train_ablation_pretrain "${DATA[@]}" "${ARCH[@]}" \
            --pretrain_objective "$OBJ" --mask_ratio 0.3 ${KEEP[@]+"${KEEP[@]}"} \
            "${COMMON[@]}" "${BUDGET[@]}" \
            --train_batch_size 64 --val_batch_size 256 \
            --seed 0 "$@"
    fi

    # Full, C1, C2, C3, C7: denoising pre-training with cli.train.
    local BUDGET=() BATCH=() NOISE=() LOSS=()
    local BOTH=(--perturb_locations --perturb_timestamps)
    local NONE=(--no-perturb_locations --no-perturb_timestamps)
    case "$TASK" in
        mobility_ad)
            BUDGET=(--num_epochs 300 --early_stopping_patience 80)
            BATCH=(--train_batch_size 64 --val_batch_size 256)
            NOISE=("${BOTH[@]}")
            case "$ARM" in
                Full)
                    # Full: seed 0 perturbs timestamps only; seeds 101-404 perturb both.
                    BUDGET=(--num_epochs 800 --early_stopping_patience 80)
                    if [[ "$SEED" == "0" ]]; then
                        NOISE=(--no-perturb_locations --perturb_timestamps)
                    fi ;;
                C2)
                    BUDGET=(--num_epochs 800 --early_stopping_patience 80)
                    BATCH=(--train_batch_size 1 --val_batch_size 1)
                    NOISE=("${NONE[@]}") ;;
            esac ;;
        next_visit)
            BUDGET=(--num_epochs 500 --early_stopping_patience 80)
            BATCH=(--train_batch_size 64 --val_batch_size 256)
            NOISE=("${BOTH[@]}")
            case "$ARM" in
                Full) NOISE=(--perturb_locations --no-perturb_timestamps) ;;
                C2)   BATCH=(--train_batch_size 96 --val_batch_size 512)
                      NOISE=("${NONE[@]}") ;;
            esac ;;
        social)
            BUDGET=(--num_epochs 500 --early_stopping_patience 80)
            BATCH=(--train_batch_size 64 --val_batch_size 256)
            NOISE=("${BOTH[@]}") ;;
        auth_log)
            BUDGET=(--num_epochs 200 --early_stopping_patience 40)
            BATCH=(--train_batch_size 64 --val_batch_size 256)
            NOISE=("${BOTH[@]}" --no-user_host_swap)
            case "$ARM" in
                Full)
                    # Full: the backbones scored by score-lanl. Seed 0 uses batch 128
                    # and logs a holdout validation loss every 200 steps.
                    BATCH=(--train_batch_size 64 --val_batch_size 64)
                    if [[ "$SEED" == "0" ]]; then
                        BATCH=(--train_batch_size 128 --val_batch_size 128 --holdout_eval_every 200)
                    fi ;;
                C2)
                    BATCH=(--train_batch_size 64 --val_batch_size 64)
                    NOISE=("${NONE[@]}" --no-user_host_swap) ;;
                C7)
                    BATCH=(--train_batch_size 32 --val_batch_size 128) ;;
            esac ;;
    esac
    case "$ARM" in
        C1) LOSS=(--no_contrastive) ;;
        C2) LOSS=(--no_bce) ;;
    esac

    set -x
    exec python -m cli.train "${DATA[@]}" "${ARCH[@]}" "${COMMON[@]}" "${BUDGET[@]}" \
        "${BATCH[@]}" "${NOISE[@]}" --no-insert_visits --no-behavioral_anomaly \
        ${LOSS[@]+"${LOSS[@]}"} --seed 0 "$@"
}

# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
finetune() {
    [[ $# -lt 2 ]] && usage
    local ARM="$1" TASK="$2"; shift 2
    check_arm "$ARM"; set_data "$TASK"; set_arch "$ARM"

    local INIT=()
    if [[ "$ARM" != "C6" ]]; then
        [[ $# -lt 1 ]] && { echo "finetune $ARM $TASK needs <pretrain_ckpt>" >&2; exit 1; }
        INIT=(--finetune --resume_from_checkpoint "$1"); shift
    fi

    case "$TASK" in
        mobility_ad)
            # Mobility AD: behavioral-anomaly fine-tuning, BCE only, EMA early stopping.
            local EPOCHS=250 BATCH=(--train_batch_size 64 --val_batch_size 256)
            case "$ARM" in
                Full|C2) EPOCHS=500 ;;
                C4|C6)   BATCH=(--train_batch_size 32 --val_batch_size 128) ;;
            esac
            set -x
            exec python -m cli.train "${DATA[@]}" "${ARCH[@]}" ${INIT[@]+"${INIT[@]}"} \
                --lr 1e-4 --weight_decay 0.01 "${BATCH[@]}" \
                --num_epochs "$EPOCHS" --early_stopping_patience 50 \
                --early_stopping_criterion bce_loss --ema_early_stopping \
                --no_contrastive --behavioral_anomaly --no-insert_visits \
                --no-perturb_locations --no-perturb_timestamps --no-user_host_swap \
                --seed 0 "$@" ;;

        next_visit)
            # Next-visit: bypass route (--no_neighbor_attn), anchor-time head.
            local SEED=0
            case "$ARM" in Full|C4|C5) SEED=1 ;; esac
            set -x
            exec python -m cli.train_next_visit "${DATA[@]}" "${ARCH[@]}" ${INIT[@]+"${INIT[@]}"} \
                --no_neighbor_attn \
                --lr 1e-4 --weight_decay 0.01 --train_batch_size 64 --val_batch_size 256 \
                --num_epochs 200 --early_stopping_patience 40 \
                --early_stopping_criterion loss --ema_early_stopping \
                --time_target log --num_gaussians 8 --time_loss_weight 0.3 \
                --use_anchor_time_in_head --anchor_time_emb_dim 64 --anchor_time_modulo daily \
                --seed "$SEED" "$@" ;;

        social)
            # Social link: no EMA, patience 40, cap 100.
            set -x
            exec python -m cli.train_social "${DATA[@]}" "${ARCH[@]}" ${INIT[@]+"${INIT[@]}"} \
                --lr 1e-4 --weight_decay 0.01 --train_batch_size 64 --val_batch_size 64 \
                --num_epochs 100 --early_stopping_patience 40 \
                --early_stopping_criterion total_loss \
                --pair_feat_norm zscore --random_neg_weight 1.0 \
                --seed 0 "$@" ;;

        auth_log)
            # Auth-log: user-host swap with inserted visits, no EMA. C1 and C2 drop
            # the same loss as in pre-training.
            if [[ "$ARM" == "Full" ]]; then
                echo "Full is scored without fine-tuning: bash $0 score-lanl <pretrain_ckpt> <out_dir>" >&2
                exit 2
            fi
            local BUDGET=(--lr 5e-5 --num_epochs 30 --early_stopping_patience 8) LOSS=()
            if [[ "$ARM" == "C6" ]]; then
                BUDGET=(--lr 1e-4 --num_epochs 60 --early_stopping_patience 12)
            fi
            case "$ARM" in
                C1) LOSS=(--no_contrastive) ;;
                C2) LOSS=(--no_bce) ;;
            esac
            set -x
            exec python -m cli.train "${DATA[@]}" "${ARCH[@]}" ${INIT[@]+"${INIT[@]}"} \
                "${BUDGET[@]}" --weight_decay 0.01 --train_batch_size 32 --val_batch_size 32 \
                --early_stopping_criterion total_loss \
                --insert_visits --no-perturb_locations --no-perturb_timestamps \
                --no-behavioral_anomaly --user_host_swap --user_host_swap_frac 0.3 \
                ${LOSS[@]+"${LOSS[@]}"} --seed 0 "$@" ;;
    esac
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# LANL: dump per-event scores, then rank-fuse the noise-detection and
# prototype scores; the user-level AP and Max F1 are reported.
score_lanl() {
    [[ $# -lt 2 ]] && usage
    local CKPT="$1" OUT="$2"; shift 2
    set -x
    python -m cli.evaluate_lanl_unsup --checkpoint "$CKPT" --out_dir "$OUT" \
        --batch_size 32 --num_workers 4 --skip_train "$@"
    python -m cli.score_lanl_unsup --dump_dir "$OUT" --out "$OUT/scores.json"
}

evaluate() {
    [[ $# -lt 2 ]] && usage
    local TASK="$1" CKPT="$2"; shift 2
    case "$TASK" in
        mobility_ad)
            set -x
            exec python -m cli.evaluate --checkpoint "$CKPT" --no_wandb --batch_size 32 "$@" ;;
        next_visit)
            set -x
            exec python -m cli.evaluate_next_visit --checkpoint "$CKPT" --no_wandb --batch_size 32 "$@" ;;
        social)
            set -x
            exec python -m cli.evaluate_social --dataset gowalla-austin-v1 \
                --resume_from_checkpoint "$CKPT" --val_batch_size 32 "$@" ;;
        auth_log)
            [[ $# -lt 1 ]] && { echo "evaluate auth_log needs <ckpt> <out_dir>" >&2; exit 1; }
            score_lanl "$CKPT" "$@" ;;
        *) echo "unknown task: $TASK" >&2; exit 2 ;;
    esac
}

[[ $# -lt 1 ]] && usage
MODE="$1"; shift
case "$MODE" in
    pretrain)   pretrain "$@" ;;
    finetune)   finetune "$@" ;;
    evaluate)   evaluate "$@" ;;
    score-lanl) score_lanl "$@" ;;
    *) echo "unknown mode: $MODE" >&2; usage ;;
esac
