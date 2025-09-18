#!/usr/bin/env bash

# Sweep aux loss coefficients and normalization flag for SFT regularized trainer
# Usage: chmod +x sweep_sft_regularized_sft.sh && ./sweep_sft_regularized_sft.sh

set -euo pipefail

cd examples

export WANDB_PROJECT="sft-regularized-sft"

# Default parameters (can be overridden via env vars)
MODEL=${MODEL:-"Qwen/Qwen3-1.7B-Base"}
DATASET=${DATASET:-"atrost/math_sft_40K_trl"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs"}
HF_USERNAME=${HF_USERNAME:-"atrost"}
RUN_NAME_BASE=${RUN_NAME_BASE:-"math_sft_40K_trl_SFT"}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-64}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-"cosine"}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
PUSH_TO_HUB=${PUSH_TO_HUB:-true}
USE_VLLM=${USE_VLLM:-false}
VLLM_HOST=${VLLM_HOST:-"127.0.0.1"}
VLLM_PORT=${VLLM_PORT:-8000}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
MAX_TOKENS=${MAX_TOKENS:-4096}
MC_CACHE_DIR=${MC_CACHE_DIR:-}

# Sweep values
AUX_LOSS_COEFS=${AUX_LOSS_COEFS:-"0.0 0.01 0.05 0.1 0.3 0.5 0.7 0.9 0.95 0.99 1.0"}
NORMALIZE_OPTIONS=${NORMALIZE_OPTIONS:-"true false"}

# Ensure output dir exists
mkdir -p "$OUTPUT_DIR"

for coef in $AUX_LOSS_COEFS; do
  for norm in $NORMALIZE_OPTIONS; do
    echo "=== Launching run: aux_loss_coef=${coef}, normalize_loss=${norm} ==="
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NO_PROXY='127.0.0.1,localhost,::1' no_proxy='127.0.0.1,localhost,::1' uv run accelerate launch sft_regularized_sft.py \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --output-dir "$OUTPUT_DIR" \
      --hf-username "$HF_USERNAME" \
      --run-name-base "$RUN_NAME_BASE" \
      --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
      --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
      --learning-rate "$LEARNING_RATE" \
      --num-train-epochs "$NUM_TRAIN_EPOCHS" \
      --weight-decay "$WEIGHT_DECAY" \
      --max-grad-norm "$MAX_GRAD_NORM" \
      --lr-scheduler-type "$LR_SCHEDULER_TYPE" \
      --warmup-ratio "$WARMUP_RATIO" \
      $( [ "${PUSH_TO_HUB}" = "true" ] && echo "--push-to-hub" ) \
      $( [ "${USE_VLLM}" = "true" ] && echo "--use-vllm" ) \
      --vllm-host "$VLLM_HOST" \
      --vllm-port "$VLLM_PORT" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --max-tokens "$MAX_TOKENS" \
      ${MC_CACHE_DIR:+--mc-cache-dir "$MC_CACHE_DIR"} \
      --aux-loss-coef "$coef" \
      $( [ "${norm}" = "true" ] && echo "--normalize-loss" )
  done
  echo "=== Finished run: aux_loss_coef=${coef}, normalize_loss=${norm} ==="
  echo "=== Waiting for 10 seconds ==="
  sleep 10
done
