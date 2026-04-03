#!/usr/bin/env bash
set -euo pipefail

# Exp12: Compare kv_only vs resid_only vs hybrid injection.
#
# Usage:
#   bash nnm/scripts/run_qwen_exp12_hybrid_injection.sh
#
# Optional env overrides:
#   MODEL="Qwen/Qwen2-1.5B"
#   INJECTION_LAYER=18
#   DATASET=scifact
#   MAX_QUERIES=50
#   MAX_CORPUS=2000
#   MAX_PAIRS=64
#   KV_ALPHA=1.0
#   RESIDUAL_ALPHA=0.10
#   RESIDUAL_SOURCE=delta    # delta|full
#   RESIDUAL_NORMALIZE=1     # 1|0
#   RESIDUAL_APPLY=all       # all|last
#   PROGRESS_EVERY=10

MODEL="${MODEL:-Qwen/Qwen2-1.5B}"
INJECTION_LAYER="${INJECTION_LAYER:-18}"
DATASET="${DATASET:-scifact}"
MAX_QUERIES="${MAX_QUERIES:-50}"
MAX_CORPUS="${MAX_CORPUS:-2000}"
MAX_PAIRS="${MAX_PAIRS:-64}"
KV_ALPHA="${KV_ALPHA:-1.0}"
RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-0.10}"
RESIDUAL_SOURCE="${RESIDUAL_SOURCE:-delta}"
RESIDUAL_NORMALIZE="${RESIDUAL_NORMALIZE:-1}"
RESIDUAL_APPLY="${RESIDUAL_APPLY:-all}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

RESIDUAL_NORMALIZE_FLAG="--residual-normalize"
if [[ "${RESIDUAL_NORMALIZE}" == "0" ]]; then
  RESIDUAL_NORMALIZE_FLAG="--no-residual-normalize"
fi

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 nnm/experiments/exp12_hybrid_kv_residual_injection_tl.py \
  --dataset "${DATASET}" \
  --split test \
  --model "${MODEL}" \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --injection-layer "${INJECTION_LAYER}" \
  --max-queries "${MAX_QUERIES}" \
  --max-corpus "${MAX_CORPUS}" \
  --max-pairs "${MAX_PAIRS}" \
  --kv-alpha "${KV_ALPHA}" \
  --residual-alpha "${RESIDUAL_ALPHA}" \
  --residual-source "${RESIDUAL_SOURCE}" \
  ${RESIDUAL_NORMALIZE_FLAG} \
  --residual-apply "${RESIDUAL_APPLY}" \
  --progress-every "${PROGRESS_EVERY}"
