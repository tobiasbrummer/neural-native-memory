#!/usr/bin/env bash
set -euo pipefail

# Compare baseline vs transformed retrieval quality on BEIR.
#
# Expects both collections to be ingested already.
#
# Usage:
#   bash nnm/scripts/run_qwen_qdrant_eval_compare.sh
#
# Optional env overrides:
#   MODEL="Qwen/Qwen2-1.5B"
#   RETRIEVAL_LAYER=26
#   INJECTION_LAYER=18
#   DATASET=scifact
#   MAX_QUERIES=100
#   TOP_K=10
#   SEARCH_K=200
#   BASELINE_COLLECTION=nnm_token_memory_qwen2_1_5b_r26_i18_baseline
#   TRANSFORMED_COLLECTION=nnm_token_memory_qwen2_1_5b_r26_i18
#   TRANSFORM_FILE=data/retrieval_transforms/Qwen_Qwen2-1.5B_r26_z1_w1.npz
#   QDRANT_URL=http://localhost:6333
#   QDRANT_TIMEOUT=300
#   MIN_ENTRY_TOKEN_HITS=3
#   SWEEP_MIN_ENTRY_TOKEN_HITS=1,2,3
#   SWEEP_ENTRY_AGG=mean_top3,max
#   EXCLUDE_TOKEN_IDS=25,6,7,8,9,10

MODEL="${MODEL:-Qwen/Qwen2-1.5B}"
RETRIEVAL_LAYER="${RETRIEVAL_LAYER:-26}"
INJECTION_LAYER="${INJECTION_LAYER:-18}"
DATASET="${DATASET:-scifact}"
MAX_QUERIES="${MAX_QUERIES:-100}"
TOP_K="${TOP_K:-10}"
SEARCH_K="${SEARCH_K:-200}"
BASELINE_COLLECTION="${BASELINE_COLLECTION:-nnm_token_memory_qwen2_1_5b_r${RETRIEVAL_LAYER}_i${INJECTION_LAYER}_baseline}"
TRANSFORMED_COLLECTION="${TRANSFORMED_COLLECTION:-nnm_token_memory_qwen2_1_5b_r${RETRIEVAL_LAYER}_i${INJECTION_LAYER}}"
TRANSFORM_FILE="${TRANSFORM_FILE:-data/retrieval_transforms/Qwen_Qwen2-1.5B_r${RETRIEVAL_LAYER}_z1_w1.npz}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_TIMEOUT="${QDRANT_TIMEOUT:-300}"
MIN_ENTRY_TOKEN_HITS="${MIN_ENTRY_TOKEN_HITS:-3}"
SWEEP_MIN_ENTRY_TOKEN_HITS="${SWEEP_MIN_ENTRY_TOKEN_HITS:-}"
SWEEP_ENTRY_AGG="${SWEEP_ENTRY_AGG:-}"
EXCLUDE_TOKEN_IDS="${EXCLUDE_TOKEN_IDS:-25,6,7,8,9,10}"
SEED="${SEED:-42}"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 nnm/scripts/eval_qdrant_beir_compare_tl.py \
  --dataset "${DATASET}" \
  --split test \
  --model "${MODEL}" \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --retrieval-layer "${RETRIEVAL_LAYER}" \
  --baseline-collection "${BASELINE_COLLECTION}" \
  --transformed-collection "${TRANSFORMED_COLLECTION}" \
  --transformed-transform-file "${TRANSFORM_FILE}" \
  --baseline-post-l2 \
  --transformed-post-l2 \
  --top-k "${TOP_K}" \
  --search-k "${SEARCH_K}" \
  --max-queries "${MAX_QUERIES}" \
  --seed "${SEED}" \
  --qdrant-url "${QDRANT_URL}" \
  --qdrant-timeout "${QDRANT_TIMEOUT}" \
  --qdrant-prefer-grpc \
  --no-qdrant-check-compatibility \
  --filter-model-id "${MODEL}" \
  --exclude-token-ids "${EXCLUDE_TOKEN_IDS}" \
  --exclude-special-token-ids \
  --group-by-entry \
  --entry-agg mean_top3 \
  --min-entry-token-hits "${MIN_ENTRY_TOKEN_HITS}" \
  --sweep-min-entry-token-hits "${SWEEP_MIN_ENTRY_TOKEN_HITS}" \
  --sweep-entry-agg "${SWEEP_ENTRY_AGG}" \
  --progress-every 10
