#!/usr/bin/env bash
set -euo pipefail

# Ready-to-run MVP pipeline for Qwen + Qdrant token memory.
#
# Usage:
#   bash nnm/scripts/run_qwen_qdrant_mvp.sh
#
# Optional env overrides:
#   MODEL="Qwen/Qwen2-1.5B"
#   RETRIEVAL_LAYER=26
#   INJECTION_LAYER=18
#   DATASET=scifact
#   MAX_ITEMS=2000
#   COLLECTION=nnm_token_memory_qwen2_1_5b_r26_i18
#   QDRANT_URL=http://localhost:6333
#   BATCH_SIZE=64
#   QDRANT_TIMEOUT=300
#   UPSERT_RETRIES=6
#   MIN_ENTRY_TOKEN_HITS=3
#   START_ITEM=0
#   RECREATE_COLLECTION=0
#   TRANSFORM_FILE=data/retrieval_transforms/Qwen_Qwen2-1.5B_r26_z1_w1.npz

MODEL="${MODEL:-Qwen/Qwen2-1.5B}"
RETRIEVAL_LAYER="${RETRIEVAL_LAYER:-26}"
INJECTION_LAYER="${INJECTION_LAYER:-18}"
DATASET="${DATASET:-scifact}"
MAX_ITEMS="${MAX_ITEMS:-2000}"
COLLECTION="${COLLECTION:-nnm_token_memory_qwen2_1_5b_r26_i18}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
BATCH_SIZE="${BATCH_SIZE:-64}"
QDRANT_TIMEOUT="${QDRANT_TIMEOUT:-300}"
UPSERT_RETRIES="${UPSERT_RETRIES:-6}"
MIN_ENTRY_TOKEN_HITS="${MIN_ENTRY_TOKEN_HITS:-3}"
START_ITEM="${START_ITEM:-0}"
RECREATE_COLLECTION="${RECREATE_COLLECTION:-0}"
TRANSFORM_FILE="${TRANSFORM_FILE:-data/retrieval_transforms/Qwen_Qwen2-1.5B_r${RETRIEVAL_LAYER}_z1_w1.npz}"

echo "[1/2] Ingesting BEIR corpus into Qdrant..."
RECREATE_FLAG=""
if [[ "${RECREATE_COLLECTION}" == "1" ]]; then
  RECREATE_FLAG="--recreate-collection"
fi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 nnm/scripts/ingest_ri_qdrant_tl.py \
  --model "${MODEL}" \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --retrieval-layer "${RETRIEVAL_LAYER}" \
  --injection-layer "${INJECTION_LAYER}" \
  --jsonl-file "data/beir_datasets/${DATASET}/corpus.jsonl" \
  --jsonl-id-field "_id" \
  --jsonl-title-field "title" \
  --jsonl-text-field "text" \
  --start-item "${START_ITEM}" \
  --max-items "${MAX_ITEMS}" \
  --qdrant-url "${QDRANT_URL}" \
  --qdrant-timeout "${QDRANT_TIMEOUT}" \
  --qdrant-prefer-grpc \
  --no-qdrant-check-compatibility \
  --retrieval-zscore \
  --retrieval-whitening \
  --retrieval-post-l2 \
  --retrieval-transform-file "${TRANSFORM_FILE}" \
  --fit-transform \
  --fit-transform-items 64 \
  --fit-transform-max-tokens 30000 \
  --fit-transform-eps 1e-5 \
  --collection "${COLLECTION}" \
  --on-disk-vectors \
  --batch-size "${BATCH_SIZE}" \
  --upsert-retries "${UPSERT_RETRIES}" \
  --retry-base-delay 2 \
  --progress-every 10 \
  --skip-existing-entry \
  ${RECREATE_FLAG}

echo "[2/2] Running a smoke query..."
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 nnm/scripts/search_ri_qdrant_tl.py \
  --model "${MODEL}" \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --retrieval-layer "${RETRIEVAL_LAYER}" \
  --query "What evidence exists about vaccines?" \
  --qdrant-url "${QDRANT_URL}" \
  --qdrant-timeout "${QDRANT_TIMEOUT}" \
  --qdrant-prefer-grpc \
  --no-qdrant-check-compatibility \
  --retrieval-transform-file "${TRANSFORM_FILE}" \
  --retrieval-post-l2 \
  --collection "${COLLECTION}" \
  --filter-model-id "${MODEL}" \
  --search-k 200 \
  --exclude-token-ids 25,6,7,8,9,10 \
  --group-by-entry \
  --entry-agg mean_top3 \
  --min-entry-token-hits "${MIN_ENTRY_TOKEN_HITS}" \
  --top-k 10

echo "Done."
