#!/usr/bin/env bash
set -euo pipefail

# Exp13 qualitative residual steering demo.
#
# Usage:
#   bash nnm/scripts/run_qwen_exp13_residual_steering_qual.sh
#
# Optional env overrides:
#   MODEL="Qwen/Qwen2-1.5B"
#   STEERING_LAYER=18
#   PROFILES="sleepy,joy,memory_hint"
#   ALPHA=1.5
#   SOURCE=delta              # delta|full
#   NORMALIZE=1               # 1|0
#   APPLY=last                # last|all
#   MAX_NEW_TOKENS=96
#   MIN_NEW_TOKENS=8
#   PROMPT_STYLE=plain        # plain|paper
#   PROMPTS_FILE=path/to/prompts.txt
#   DO_SAMPLE=1               # 1|0
#   TEMPERATURE=0.8
#   TOP_P=0.9
#   REPETITION_PENALTY=1.10
#   NO_REPEAT_NGRAM_SIZE=3

MODEL="${MODEL:-Qwen/Qwen2-1.5B}"
STEERING_LAYER="${STEERING_LAYER:-18}"
PROFILES="${PROFILES:-sleepy,joy,memory_hint}"
ALPHA="${ALPHA:-1.5}"
SOURCE="${SOURCE:-delta}"
NORMALIZE="${NORMALIZE:-1}"
APPLY="${APPLY:-last}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-8}"
PROMPT_STYLE="${PROMPT_STYLE:-plain}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
DO_SAMPLE="${DO_SAMPLE:-1}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.10}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-3}"

NORM_FLAG="--normalize"
if [[ "${NORMALIZE}" == "0" ]]; then
  NORM_FLAG="--no-normalize"
fi

PROMPTS_FILE_ARGS=()
if [[ -n "${PROMPTS_FILE}" ]]; then
  PROMPTS_FILE_ARGS=(--prompts-file "${PROMPTS_FILE}")
fi

SAMPLE_FLAG="--do-sample"
if [[ "${DO_SAMPLE}" == "0" ]]; then
  SAMPLE_FLAG="--no-do-sample"
fi

python3 nnm/experiments/exp13_residual_steering_qualitative_tl.py \
  --model "${MODEL}" \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --steering-layer "${STEERING_LAYER}" \
  --profiles "${PROFILES}" \
  --alpha "${ALPHA}" \
  --source "${SOURCE}" \
  ${NORM_FLAG} \
  --apply "${APPLY}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --min-new-tokens "${MIN_NEW_TOKENS}" \
  ${SAMPLE_FLAG} \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --repetition-penalty "${REPETITION_PENALTY}" \
  --no-repeat-ngram-size "${NO_REPEAT_NGRAM_SIZE}" \
  --prompt-style "${PROMPT_STYLE}" \
  "${PROMPTS_FILE_ARGS[@]}"
