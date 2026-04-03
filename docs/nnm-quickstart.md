# Neural Native Memory (NNM)

This is a clean project area for building the paper-near KV-Embedding pipeline
as the foundation for Neural Native Memory.

## Focus (Phase 1)

Implement KV-Embedding as close to the paper as possible:

1. Compression-oriented prompt
2. Automated layer selection via TwoNN intrinsic dimensionality
3. KV re-routing using the last token as a virtual prefix
4. Prefix attention bias (`b = 1.0`)
5. Pooling: `Normalize((Last + Mean) / 2)`

TransformerLens is used for stable access to K/V and hidden states.

## Quick Start (local)

Example script:

```bash
python nnm/scripts/run_kvembed_tl.py
```

You can point it to a small corpus for layer selection:

```bash
python nnm/scripts/run_kvembed_tl.py --id-corpus path/to/texts.txt
```

## Structure

- `nnm/kvembed/`: core KV-Embedding implementation
- `nnm/scripts/`: small entrypoints for running experiments
- `nnm/storage/`: storage backends (Qdrant)

## Qdrant MVP (R_t + I_t)

Start local Qdrant:

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

If you already had an older Qdrant image running, recreate it once:

```bash
docker compose -f docker-compose.qdrant.yml down
docker compose -f docker-compose.qdrant.yml up -d --force-recreate
```

Install client dependency (once):

```bash
uv pip install --python .venv_nnm/bin/python qdrant-client
```

Ingest token-level retrieval/injection vectors:

```bash
python nnm/scripts/ingest_ri_qdrant_tl.py \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --retrieval-layer 26 \
  --injection-layer 18 \
  --text "Die Muellabfuhr ist jeden Dienstag um 7 Uhr." \
  --collection nnm_token_memory \
  --on-disk-vectors
```

Ingest directly from BEIR JSONL (keeps `_id` as `entity_id`):

```bash
python nnm/scripts/ingest_ri_qdrant_tl.py \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --retrieval-layer 26 \
  --injection-layer 18 \
  --jsonl-file data/beir_datasets/scifact/corpus.jsonl \
  --jsonl-id-field _id \
  --jsonl-title-field title \
  --jsonl-text-field text \
  --max-items 2000 \
  --qdrant-timeout 300 \
  --qdrant-prefer-grpc \
  --no-qdrant-check-compatibility \
  --retrieval-zscore \
  --retrieval-whitening \
  --retrieval-post-l2 \
  --retrieval-transform-file data/retrieval_transforms/Qwen_Qwen2-1.5B_r26_z1_w1.npz \
  --fit-transform \
  --fit-transform-items 64 \
  --fit-transform-max-tokens 30000 \
  --collection nnm_token_memory_qwen2_1_5b_r26_i18 \
  --batch-size 64 \
  --upsert-retries 6 \
  --retry-base-delay 2 \
  --on-disk-vectors \
  --progress-every 10
```

Resume-friendly options:

```bash
# Skip first N source items (0-based)
--start-item 50

# Skip already ingested deterministic entries
--skip-existing-entry
```

Search retrieval vectors:

```bash
python nnm/scripts/search_ri_qdrant_tl.py \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --retrieval-layer 26 \
  --query "Wann ist Muellabfuhr?" \
  --retrieval-transform-file data/retrieval_transforms/Qwen_Qwen2-1.5B_r26_z1_w1.npz \
  --retrieval-post-l2 \
  --collection nnm_token_memory \
  --search-k 200 \
  --exclude-token-ids 25,6,7,8,9,10 \
  --group-by-entry \
  --entry-agg mean_top3 \
  --min-entry-token-hits 3 \
  --top-k 10
```

Compare baseline vs z-score+whitening on the same BEIR query set:

```bash
# 1) Baseline ingest (no z-score / no whitening)
python nnm/scripts/ingest_ri_qdrant_tl.py \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --retrieval-layer 26 \
  --injection-layer 18 \
  --jsonl-file data/beir_datasets/scifact/corpus.jsonl \
  --jsonl-id-field _id \
  --jsonl-title-field title \
  --jsonl-text-field text \
  --max-items 2000 \
  --collection nnm_token_memory_qwen2_1_5b_r26_i18_baseline \
  --no-retrieval-zscore \
  --no-retrieval-whitening \
  --retrieval-post-l2 \
  --batch-size 64 \
  --on-disk-vectors

# 2) Transformed ingest (z-score + whitening)
python nnm/scripts/ingest_ri_qdrant_tl.py \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --retrieval-layer 26 \
  --injection-layer 18 \
  --jsonl-file data/beir_datasets/scifact/corpus.jsonl \
  --jsonl-id-field _id \
  --jsonl-title-field title \
  --jsonl-text-field text \
  --max-items 2000 \
  --collection nnm_token_memory_qwen2_1_5b_r26_i18 \
  --retrieval-zscore \
  --retrieval-whitening \
  --retrieval-post-l2 \
  --retrieval-transform-file data/retrieval_transforms/Qwen_Qwen2-1.5B_r26_z1_w1.npz \
  --fit-transform \
  --batch-size 64 \
  --on-disk-vectors

# 3) Compare both collections with BEIR metrics (same query ids)
python nnm/scripts/eval_qdrant_beir_compare_tl.py \
  --dataset scifact \
  --split test \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --retrieval-layer 26 \
  --baseline-collection nnm_token_memory_qwen2_1_5b_r26_i18_baseline \
  --transformed-collection nnm_token_memory_qwen2_1_5b_r26_i18 \
  --transformed-transform-file data/retrieval_transforms/Qwen_Qwen2-1.5B_r26_z1_w1.npz \
  --top-k 10 \
  --search-k 200 \
  --max-queries 100 \
  --exclude-token-ids 25,6,7,8,9,10 \
  --group-by-entry \
  --entry-agg mean_top3 \
  --min-entry-token-hits 3

# Optional tuning sweep in the same run:
#   --sweep-min-entry-token-hits 1,2,3 \
#   --sweep-entry-agg mean_top3,max
```

Shortcut runner (expects both collections to exist):

```bash
bash nnm/scripts/run_qwen_qdrant_eval_compare.sh
```

Ready-made Qwen run script:

```bash
bash nnm/scripts/run_qwen_qdrant_mvp.sh
```

Resume example after interruption at item 50:

```bash
START_ITEM=50 RECREATE_COLLECTION=0 bash nnm/scripts/run_qwen_qdrant_mvp.sh
```
