# Neural Native Memory (NNM)

Research project exploring training-free memory systems for LLMs using internal model representations instead of traditional RAG pipelines.

## Core Idea

Instead of encoding text into external embedding vectors (lossy), this approach stores and retrieves the LLM's own internal states -- KV-Cache entries, hidden states, delta vectors -- enabling lossless semantic storage and direct neural injection.

## Project History

This project evolved through three phases:

1. **KV-Embedding** (Jan 2026): Implementation of [arXiv:2601.01046](https://arxiv.org/abs/2601.01046) -- training-free text embedding via KV re-routing. Validated core hypotheses (NDCG@10 = 0.82 on BEIR Scifact with whitening). Code in `experiments/legacy/` and `src/legacy/`.

2. **ktransformers attempt** (Jan 2026): Brief exploration of ktransformers for KV-layer access. Abandoned in favor of TransformerLens.

3. **Neural Native Memory** (Feb 2026, current): Clean rewrite using TransformerLens for stable access to internal model states. Implements the full NNM pipeline: layer selection via TwoNN intrinsic dimensionality, token-level retrieval/injection vectors, Qdrant storage with z-score + whitening normalization.

## Key Results

| Finding | Detail |
|---------|--------|
| Whitening is essential | Raw LLM embeddings are highly anisotropic. PCA-Whitening doubles NDCG (0.40 -> 0.82) |
| 4-bit quantization works | Qwen3-VL-8B-Instruct at 4-bit produces valid embeddings |
| INT8 delta compression | 4x storage reduction with minimal quality loss |
| Separate retrieval/injection layers | Retrieval benefits from later layers, injection from earlier ones |

## Project Structure

```
neural-native-memory/
├── src/
│   ├── kvembed/              # Core KV-Embedding implementation (TransformerLens)
│   │   ├── config.py         # Model/layer configuration
│   │   ├── kv_cache_tl.py    # KV cache extraction
│   │   ├── layer_selection.py # TwoNN-based layer selection
│   │   ├── prompts.py        # Compression prompts
│   │   └── transformerlens_backend.py
│   ├── storage/
│   │   ├── qdrant_store.py   # Qdrant vector storage
│   │   └── retrieval_transform.py  # Z-score, whitening, L2 normalization
│   └── legacy/               # Phase 1 code (HuggingFace-based, reference only)
├── experiments/
│   ├── nnm/                  # Current experiments (TransformerLens-based)
│   │   ├── exp1-16           # Numbered experiments
│   │   └── common.py         # Shared experiment utilities
│   └── legacy/               # Phase 1 experiments (1-10, steering, debug)
├── scripts/                  # Runner scripts for Qdrant MVP
├── docs/
│   ├── concept-nnma.md       # Full architecture whitepaper
│   ├── plans/                # Design documents
│   └── research/             # Paper extracts (txt)
├── docker-compose.qdrant.yml # Local Qdrant instance
├── requirements.txt
└── PLANNING.md
```

## Setup

### Dependencies

```bash
pip install transformerlens qdrant-client torch numpy
```

For GPU (recommended):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Qdrant (for storage experiments)

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

### External repos (clone separately if needed)

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) -- used for model internals access
- [llama.cpp](https://github.com/ggerganov/llama.cpp) -- used in Phase 1 for GGUF KV-layer experiments

## Quick Start

```bash
# Run a basic KV-Embedding experiment
python scripts/run_kvembed_tl.py

# Ingest into Qdrant
python scripts/ingest_ri_qdrant_tl.py \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --retrieval-layer 26 \
  --injection-layer 18 \
  --text "Some text to store"

# Search
python scripts/search_ri_qdrant_tl.py \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --retrieval-layer 26 \
  --query "search query"
```

See `docs/nnm-quickstart.md` for detailed usage and BEIR evaluation instructions.

## References

- [KV-Embedding: Training-free Text Embedding via Internal KV Re-routing](https://arxiv.org/abs/2601.01046)
- [Deconstructed Vectors concept](docs/concept-nnma.md) -- full NNMA architecture vision
