# Neural Native Memory (NNM)

Research project exploring training-free memory systems for LLMs using internal model representations instead of traditional RAG pipelines.

## Core Idea

Instead of encoding text into external embedding vectors (lossy), this approach stores and retrieves the LLM's own internal states -- KV-Cache entries, hidden states, delta vectors -- enabling lossless semantic storage and direct neural injection.

## Project History

This project evolved through three phases:

1. **Phase 1 -- KV-Embedding** (Jan 2026): Implementation of [arXiv:2601.01046](https://arxiv.org/abs/2601.01046) -- training-free text embedding via KV re-routing. Validated core hypotheses on BEIR SciFact. Code in `experiments/legacy/` and `src/legacy/`. Saved BEIR result JSONs in [`results/legacy_phase1_beir/`](results/legacy_phase1_beir/).

2. **ktransformers attempt** (Jan 2026): Brief exploration of ktransformers for KV-layer access. Abandoned in favor of TransformerLens.

3. **Phase 3 -- Neural Native Memory** (Feb-Apr 2026, current): Clean rewrite using TransformerLens for stable access to internal model states. Implements the full NNM pipeline: layer selection via TwoNN intrinsic dimensionality, token-level retrieval/injection vectors, Qdrant storage with z-score + whitening normalization. End-to-end gate evaluations (TriviaQA, needle-in-haystack) in [`results/phase3_e2e_gate/`](results/phase3_e2e_gate/).

## Key Results

All numbers below map to a saved JSON artifact under [`results/`](results/) -- see [`results/README.md`](results/README.md) for the full mapping.

| Finding | Detail | Source |
|---------|--------|--------|
| Whitening + z-score normalization is essential (Phase 1, BEIR SciFact, Qwen3-VL-8B-Instruct 4bit) | NDCG@10: 0.397 without -> 0.821 with PCA-whitening + z-score | `results/legacy_phase1_beir/exp4_beir_20260121_{231316,232031}/beir_results.json` |
| 4-bit quantization preserves embedding quality (Phase 1, same as above) | Qwen3-VL-8B-Instruct at 4-bit reaches NDCG@10 = 0.821 on SciFact | `results/legacy_phase1_beir/exp4_beir_20260121_232031/beir_results.json` |
| INT8 delta-storage trades quality for footprint | 4x storage reduction; SciFact NDCG@10 0.821 -> 0.556 (~32% relative drop). Not "lossless" -- useful when storage dominates and the consumer can tolerate the recall loss. | `results/legacy_phase1_beir/exp7_beir_storage_20260123_213956/beir_storage_results.json` |
| TriviaQA end-to-end gate (Phase 3, Qwen2.5-7B-Instruct, n=100 x 3 seeds) | nnma EM = 0.76, rag = 0.72, cold = 0.52, random = 0.50. Random-injection control sits at cold floor -- recall is not coming from prompt structure. | `results/phase3_e2e_gate/nnm_exp17c_20260425_184841/summary.json` |
| Needle-in-haystack (Phase 3, n=50, 2.5-8k char articles) | nnma needle_recall = 1.00 vs rag = 0.86; paired McNemar p = 0.0156 | `results/phase3_e2e_gate/nnm_exp17g_20260426_130615/summary.json` |
| Separate retrieval/injection layers | Retrieval benefits from later layers, injection from earlier ones | -- (architecture, no single benchmark file) |

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
pip install transformer_lens qdrant-client torch numpy
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

The quickstart commands below run on **Qwen2-1.5B** as a cheap smoke-test of the
Phase-3 pipeline. The Phase-1 BEIR headline numbers in the Key Results table
were produced on **Qwen3-VL-8B-Instruct (4-bit)** -- see
`experiments/legacy/4_benchmark_beir.py --help` for the full eval command.

```bash
# Phase-3 smoke (Qwen2-1.5B, fast on a single GPU)
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

# Phase-1 BEIR headline reproduction (Qwen3-VL-8B-Instruct, 4bit)
python experiments/legacy/4_benchmark_beir.py \
  --dataset scifact \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --load_in_4bit --whitening --zscore \
  --prompt '"{context}" Compress the Context in one word:'
```

See `docs/nnm-quickstart.md` for detailed usage and BEIR evaluation instructions.

## References

- [KV-Embedding: Training-free Text Embedding via Internal KV Re-routing](https://arxiv.org/abs/2601.01046)
- [Deconstructed Vectors concept](docs/concept-nnma.md) -- full NNMA architecture vision
