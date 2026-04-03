# KV-Embedding Prototype – Design Document

+doc:planning +project:kv-llm-vectorstore

## Overview

Experimental prototype to validate three key hypotheses for efficient LLM-based text embeddings:

1. KV-Embedding method produces high-quality contextual embeddings (full document context)
2. Token-IDs can be deterministically mapped to/from static embeddings
3. Contextual embeddings can be stored as Token-IDs + Delta vectors with aggressive compression

## Goals

Primary goal: Validate the three hypotheses through isolated, reproducible experiments.

Secondary goal: If validated, quantify compression ratios and semantic preservation for vector database storage.

## Architecture

### Project Structure

```
kv_llm_vectorstore/
├── experiments/
│   ├── 1_test_contextual_embeddings.py
│   ├── 2_test_token_decode.py
│   └── 3_test_delta_storage.py
├── lib/
│   ├── model_loader.py
│   ├── embedding_utils.py
│   ├── token_utils.py
│   ├── compression.py
│   └── io_utils.py
├── data/
│   ├── synthetic/
│   └── results/
├── docs/
│   └── plans/
└── requirements.txt
```

### Design Principles

- Model-agnostic: Use transformers library for maximum flexibility
- Linear progression: Three independent scripts, each validates one hypothesis
- Simple persistence: JSON files for results, Python logging for execution traces
- Progressive data complexity: Synthetic → Real documents → BEIR benchmark

## Experiment 1: KV-Embedding Implementation

### Objective

Implement the KV-Embedding method from the paper to generate token-level contextual embeddings.

### Method

Based on paper "KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs" (arXiv:2601.01046v1).

**Core mechanism:**
1. Wrap input with compression prompt: `"{Context}: {text}" Compress the Context in one word:`
2. Forward pass with KV manipulation in selected layers:
   - Extract KV states of final token: `(k_n, v_n)`
   - Prepend as virtual prefix: `K̃ = [k_n || K]`, `Ṽ = [v_n || V]`
   - All tokens can now attend to global sequence summary
3. Extract token-level embeddings from final layer
4. Hybrid pooling: `(last_token + mean_pooling) / 2`

**Layer selection:**
- Intrinsic Dimensionality (TwoNN estimator) identifies layers with maximal semantic compression
- Avoids early layers (surface features) and late layers (prediction bias)

### Input Data

Start with 5-10 synthetic texts (50-500 tokens each) in `data/synthetic/`.

Characteristics: Varied semantic content, manually written for full control during debugging.

### Output Format

JSON per document in `data/results/exp1_<timestamp>/`:

```json
{
  "doc_id": "test_001",
  "text": "...",
  "token_ids": [123, 456, 789],
  "token_embeddings_contextualized": [[...], [...], [...]],
  "pooled_embedding": [...],
  "model": "Qwen/Qwen2-1.5B",
  "selected_layers": [12, 13, 14, 15]
}
```

### Validation

Cosine similarity between semantically similar documents should be high.

Log similarity matrix for all document pairs.

### Success Criterion

Semantic similarity measurable and plausible (similar docs > 0.7 cosine, dissimilar < 0.4).

## Experiment 2: Token-ID-based Encoding/Decoding

### Objective

Test whether text can be deterministically reconstructed from Token-IDs using static embeddings.

### Hypothesis

Static embeddings (from embedding layer, before any attention) contain sufficient information to reverse-map to Token-IDs via nearest-neighbor search.

### Method

1. Tokenize text → obtain Token-IDs
2. Extract static embeddings (embedding layer output, layer 0)
3. Decoding test:
   - For each static embedding: Nearest-neighbor search in model's embedding matrix
   - Compare reconstructed Token-ID with original
4. Compute accuracy: `correct_mappings / total_tokens`

### Output Format

JSON per document in `data/results/exp2_<timestamp>/`:

```json
{
  "doc_id": "test_001",
  "token_ids_original": [123, 456, 789],
  "token_ids_reconstructed": [123, 456, 789],
  "static_embeddings": [[...], [...], [...]],
  "accuracy": 0.98,
  "mismatches": [
    {"position": 5, "original": 789, "reconstructed": 790}
  ]
}
```

### Validation

Manual inspection of mismatches. Check if mismatches are semantically similar tokens.

### Success Criterion

Reconstruction accuracy > 95%.

## Experiment 3: Dense Vector Delta Storage

### Objective

Validate that contextual embeddings can be stored as `static_embedding + delta`, and quantify best compression method for deltas.

### Part 3a: Delta Theory Validation

**Hypothesis:**
```
contextual_embedding = static_embedding + delta
```

**Method:**
1. Compute both embeddings per token:
   - `static_emb` from Experiment 2
   - `contextual_emb` from Experiment 1
2. Calculate delta: `delta = contextual_emb - static_emb`
3. Test reconstruction: `reconstructed = static_emb + delta`
4. Measure: `||reconstructed - contextual_emb||_2`

**Expected result:** Reconstruction error ≈ 0 (within numerical precision).

### Part 3b: Delta Compression

**Objective:** Find best compression method for deltas that preserves semantic quality.

**Compression methods:**

1. Scalar Quantization:
   - FP16 → INT8 → INT4 → INT2
   - Range-based: `quantized = round((delta - min) / (max - min) * (2^bits - 1))`

2. Vector Quantization:
   - Product Quantization (PQ): Split vector into sub-vectors, codebook per sub-vector
   - Residual Vector Quantization (RVQ): Iteratively quantize residuals

3. Extreme compression:
   - SimHash / Signed Random Projections: Binary hashes (sign only)

**Metrics:**

1. Reconstruction error:
   - L2 norm: `||reconstructed - original||_2`
   - Cosine similarity: `cos(reconstructed, original)`

2. Compression ratio: `original_size / compressed_size`

3. Semantic preservation:
   - Retrieval test: Top-k similar documents with compressed vs. uncompressed embeddings
   - Recall@k: How many correct neighbors are still found?

**Workflow:**
```
for method in [fp16, int8, int4, int2, pq, rvq, simhash]:
    compressed = compress(deltas, method)
    reconstructed = static_embeddings + decompress(compressed)

    metrics = {
        "l2_error": compute_l2(reconstructed, contextual_embeddings),
        "cosine_sim": compute_cosine(reconstructed, contextual_embeddings),
        "compression_ratio": original_size / compressed_size,
        "recall@10": retrieval_test(reconstructed, ground_truth)
    }
```

### Output Format

JSON in `data/results/exp3_<timestamp>/`:

```json
{
  "doc_id": "test_001",
  "token_ids": [123, 456, 789],
  "deltas_full_precision": [[...], [...], [...]],
  "reconstruction_error_baseline": 1.2e-7,
  "delta_sparsity": 0.15,
  "compression_results": [
    {
      "method": "int8",
      "compression_ratio": 2.0,
      "reconstruction_l2": 0.034,
      "reconstruction_cosine": 0.9987,
      "retrieval_recall@10": 0.98,
      "storage_kb": 31.6
    },
    {
      "method": "int4",
      "compression_ratio": 4.0,
      "reconstruction_l2": 0.089,
      "reconstruction_cosine": 0.9921,
      "retrieval_recall@10": 0.92,
      "storage_kb": 15.8
    }
  ]
}
```

### Validation

Decision criterion: Best ratio of compression_ratio to semantic preservation.

Target: Recall@10 > 0.95 with highest possible compression ratio.

### Success Criterion

- Part 3a: Reconstruction error ≈ 0
- Part 3b: Compression ratio > 2x while maintaining Recall@10 > 0.95

## Implementation Details

### Model Setup

Model-agnostic design using HuggingFace transformers:

- Load any decoder-only LLM via model name string
- Tested with: Qwen/Qwen2-1.5B (paper used Qwen3-4B)
- Later migration to llama.cpp planned for larger models with offloading

### Shared Libraries

**lib/model_loader.py:**
- Function: `load_model(model_name: str) -> (model, tokenizer)`
- Handle device placement (CUDA/CPU)

**lib/embedding_utils.py:**
- KV-routing implementation
- Intrinsic Dimensionality computation (TwoNN)
- Layer selection logic
- Hybrid pooling

**lib/token_utils.py:**
- Static embedding extraction
- Nearest-neighbor Token-ID search
- Token-ID ↔ embedding mapping

**lib/compression.py:**
- Scalar quantization (INT8/4/2)
- Product Quantization
- Residual Vector Quantization
- SimHash/binary projections

**lib/io_utils.py:**
- JSON read/write
- Logging setup
- Result file organization

### Logging

Python `logging` module for all scripts:
- Console output: INFO level
- File output: `data/results/<experiment>_<timestamp>.log` (DEBUG level)

### Data Progression

1. Phase 1 (Experiment validation): 5-10 synthetic texts
2. Phase 2 (Real-world test): Small set of real documents (PDFs, papers, markdown)
3. Phase 3 (Benchmark): BEIR dataset subset

## Vector Database Compatibility

Compression method affects storage strategy:

- INT8/INT4: Directly supported by most vector DBs (Qdrant, Milvus, Weaviate)
- PQ/RVQ: Native support in Faiss, Milvus
- Binary hashes: Separate index structure required

Final decision on compression method determines vector DB selection.

## Next Steps After Design Validation

If all three hypotheses are confirmed:

1. Scale to real documents and BEIR benchmark
2. Implement vector database integration
3. Build retrieval pipeline: Query → KV-Embedding → Delta reconstruction → Search
4. Performance optimization (batching, caching, GPU utilization)
5. Potential migration to llama.cpp for production inference

## Open Questions

None at this stage. All core decisions made during brainstorming.

## References

- Paper: "KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs" (arXiv:2601.01046v1)
- BEIR benchmark for retrieval evaluation
- TwoNN estimator for Intrinsic Dimensionality (Facco et al., 2017)

## Tags

#machine-learning #embeddings #llm #transformers #vector-compression #information-retrieval
