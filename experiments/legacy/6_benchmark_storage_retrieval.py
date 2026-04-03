#!/usr/bin/env python3
"""
Experiment 6: Storage & Retrieval Benchmark
Objective: Compare storage size and runtime latency between:
  Method A: Full FP32 Contextual Embeddings
  Method B: Token IDs + INT8 Deltas (Proposed)

Metrics:
  - Storage Size (MB)
  - Retrieval Latency (ms/token) - Simulated (Whitening)
  - Reconstruction Latency (ms/token) - Simulated (Dequantize + Project)
"""

import sys
import time
import json
import logging
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Project imports
from lib.io_utils import create_results_dir, load_test_documents, setup_logging, save_json
from lib.model_loader import load_model, get_device
from lib.embedding_utils import KVEmbeddingExtractor
from lib.token_utils import extract_static_embeddings
from lib.compression import scalar_quantize, scalar_dequantize
from lib.virtual_prefix import project_hidden_to_kv

# =============================================================================
# Benchmarking Utils
# =============================================================================

@dataclass
class BenchmarkResult:
    method: str
    storage_size_mb: float
    retrieval_latency_ms: float
    reconstruction_latency_ms: float
    total_tokens: int

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start

# =============================================================================
# Core Benchmark Logic
# =============================================================================

def simulate_retrieval(
    deltas: np.ndarray,
    whitening_matrix: np.ndarray,
    whitening_mean: np.ndarray,
    n_iters: int = 10
) -> float:
    """
    Simulate retrieval latency: Load -> (Dequantize if needed) -> Whiten -> Dot Product
    For this benchmark, we assume data is already in memory (measuring compute bound).
    """
    # Simulate whitening transform: (x - mu) @ W
    # We do a batch matrix multiplication to simulate processing a chunk of tokens
    
    start_time = time.perf_counter()
    for _ in range(n_iters):
        # 1. Center
        centered = deltas - whitening_mean
        # 2. Transform (Whiten)
        whitened = centered @ whitening_matrix
        # 3. Similarity (Dot product with some query vector - simulated)
        _ = whitened @ np.ones((whitened.shape[1],), dtype=np.float32)
        
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / n_iters
    return avg_time * 1000  # ms

def simulate_reconstruction_baseline(
    embeddings: np.ndarray,
    layer_idx: int,
    model,
    n_iters: int = 10
) -> float:
    """
    Baseline Reconstruction: H -> Project -> K,V
    """
    start_time = time.perf_counter()
    for _ in range(n_iters):
        # 1. Project to K, V (using our lib function logic)
        # We simulate the operations: K = H @ W_k, V = H @ W_v
        # Note: project_hidden_to_kv includes torch conversion overhead, we want pure compute mostly
        # but let's stick closer to reality and use helper if possible or simulate math.
        
        # Simulating just the math to avoid torch overhead dominance in this micro-benchmark
        # unless we want to measure torch overhead too.
        # Let's do raw numpy matmul to be fair to "compute" part, or torch if we use torch for proposed.
        # Let's use the actual project_hidden_to_kv function data flow simulation
        
        # H is [seq, hidden]
        # K = H @ W_k.T
        # V = H @ W_v.T
        # For simulation, just random weights
        hidden_dim = embeddings.shape[1]
        dummy_W = np.eye(hidden_dim, dtype=np.float32)
        
        k = embeddings @ dummy_W
        v = embeddings @ dummy_W
        
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / n_iters
    return avg_time * 1000 # ms

def simulate_reconstruction_proposed(
    deltas_int8: np.ndarray,
    quant_params: Dict,
    static_embeddings: np.ndarray,
    original_shape: Tuple,
    n_iters: int = 10
) -> float:
    """
    Proposed Reconstruction: Int8 Delta -> Dequantize -> Add Static -> Project
    """
    start_time = time.perf_counter()
    for _ in range(n_iters):
        # 1. Dequantize
        # scalar_dequantize involves: (q * scale) + min
        deltas_float = scalar_dequantize(deltas_int8, quant_params, original_shape)
        
        # 2. Add Static
        reconstructed_H = static_embeddings + deltas_float
        
        # 3. Project (same dummy math as baseline)
        hidden_dim = reconstructed_H.shape[1]
        dummy_W = np.eye(hidden_dim, dtype=np.float32)
        
        k = reconstructed_H @ dummy_W
        v = reconstructed_H @ dummy_W
        
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / n_iters
    return avg_time * 1000 # ms


def main():
    # Setup
    results_dir = create_results_dir("exp6")
    logger = setup_logging("exp6_benchmark", results_dir)
    logger.info("Starting Storage & Retrieval Benchmark")
    
    # 1. Load Data
    documents = load_test_documents()
    logger.info(f"Loaded {len(documents)} documents")
    model, tokenizer = load_model()
    
    # Extract Embeddings
    logger.info("Extracting embeddings...")
    texts = [doc["text"] for doc in documents]
    kv_extractor = KVEmbeddingExtractor(model, tokenizer)
    contextual_data = kv_extractor.extract_embeddings(texts, return_token_embeddings=True)
    
    # Extract Static Embeddings (for delta computation)
    static_data = extract_static_embeddings(model, tokenizer, texts, use_prompt=True)
    
    # Collect all tokens into one big array for benchmarking
    all_contextual = []
    all_static = []
    all_tokens = []
    
    for i in range(len(documents)):
        c_emb = contextual_data["token_embeddings"][i]
        s_emb = static_data["static_embeddings"][i]
        t_ids = static_data["token_ids"][i]
        
        # Length check logic from exp3
        min_len = min(len(c_emb), len(s_emb))
        c_emb = c_emb[:min_len]
        s_emb = s_emb[:min_len]
        t_ids = t_ids[:min_len]
        
        all_contextual.append(c_emb)
        all_static.append(s_emb)
        all_tokens.append(t_ids)
        
    # Flat arrays
    flat_contextual = np.vstack(all_contextual).astype(np.float32)
    flat_static = np.vstack(all_static).astype(np.float32)
    flat_tokens = np.concatenate(all_tokens).astype(np.int32)
    flat_deltas = flat_contextual - flat_static
    
    n_tokens, hidden_dim = flat_contextual.shape
    logger.info(f"Total tokens: {n_tokens}, Hidden Dim: {hidden_dim}")
    
    # =========================================================================
    # Method A: Baseline (Full FP32)
    # =========================================================================
    logger.info("\n--- Method A: Baseline (Full FP32) ---")
    
    # 1. Storage Size
    # We save to uncompressed .npz to mimic "Store on disk"
    baseline_path = results_dir / "baseline.npz"
    np.savez(baseline_path, embeddings=flat_contextual)
    baseline_size_mb = baseline_path.stat().st_size / (1024 * 1024)
    logger.info(f"Storage Size: {baseline_size_mb:.2f} MB")
    
    # 2. Latency
    # For baseline search, we assume we assume whitening on full vectors
    # W parameters - random for benchmark
    W = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
    mu = np.zeros((hidden_dim,), dtype=np.float32)
    
    baseline_retrieval_ms = simulate_retrieval(flat_contextual, W, mu)
    baseline_recon_ms = simulate_reconstruction_baseline(flat_contextual, 0, model)
    
    logger.info(f"Retrieval Latency (batch): {baseline_retrieval_ms:.2f} ms")
    logger.info(f"Reconstruction Latency (batch): {baseline_recon_ms:.2f} ms")
    
    
    # =========================================================================
    # Method B: Proposed (Token IDs + INT8 Deltas)
    # =========================================================================
    logger.info("\n--- Method B: Proposed (TokenID + INT8 Deltas) ---")
    
    # 1. Quantization
    q_start = time.perf_counter()
    deltas_int8, quant_params = scalar_quantize(flat_deltas, bits=8)
    q_time = time.perf_counter() - q_start
    logger.info(f"Quantization logic overhead: {q_time*1000:.2f} ms")
    
    # 2. Storage Size
    proposed_path = results_dir / "proposed.npz"
    # Save: token_ids, int8_deltas, quant_params (scale/min per feature or scalar?)
    # Our scalar_quantize returns min/step vectors.
    np.savez(
        proposed_path,
        token_ids=flat_tokens,
        deltas=deltas_int8,
        q_min=quant_params.min_val,
        q_max=quant_params.max_val
    )
    proposed_size_mb = proposed_path.stat().st_size / (1024 * 1024)
    logger.info(f"Storage Size: {proposed_size_mb:.2f} MB")
    
    # 3. Latency
    # Retrieval: Dequantize -> Whiten -> Dot
    # NOTE: In reality, we could do "Whiten on INT8" directly if we merge transforms,
    # but for "Lazy" proposed plan: Dequant -> Whiten.
    
    # Helper to measure dequant + whiten
    def simulate_proposed_retrieval_full_pipeline(iters=10):
        start = time.perf_counter()
        for _ in range(iters):
            # Dequant
            d_float = scalar_dequantize(deltas_int8, quant_params, flat_deltas.shape)
            # Whiten
            centered = d_float - mu
            whitened = centered @ W
            # Dot
            _ = whitened @ np.ones((whitened.shape[1],), dtype=np.float32)
        avg = (time.perf_counter() - start) / iters
        return avg * 1000

    proposed_retrieval_ms = simulate_proposed_retrieval_full_pipeline()
    proposed_recon_ms = simulate_reconstruction_proposed(
        deltas_int8, quant_params, flat_static, flat_deltas.shape
    )
    
    logger.info(f"Retrieval Latency (batch): {proposed_retrieval_ms:.2f} ms")
    logger.info(f"Reconstruction Latency (batch): {proposed_recon_ms:.2f} ms")
    
    
    # =========================================================================
    # Comparison
    # =========================================================================
    ratio = baseline_size_mb / proposed_size_mb
    logger.info("\n" + "="*40)
    logger.info("FINAL COMPARISON")
    logger.info("="*40)
    logger.info(f"{'Metric':<20} {'Baseline':>10} {'Proposed':>10} {'Impact':>10}")
    logger.info("-" * 55)
    logger.info(f"{'Size (MB)':<20} {baseline_size_mb:>10.2f} {proposed_size_mb:>10.2f} {ratio:.1f}x smaller")
    logger.info(f"{'Retrieval (ms)':<20} {baseline_retrieval_ms:>10.2f} {proposed_retrieval_ms:>10.2f} {proposed_retrieval_ms/baseline_retrieval_ms:.1f}x slower")
    logger.info(f"{'Recon (ms)':<20} {baseline_recon_ms:>10.2f} {proposed_recon_ms:>10.2f} {proposed_recon_ms/baseline_recon_ms:.1f}x slower")
    
    # Save results
    results = {
        "baseline": {
            "size_mb": baseline_size_mb,
            "retrieval_ms": baseline_retrieval_ms,
            "recon_ms": baseline_recon_ms
        },
        "proposed": {
            "size_mb": proposed_size_mb,
            "retrieval_ms": proposed_retrieval_ms,
            "recon_ms": proposed_recon_ms
        },
        "compression_ratio": ratio
    }
    save_json(results, results_dir / "results.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
