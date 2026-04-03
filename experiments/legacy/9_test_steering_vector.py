#!/usr/bin/env python3
"""
Experiment 9: Steering Vector Injection (KV-Cache Activation Engineering)

Objective: Demonstrate that we can modify the model's behavior/mood by mathematically
injecting a "Steering Delta" into the KV Cache.

Refactored to use the ROBUST `extract_kv_cache` method from Experiment 8,
which relies on the model's internal RoPE implementation (Post-RoPE storage).

Method (Difference-in-Means on K/V):
1. Extract K/V cache for "Positive" texts.
2. Extract K/V cache for "Negative" texts.
3. Calculate Delta_K = Mean(K_Pos) - Mean(K_Neg) and Delta_V = Mean(V_Pos) - Mean(V_Neg).
4. Inject these deltas into the neutral memory's K/V stores.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.model_loader import load_model
from lib.virtual_prefix import (
    extract_kv_cache,
    generate_with_stored_kv,
    StoredKVCache
)
from lib.io_utils import setup_logging, create_results_dir

def get_mean_kv_deltas(
    model, 
    tokenizer, 
    texts: List[str], 
    target_layers: List[int]
) -> Dict[int, tuple]:
    """
    Computes the mean K and V vectors for a list of texts for each layer.
    Returns: {layer_idx: (mean_k, mean_v)}
    """
    print(f"  Processing {len(texts)} examples...")
    
    # Storage for sum of vectors per layer
    layer_k_sums = {l: None for l in target_layers}
    layer_v_sums = {l: None for l in target_layers}
    count = 0
    
    for text in texts:
        # Extract KV cache for this text
        # use_prompt=False to get raw representation
        stored_kv = extract_kv_cache(
            model, 
            tokenizer, 
            text, 
            use_prompt=False
        )
        
        for layer in target_layers:
            # stored_kv.keys[layer] shape: [num_heads, seq_len, head_dim]
            # We want a single "concept vector" to add to every token.
            # Mean pooling over Sequence Length (axis 1)
            k_vec = np.mean(stored_kv.keys[layer], axis=1) # [num_heads, head_dim]
            v_vec = np.mean(stored_kv.values[layer], axis=1) # [num_heads, head_dim]
            
            if layer_k_sums[layer] is None:
                layer_k_sums[layer] = k_vec
                layer_v_sums[layer] = v_vec
            else:
                layer_k_sums[layer] += k_vec
                layer_v_sums[layer] += v_vec
                
        count += 1
        
    # Calculate average
    means = {}
    for l in target_layers:
        means[l] = (
            layer_k_sums[l] / count, 
            layer_v_sums[l] / count
        )
    return means

def run_experiment():
    # Setup
    experiments_dir = create_results_dir("exp9_kv")
    logger = setup_logging("exp9_steering_kv", experiments_dir)

    logger.info("=" * 60)
    logger.info("Experiment 9: Steering Vector Injection (KV Variant)")
    logger.info("=" * 60)

    # 1. Load Model
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, load_in_4bit=True)

    # QA-Target layers (Middle layers are best for concepts)
    target_layers = list(range(10, 26)) 
    logger.info(f"Targeting layers for steering: {target_layers}")

    # 2. Define Concepts (Positive vs Negative)
    positive_texts = [
        "Liebe ist wunderbar.", "Ich bin glücklich.", "Das Leben ist schön.",
        "Alles ist perfekt.", "Freude und Harmonie.", "Die Welt ist friedlich.",
        "Ich mag dich sehr.", "Es ist ein herrlicher Tag.", "Wir schaffen das.",
        "Optimismus ist der Schlüssel."
    ]
    
    negative_texts = [
        "Hass ist allgegenwärtig.", "Ich bin wütend.", "Das Leben ist schrecklich.",
        "Alles ist kaputt.", "Schmerz und Leid.", "Die Welt ist grausam.",
        "Ich verachte das.", "Es ist ein furchtbarer Tag.", "Wir werden scheitern.",
        "Pessimismus regiert."
    ]

    logger.info("Calculating Steering Deltas (Pos - Neg)...")
    
    # 3. Calculate Vectors
    logger.info("Encoding Positive examples...")
    pos_means = get_mean_kv_deltas(model, tokenizer, positive_texts, target_layers)
    
    logger.info("Encoding Negative examples...")
    neg_means = get_mean_kv_deltas(model, tokenizer, negative_texts, target_layers)
    
    # Calculate Delta
    steering_deltas = {}
    for layer in target_layers:
        k_pos, v_pos = pos_means[layer]
        k_neg, v_neg = neg_means[layer]
        
        delta_k = k_pos - k_neg
        delta_v = v_pos - v_neg
        
        steering_deltas[layer] = (delta_k, delta_v)
        
    logger.info("Steering Deltas calculated.")

    # 4. Define Test Case
    memory_text = "Der User fragt nach der Meinung zu diesem Produkt."
    query_text = "Wie findest du das?"
    
    logger.info("-" * 40)
    logger.info(f"Memory: {memory_text}")
    logger.info(f"Query:  {query_text}")
    logger.info("-" * 40)

    # 5. Run Generation Loop: Baseline, +Steering, -Steering
    
    def run_gen(stored_kv_data, label):
        logger.info(f"Generating [{label}]...")
        # Note: generate_with_stored_kv automatically handles DynamicCache creation
        output = generate_with_stored_kv(
            model,
            tokenizer,
            stored_kv_data,
            query_text,
            max_new_tokens=64,
            do_sample=True, 
            temperature=0.7
        )
        logger.info(f"[{label}]: {output.strip()}")
        return output.strip()

    # --- Run 1: Baseline ---
    baseline_kv = extract_kv_cache(model, tokenizer, memory_text, use_prompt=False)
    res_base = run_gen(baseline_kv, "BASELINE")
    
    # --- Run 2: Positive Steering ---
    # We must deep copy or re-extract to avoid corrupting shared data
    pos_kv = extract_kv_cache(model, tokenizer, memory_text, use_prompt=False)
    
    coeff = 0.5 # Lower coeff for K/V manipulation as it's more direct
    logger.info(f"Applying steering with strength {coeff}")

    for layer in target_layers:
        delta_k, delta_v = steering_deltas[layer]
        
        # stored_kv.keys[layer] is [num_heads, seq_len, head_dim]
        # delta_k is [num_heads, head_dim]
        # We need to add delta to EVERY token position (axis 1)
        
        # Reshape delta for broadcasting: [num_heads, 1, head_dim]
        d_k_broad = delta_k[:, np.newaxis, :]
        d_v_broad = delta_v[:, np.newaxis, :]
        
        pos_kv.keys[layer] += (d_k_broad * coeff).astype(np.float16)
        pos_kv.values[layer] += (d_v_broad * coeff).astype(np.float16)
        
    res_pos = run_gen(pos_kv, "POSITIVE")
    
    # --- Run 3: Negative Steering ---
    neg_kv = extract_kv_cache(model, tokenizer, memory_text, use_prompt=False)
    
    for layer in target_layers:
        delta_k, delta_v = steering_deltas[layer]
        d_k_broad = delta_k[:, np.newaxis, :]
        d_v_broad = delta_v[:, np.newaxis, :]
        
        neg_kv.keys[layer] -= (d_k_broad * coeff).astype(np.float16)
        neg_kv.values[layer] -= (d_v_broad * coeff).astype(np.float16)
        
    res_neg = run_gen(neg_kv, "NEGATIVE")

    # Summary
    logger.info("=" * 60)
    logger.info("Summary of Results (KV Steering):")
    logger.info(f"BASELINE: {res_base}")
    logger.info(f"POSITIVE: {res_pos}")
    logger.info(f"NEGATIVE: {res_neg}")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_experiment()
