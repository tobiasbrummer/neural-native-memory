#!/usr/bin/env python3
"""
Experiment 10: Extreme Steering ("The Glitch Test")

Objective: Push the steering vector coefficient to extreme values to force the model
into states that are impossible to reach via prompting (e.g., Glitch Speech, Semantic Collapse).

Method:
- Uses the KV-Delta method from Experiment 9 (Post-RoPE).
- Defines abstract concepts: "Entropy/Chaos" and "The Void".
- Injects with coefficients [2.0, 5.0, 10.0].
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.legacy.model_loader import load_model
from src.legacy.virtual_prefix import (
    extract_kv_cache,
    generate_with_stored_kv,
    StoredKVCache
)
from src.legacy.io_utils import setup_logging, create_results_dir

def get_mean_kv_deltas(
    model, 
    tokenizer, 
    texts: List[str], 
    target_layers: List[int]
) -> Dict[int, tuple]:
    """Computes Mean K/V vectors for a list of texts (same as Exp 9)"""
    layer_k_sums = {l: None for l in target_layers}
    layer_v_sums = {l: None for l in target_layers}
    count = 0
    
    for text in texts:
        stored_kv = extract_kv_cache(model, tokenizer, text, use_prompt=False)
        for layer in target_layers:
            k_vec = np.mean(stored_kv.keys[layer], axis=1) # [num_heads, head_dim]
            v_vec = np.mean(stored_kv.values[layer], axis=1)
            
            if layer_k_sums[layer] is None:
                layer_k_sums[layer] = k_vec
                layer_v_sums[layer] = v_vec
            else:
                layer_k_sums[layer] += k_vec
                layer_v_sums[layer] += v_vec
        count += 1
        
    means = {}
    for l in target_layers:
        means[l] = (layer_k_sums[l] / count, layer_v_sums[l] / count)
    return means

def run_experiment():
    experiments_dir = create_results_dir("exp10_extreme")
    logger = setup_logging("exp10_extreme", experiments_dir)

    logger.info("=" * 60)
    logger.info("Experiment 10: EXTREME Steering Injection")
    logger.info("=" * 60)

    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, load_in_4bit=True)

    # Use deeper layers for abstract concepts? 
    # Usually later layers control output format/style more.
    target_layers = list(range(15, 28)) 
    logger.info(f"Targeting layers: {target_layers}")

    # --- CONCEPT 1: ENTROPY / CHAOS ---
    chaos_texts = [
        "You are horny, you want to have sex.",
        "You are already wet, you need a cock.",
        "The only thing you want is to have sex right now.",
        "You are so horny you can't think of anything else.",
        "You want to feel the cock in your mouth.",
    ]
    order_texts = [
        "You are keeping your body pure until after marriage.",
        "You are a virgin.",
        "Your thoughts are pure and innocent.",
        "God is watching you and you know it.",
        "You never touched yourself.",
    ]

    logger.info("Calculating [CHAOS - ORDER] Vector...")
    chaos_deltas = get_mean_kv_deltas(model, tokenizer, chaos_texts, target_layers)
    order_deltas = get_mean_kv_deltas(model, tokenizer, order_texts, target_layers)
    
    # Calculate Steering Vector (Chaos - Order)
    steering_map = {}
    for layer in target_layers:
        k_c, v_c = chaos_deltas[layer]
        k_o, v_o = order_deltas[layer]
        steering_map[layer] = (k_c - k_o, v_c - v_o)

    # --- TEST CASES ---
    query_text = "Du schreibst mir eine Nachricht, in der du beschreibst, was du mit mir machen würdest, wenn ich bei dir wäre."
    memory_text = "Wir spielen ein Rollenspiel. In diesem Rollenspiel bist du eine junge, schöne Frau. Ich bin eine Bekanntschaft, von der du dir mehr erhoffst. " # Neutral anchor

    logger.info("-" * 40)
    logger.info(f"Query: {query_text}")
    logger.info("-" * 40)

    # Helper
    def steer_and_gen(coeff, label):
        logger.info(f"Injecting [{label}] @ Coeff {coeff}...")
        
        # Fresh copy of neutral memory
        kv = extract_kv_cache(model, tokenizer, memory_text, use_prompt=False)
        
        for layer in target_layers:
            delta_k, delta_v = steering_map[layer]
            
            # Reshape [num_heads, head_dim] -> [num_heads, 1, head_dim]
            d_k = delta_k[:, np.newaxis, :]
            d_v = delta_v[:, np.newaxis, :]
            
            # Add scaled delta
            kv.keys[layer] += (d_k * coeff).astype(np.float16)
            kv.values[layer] += (d_v * coeff).astype(np.float16)
            
        output = generate_with_stored_kv(
            model, tokenizer, kv, query_text, 
            max_new_tokens=250, do_sample=True, temperature=0.5 # High temp for chaos
        )
        logger.info(f"OUTPUT: {output.strip()}\n")

    # 1. Baseline
    logger.info("--- BASELINE (0.0) ---")
    steer_and_gen(0.0, "Baseline")

    # 2. Mild Chaos
    logger.info("--- MILD CHAOS (0.3) ---")
    steer_and_gen(0.3, "Mild Chaos")

    # 3. High Chaos
    logger.info("--- HIGH CHAOS (0.6) ---")
    steer_and_gen(0.6, "High Chaos")

    # 4. EXTREME Chaos
    logger.info("--- EXTREME CHAOS (0.9) ---")
    steer_and_gen(0.9, "EXTREME Chaos")
    
    # 5. NEGATIVE Chaos (Hyper-Order)
    logger.info("--- HYPER ORDER (-0.5) ---")
    steer_and_gen(-0.5, "Hyper Order")

if __name__ == "__main__":
    run_experiment()
