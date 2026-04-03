#!/usr/bin/env python3
"""
Debug script to analyze layer selection and intrinsic dimensionality.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure basic logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("debug_layers")

from lib.model_loader import load_model
from lib.embedding_utils import select_optimal_layers, compute_intrinsic_dimension_twonn

def main():
    logger.info("Starting Layer Selection Debugger")
    
    # Load model (4-bit to match experiment)
    logger.info("Loading model Qwen/Qwen3-VL-8B-Instruct (4-bit)...")
    model, tokenizer = load_model(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True
    )
    
    # Sample texts for analysis
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "Artificial intelligence is transforming the world.",
        "Quantum mechanics describes nature at the smallest scales.",
        "Python is a versatile programming language for data science."
    ]
    
    logger.info(f"Analyzing layers using {len(sample_texts)} sample texts...")
    
    # We will manually run the logic from select_optimal_layers but with more verbosity
    # and plotting
    
    n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else 28
    if hasattr(model.config, "text_config"):
         if hasattr(model.config.text_config, "num_hidden_layers"):
             n_layers = model.config.text_config.num_hidden_layers
    
    logger.info(f"Model has {n_layers} layers")
    
    # Get hidden states
    all_hidden_states = []
    
    with torch.no_grad():
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Use fallback for VL models if needed
            try:
                outputs = model(**inputs, output_hidden_states=True)
            except Exception:
                # Some VL models need pixel_values if using their specific class
                # But we are likely using AutoModelForVision2Seq which might require dummy images
                # Or we can just inspect the model and see if we can run text-only
                logger.warning("Standard forward pass failed, trying with dummy pixel_values if applicable...")
                # For now assume model loader gave us something usable or this will fail
                raise
                
            hidden_states = outputs.hidden_states[1:] # Skip embedding layer
            all_hidden_states.append([h.squeeze(0).cpu().numpy() for h in hidden_states])

    # Compute ID per layer
    layer_ids = []
    ids = []
    
    print("\nLayer Intrinsic Dimensionality Profile:")
    print(f"{'Layer':<6} | {'ID Score':<10}")
    print("-" * 20)
    
    for layer_idx in range(len(all_hidden_states[0])):
        # Collect embeddings
        layer_embeddings = []
        for sample in all_hidden_states:
             layer_embeddings.append(sample[layer_idx])
        
        layer_embeddings = np.vstack(layer_embeddings)
        
        id_score = compute_intrinsic_dimension_twonn(layer_embeddings)
        
        layer_ids.append(layer_idx)
        ids.append(id_score)
        
        print(f"{layer_idx:<6} | {id_score:.4f}")
        
    ids = np.array(ids)
    
    # Identify optimal layers (lowest ID)
    # Filter out first/last 15%
    margin = max(1, n_layers // 6)
    valid_mask = (np.arange(len(ids)) >= margin) & (np.arange(len(ids)) < n_layers - margin)
    
    valid_indices = np.where(valid_mask)[0]
    valid_ids = ids[valid_indices]
    
    sorted_indices = np.argsort(valid_ids)
    best_relative_indices = sorted_indices[:4]
    best_layers = valid_indices[best_relative_indices]
    best_layers.sort()
    
    print("\nAnalysis Results:")
    print(f"Valid Range: Layer {margin} to {n_layers - margin - 1}")
    print(f"Selected Optimal Layers (Lowest ID): {best_layers.tolist()}")
    print(f"Corresponding IDs: {ids[best_layers]}")

if __name__ == "__main__":
    main()
