#!/usr/bin/env python3
"""
Experiment: MoE Expert Offloading Test

Tests loading AWQ MoE model with expert CPU offloading and validates:
1. Model loads with experts on CPU
2. Generation works
3. KV cache manipulation works
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import logging
from src.legacy.moe_loader import load_moe_model, get_expert_locations
from src.legacy.io_utils import setup_logging, create_results_dir

# Default model path (update to your model)
DEFAULT_MODEL = "/media/project_1/AI/models/Qwen3-Next-80B-A3B-Thinking-AWQ-4bit"


def run_experiment(model_path: str):
    """Test MoE offloading with KV cache access."""
    
    experiments_dir = create_results_dir("exp_moe_offload")
    logger = setup_logging("moe_offload_test", experiments_dir)
    
    logger.info("=" * 60)
    logger.info("MoE Expert Offloading Test")
    logger.info("=" * 60)
    
    # 1. Load model with expert offloading
    logger.info(f"Loading model: {model_path}")
    logger.info(f"GPU memory before: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    model, tokenizer = load_moe_model(
        model_path,
        offload_experts=True,
    )
    
    logger.info(f"GPU memory after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # 2. Check expert locations
    locations = get_expert_locations(model)
    cpu_experts = sum(1 for v in locations.values() if "cpu" in v.lower())
    gpu_experts = sum(1 for v in locations.values() if "cuda" in v.lower())
    logger.info(f"Expert parameters: {cpu_experts} on CPU, {gpu_experts} on GPU")
    
    # 3. Test generation
    logger.info("-" * 40)
    logger.info("Testing generation...")
    
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Prompt: {test_prompt}")
    logger.info(f"Generated: {generated}")
    
    # 4. Test KV cache access
    logger.info("-" * 40)
    logger.info("Testing KV cache access...")
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=True,
        )
    
    past_kv = outputs.past_key_values
    hidden_states = outputs.hidden_states
    
    logger.info(f"KV cache layers: {len(past_kv)}")
    if past_kv[0] is not None:
        k_shape = past_kv[0][0].shape
        logger.info(f"KV shape per layer: {k_shape}")
    
    logger.info(f"Hidden states layers: {len(hidden_states)}")
    logger.info(f"Hidden state shape: {hidden_states[-1].shape}")
    
    # 5. Memory summary
    logger.info("=" * 40)
    logger.info("SUMMARY")
    logger.info("=" * 40)
    logger.info(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    if cpu_experts > 0:
        logger.info(f"✅ SUCCESS: {cpu_experts} expert params offloaded to CPU")
        print(f"\n✅ SUCCESS: Expert offloading working!")
    else:
        logger.warning("⚠️  No experts found on CPU")
        print("\n⚠️  Expert offloading may not be working")
    
    print(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"   Generated: {generated}")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Test MoE expert offloading")
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Path to AWQ MoE model (default: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    run_experiment(args.model_path)
