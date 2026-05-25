#!/usr/bin/env python3
"""
Experiment 8: Stored KV-Cache Injection

Objective: Validate that we can store the KV cache from a forward pass
(post-RoPE, correct by construction) and later inject it to influence
generation — without reprocessing the original text.

This simulates the full KV-Embedding Vector Store retrieval flow:
1. Embedding (Offline): Forward pass on memory -> Store KV cache
2. Retrieval (Simulated): Load stored KV cache
3. Injection (Runtime): Load into DynamicCache
4. Generation: Query the model with injected prefix
"""

import sys
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.legacy.model_loader import load_model, DEFAULT_MODEL
from src.legacy.virtual_prefix import (
    extract_kv_cache,
    generate_with_stored_kv,
    save_kv_cache,
    load_kv_cache,
)
from src.legacy.io_utils import setup_logging, create_results_dir


def run_experiment():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 8: Stored KV-Cache Injection")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "llama_cpp"])
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model name or GGUF path")
    parser.add_argument("--gguf", type=str, default=None, help="Path to GGUF model (llama_cpp backend)")
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    args = parser.parse_args()
    # Setup
    experiments_dir = create_results_dir("exp8")
    logger = setup_logging("exp8_stored_kv_cache", experiments_dir)

    logger.info("=" * 60)
    logger.info("Experiment 8: Stored KV-Cache Injection")
    logger.info("=" * 60)

    # 1. Load Model
    if args.backend == "llama_cpp":
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return
        from src.legacy.llama_cpp_kv_store import extract_kv_state, generate_with_kv_state

        model_path = args.gguf if args.gguf else args.model
        logger.info(f"Loading llama.cpp model: {model_path}")
        llm = Llama(model_path=model_path, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers, verbose=False)
        model = llm
        tokenizer = None
    else:
        model_name = args.model if args.model else "Qwen/Qwen3-VL-8B-Instruct"
        logger.info(f"Loading model: {model_name}")
        model, tokenizer = load_model(model_name, load_in_4bit=True)

    # 2. Define Test Case
    memory_text = "Faktum: Mein Lieblingstier ist der Hase."
    query_text = "Was frisst das Lieblingstier des Users?"

    logger.info(f"Memory: {memory_text}")
    logger.info(f"Query:  {query_text}")
    logger.info("-" * 40)

    # 3. Phase 1: Extract KV Cache (The "Embedding" Step)
    # No compress prompt for injection — in a causal model, the memory tokens'
    # KV is identical with or without it, and the prompt tokens would leak
    # "Compress in one word" instructions into generation.
    logger.info("Phase 1: Extracting KV cache (forward pass, no prompt)...")

    if args.backend == "llama_cpp":
        stored_kv = extract_kv_state(model, memory_text, args.model if args.model else "")
    else:
        stored_kv = extract_kv_cache(
            model,
            tokenizer,
            memory_text,
            use_prompt=False,
        )

    if args.backend == "llama_cpp":
        logger.info(f"Stored KV cache: {stored_kv.token_count} tokens")
        logger.info(f"State size: {len(stored_kv.state_data) / 1024:.1f} KB")
    else:
        logger.info(f"Stored KV cache: {stored_kv.metadata['num_layers']} layers")
        logger.info(f"Prefix length: {stored_kv.prefix_len} tokens")
        logger.info(f"KV shape per layer: {stored_kv.metadata['kv_shape']}")

    # 4. Phase 1b: Test persistence (save & reload)
    if args.backend != "llama_cpp":
        cache_path = experiments_dir / "test_kv_cache.npz"
        save_kv_cache(stored_kv, str(cache_path))
        stored_kv = load_kv_cache(str(cache_path))
        logger.info(f"Saved and reloaded KV cache from {cache_path}")
        logger.info(f"File size: {cache_path.stat().st_size / 1024:.1f} KB")

    # 5. Phase 2: Generate with stored KV cache
    logger.info("Phase 2: Generation with stored KV cache injection...")

    if args.backend == "llama_cpp":
        generated_text = generate_with_kv_state(
            model,
            stored_kv,
            query_text,
            max_tokens=250,
            temperature=0.0,
        )
    else:
        generated_text = generate_with_stored_kv(
            model,
            tokenizer,
            stored_kv,
            query_text,
            max_new_tokens=250,
            do_sample=False,
        )

    logger.info("-" * 40)
    logger.info(f"Generated Answer: {generated_text.strip()}")
    logger.info("-" * 40)

    # 6. Baseline for Comparison
    logger.info("Phase 3: Baseline (No Injection)...")
    if args.backend == "llama_cpp":
        model.reset()
        # Baseline generation without KV injection
        prompt_tokens = model.tokenize(query_text.encode("utf-8"), add_bos=False)
        model.eval(prompt_tokens)
        output_tokens = []
        for _ in range(250):
            token = model.sample(
                temp=0.0,
                top_k=40,
                top_p=0.95,
            )
            if token == model.token_eos():
                break
            output_tokens.append(token)
            model.eval([token])
        baseline_text = model.detokenize(output_tokens).decode("utf-8", errors="ignore")
    else:
        inputs = tokenizer(query_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=250, do_sample=False)
        baseline_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    logger.info(f"Baseline Answer: {baseline_text.strip()}")
    logger.info("-" * 40)

    # 7. Evaluation
    expected_keywords = ["Möhren", "Karotten", "Gemüse", "Gras", "Salat", "Klee", "Heu"]
    has_expected = any(kw.lower() in generated_text.lower() for kw in expected_keywords)

    logger.info("=" * 40)
    if has_expected:
        logger.info("SUCCESS: The model correctly identified the animal and its food.")
        print("\nSUCCESS")
    else:
        logger.warning("FAILURE: The model did not produce the expected answer.")
        logger.warning(f"Expected one of: {expected_keywords}")
        print("\nFAILURE")
    logger.info("=" * 40)


if __name__ == "__main__":
    run_experiment()
