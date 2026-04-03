#!/usr/bin/env python3
"""
Experiment: llama-cpp-python KV Cache Injection Test.

Replicates Experiment 8 results using llama-cpp-python's state API
instead of Transformers DynamicCache.

Goal: Store document KV cache, inject it later, and verify the model
can recall information from the stored context.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.io_utils import setup_logging, create_results_dir

# Default model path
DEFAULT_MODEL = "/media/project_1/AI/models/phi-4-Q4_K_M.gguf"


def run_experiment(model_path: str):
    """Run the KV cache injection experiment."""
    
    logger = logging.getLogger("llama_cpp_kv_test")
    logger.info("=" * 60)
    logger.info("llama-cpp-python KV Cache Injection Test")
    logger.info("=" * 60)
    
    # Import here to avoid issues if llama-cpp-python not installed
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        return
    
    from lib.llama_cpp_kv_store import (
        extract_kv_state,
        inject_kv_state,
        save_kv_state,
        load_kv_state,
        generate_with_kv_state,
    )
    
    # Test context with unique information
    test_context = """## Company Policy Document

Employee Benefits Update - January 2026

The company has updated its remote work policy. All employees are now entitled to:
- 4 days of remote work per week (instead of the previous 2 days)
- A home office stipend of €500 per year
- Flexible working hours between 7 AM and 8 PM

The new vacation policy grants 30 vacation days per year, up from 25.

Important: The company cafeteria now offers a vegetarian menu on Wednesdays.
The CEO's favorite coffee is Ethiopian Yirgacheffe.

---
"""

    test_questions = [
        ("How many remote work days are employees entitled to?", "4"),
        ("What is the home office stipend per year?", "500"),
        ("What is the CEO's favorite coffee?", "Ethiopian Yirgacheffe"),
    ]
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    logger.info("This may take a while for large models...")
    
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,       # Reduced context window
        n_gpu_layers=20,  # Limit GPU layers to fit in VRAM
        verbose=False,
    )
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")
    
    # Step 1: Process context and extract KV state
    logger.info("\n--- Step 1: Extract KV State from Context ---")
    t0 = time.time()
    kv_state = extract_kv_state(llm, test_context, model_path)
    extract_time = time.time() - t0
    logger.info(f"KV extraction took {extract_time:.2f}s")
    logger.info(f"State size: {len(kv_state.state_data) / 1024:.1f} KB")
    
    # Step 2: Save KV state to disk
    logger.info("\n--- Step 2: Save KV State to Disk ---")
    results_dir = Path(__file__).parent.parent / "data" / "kv_states"
    results_dir.mkdir(parents=True, exist_ok=True)
    state_path = results_dir / "test_context_llama_cpp.pkl"
    save_kv_state(kv_state, str(state_path))
    
    # Step 3: Clear KV cache (simulate new session)
    logger.info("\n--- Step 3: Clear KV Cache (simulate new session) ---")
    llm.reset()  # Clear all state
    
    # Step 4: Load KV state and test recall
    logger.info("\n--- Step 4: Load KV State and Test Recall ---")
    loaded_state = load_kv_state(str(state_path))
    
    # Test each question
    results = []
    for question, expected_keyword in test_questions:
        logger.info(f"\nQuestion: {question}")
        
        # Format as chat/completion
        prompt = f"Based on the document above, answer briefly: {question}\nAnswer:"
        
        t0 = time.time()
        response = generate_with_kv_state(
            llm,
            loaded_state,
            prompt,
            max_tokens=50,
            temperature=0.0,
        )
        gen_time = time.time() - t0
        
        response = response.strip()
        success = expected_keyword.lower() in response.lower()
        
        logger.info(f"Response: {response}")
        logger.info(f"Contains '{expected_keyword}': {'✓' if success else '✗'}")
        logger.info(f"Generation time: {gen_time:.2f}s")
        
        results.append({
            "question": question,
            "response": response,
            "expected": expected_keyword,
            "success": success,
            "time": gen_time,
        })
        
        # Reset for next question
        llm.reset()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Success rate: {success_count}/{len(results)}")
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        logger.info(f"  {status} {r['question'][:40]}...")
    
    logger.info("\nKV State Info:")
    logger.info(f"  Context tokens: {kv_state.token_count}")
    logger.info(f"  State size on disk: {state_path.stat().st_size / 1024:.1f} KB")
    
    if success_count == len(results):
        logger.info("\n🎉 SUCCESS: All questions answered correctly using stored KV cache!")
    else:
        logger.info("\n⚠️ PARTIAL: Some questions failed - may need different prompt format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama-cpp-python KV Cache Test")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=DEFAULT_MODEL,
        help="Path to GGUF model file",
    )
    args = parser.parse_args()
    
    # Setup logging
    results_dir = create_results_dir("exp_llama_cpp_kv")
    log_file = results_dir / "llama_cpp_kv_test.log"
    setup_logging("llama_cpp_kv_test", results_dir)
    
    run_experiment(args.model_path)
