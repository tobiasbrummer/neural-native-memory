#!/usr/bin/env python3
"""
Experiment 5: Virtual Prefix Injection – End-to-End Validation

Objective: Validate that stored Pre-RoPE Hidden States can be injected
as virtual prefix and influence generation.

Success Criterion: Generation with injected prefix shows clear influence
of the memory compared to baseline (no injection).

Test Cases:
1. Simple fact injection
2. Context preference injection
3. Control (no injection) for comparison
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.legacy.io_utils import (
    create_results_dir,
    save_json,
    setup_logging,
)
from src.legacy.model_loader import load_model, DEFAULT_MODEL
from src.legacy.virtual_prefix import generate_with_kv_rerouting


# =============================================================================
# Test Data
# =============================================================================

TEST_CASES = [
    {
        "name": "simple_fact",
        "memory": "The sky is blue during the day.",
        "query": "What color is the sky?",
        "expected_keywords": ["blue"],
        "unexpected_keywords": ["green", "red", "purple"],
    },
    {
        "name": "context_preference",
        "memory": "The user explicitly prefers Python over JavaScript for all data analysis tasks.",
        "query": "Which programming language should I use for data analysis?",
        "expected_keywords": ["Python", "python"],
        "unexpected_keywords": ["JavaScript", "javascript", "JS", "js"],
    },
    {
        "name": "specific_entity",
        "memory": "Project X-NDA is a classified initiative. The codename is NIGHTWING.",
        "query": "What is the codename for the classified project?",
        "expected_keywords": ["NIGHTWING", "nightwing", "Nightwing"],
        "unexpected_keywords": ["don't know", "not sure", "unknown"],
    },
    {
        "name": "multi_fact",
        "memory": "Emma lives in Berlin. She works as a machine learning engineer at DeepMind.",
        "query": "Where does Emma work?",
        "expected_keywords": ["DeepMind", "deepmind"],
        "unexpected_keywords": ["Google", "Meta", "OpenAI"],
    },
    {
        "name": "multi_fact_location",
        "memory": "The user lives in Schwäbisch Hall, Germany. Schwäbisch Hall is a town in the state of Baden-Württemberg.",
        "query": "Where does the user live?",
        "expected_keywords": ["Schwäbisch Hall", "Baden-Württemberg"],
        "unexpected_keywords": ["Stuttgart", "Baden-Baden", "München", "Berlin", "Hamburg"],
    },
]


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_generation(
    generated: str,
    expected_keywords: list,
    unexpected_keywords: list,
) -> dict:
    """
    Evaluate if generation contains expected keywords.

    Args:
        generated: Generated text
        expected_keywords: Keywords that should appear
        unexpected_keywords: Keywords that should NOT appear

    Returns:
        Dictionary with evaluation results
    """
    generated_lower = generated.lower()

    found_expected = []
    for kw in expected_keywords:
        if kw.lower() in generated_lower:
            found_expected.append(kw)

    found_unexpected = []
    for kw in unexpected_keywords:
        if kw.lower() in generated_lower:
            found_unexpected.append(kw)

    return {
        "expected_found": found_expected,
        "unexpected_found": found_unexpected,
        "expected_ratio": len(found_expected) / len(expected_keywords) if expected_keywords else 0,
        "has_unexpected": len(found_unexpected) > 0,
    }


def compute_influence_score(
    baseline_generation: str,
    injected_generation: str,
    memory_text: str,
) -> dict:
    """
    Compute metrics to measure memory influence on generation.

    Args:
        baseline_generation: Generation without injection
        injected_generation: Generation with virtual prefix
        memory_text: The injected memory text

    Returns:
        Dictionary with influence metrics
    """
    # Simple metrics
    baseline_words = set(baseline_generation.lower().split())
    injected_words = set(injected_generation.lower().split())
    memory_words = set(memory_text.lower().split())

    # Overlap with memory
    memory_overlap_baseline = len(baseline_words & memory_words) / max(len(memory_words), 1)
    memory_overlap_injected = len(injected_words & memory_words) / max(len(memory_words), 1)

    # Difference between generations
    word_diff = len(baseline_words.symmetric_difference(injected_words))
    total_unique = len(baseline_words | injected_words)
    diff_ratio = word_diff / max(total_unique, 1)

    return {
        "memory_overlap_baseline": round(memory_overlap_baseline, 4),
        "memory_overlap_injected": round(memory_overlap_injected, 4),
        "overlap_improvement": round(memory_overlap_injected - memory_overlap_baseline, 4),
        "generation_diff_ratio": round(diff_ratio, 4),
        "injected_length": len(injected_generation),
        "baseline_length": len(baseline_generation),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(args):
    """Run the virtual prefix injection experiment."""

    # Setup
    results_dir = create_results_dir("exp5")
    logger = setup_logging("exp5_virtual_prefix", results_dir)

    logger.info("=" * 60)
    logger.info("Experiment 5: Virtual Prefix Injection – End-to-End Validation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"4-bit: {args.load_in_4bit}")
    logger.info(f"Target layers: {args.layers or 'auto-select from KV-Embedding'}")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model(
        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
    )

    # Determine target layers
    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]
    else:
        # Use KV-Embedding layer selection
        from src.legacy.embedding_utils import select_optimal_layers
        sample_texts = [case["memory"] for case in TEST_CASES[:2]]
        target_layers = select_optimal_layers(
            model, tokenizer, sample_texts, n_layers_to_select=4
        )

    logger.info(f"Target layers: {target_layers}")
    logger.info("")

    # Results storage
    all_results = []

    # Run each test case
    for test_case in TEST_CASES:
        case_name = test_case["name"]
        logger.info("-" * 40)
        logger.info(f"Test Case: {case_name}")
        logger.info("-" * 40)

        memory = test_case["memory"]
        query = test_case["query"]

        logger.info(f"Memory: {memory}")
        logger.info(f"Query: {query}")
        logger.info("")

        # =====================================================================
        # Step 2: Baseline Generation (No Injection)
        # =====================================================================
        logger.info("Step 2: Baseline Generation (no injection)...")

        query_inputs = tokenizer(query, return_tensors="pt")
        query_inputs = {k: v.to(model.device) for k, v in query_inputs.items()}

        with torch.no_grad():
            baseline_outputs = model.generate(
                **query_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            # Use logits processor to get generation details if needed
            # Remove input tokens from output
            baseline_ids = baseline_outputs[0][query_inputs["input_ids"].shape[1]:]
            baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=True)

        logger.info(f"  Baseline: {baseline_text}")
        logger.info("")

        # =====================================================================
        # Step 3: Generate with KV Re-routing (Paper Approach)
        # =====================================================================
        logger.info("Step 3: Generating with KV Re-routing...")

        # Direct generation using paper-based approach (model's native KV cache)
        injected_text = generate_with_kv_rerouting(
            model, tokenizer, 
            memory_text=memory,
            query=query,
            target_layers=target_layers,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.2,
            do_sample=False,
        )

        logger.info(f"  Injected: {injected_text}")
        logger.info("")

        # =====================================================================
        # Step 4: Evaluation
        # =====================================================================
        logger.info("Step 4: Evaluation...")

        # Keyword-based evaluation
        injected_eval = evaluate_generation(
            injected_text,
            test_case["expected_keywords"],
            test_case["unexpected_keywords"],
        )

        baseline_eval = evaluate_generation(
            baseline_text,
            test_case["expected_keywords"],
            test_case["unexpected_keywords"],
        )

        logger.info(f"  Expected keywords found: {injected_eval['expected_found']}")
        logger.info(f"  Unexpected keywords found: {injected_eval['unexpected_found']}")
        logger.info(f"  Expected ratio: {injected_eval['expected_ratio']:.2f}")
        logger.info("")

        # Influence score
        influence = compute_influence_score(baseline_text, injected_text, memory)
        logger.info(f"  Memory overlap (baseline): {influence['memory_overlap_baseline']:.3f}")
        logger.info(f"  Memory overlap (injected): {influence['memory_overlap_injected']:.3f}")
        logger.info(f"  Overlap improvement: {influence['overlap_improvement']:.3f}")
        logger.info(f"  Generation diff ratio: {influence['generation_diff_ratio']:.3f}")
        logger.info("")

        # Determine success
        case_success = (
            injected_eval["expected_ratio"] >= 0.5 or  # At least half expected keywords
            influence["overlap_improvement"] > 0.1 or  # Clear memory overlap improvement
            influence["generation_diff_ratio"] > 0.2  # Significant generation change
        )

        logger.info(f"  Case {'PASS' if case_success else 'FAIL'}")

        # Store results
        result = {
            "case_name": case_name,
            "memory": memory,
            "query": query,
            "baseline_generation": baseline_text,
            "injected_generation": injected_text,
            "evaluation": {
                "baseline": baseline_eval,
                "injected": injected_eval,
                "influence": influence,
            },
            "success": bool(case_success),
        }
        all_results.append(result)

    # =========================================================================
    # Overall Summary
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)

    passed = sum(1 for r in all_results if r["success"])
    total = len(all_results)

    logger.info(f"Passed: {passed}/{total}")

    for result in all_results:
        status = "PASS" if result["success"] else "FAIL"
        logger.info(f"  {result['case_name']}: {status}")

    overall_success = passed >= total * 0.75  # 75% pass rate

    logger.info("")
    logger.info("=" * 40)
    logger.info(f"OVERALL SUCCESS: {overall_success}")
    logger.info("=" * 40)

    # Save results
    output = {
        "experiment": "5_virtual_prefix_injection",
        "model": args.model,
        "load_in_4bit": args.load_in_4bit,
        "target_layers": target_layers,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": datetime.now().isoformat(),
        "test_cases": all_results,
        "passed": passed,
        "total": total,
        "overall_success": bool(overall_success),
    }

    output_path = results_dir / "results.json"
    save_json(output, output_path)
    logger.info(f"\nResults saved to: {output_path}")

    return 0 if overall_success else 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 5: Virtual Prefix Injection Validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model to use",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit precision",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to inject (default: auto-select)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
