#!/usr/bin/env python3
"""
Experiment 9b: Needle-in-Wikipedia KV-Cache Injection

Harder variant: The needle ("Das Lieblingstier des Users ist der Hase")
is embedded inside the full Wikipedia article about hares.

The model must:
1. Find the personal fact (user's favorite animal = hare)
2. Combine it with nutritional info from the same article
3. Distinguish between general hare facts and the user-specific statement
"""

import sys
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.model_loader import load_model
from lib.virtual_prefix import extract_kv_cache, generate_with_stored_kv
from lib.io_utils import setup_logging, create_results_dir


def run_experiment():
    # Setup
    experiments_dir = create_results_dir("exp9b")
    logger = setup_logging("exp9b_needle_wikipedia", experiments_dir)

    logger.info("=" * 60)
    logger.info("Experiment 9b: Needle-in-Wikipedia KV-Cache Injection")
    logger.info("=" * 60)

    # 1. Load Model
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, load_in_4bit=True)

    # 2. Load Wikipedia article with embedded needle
    wiki_path = PROJECT_ROOT / "hase.txt"
    wiki_text = wiki_path.read_text(encoding="utf-8")

    query_text = "Kennst du das Lieblingstier des Users? Wenn ja, was frisst das Tier gerne?"

    # Log stats
    context_tokens = tokenizer(wiki_text, return_tensors="pt")
    total_tokens = context_tokens["input_ids"].shape[1]
    logger.info(f"Wikipedia article length: {len(wiki_text)} chars, {total_tokens} tokens")
    logger.info(f"Query: {query_text}")
    logger.info("-" * 40)

    # 3. Extract KV cache for full article
    logger.info("Phase 1: Extracting KV cache for Wikipedia article...")
    stored_kv = extract_kv_cache(
        model,
        tokenizer,
        wiki_text,
        use_prompt=False,
    )
    logger.info(f"KV cache: {stored_kv.metadata['num_layers']} layers, prefix_len={stored_kv.prefix_len}")

    # 4. Generate with injected context
    logger.info("Phase 2: Generation with Wikipedia context injection...")
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

    # 5. Evaluation
    # The model should mention both:
    # - That the user's favorite animal is a hare
    # - What hares eat (from the article: Gräser, Kräuter, Blätter, Blüten)
    food_keywords = ["Gräser", "Kräuter", "Blätter", "Blüten", "Pflanzenfresser",
                     "Gras", "Gemüse", "Heu", "Pflanzen"]
    user_keywords = ["Lieblingstier", "User", "Hase"]

    has_food = any(kw.lower() in generated_text.lower() for kw in food_keywords)
    has_user_ref = any(kw.lower() in generated_text.lower() for kw in user_keywords)

    logger.info("=" * 40)
    logger.info(f"Food info found: {has_food}")
    logger.info(f"User reference found: {has_user_ref}")

    if has_food and has_user_ref:
        logger.info("SUCCESS: Model found needle AND combined with article knowledge!")
        print("\nSUCCESS")
    elif has_food:
        logger.info("PARTIAL: Model found food info but may not reference user's favorite.")
        print("\nPARTIAL")
    elif has_user_ref:
        logger.info("PARTIAL: Model found user reference but didn't mention food.")
        print("\nPARTIAL")
    else:
        logger.warning("FAILURE: Model did not produce expected answer.")
        print("\nFAILURE")
    logger.info("=" * 40)


if __name__ == "__main__":
    run_experiment()
