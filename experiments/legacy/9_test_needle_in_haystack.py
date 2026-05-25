#!/usr/bin/env python3
"""
Experiment 9: Needle-in-a-Haystack KV-Cache Injection

Objective: Validate that the model can find and use a specific fact
when it's buried in a large context of unrelated documents.

Setup:
- 20 BEIR SciFact documents as "haystack"
- 1 custom fact ("Hase") injected at position 10
- All 21 texts concatenated and injected as one KV cache
- Query asks about the custom fact

Success: Model correctly answers based on the injected fact,
demonstrating it can attend to relevant information in a large prefix.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from src.legacy.model_loader import load_model
from src.legacy.virtual_prefix import extract_kv_cache, generate_with_stored_kv
from src.legacy.io_utils import setup_logging, create_results_dir


def load_beir_documents(dataset: str = "scifact", n_docs: int = 20) -> list:
    """Load n documents from a BEIR dataset."""
    data_path = Path("data/beir_datasets") / dataset
    if not data_path.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        util.download_and_unzip(url, "data/beir_datasets")

    corpus, _, _ = GenericDataLoader(data_path).load(split="test")

    # Take first n_docs, combine title + text
    docs = []
    for i, (doc_id, doc) in enumerate(corpus.items()):
        if i >= n_docs:
            break
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        # Truncate long documents to keep total context manageable
        full_text = f"{title}. {text}" if title else text
        # Limit each doc to ~200 tokens worth of text (~800 chars)
        if len(full_text) > 800:
            full_text = full_text[:800] + "..."
        docs.append(full_text)

    return docs


def run_experiment():
    # Setup
    experiments_dir = create_results_dir("exp9")
    logger = setup_logging("exp9_needle_in_haystack", experiments_dir)

    logger.info("=" * 60)
    logger.info("Experiment 9: Needle-in-a-Haystack KV-Cache Injection")
    logger.info("=" * 60)

    # 1. Load Model
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, load_in_4bit=True)

    # 2. Load BEIR documents (haystack)
    logger.info("Loading BEIR SciFact documents...")
    beir_docs = load_beir_documents("scifact", n_docs=20)
    logger.info(f"Loaded {len(beir_docs)} documents")

    # 3. Define needle (custom fact)
    needle = "Faktum: Mein Lieblingstier ist der Hase."
    query_text = "Was frisst das Lieblingstier des Users?"
    needle_position = 10  # Insert after the 10th document

    # 4. Build combined context
    all_docs = beir_docs[:needle_position] + [needle] + beir_docs[needle_position:]
    combined_context = "\n\n".join(all_docs)

    # Log context stats
    context_tokens = tokenizer(combined_context, return_tensors="pt")
    total_tokens = context_tokens["input_ids"].shape[1]
    logger.info(f"Total documents: {len(all_docs)}")
    logger.info(f"Needle at position: {needle_position + 1} of {len(all_docs)}")
    logger.info(f"Total context tokens: {total_tokens}")
    logger.info(f"Context chars: {len(combined_context)}")
    logger.info("-" * 40)

    # 5. Extract KV cache for entire context
    logger.info("Phase 1: Extracting KV cache for full context...")
    stored_kv = extract_kv_cache(
        model,
        tokenizer,
        combined_context,
        use_prompt=False,
    )
    logger.info(f"KV cache: {stored_kv.metadata['num_layers']} layers, prefix_len={stored_kv.prefix_len}")

    # 6. Generate with injected context
    logger.info("Phase 2: Generation with full context injection...")
    generated_text = generate_with_stored_kv(
        model,
        tokenizer,
        stored_kv,
        query_text,
        max_new_tokens=250,
        do_sample=False,
    )

    logger.info("-" * 40)
    logger.info(f"Query: {query_text}")
    logger.info(f"Generated Answer: {generated_text.strip()}")
    logger.info("-" * 40)

    # 7. Baseline (no injection)
    logger.info("Phase 3: Baseline (no context)...")
    inputs = tokenizer(query_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    baseline_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    logger.info(f"Baseline Answer: {baseline_text.strip()}")
    logger.info("-" * 40)

    # 8. Evaluation
    expected_keywords = ["Möhren", "Karotten", "Gemüse", "Gras", "Salat", "Klee", "Heu",
                         "Hase", "Kaninchen"]
    has_expected = any(kw.lower() in generated_text.lower() for kw in expected_keywords)

    # Check if baseline does NOT have the answer (confirms injection is needed)
    baseline_has_answer = any(kw.lower() in baseline_text.lower() for kw in expected_keywords[:7])

    logger.info("=" * 40)
    logger.info(f"Needle found in generation: {has_expected}")
    logger.info(f"Baseline has answer (should be False): {baseline_has_answer}")

    if has_expected and not baseline_has_answer:
        logger.info("SUCCESS: Model found the needle in the haystack!")
        print("\nSUCCESS")
    elif has_expected and baseline_has_answer:
        logger.info("PARTIAL: Model answered correctly, but baseline also knows (not conclusive)")
        print("\nPARTIAL")
    else:
        logger.warning("FAILURE: Model did not find the needle.")
        logger.warning(f"Expected one of: {expected_keywords}")
        print("\nFAILURE")
    logger.info("=" * 40)


if __name__ == "__main__":
    run_experiment()
