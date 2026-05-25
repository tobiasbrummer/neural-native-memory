#!/usr/bin/env python3
"""
Experiment 2: Token-ID Encoding/Decoding Test

Objective: Validate that token IDs can be deterministically reconstructed
from static embeddings via nearest-neighbor search.

Hypothesis: Static embeddings (layer 0) contain sufficient information
to reverse-map to token IDs with high accuracy.

Success Criterion: Reconstruction accuracy > 95%.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.legacy.io_utils import (
    create_results_dir,
    load_test_documents,
    save_json,
    setup_logging,
)
from src.legacy.model_loader import load_model, DEFAULT_MODEL
from src.legacy.token_utils import (
    TokenEmbeddingIndex,
    extract_static_embeddings,
    compute_reconstruction_accuracy,
)


def main():
    # Setup
    results_dir = create_results_dir("exp2")
    logger = setup_logging("exp2_token_decode", results_dir)
    
    logger.info("=" * 60)
    logger.info("Experiment 2: Token-ID Encoding/Decoding Test")
    logger.info("=" * 60)
    
    # Load test documents
    documents = load_test_documents()
    logger.info(f"Loaded {len(documents)} test documents")
    
    # Load model
    logger.info("Loading model...")
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "llama"])
    parser.add_argument("--model", type=str, default=None, help="HF model name or GGUF path")
    parser.add_argument("--gguf", type=str, default=None, help="Path to GGUF model (llama backend)")
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    args, _ = parser.parse_known_args()

    if args.backend == "llama":
        from src.legacy.llama_raw import LlamaModel
        from src.legacy.llama_embedding_utils import extract_static_embeddings_llama

        model_path = args.gguf if args.gguf else (args.model if args.model else "")
        if not model_path:
            raise ValueError("Please provide --gguf or --model for llama backend")
        model = LlamaModel(model_path, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)
        tokenizer = None
    else:
        model_name = args.model if args.model else DEFAULT_MODEL
        model, tokenizer = load_model(model_name=model_name)
    
    # Build embedding index
    logger.info("Building token embedding index...")
    if args.backend == "llama":
        embed_matrix, _, _ = model.get_embed_matrix()
        index = TokenEmbeddingIndex(model=None, embedding_matrix=embed_matrix, use_faiss=False)
    else:
        index = TokenEmbeddingIndex(model, use_faiss=False)
    
    # Extract static embeddings for all documents
    logger.info("Extracting static embeddings...")
    texts = [doc["text"] for doc in documents]
    if args.backend == "llama":
        static_data = extract_static_embeddings_llama(model, texts)
    else:
        static_data = extract_static_embeddings(model, tokenizer, texts)
    
    # Reconstruct and evaluate
    total_tokens = 0
    total_correct = 0
    all_results = []
    
    logger.info("\nReconstructing token IDs...")
    
    for i, doc in enumerate(documents):
        original_ids = static_data["token_ids"][i]
        static_embeddings = static_data["static_embeddings"][i]
        
        # Reconstruct via nearest neighbor
        reconstructed_ids, similarities = index.find_nearest_tokens_batch(
            static_embeddings, k=1
        )
        reconstructed_ids = reconstructed_ids.squeeze(-1)
        similarities = similarities.squeeze(-1)
        
        # Compute accuracy
        accuracy, mismatches = compute_reconstruction_accuracy(
            original_ids, reconstructed_ids
        )
        
        total_tokens += len(original_ids)
        total_correct += int(accuracy * len(original_ids))
        
        # Log results
        logger.info(f"\n{doc['doc_id']}:")
        logger.info(f"  Tokens: {len(original_ids)}")
        logger.info(f"  Accuracy: {accuracy:.4f} ({int(accuracy * len(original_ids))}/{len(original_ids)})")
        
        if mismatches:
            logger.info(f"  Mismatches: {len(mismatches)}")
            for mm in mismatches[:5]:  # Show first 5 mismatches
                if tokenizer is not None:
                    orig_token = tokenizer.decode([mm["original"]])
                    recon_token = tokenizer.decode([mm["reconstructed"]])
                else:
                    orig_token = model.detokenize([mm["original"]])
                    recon_token = model.detokenize([mm["reconstructed"]])
                logger.info(f"    pos {mm['position']}: '{orig_token}' ({mm['original']}) -> "
                           f"'{recon_token}' ({mm['reconstructed']})")
            if len(mismatches) > 5:
                logger.info(f"    ... and {len(mismatches) - 5} more")
        
        # Store result
        result = {
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "n_tokens": len(original_ids),
            "token_ids_original": original_ids.tolist(),
            "token_ids_reconstructed": reconstructed_ids.tolist(),
            "similarities": similarities.tolist(),
            "accuracy": accuracy,
            "mismatches": mismatches,
        }
        all_results.append(result)
    
    # Overall statistics
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    logger.info("\n" + "=" * 40)
    logger.info("Overall Results")
    logger.info("=" * 40)
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Correct reconstructions: {total_correct}")
    logger.info(f"Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    
    # Evaluate success criterion
    success = overall_accuracy >= 0.95
    
    logger.info("\n" + "=" * 40)
    logger.info("Success Evaluation")
    logger.info("=" * 40)
    logger.info(f"Accuracy >= 95%: {success} ({overall_accuracy * 100:.2f}%)")
    logger.info(f"OVERALL SUCCESS: {success}")
    
    # Analyze mismatch patterns
    all_mismatches = []
    for result in all_results:
        for mm in result["mismatches"]:
            mm["doc_id"] = result["doc_id"]
            all_mismatches.append(mm)
    
    if all_mismatches:
        logger.info(f"\nTotal mismatches: {len(all_mismatches)}")
        
        # Check if mismatches are semantically similar
        logger.info("\nAnalyzing mismatch patterns:")
        for mm in all_mismatches[:10]:
            if tokenizer is not None:
                orig_token = tokenizer.decode([mm["original"]])
                recon_token = tokenizer.decode([mm["reconstructed"]])
            else:
                orig_token = model.detokenize([mm["original"]])
                recon_token = model.detokenize([mm["reconstructed"]])
            logger.info(f"  '{orig_token}' -> '{recon_token}'")
    
    # Save results
    output = {
        "experiment": "2_token_decode",
        "model": model.config.name_or_path if hasattr(model, "config") and hasattr(model.config, "name_or_path") else getattr(model, "model_path", "unknown"),
        "vocab_size": index.vocab_size,
        "hidden_dim": index.hidden_size,
        "n_documents": len(documents),
        "total_tokens": total_tokens,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "documents": all_results,
        "all_mismatches": all_mismatches,
        "success": success,
    }
    
    output_path = results_dir / "results.json"
    save_json(output, output_path)
    logger.info(f"\nResults saved to: {output_path}")
    
    logger.info("Experiment 2 complete!")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
