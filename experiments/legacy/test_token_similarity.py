#!/usr/bin/env python3
"""
Token-Level Similarity Test

Compares two sentences at the token level using KV-Embeddings with whitening.
Uses Scifact corpus for PCA fitting (whitening).
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from lib.io_utils import create_results_dir, setup_logging
from lib.model_loader import load_model, DEFAULT_MODEL
from lib.embedding_utils import KVEmbeddingExtractor, apply_zscore

logger = None


def load_scifact_texts(corpus_path: str, limit: int = 512):
    """Load texts from Scifact corpus for PCA fitting."""
    texts = []
    with open(corpus_path, 'r') as f:
        for line in f:
            if len(texts) >= limit:
                break
            doc = json.loads(line)
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            if len(text) > 50:
                texts.append(text)
    return texts


def fit_pca_on_corpus(extractor: KVEmbeddingExtractor, texts: list, batch_size: int = 32):
    """Fit PCA whitening on corpus embeddings."""
    logger.info(f"Fitting PCA on {len(texts)} corpus samples...")
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = extractor.extract_embeddings(batch, return_token_embeddings=False)
        all_embeddings.append(results["pooled_embeddings"])
        if i % (batch_size * 4) == 0:
            logger.info(f"  Encoded {i}/{len(texts)}...")
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    n_components = min(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(embeddings)
    
    logger.info(f"PCA fitted: {n_components} components")
    return pca


def extract_token_embeddings(extractor: KVEmbeddingExtractor, text: str, tokenizer, pca_model=None, apply_zs=True):
    """Extract token embeddings for a single text with optional whitening.
    
    Returns only the content tokens (excludes prompt suffix).
    """
    results = extractor.extract_embeddings([text], return_token_embeddings=True)
    
    token_embs = results["token_embeddings"][0]  # Shape: (seq_len, hidden_size)
    token_ids = results["token_ids"][0]
    
    # Find where the actual text ends and prompt suffix begins
    # The prompt format is: "{text}" Compress the Context in one word:
    # So we need to find the closing quote after the text
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    # Find the end of the quoted text (look for ." or just ")
    content_end_idx = len(token_ids)  # Default: all tokens
    
    # Strategy: Find the pattern where the text ends
    # The text is wrapped in quotes, so look for the closing quote followed by space
    for i, tok in enumerate(decoded_tokens):
        if tok.strip() == 'Compress' or tok.strip() == 'Com':
            # Found start of prompt suffix, cut here
            # Go back to include the closing quote and period if present
            content_end_idx = i
            break
    
    # Also skip the opening quote at the beginning
    content_start_idx = 0
    if decoded_tokens[0].strip() == '"':
        content_start_idx = 1
    
    # Slice to content only
    token_embs = token_embs[content_start_idx:content_end_idx]
    token_ids = token_ids[content_start_idx:content_end_idx]
    
    # Apply whitening if PCA model provided
    if pca_model is not None:
        token_embs = pca_model.transform(token_embs)
    
    # Apply Z-Score normalization
    if apply_zs:
        token_embs = apply_zscore(token_embs)
    
    return token_embs, token_ids


def decode_tokens(tokenizer, token_ids):
    """Decode token IDs to readable strings."""
    return [tokenizer.decode([tid]) for tid in token_ids]


def compute_token_similarity_matrix(embs1: np.ndarray, embs2: np.ndarray):
    """Compute cosine similarity between all token pairs."""
    return cosine_similarity(embs1, embs2)


def print_similarity_analysis(tokenizer, tokens1, tokens2, embs1, embs2, sim_matrix):
    """Print detailed similarity analysis."""
    decoded1 = decode_tokens(tokenizer, tokens1)
    decoded2 = decode_tokens(tokenizer, tokens2)
    
    print("\n" + "=" * 80)
    print("TOKEN-LEVEL SIMILARITY ANALYSIS")
    print("=" * 80)
    
    print(f"\nSatz 1 ({len(tokens1)} tokens): {decoded1}")
    print(f"Satz 2 ({len(tokens2)} tokens): {decoded2}")
    
    # Print similarity matrix
    print("\n" + "-" * 80)
    print("SIMILARITY MATRIX (Rows=Satz1, Cols=Satz2)")
    print("-" * 80)
    
    # Header
    header = "            " + " ".join([f"{d[:7]:>8}" for d in decoded2])
    print(header)
    
    # Rows
    for i, tok1 in enumerate(decoded1):
        row = f"{tok1[:10]:>10}  " + " ".join([f"{sim_matrix[i, j]:>8.3f}" for j in range(len(decoded2))])
        print(row)
    
    # Find best matches per token in Satz 1
    print("\n" + "-" * 80)
    print("BEST MATCHES (für jeden Token in Satz 1)")
    print("-" * 80)
    
    for i, tok1 in enumerate(decoded1):
        best_j = np.argmax(sim_matrix[i, :])
        best_sim = sim_matrix[i, best_j]
        best_tok2 = decoded2[best_j]
        print(f"  '{tok1}' → '{best_tok2}' (sim={best_sim:.3f})")
    
    # Diagonal alignment (if same length or aligned)
    print("\n" + "-" * 80)
    print("ALIGNED TOKENS (Position-by-Position)")
    print("-" * 80)
    
    min_len = min(len(tokens1), len(tokens2))
    for i in range(min_len):
        sim = sim_matrix[i, i]
        print(f"  Pos {i:2d}: '{decoded1[i]}' ↔ '{decoded2[i]}' = {sim:+.3f}")
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)
    
    # Average similarity
    avg_sim = np.mean(sim_matrix)
    max_sim = np.max(sim_matrix)
    min_sim = np.min(sim_matrix)
    
    # Diagonal average (aligned tokens)
    diag_avg = np.mean([sim_matrix[i, i] for i in range(min_len)])
    
    print(f"  Average similarity (all pairs): {avg_sim:.3f}")
    print(f"  Max similarity: {max_sim:.3f}")
    print(f"  Min similarity: {min_sim:.3f}")
    print(f"  Diagonal average (aligned): {diag_avg:.3f}")
    
    # Best overall match
    best_i, best_j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    print(f"  Best pair: '{decoded1[best_i]}' ↔ '{decoded2[best_j]}' = {sim_matrix[best_i, best_j]:.3f}")


def main():
    global logger
    
    parser = argparse.ArgumentParser(description="Token-Level Similarity Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Model to use")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Use 4-bit quantization")
    parser.add_argument("--corpus_file", type=str, default="data/beir_datasets/scifact/corpus.jsonl",
                        help="Corpus file for PCA fitting")
    parser.add_argument("--pca_samples", type=int, default=256, help="Number of samples for PCA fitting")
    parser.add_argument("--no_whitening", action="store_true", help="Disable whitening")
    parser.add_argument("--no_zscore", action="store_true", help="Disable z-score normalization")
    args = parser.parse_args()
    
    # Setup
    results_dir = create_results_dir("token_similarity")
    logger = setup_logging("token_similarity", results_dir)
    
    # The two sentences to compare
    sentence1 = "Die Bank informiert den Kunden 8 Wochen im Voraus über Zinsänderungen."
    sentence2 = "Die Bank informiert auf Nachfrage über Zinsänderungen."
    
    logger.info("=" * 60)
    logger.info("Token-Level Similarity Test")
    logger.info(f"Model: {args.model}")
    logger.info(f"Whitening: {not args.no_whitening}")
    logger.info(f"Z-Score: {not args.no_zscore}")
    logger.info("=" * 60)
    
    logger.info(f"\nSatz 1: {sentence1}")
    logger.info(f"Satz 2: {sentence2}")
    
    # Load model
    logger.info("\nLoading model...")
    model, tokenizer = load_model(args.model, load_in_4bit=args.load_in_4bit)
    
    # Initialize extractor
    extractor = KVEmbeddingExtractor(model, tokenizer)
    
    # Fit PCA if whitening enabled
    pca_model = None
    if not args.no_whitening:
        logger.info(f"\nLoading corpus from {args.corpus_file}...")
        corpus_texts = load_scifact_texts(args.corpus_file, limit=args.pca_samples)
        pca_model = fit_pca_on_corpus(extractor, corpus_texts)
    
    # Extract token embeddings for both sentences
    logger.info("\nExtracting token embeddings for Satz 1...")
    embs1, tokens1 = extract_token_embeddings(
        extractor, sentence1, tokenizer,
        pca_model=pca_model, 
        apply_zs=not args.no_zscore
    )
    
    logger.info("Extracting token embeddings for Satz 2...")
    embs2, tokens2 = extract_token_embeddings(
        extractor, sentence2, tokenizer,
        pca_model=pca_model, 
        apply_zs=not args.no_zscore
    )
    
    logger.info(f"Satz 1: {len(tokens1)} tokens, embedding shape {embs1.shape}")
    logger.info(f"Satz 2: {len(tokens2)} tokens, embedding shape {embs2.shape}")
    
    # Compute similarity matrix
    logger.info("\nComputing similarity matrix...")
    sim_matrix = compute_token_similarity_matrix(embs1, embs2)
    
    # Print analysis
    print_similarity_analysis(tokenizer, tokens1, tokens2, embs1, embs2, sim_matrix)
    
    # Save results
    output = {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "tokens1": decode_tokens(tokenizer, tokens1),
        "tokens2": decode_tokens(tokenizer, tokens2),
        "similarity_matrix": sim_matrix.tolist(),
        "config": {
            "model": args.model,
            "whitening": not args.no_whitening,
            "zscore": not args.no_zscore,
            "pca_samples": args.pca_samples if not args.no_whitening else 0,
        }
    }
    
    output_path = results_dir / "token_similarity_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
