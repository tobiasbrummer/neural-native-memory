#!/usr/bin/env python3
"""
Experiment 4: BEIR Benchmark Integration

Objective: Evaluate KV-Embedding performance on a standard retrieval benchmark (SciFact).
"""

import sys
import logging
from pathlib import Path
import argparse
from typing import List, Dict, Union
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.datasets.data_loader import GenericDataLoader

from sklearn.decomposition import PCA
from src.legacy.io_utils import create_results_dir, setup_logging, save_json
from src.legacy.model_loader import load_model, DEFAULT_MODEL
from src.legacy.embedding_utils import (
    KVEmbeddingExtractor, 
    apply_zscore
)

# Set up logging for BEIR
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


class KVEmbeddingModel:
    """Wrapper for KV-Embedding to be compatible with BEIR."""
    
    def __init__(self, model_name, load_in_4bit=False, whitening=False, zscore=False, prompt_template=None):
        self.model, self.tokenizer = load_model(model_name, load_in_4bit=load_in_4bit)
        self.extractor = KVEmbeddingExtractor(self.model, self.tokenizer)
        self.whitening = whitening
        self.zscore = zscore
        self.prompt_template = prompt_template
        
        # Override prompt if needed
        if prompt_template:
            self.extractor._wrap_with_prompt = lambda text: prompt_template.format(context=text)
            
        self.pca_model = None # Store PCA model for consistency
        
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        """Encode queries using the KV-Embedding extractor."""
        return self._encode(queries, batch_size=batch_size, is_query=True)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        """Encode corpus documents."""
        # BEIR corpus is a list of dicts {"title":..., "text":...}
        texts = [f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc in corpus]
        return self._encode(texts, batch_size=batch_size, is_query=False)

    def _encode(self, texts: List[str], batch_size: int = 8, is_query: bool = False) -> np.ndarray:
        all_embeddings = []
        
        # Batch processing
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            results = self.extractor.extract_embeddings(batch_texts, return_token_embeddings=False)
            all_embeddings.append(results["pooled_embeddings"])
            
            if i % (batch_size * 5) == 0:
                logger.info(f"Encoded {i}/{len(texts)} texts...")
                
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Post-Processing
        # Correct Logic: Fit PCA on Corpus, Apply to Queries
        if self.whitening:
            if not is_query: # Corpus -> Fit
                logger.info(f"Fitting PCA Whitening on corpus (N={len(embeddings)})...")
                n_components = min(embeddings.shape[0], embeddings.shape[1])
                self.pca_model = PCA(n_components=n_components, whiten=True)
                embeddings = self.pca_model.fit_transform(embeddings)
            else: # Query -> Transform
                if self.pca_model:
                    logger.info("Applying PCA Whitening to queries...")
                    embeddings = self.pca_model.transform(embeddings)
                else:
                    logger.warning("PCA Whitening requested but no model fitted (Queries encoded before Corpus?). Skipping.")
                
        if self.zscore:
            # Z-Score is typically per-batch or global? 
            # In Experiment 1 we did per-dataset. 
            # Ideally we calculate mean/std on corpus and apply to queries.
            # But standard zscore is just (x - mean) / std.
            # For this prototype we'll keep it simple: Per-set normalization. 
            # NOTE: This is technically leakage or mismatch if query distribution differs, but standard in simple baselines.
            embeddings = apply_zscore(embeddings)
            
        return embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 4: BEIR Benchmark")
    parser.add_argument("--dataset", type=str, default="scifact", help="BEIR dataset to download and evaluate")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load key model in 4-bit")
    parser.add_argument("--whitening", action="store_true", help="Apply Whitening")
    parser.add_argument("--zscore", action="store_true", help="Apply Z-Score")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt template")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents/queries for testing")
    return parser.parse_args()

def main():
    args = parse_args()
    results_dir = create_results_dir("exp4_beir")
    
    # 1. Download/Load Dataset
    data_path = Path("data/beir_datasets") / args.dataset
    if not data_path.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
        logger.info(f"Downloading dataset {args.dataset}...")
        util.download_and_unzip(url, "data/beir_datasets")
    
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    if args.limit:
        logger.info(f"Limiting to {args.limit} queries (and their relevant docs)...")
        # 1. Select top N queries
        limited_query_ids = list(queries.keys())[:args.limit]
        queries = {qid: queries[qid] for qid in limited_query_ids}
        
        # 2. Find all relevant doc IDs for these queries
        relevant_doc_ids = set()
        limited_qrels = {}
        
        for qid in limited_query_ids:
            if qid in qrels:
                limited_qrels[qid] = qrels[qid]
                relevant_doc_ids.update(qrels[qid].keys())
        
        # 3. Add some random negative docs to corpus if needed to reach a minimum size
        # Or just keep it to relevant ones + some extras for distraction
        all_corpus_ids = list(corpus.keys())
        extra_count = max(0, args.limit * 5 - len(relevant_doc_ids)) # Ensure some negatives
        for doc_id in all_corpus_ids:
            if extra_count <= 0:
                break
            if doc_id not in relevant_doc_ids:
                relevant_doc_ids.add(doc_id)
                extra_count -= 1
        
        corpus = {doc_id: corpus[doc_id] for doc_id in relevant_doc_ids if doc_id in corpus}
        qrels = limited_qrels
        
    logger.info(f"Loaded {len(corpus)} documents and {len(queries)} queries.")
    
    # 2. Initialize Model
    logger.info("Initializing KV-Embedding Model...")
    kv_model = KVEmbeddingModel(
        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
        whitening=args.whitening,
        zscore=args.zscore,
        prompt_template=args.prompt
    )
    
    # Pre-fit PCA if whitening is enabled
    if args.whitening:
        logger.info("Pre-fitting PCA on corpus subset for whitening...")
        # Sample representative subset (e.g., 2048 docs or all if less)
        # For N > D stability
        sample_texts = [f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc in list(corpus.values())[:2048]]
        
        # We need to extract raw embeddings first (without whitening/zscore, which are flags in the model)
        # So we temporarily disable them or access the raw extractor
        # kv_model.whitening is True, so encode_corpus would try to use it (and fail or warn)
        # We manualy extract using the inner extractor
        
        logger.info(f"Extracting embeddings for PCA fit (N={len(sample_texts)})...")
        # Reuse internal batching logic is hard without duplicating code
        # Let's just use kv_model._encode but with flags temporarily disabled
        
        temp_whitening = kv_model.whitening
        temp_zscore = kv_model.zscore
        kv_model.whitening = False
        kv_model.zscore = False
        
        raw_embeddings = kv_model._encode(sample_texts, batch_size=32, is_query=False)
        
        # Fit PCA
        logger.info("Fitting PCA...")
        n_components = min(raw_embeddings.shape[0], raw_embeddings.shape[1])
        pca = PCA(n_components=n_components, whiten=True)
        pca.fit(raw_embeddings)
        
        # Assign to model
        kv_model.pca_model = pca
        
        # Restore flags
        kv_model.whitening = temp_whitening
        kv_model.zscore = temp_zscore
        logger.info("PCA fitted and assigned to model.")

    # 3. Evaluate
    retriever = EvaluateRetrieval(DenseRetrievalExactSearch(kv_model, batch_size=32), score_function="cos_sim")
    
    logger.info("Starting Retrieval Evaluation...")
    results = retriever.retrieve(corpus, queries)
    
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    
    # Print Results
    logger.info("\nResults:")
    logger.info(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    logger.info(f"Recall@100: {recall['Recall@100']:.4f}")
    
    # Save Results
    output = {
        "dataset": args.dataset,
        "model": args.model,
        "config": {
            "load_in_4bit": args.load_in_4bit,
            "whitening": args.whitening,
            "zscore": args.zscore,
            "prompt": args.prompt
        },
        "metrics": {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision
        }
    }
    save_json(output, results_dir / "beir_results.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
