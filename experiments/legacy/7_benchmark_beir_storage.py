#!/usr/bin/env python3
"""
Experiment 7: BEIR Storage & Quality Benchmark
Objective: Validate that INT8 quantization of Deltas + Lazy Whitening does not degrade search relevance on standard benchmarks.
"""

import sys
import logging
from pathlib import Path
import argparse
import time
from typing import List, Dict, Union, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from beir import util, LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.datasets.data_loader import GenericDataLoader

from sklearn.decomposition import PCA
from lib.io_utils import create_results_dir, save_json
from lib.model_loader import load_model, DEFAULT_MODEL
from lib.embedding_utils import KVEmbeddingExtractor
from lib.token_utils import extract_static_embeddings
from lib.compression import scalar_quantize, scalar_dequantize

# Set up logging for BEIR
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


class StorageAwareKVModel:
    """
    KV-Embedding Model that simulates the storage pipeline:
    1. Compute Deltas = H - E[id]
    2. fit_whitening(Deltas) [Calibration Phase]
    3. Quantize Deltas -> INT8 [Storage Phase]
    4. Dequantize -> Whiten -> Score [Retrieval Phase]
    """
    
    def __init__(self, model_name, load_in_4bit=False, quantize_int8=True):
        self.model, self.tokenizer = load_model(model_name, load_in_4bit=load_in_4bit)
        self.extractor = KVEmbeddingExtractor(self.model, self.tokenizer)
        self.quantize_int8 = quantize_int8
        
        self.pca_model = None
        self.whitening_mean = None
        self.whitening_components = None # W matrix
        
        # Statistics
        self.total_storage_bytes = 0
        self.total_tokens_encoded = 0
    
    def calibrate(self, corpus: Dict[str, Dict], n_samples: int = 2048):
        """
        Calibrate whitening parameters on a subset of the corpus.
        This must be called BEFORE encoding the full corpus.
        """
        logger.info(f"Calibrating on {n_samples} documents...")
        sample_ids = list(corpus.keys())[:n_samples]
        sample_texts = [f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip() for did in sample_ids]
        
        # 1. Extract Embeddings (H)
        # We need token_embeddings=True to get raw H for delta computation
        results = self.extractor.extract_embeddings(sample_texts, return_token_embeddings=True)
        token_embs_list = results["token_embeddings"]
        
        # 2. Extract Static Embeddings (E)
        static_data = extract_static_embeddings(self.model, self.tokenizer, sample_texts, use_prompt=True)
        static_embs_list = static_data["static_embeddings"]
        
        # 3. Compute Deltas
        all_deltas = []
        for i in range(len(sample_texts)):
            h = token_embs_list[i]
            e = static_embs_list[i]
            # Length align
            min_len = min(len(h), len(e))
            delta = h[:min_len] - e[:min_len]
            all_deltas.append(delta)
            
        all_deltas_flat = np.vstack(all_deltas).astype(np.float32)
        
        # 4. Fit PCA (Whitening)
        logger.info(f"Fitting PCA on {len(all_deltas_flat)} deltas...")
        n_components = min(all_deltas_flat.shape[0], all_deltas_flat.shape[1])
        self.pca_model = PCA(n_components=n_components, whiten=True)
        self.pca_model.fit(all_deltas_flat)
        
        self.whitening_mean = self.pca_model.mean_
        self.whitening_components = self.pca_model.components_.T # Transpose for x @ W
        # PCA transform is (x - mean) @ components_.T / sqrt(explained_variance)
        # But sklearn whiten=True already handles scale. 
        # Actually sklearn transform is: (X - mean) @ components_.T / sqrt(explained_variance)
        # Let's trust sklearn's transform method for now to avoid manual math errors
        
        logger.info("Calibration complete.")

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        """
        Encode queries:
        Query -> H -> Delta -> Whiten -> Pool -> Return
        (No quantization for queries usually, precision is cheap here)
        """
        return self._encode(queries, batch_size=batch_size, is_query=True)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        """
        Encode corpus:
        Doc -> H -> Delta -> Quantize(INT8) -> Measure Size -> Dequantize -> Whiten -> Pool
        """
        texts = [f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc in corpus]
        return self._encode(texts, batch_size=batch_size, is_query=False)

    def _encode(self, texts: List[str], batch_size: int = 8, is_query: bool = False) -> np.ndarray:
        if self.pca_model is None:
            raise RuntimeError("Model must be calibrated before encoding! Call calibrate() first.")
            
        final_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 1. Extract H
            h_results = self.extractor.extract_embeddings(batch_texts, return_token_embeddings=True)
            batch_h = h_results["token_embeddings"]
            
            # 2. Extract E (Static)
            s_results = extract_static_embeddings(self.model, self.tokenizer, batch_texts, use_prompt=True)
            batch_s = s_results["static_embeddings"]
            
            # Process per document
            batch_whitened_pooled = []
            
            for j in range(len(batch_texts)):
                h = batch_h[j]
                e = batch_s[j]
                min_len = min(len(h), len(e))
                h = h[:min_len]
                e = e[:min_len]
                
                # 3. Compute Delta
                delta = h - e
                
                if not is_query and self.quantize_int8:
                    # SIMULATE STORAGE: Quantize -> Count Bytes -> Dequantize
                    q_delta, params = scalar_quantize(delta, bits=8)
                    
                    # Track stats
                    self.total_tokens_encoded += len(delta)
                    self.total_storage_bytes += q_delta.nbytes + 40 # approx struct overhead
                    
                    # Dequantize for search
                    delta_search = scalar_dequantize(q_delta, params, delta.shape)
                else:
                    delta_search = delta
                
                # 4. Whiten
                # PCA transform expects (n_samples, n_features)
                delta_whitened = self.pca_model.transform(delta_search)
                
                # 5. Pool (Mean Pooling over tokens)
                # For dense retrieval, we need one vector per doc
                pooled = np.mean(delta_whitened, axis=0)
                batch_whitened_pooled.append(pooled)
            
            final_embeddings.append(np.stack(batch_whitened_pooled))
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Encoded {i}/{len(texts)}...")
                
        return np.concatenate(final_embeddings, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 7: BEIR Storage Benchmark")
    parser.add_argument("--dataset", type=str, default="scifact", help="BEIR dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit documents")
    parser.add_argument("--quantize_int8", action="store_true", default=True, help="Enable INT8 quantization")
    parser.add_argument("--no_quantize", action="store_false", dest="quantize_int8", help="Disable quantization (Float32 Baseline)")
    return parser.parse_args()

def main():
    args = parse_args()
    results_dir = create_results_dir("exp7_beir_storage")
    
    # 1. Load Data
    data_path = Path("data/beir_datasets") / args.dataset
    if not data_path.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
        logger.info(f"Downloading dataset {args.dataset}...")
        util.download_and_unzip(url, "data/beir_datasets")
        
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # Optional Limit
    if args.limit:
        logger.info(f"Limiting to {args.limit} queries...")
        # (Same limiting logic as exp4 - simplified here)
        limit_qids = list(queries.keys())[:args.limit]
        queries = {k: queries[k] for k in limit_qids}
        rel_docs = set()
        for qid in limit_qids:
            if qid in qrels:
                rel_docs.update(qrels[qid].keys())
        # Add randoms
        extra = args.limit * 5
        corpus_ids = list(corpus.keys())
        for cid in corpus_ids:
            if len(rel_docs) > len(corpus_ids) or extra <= 0: break
            if cid not in rel_docs:
                rel_docs.add(cid)
                extra -= 1
        corpus = {k: v for k, v in corpus.items() if k in rel_docs}
        qrels = {k: v for k, v in qrels.items() if k in limit_qids}
    
    logger.info(f"Loaded {len(corpus)} docs, {len(queries)} queries")
    
    # 2. Init Model
    model = StorageAwareKVModel(DEFAULT_MODEL, quantize_int8=args.quantize_int8)
    
    # 3. Calibrate
    model.calibrate(corpus, n_samples=2048)
    
    # 4. Evaluate
    retriever = EvaluateRetrieval(DenseRetrievalExactSearch(model, batch_size=32), score_function="cos_sim")
    
    start_time = time.perf_counter()
    results = retriever.retrieve(corpus, queries)
    duration = time.perf_counter() - start_time
    
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    
    # 5. Report
    logger.info("\nResults:")
    logger.info(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    logger.info(f"Recall@100: {recall['Recall@100']:.4f}")
    logger.info(f"Total Encoding Time: {duration:.2f}s")
    
    if args.quantize_int8:
        avg_bytes_per_token = model.total_storage_bytes / max(1, model.total_tokens_encoded)
        logger.info(f"Total Stored Size: {model.total_storage_bytes / 1024 / 1024:.2f} MB")
        logger.info(f"Avg Bytes/Token: {avg_bytes_per_token:.2f}")
    
    # Save
    out = {
        "dataset": args.dataset,
        "quantize_int8": args.quantize_int8,
        "metrics": {"ndcg": ndcg, "recall": recall},
        "storage": {
            "total_mb": model.total_storage_bytes / 1024 / 1024 if args.quantize_int8 else 0,
            "bytes_per_token": model.total_storage_bytes / max(1, model.total_tokens_encoded) if args.quantize_int8 else 0
        },
        "time_sec": duration
    }
    save_json(out, results_dir / "beir_storage_results.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
