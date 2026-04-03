#!/usr/bin/env python3
"""Experiment 16 (NNM): Hybrid Sparse+Dense Retrieval on BEIR/SciFact.

Validates the NNMA search pipeline by combining:
  - Token-ID BM25 (sparse, lexical matching on subword token IDs)
  - Semantic Delta matching (dense, whitened cosine similarity)
  - Hybrid score fusion (linear interpolation + Reciprocal Rank Fusion)

Self-contained: no Qdrant dependency. All retrieval computed in-memory.

Comparison target: eval_compare baseline (NDCG@10 = 0.023 raw, 0.273 whitened).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _setup_logging(name: str, results_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(results_dir / "log.txt")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _create_results_dir(prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("results") / f"{prefix}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(data: object, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# BEIR data loading
# ---------------------------------------------------------------------------

def _safe_text(doc: dict) -> str:
    title = str(doc.get("title", "") or "").strip()
    text = str(doc.get("text", "") or "").strip()
    joined = f"{title} {text}".strip()
    return joined if joined else text


def load_beir_corpus(dataset: str, beir_path: Optional[str]) -> Dict[str, str]:
    """Load BEIR corpus as {doc_id: text}. Tries local JSONL, then HuggingFace."""
    if beir_path:
        corpus_path = Path(beir_path)
    else:
        corpus_path = REPO_ROOT / "data" / "beir_datasets" / dataset / "corpus.jsonl"

    corpus: Dict[str, str] = {}

    if corpus_path.exists():
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = _safe_text(row)
                if text:
                    corpus[str(row["_id"])] = text
        return corpus

    # Fallback: HuggingFace datasets
    from datasets import load_dataset
    ds = load_dataset("BeIR/" + dataset, "corpus", split="corpus")
    for row in ds:
        text = _safe_text(row)
        if text:
            corpus[str(row["_id"])] = text
    return corpus


def load_beir_queries(dataset: str, beir_path: Optional[str]) -> Dict[str, str]:
    """Load BEIR queries as {query_id: text}."""
    if beir_path:
        queries_path = Path(beir_path).parent / "queries.jsonl"
    else:
        queries_path = REPO_ROOT / "data" / "beir_datasets" / dataset / "queries.jsonl"

    queries: Dict[str, str] = {}

    if queries_path.exists():
        with queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get("text", "")).strip()
                if text:
                    queries[str(row["_id"])] = text
        return queries

    from datasets import load_dataset
    ds = load_dataset("BeIR/" + dataset, "queries", split="queries")
    for row in ds:
        text = str(row.get("text", "")).strip()
        if text:
            queries[str(row["_id"])] = text
    return queries


def load_beir_qrels(dataset: str, split: str, beir_path: Optional[str]) -> Dict[str, Dict[str, int]]:
    """Load BEIR qrels as {query_id: {doc_id: relevance_score}}."""
    if beir_path:
        qrels_path = Path(beir_path).parent / "qrels" / f"{split}.tsv"
    else:
        qrels_path = REPO_ROOT / "data" / "beir_datasets" / dataset / "qrels" / f"{split}.tsv"

    qrels: Dict[str, Dict[str, int]] = {}

    if qrels_path.exists():
        import csv
        with qrels_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid = str(row["query-id"])
                did = str(row["corpus-id"])
                score = int(float(row["score"]))
                if score > 0:
                    qrels.setdefault(qid, {})[did] = score
        return qrels

    from datasets import load_dataset
    ds = load_dataset("BeIR/" + dataset, "default", split=split)
    for row in ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(float(row["score"]))
        if score > 0:
            qrels.setdefault(qid, {})[did] = score
    return qrels


# ---------------------------------------------------------------------------
# Evaluation metrics (NDCG, Recall, MRR, Hit Rate)
# ---------------------------------------------------------------------------

def _dcg_at_k(rels: Sequence[float], k: int) -> float:
    return sum(
        (2.0 ** rels[i] - 1.0) / math.log2(i + 2)
        for i in range(min(len(rels), k))
    )


def evaluate_ranking(
    ranked_doc_ids: List[str],
    qrels_for_query: Dict[str, int],
    top_k: int = 10,
) -> Dict[str, float]:
    """Compute NDCG@k, Recall@k, MRR@k, Hit@k for a single query."""
    rels = [float(qrels_for_query.get(did, 0)) for did in ranked_doc_ids[:top_k]]

    # NDCG
    dcg = _dcg_at_k(rels, top_k)
    ideal_rels = sorted(qrels_for_query.values(), reverse=True)
    idcg = _dcg_at_k([float(r) for r in ideal_rels], top_k)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Recall
    n_relevant = len(qrels_for_query)
    n_retrieved_relevant = sum(1 for did in ranked_doc_ids[:top_k] if did in qrels_for_query)
    recall = n_retrieved_relevant / n_relevant if n_relevant > 0 else 0.0

    # MRR
    mrr = 0.0
    for i, did in enumerate(ranked_doc_ids[:top_k]):
        if did in qrels_for_query:
            mrr = 1.0 / (i + 1)
            break

    # Hit Rate
    hit = 1.0 if any(did in qrels_for_query for did in ranked_doc_ids[:top_k]) else 0.0

    return {"ndcg@10": ndcg, "recall@10": recall, "mrr@10": mrr, "hit_rate@10": hit}


def aggregate_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: float(np.mean([q[k] for q in per_query])) for k in keys}


# ---------------------------------------------------------------------------
# Phase 2: Token-ID BM25
# ---------------------------------------------------------------------------

@dataclass
class BM25Index:
    """BM25 inverted index on subword token IDs."""
    doc_ids: List[str]
    doc_token_counts: List[Counter]  # token_id -> count per doc
    doc_lengths: List[int]           # number of tokens per doc
    avg_dl: float                    # average document length
    idf: Dict[int, float]           # token_id -> IDF score
    n_docs: int

    @staticmethod
    def build(
        doc_ids: List[str],
        doc_token_ids: List[List[int]],
        exclude_token_ids: Optional[set] = None,
    ) -> "BM25Index":
        """Build BM25 index from tokenized documents."""
        if exclude_token_ids is None:
            exclude_token_ids = set()

        n = len(doc_ids)
        doc_counts = []
        doc_lengths = []
        df: Counter = Counter()  # document frequency per token

        for tokens in doc_token_ids:
            filtered = [t for t in tokens if t not in exclude_token_ids]
            counts = Counter(filtered)
            doc_counts.append(counts)
            doc_lengths.append(len(filtered))
            for token_id in counts:
                df[token_id] += 1

        avg_dl = sum(doc_lengths) / max(n, 1)

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)  (Robertson BM25 variant)
        idf = {}
        for token_id, freq in df.items():
            idf[token_id] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

        return BM25Index(
            doc_ids=doc_ids,
            doc_token_counts=doc_counts,
            doc_lengths=doc_lengths,
            avg_dl=avg_dl,
            idf=idf,
            n_docs=n,
        )

    def score_query(
        self,
        query_token_ids: List[int],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Tuple[str, float]]:
        """Score all documents against a query. Returns sorted (doc_id, score) pairs."""
        query_counts = Counter(query_token_ids)
        scores = []

        for i in range(self.n_docs):
            score = 0.0
            dl = self.doc_lengths[i]
            doc_counts = self.doc_token_counts[i]

            for token_id, qf in query_counts.items():
                if token_id not in self.idf:
                    continue
                tf = doc_counts.get(token_id, 0)
                if tf == 0:
                    continue
                idf_val = self.idf[token_id]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avg_dl))
                score += idf_val * tf_norm

            scores.append((self.doc_ids[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ---------------------------------------------------------------------------
# Phase 3: Semantic Delta Retrieval
# ---------------------------------------------------------------------------

def extract_token_deltas(
    model,
    text: str,
    retrieval_layer: int,
    prepend_bos: bool,
    max_tokens: int = 128,
) -> np.ndarray:
    """Extract per-token semantic deltas (contextual - static) at retrieval_layer.

    Returns (n_tokens, d_model) float32 array.
    Truncates to max_tokens to avoid OOM.
    """
    layer_name = f"blocks.{retrieval_layer}.hook_resid_post"
    tokens = model.to_tokens(text, prepend_bos=prepend_bos)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            return_type=None,
            names_filter=[layer_name],
            remove_batch_dim=False,
            prepend_bos=False,
        )

    contextual = cache[layer_name][0].detach().to(torch.float32).cpu().numpy()  # (seq, d)
    static = model.W_E[tokens[0]].detach().to(torch.float32).cpu().numpy()      # (seq, d)
    delta = (contextual - static).astype(np.float32)

    if delta.shape[0] > max_tokens:
        delta = delta[:max_tokens]

    return delta


def fit_whitening_transform(
    deltas: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Z-Score + PCA Whitening. Returns (mean, std, whitening_matrix)."""
    x = deltas.astype(np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    std = centered.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    normed = centered / std

    n = max(1, int(normed.shape[0] - 1))
    cov = (normed.T @ normed) / float(n)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, eps)
    inv_sqrt = 1.0 / np.sqrt(evals)
    W = (evecs * inv_sqrt) @ evecs.T

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
        W.astype(np.float32),
    )


def apply_whitening(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    W: np.ndarray,
    l2_normalize: bool = True,
) -> np.ndarray:
    """Apply Z-Score + PCA Whitening + optional L2 normalization."""
    y = x.astype(np.float32, copy=True)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    y = (y - mean.reshape(1, -1)) / std.reshape(1, -1)
    y = y @ W
    if l2_normalize:
        norms = np.linalg.norm(y, axis=1, keepdims=True)
        y = y / np.maximum(norms, 1e-10)
    if was_1d := (x.ndim == 1):
        return y.squeeze(0)
    return y


def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query vectors and corpus vectors.

    Args:
        queries: (n_queries, d_model)
        corpus: (n_corpus, d_model)

    Returns: (n_queries, n_corpus)
    """
    # L2 normalize
    q_norm = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-10)
    c_norm = corpus / np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-10)
    return q_norm @ c_norm.T


# ---------------------------------------------------------------------------
# Phase 3b: ColBERT-style in-memory scoring
# ---------------------------------------------------------------------------

def batch_score_colbert(
    query_tokens: np.ndarray,      # (q_len, d) – L2-normed
    all_doc_tokens: np.ndarray,    # (N_total, d) – L2-normed flat matrix
    doc_boundaries: List[int],     # length n_docs+1, token start/end per doc
    top_q_tokens: int = 3,
    min_token_hits: int = 2,
) -> np.ndarray:
    """Score all docs for one query using ColBERT MaxSim + mean_top_k aggregation.

    For each query token, find the best-matching doc token (MaxSim).
    Sort those per-query-token scores, average the top-k.
    Docs where fewer than min_token_hits query tokens get a positive MaxSim are zeroed.

    Returns (n_docs,) float32 scores.
    """
    sim_all = query_tokens @ all_doc_tokens.T  # (q_len, N_total)
    n_docs = len(doc_boundaries) - 1
    scores = np.zeros(n_docs, dtype=np.float32)

    for i in range(n_docs):
        start, end = doc_boundaries[i], doc_boundaries[i + 1]
        if end <= start:
            continue
        max_per_q = sim_all[:, start:end].max(axis=1)      # (q_len,)
        if int((max_per_q > 0.0).sum()) < min_token_hits:
            continue
        sorted_max = np.sort(max_per_q)[::-1]
        k = min(top_q_tokens, len(sorted_max))
        scores[i] = float(sorted_max[:k].mean())

    return scores


# ---------------------------------------------------------------------------
# Phase 4: Hybrid Score Fusion
# ---------------------------------------------------------------------------

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1] per query."""
    mn = scores.min()
    mx = scores.max()
    if mx - mn < 1e-10:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def hybrid_linear(
    sparse_scores: np.ndarray,
    dense_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linear interpolation: alpha * sparse + (1 - alpha) * dense."""
    s_norm = min_max_normalize(sparse_scores)
    d_norm = min_max_normalize(dense_scores)
    return alpha * s_norm + (1.0 - alpha) * d_norm


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF(d) = sum over rankings of 1 / (k + rank(d))
    """
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp16: Hybrid Sparse+Dense Retrieval on BEIR"
    )
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    # Data
    parser.add_argument("--beir-dataset", type=str, default="scifact",
                        help="BEIR dataset name")
    parser.add_argument("--beir-path", type=str, default=None,
                        help="Path to local corpus.jsonl")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-corpus", type=int, default=2000,
                        help="Max corpus documents")
    parser.add_argument("--max-queries", type=int, default=100,
                        help="Max queries to evaluate")
    # Retrieval
    parser.add_argument("--retrieval-layer", type=int, default=None,
                        help="Layer for delta extraction (default: 90%% of depth)")
    parser.add_argument("--max-doc-tokens", type=int, default=128,
                        help="Max tokens per document for token-level extraction")
    parser.add_argument("--top-q-tokens", type=int, default=3,
                        help="Mean of top-k query token MaxSim scores (ColBERT aggregation)")
    parser.add_argument("--min-token-hits", type=int, default=2,
                        help="Min query tokens with positive MaxSim to score a doc")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K for evaluation metrics")
    # BM25
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    # Hybrid
    parser.add_argument("--alpha-steps", type=int, default=11,
                        help="Number of alpha values to sweep (0.0 to 1.0)")
    # Phase control
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip Phase 3 (semantic deltas). Only run BM25.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = _create_results_dir("nnm_exp16")
    logger = _setup_logging("nnm_exp16_hybrid_retrieval_tl", results_dir)

    # ===================================================================
    # Phase 1: Load BEIR data
    # ===================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Load BEIR/%s data", args.beir_dataset)
    logger.info("=" * 60)

    corpus = load_beir_corpus(args.beir_dataset, args.beir_path)
    queries = load_beir_queries(args.beir_dataset, args.beir_path)
    qrels = load_beir_qrels(args.beir_dataset, args.split, args.beir_path)

    logger.info("Loaded: %d corpus docs, %d queries, %d qrels",
                len(corpus), len(queries), len(qrels))

    # Filter queries to those with qrels
    valid_qids = [qid for qid in queries if qid in qrels]
    if args.max_queries and len(valid_qids) > args.max_queries:
        import random
        random.seed(42)
        valid_qids = sorted(random.sample(valid_qids, args.max_queries))
    logger.info("Using %d queries with relevance judgments", len(valid_qids))

    # Subset corpus
    corpus_ids = list(corpus.keys())
    if args.max_corpus and len(corpus_ids) > args.max_corpus:
        # Keep all relevant docs, sample the rest
        relevant_doc_ids = set()
        for qid in valid_qids:
            relevant_doc_ids.update(qrels[qid].keys())

        other_ids = [did for did in corpus_ids if did not in relevant_doc_ids]
        import random
        random.seed(42)
        n_sample = min(args.max_corpus - len(relevant_doc_ids), len(other_ids))
        sampled = random.sample(other_ids, max(0, n_sample))
        corpus_ids = sorted(list(relevant_doc_ids & set(corpus_ids)) + sampled)

    corpus_texts = {did: corpus[did] for did in corpus_ids if did in corpus}
    corpus_id_list = list(corpus_texts.keys())
    logger.info("Corpus subset: %d documents (incl. all relevant docs)", len(corpus_id_list))

    # ===================================================================
    # Phase 2: Token-ID BM25
    # ===================================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Token-ID BM25 Retrieval")
    logger.info("=" * 60)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    # Get special/exclude token IDs
    exclude_ids = set(tokenizer.all_special_ids)
    logger.info("Excluding %d special token IDs from BM25", len(exclude_ids))

    # Tokenize corpus
    logger.info("Tokenizing %d corpus documents...", len(corpus_id_list))
    corpus_token_ids: List[List[int]] = []
    for i, did in enumerate(corpus_id_list):
        tokens = tokenizer.encode(corpus_texts[did], add_special_tokens=False)
        corpus_token_ids.append(tokens)
        if (i + 1) % 500 == 0:
            logger.info("  ... %d/%d tokenized", i + 1, len(corpus_id_list))

    # Build BM25 index
    logger.info("Building BM25 index...")
    bm25 = BM25Index.build(corpus_id_list, corpus_token_ids, exclude_ids)
    logger.info("BM25 index: %d docs, %d unique tokens, avg_dl=%.1f",
                bm25.n_docs, len(bm25.idf), bm25.avg_dl)

    # Tokenize and score queries
    logger.info("Scoring %d queries with BM25...", len(valid_qids))
    bm25_rankings: Dict[str, List[Tuple[str, float]]] = {}
    bm25_per_query_metrics: List[Dict[str, float]] = []

    for qid in valid_qids:
        q_tokens = tokenizer.encode(queries[qid], add_special_tokens=False)
        ranked = bm25.score_query(q_tokens, k1=args.bm25_k1, b=args.bm25_b)
        bm25_rankings[qid] = ranked
        ranked_ids = [did for did, _ in ranked]
        metrics = evaluate_ranking(ranked_ids, qrels[qid], top_k=args.top_k)
        bm25_per_query_metrics.append(metrics)

    bm25_metrics = aggregate_metrics(bm25_per_query_metrics)
    logger.info("BM25 results: %s", {k: f"{v:.4f}" for k, v in bm25_metrics.items()})

    # ===================================================================
    # Phase 3: Token-Level Semantic Delta Retrieval (optional)
    # ===================================================================
    dense_metrics: Optional[Dict[str, float]] = None
    colbert_score_matrix: Optional[np.ndarray] = None
    retrieval_layer: int = 0

    if not args.skip_dense:
        logger.info("=" * 60)
        logger.info("PHASE 3: Token-Level Semantic Delta Retrieval (ColBERT-style)")
        logger.info("=" * 60)

        from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder

        config = KVEmbeddingConfig(
            model_name=args.model,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            load_in_4bit=args.load_in_4bit,
        )
        embedder = TransformerLensKVEmbedder(config)
        model = embedder.model
        prepend_bos = embedder._prepend_bos

        # Layer default: ~90% of depth (matches eval_compare Qwen2-1.5B layer 26/28)
        n_layers = model.cfg.n_layers
        retrieval_layer = args.retrieval_layer if args.retrieval_layer is not None else int(n_layers * 0.9)
        logger.info("Using retrieval layer %d (of %d, %.0f%%)",
                    retrieval_layer, n_layers, retrieval_layer / n_layers * 100)

        # --- Extract per-token deltas for corpus ---
        logger.info("Extracting token-level deltas for %d corpus docs (max %d tokens each)...",
                    len(corpus_id_list), args.max_doc_tokens)
        corpus_token_arrays: List[np.ndarray] = []
        for i, did in enumerate(corpus_id_list):
            arr = extract_token_deltas(
                model, corpus_texts[did], retrieval_layer, prepend_bos, args.max_doc_tokens
            )
            corpus_token_arrays.append(arr)
            if (i + 1) % 100 == 0:
                logger.info("  ... %d/%d corpus token-deltas extracted", i + 1, len(corpus_id_list))

        # --- Fit whitening on all corpus tokens (subsample if needed) ---
        all_tokens_flat = np.concatenate(corpus_token_arrays, axis=0)  # (N_total, d)
        logger.info("Total corpus tokens for whitening fit: %d", all_tokens_flat.shape[0])

        max_fit_tokens = 50_000
        if all_tokens_flat.shape[0] > max_fit_tokens:
            rng = np.random.RandomState(42)
            idx = rng.choice(all_tokens_flat.shape[0], max_fit_tokens, replace=False)
            fit_sample = all_tokens_flat[idx]
        else:
            fit_sample = all_tokens_flat

        logger.info("Fitting whitening transform on %d token samples...", fit_sample.shape[0])
        w_mean, w_std, W = fit_whitening_transform(fit_sample)
        logger.info("Whitening matrix shape: %s", W.shape)

        # --- Apply whitening + L2-normalize each token, build flat matrix ---
        corpus_whitened_arrays = [
            apply_whitening(arr, w_mean, w_std, W, l2_normalize=True)
            for arr in corpus_token_arrays
        ]
        doc_boundaries: List[int] = [0]
        for arr in corpus_whitened_arrays:
            doc_boundaries.append(doc_boundaries[-1] + arr.shape[0])

        T_all = np.concatenate(corpus_whitened_arrays, axis=0).astype(np.float32)
        logger.info("Flat token matrix: %s  (%.1f MB)",
                    T_all.shape, T_all.nbytes / 1e6)

        # --- Extract query token deltas ---
        logger.info("Extracting token-level deltas for %d queries...", len(valid_qids))
        query_token_arrays: Dict[str, np.ndarray] = {}
        for i, qid in enumerate(valid_qids):
            arr = extract_token_deltas(
                model, queries[qid], retrieval_layer, prepend_bos, args.max_doc_tokens
            )
            query_token_arrays[qid] = apply_whitening(arr, w_mean, w_std, W, l2_normalize=True)
            if (i + 1) % 50 == 0:
                logger.info("  ... %d/%d query token-deltas extracted", i + 1, len(valid_qids))

        # --- ColBERT scoring ---
        logger.info("Scoring with ColBERT MaxSim (top_q=%d, min_hits=%d)...",
                    args.top_q_tokens, args.min_token_hits)
        colbert_score_matrix = np.zeros((len(valid_qids), len(corpus_id_list)), dtype=np.float32)
        for qi, qid in enumerate(valid_qids):
            Q = query_token_arrays[qid].astype(np.float32)
            colbert_score_matrix[qi] = batch_score_colbert(
                Q, T_all, doc_boundaries, args.top_q_tokens, args.min_token_hits
            )

        dense_per_query: List[Dict[str, float]] = []
        dense_rankings: Dict[str, List[Tuple[str, float]]] = {}
        for qi, qid in enumerate(valid_qids):
            sorted_idx = np.argsort(-colbert_score_matrix[qi])
            ranked = [(corpus_id_list[j], float(colbert_score_matrix[qi, j])) for j in sorted_idx]
            dense_rankings[qid] = ranked
            ranked_ids = [did for did, _ in ranked]
            metrics = evaluate_ranking(ranked_ids, qrels[qid], top_k=args.top_k)
            dense_per_query.append(metrics)

        dense_metrics = aggregate_metrics(dense_per_query)
        logger.info("Dense (ColBERT token-level) results: %s",
                    {k: f"{v:.4f}" for k, v in dense_metrics.items()})

        # ==================================================================
        # Phase 4: Hybrid Score Fusion
        # ==================================================================
        logger.info("=" * 60)
        logger.info("PHASE 4: Hybrid Score Fusion")
        logger.info("=" * 60)

        # BM25 score matrix
        bm25_score_matrix = np.zeros((len(valid_qids), len(corpus_id_list)), dtype=np.float32)
        doc_id_to_idx = {did: i for i, did in enumerate(corpus_id_list)}
        for qi, qid in enumerate(valid_qids):
            for did, score in bm25_rankings[qid]:
                if did in doc_id_to_idx:
                    bm25_score_matrix[qi, doc_id_to_idx[did]] = score

        # Alpha sweep (linear interpolation: alpha * BM25 + (1-alpha) * ColBERT)
        alphas = np.linspace(0.0, 1.0, args.alpha_steps)
        alpha_results = []

        for alpha in alphas:
            hybrid_per_query: List[Dict[str, float]] = []
            for qi, qid in enumerate(valid_qids):
                hybrid_scores = hybrid_linear(
                    bm25_score_matrix[qi],
                    colbert_score_matrix[qi],
                    alpha=alpha,
                )
                sorted_idx = np.argsort(-hybrid_scores)
                ranked_ids = [corpus_id_list[j] for j in sorted_idx]
                metrics = evaluate_ranking(ranked_ids, qrels[qid], top_k=args.top_k)
                hybrid_per_query.append(metrics)

            hybrid_metrics = aggregate_metrics(hybrid_per_query)
            alpha_results.append({"alpha": float(alpha), "metrics": hybrid_metrics})
            logger.info("  alpha=%.2f: NDCG@10=%.4f  Recall@10=%.4f  Hit@10=%.4f",
                        alpha, hybrid_metrics["ndcg@10"],
                        hybrid_metrics["recall@10"], hybrid_metrics["hit_rate@10"])

        best_alpha_result = max(alpha_results, key=lambda r: r["metrics"]["ndcg@10"])
        logger.info("Best alpha: %.2f -> NDCG@10=%.4f",
                    best_alpha_result["alpha"], best_alpha_result["metrics"]["ndcg@10"])

        # Reciprocal Rank Fusion
        logger.info("Computing Reciprocal Rank Fusion...")
        rrf_per_query: List[Dict[str, float]] = []
        for qi, qid in enumerate(valid_qids):
            bm25_ranking = [did for did, _ in bm25_rankings[qid]]
            dense_ranking = [did for did, _ in dense_rankings[qid]]
            rrf_ranked = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)
            ranked_ids = [did for did, _ in rrf_ranked]
            metrics = evaluate_ranking(ranked_ids, qrels[qid], top_k=args.top_k)
            rrf_per_query.append(metrics)

        rrf_metrics = aggregate_metrics(rrf_per_query)
        logger.info("RRF results: %s", {k: f"{v:.4f}" for k, v in rrf_metrics.items()})

    # ===================================================================
    # Phase 5: Summary
    # ===================================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: Summary")
    logger.info("=" * 60)

    output: Dict[str, Any] = {
        "experiment": "nnm_exp16_hybrid_retrieval_tl",
        "model": args.model,
        "dataset": args.beir_dataset,
        "split": args.split,
        "config": {
            "dtype": args.dtype,
            "load_in_4bit": args.load_in_4bit,
            "max_corpus": len(corpus_id_list),
            "max_queries": len(valid_qids),
            "top_k": args.top_k,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "bm25_vocab_size": len(bm25.idf),
            "bm25_avg_dl": bm25.avg_dl,
        },
        "bm25": bm25_metrics,
    }

    # Summary table
    logger.info("")
    logger.info("%-30s  NDCG@10   Recall@10  Hit@10", "Method")
    logger.info("-" * 70)
    logger.info("%-30s  %.4f    %.4f     %.4f", "eval_compare (Qwen2-1.5B token)",
                0.273, 0.334, 0.370)
    logger.info("%-30s  %.4f    %.4f     %.4f", "BM25 (Token-ID)",
                bm25_metrics["ndcg@10"], bm25_metrics["recall@10"],
                bm25_metrics["hit_rate@10"])

    if not args.skip_dense and dense_metrics:
        output["dense_colbert"] = dense_metrics
        output["retrieval_layer"] = retrieval_layer
        output["colbert_config"] = {
            "max_doc_tokens": args.max_doc_tokens,
            "top_q_tokens": args.top_q_tokens,
            "min_token_hits": args.min_token_hits,
        }

        logger.info("%-30s  %.4f    %.4f     %.4f",
                    f"Dense ColBERT (layer {retrieval_layer})",
                    dense_metrics["ndcg@10"], dense_metrics["recall@10"],
                    dense_metrics["hit_rate@10"])

        output["hybrid_alpha_sweep"] = alpha_results
        output["hybrid_best_alpha"] = best_alpha_result
        output["hybrid_rrf"] = rrf_metrics

        logger.info("%-30s  %.4f    %.4f     %.4f",
                    f"Hybrid (alpha={best_alpha_result['alpha']:.2f})",
                    best_alpha_result["metrics"]["ndcg@10"],
                    best_alpha_result["metrics"]["recall@10"],
                    best_alpha_result["metrics"]["hit_rate@10"])
        logger.info("%-30s  %.4f    %.4f     %.4f", "Hybrid (RRF k=60)",
                    rrf_metrics["ndcg@10"], rrf_metrics["recall@10"],
                    rrf_metrics["hit_rate@10"])

    logger.info("-" * 65)

    _save_json(output, results_dir / "results.json")
    logger.info("Results saved to %s", results_dir / "results.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
