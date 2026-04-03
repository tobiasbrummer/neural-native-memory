"""
Token utilities for KV-Embedding experiments.

Handles static embedding extraction and token-ID reconstruction via nearest-neighbor search.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_loader import get_embedding_matrix

logger = logging.getLogger(__name__)


class TokenEmbeddingIndex:
    """
    Index for mapping between token IDs and static embeddings.
    
    Enables:
    - Fast lookup: token_id -> static_embedding
    - Reverse mapping: static_embedding -> token_id (via nearest neighbor)
    """
    
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        use_faiss: bool = False,
        embedding_matrix: Optional[np.ndarray] = None,
    ):
        """
        Initialize the token embedding index.
        
        Args:
            model: The language model
            use_faiss: Whether to use FAISS for fast NN search (requires faiss-gpu)
            embedding_matrix: Optional precomputed embedding matrix (vocab_size, hidden_size)
        """
        if embedding_matrix is not None:
            self.embedding_matrix = np.asarray(embedding_matrix, dtype=np.float32)
        else:
            if model is None:
                raise ValueError("Either model or embedding_matrix must be provided")
            self.embedding_matrix = get_embedding_matrix(model).cpu().float().numpy()
        self.vocab_size, self.hidden_size = self.embedding_matrix.shape
        self.use_faiss = use_faiss
        self._faiss_index = None
        
        logger.info(f"Initialized embedding index: {self.vocab_size} tokens, "
                    f"{self.hidden_size} dimensions")
        
        # Precompute normalized embeddings for cosine similarity
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        self._normalized = self.embedding_matrix / np.maximum(norms, 1e-10)
        
        if use_faiss:
            self._init_faiss_index()
    
    def _init_faiss_index(self) -> None:
        """Initialize FAISS index for fast similarity search."""
        try:
            import faiss
            
            # Use inner product on normalized vectors = cosine similarity
            self._faiss_index = faiss.IndexFlatIP(self.hidden_size)
            self._faiss_index.add(self._normalized.astype(np.float32))
            
            logger.info("FAISS index initialized")
        except ImportError:
            logger.warning("FAISS not available, falling back to numpy search")
            self.use_faiss = False
    
    def get_embedding(self, token_id: int) -> np.ndarray:
        """
        Get the static embedding for a token ID.
        
        Args:
            token_id: The token ID
        
        Returns:
            Static embedding vector
        """
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} out of range [0, {self.vocab_size})")
        return self.embedding_matrix[token_id]
    
    def get_embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Get static embeddings for multiple token IDs.
        
        Args:
            token_ids: Array of token IDs
        
        Returns:
            Array of shape (n_tokens, hidden_size)
        """
        return self.embedding_matrix[token_ids]
    
    def find_nearest_token(
        self,
        embedding: np.ndarray,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the nearest token ID(s) for an embedding.
        
        Args:
            embedding: Query embedding of shape (hidden_size,)
            k: Number of nearest neighbors to return
        
        Returns:
            Tuple of (token_ids, similarities) of shape (k,) each
        """
        # Normalize query
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            normalized = embedding / norm
        else:
            normalized = embedding
        
        if self.use_faiss and self._faiss_index is not None:
            similarities, indices = self._faiss_index.search(
                normalized.reshape(1, -1).astype(np.float32),
                k
            )
            return indices[0], similarities[0]
        else:
            # Cosine similarity via dot product on normalized vectors
            similarities = self._normalized @ normalized
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            return top_k_indices, similarities[top_k_indices]
    
    def find_nearest_tokens_batch(
        self,
        embeddings: np.ndarray,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest token IDs for a batch of embeddings.
        
        Args:
            embeddings: Query embeddings of shape (n_queries, hidden_size)
            k: Number of nearest neighbors per query
        
        Returns:
            Tuple of (token_ids, similarities) of shape (n_queries, k) each
        """
        # Normalize queries
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-10)
        
        if self.use_faiss and self._faiss_index is not None:
            similarities, indices = self._faiss_index.search(
                normalized.astype(np.float32),
                k
            )
            return indices, similarities
        else:
            # Batch cosine similarity
            similarities = normalized @ self._normalized.T  # (n_queries, vocab_size)
            
            # Get top-k for each query
            if k == 1:
                indices = np.argmax(similarities, axis=1).reshape(-1, 1)
                top_sims = np.take_along_axis(similarities, indices, axis=1)
            else:
                indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
                top_sims = np.take_along_axis(similarities, indices, axis=1)
            
            return indices, top_sims


def extract_static_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    use_prompt: bool = False,
    prompt_template: str = '"{context}" Compress the Context in one word:',
    normalize: bool = True,
) -> Dict[str, any]:
    """
    Extract static embeddings (layer 0) for texts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of input texts
        use_prompt: If True, wrap texts with prompt template (for alignment with contextual)
        prompt_template: Prompt template with {context} placeholder
        normalize: If True, L2-normalize embeddings (for fair comparison with normalized contextual)
    
    Returns:
        Dictionary with:
        - token_ids: List of token ID arrays
        - static_embeddings: List of embedding arrays (seq_len, hidden_size)
    """
    device = next(model.parameters()).device
    embedding_layer = get_embedding_matrix(model)
    
    results = {
        "token_ids": [],
        "static_embeddings": [],
    }
    
    for text in texts:
        # Apply prompt if requested
        if use_prompt:
            processed_text = prompt_template.format(context=text)
        else:
            processed_text = text
        
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=2048)
        token_ids = inputs["input_ids"].squeeze(0)
        
        # Get static embeddings directly from embedding layer
        static_embs = embedding_layer[token_ids].cpu().float().numpy()
        
        # L2 normalize if requested (for fair comparison with normalized contextual)
        if normalize:
            norms = np.linalg.norm(static_embs, axis=1, keepdims=True)
            static_embs = static_embs / np.maximum(norms, 1e-10)
        
        results["token_ids"].append(token_ids.cpu().numpy())
        results["static_embeddings"].append(static_embs)
    
    return results


def reconstruct_token_ids(
    static_embeddings: np.ndarray,
    index: TokenEmbeddingIndex,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Reconstruct token IDs from static embeddings via nearest-neighbor search.
    
    Args:
        static_embeddings: Array of shape (seq_len, hidden_size)
        index: Token embedding index
    
    Returns:
        Tuple of (reconstructed_ids, similarities, accuracy placeholder)
    """
    indices, similarities = index.find_nearest_tokens_batch(static_embeddings, k=1)
    reconstructed_ids = indices.squeeze(-1)
    
    return reconstructed_ids, similarities.squeeze(-1), 0.0


def compute_reconstruction_accuracy(
    original_ids: np.ndarray,
    reconstructed_ids: np.ndarray,
) -> Tuple[float, List[Dict]]:
    """
    Compute reconstruction accuracy and identify mismatches.
    
    Args:
        original_ids: Original token IDs
        reconstructed_ids: Reconstructed token IDs
    
    Returns:
        Tuple of (accuracy, list of mismatch dictionaries)
    """
    matches = original_ids == reconstructed_ids
    accuracy = np.mean(matches)
    
    mismatches = []
    for i, (orig, recon) in enumerate(zip(original_ids, reconstructed_ids)):
        if orig != recon:
            mismatches.append({
                "position": int(i),
                "original": int(orig),
                "reconstructed": int(recon),
            })
    
    return float(accuracy), mismatches
