"""
Embedding utilities for KV-Embedding experiments.

Implements the core KV-Embedding method from the paper:
"KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs"
(arXiv:2601.01046v1)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Compression prompt template from paper
COMPRESSION_PROMPT = '"{context}" Compress the Context in one word:'


def compute_intrinsic_dimension_twonn(embeddings: np.ndarray) -> float:
    """
    Compute intrinsic dimensionality using TwoNN estimator.
    
    Based on Facco et al., 2017: "Estimating the intrinsic dimension of
    datasets by a minimal neighborhood information"
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
    
    Returns:
        Estimated intrinsic dimension
    """
    n_samples = embeddings.shape[0]
    
    if n_samples < 3:
        logger.warning("Not enough samples for TwoNN estimation")
        return float("nan")
    
    # Compute pairwise distances
    from sklearn.neighbors import NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    
    # distances[:, 0] is self (0), [:, 1] is r1, [:, 2] is r2
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # Avoid division by zero
    valid = r1 > 1e-10
    if not np.any(valid):
        return float("nan")
    
    mu = r2[valid] / r1[valid]
    
    # TwoNN estimator: d = n / sum(log(mu))
    n_valid = len(mu)
    log_mu_sum = np.sum(np.log(mu))
    
    if log_mu_sum <= 0:
        return float("nan")
    
    intrinsic_dim = n_valid / log_mu_sum
    
    return intrinsic_dim


def select_optimal_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample_texts: List[str],
    n_layers_to_select: int = 4,
) -> List[int]:
    """
    Select layers with optimal semantic compression using intrinsic dimensionality.
    
    The paper suggests that layers with lower intrinsic dimensionality contain
    more compressed semantic information.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        sample_texts: Sample texts to analyze
        n_layers_to_select: Number of layers to select
    
    Returns:
        List of layer indices to use for KV-routing
    """
    # Safe layer count retrieval
    if hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    elif hasattr(model.config, "num_layers"):
        n_layers = model.config.num_layers
    elif hasattr(model.config, "text_config"):
         # For VL models like Qwen-VL, the text config holds the layers
         if hasattr(model.config.text_config, "num_hidden_layers"):
             n_layers = model.config.text_config.num_hidden_layers
         elif hasattr(model.config.text_config, "num_layers"):
             n_layers = model.config.text_config.num_layers
         else:
             n_layers = 28 # Default fallback
    else:
        n_layers = 28 # Fallback if everything fails (usually 32 or 28 for small models, 80 for large)
        logger.warning(f"Could not determine num_layers from config, using default {n_layers}")
    
    logger.info(f"Analyzing {n_layers} layers for optimal selection...")
    
    # Get hidden states for all layers
    all_hidden_states = []
    
    with torch.no_grad():
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of (n_layers + 1,) tensors of shape (batch, seq_len, hidden)
            # Skip layer 0 (embedding layer)
            hidden_states = outputs.hidden_states[1:]
            
            all_hidden_states.append([h.squeeze(0).cpu().numpy() for h in hidden_states])
    
    # Compute intrinsic dimension per layer
    layer_dims = []
    
    for layer_idx in range(n_layers):
        # Collect all token embeddings from this layer across samples
        layer_embeddings = []
        for sample_states in all_hidden_states:
            layer_embeddings.append(sample_states[layer_idx])
        
        layer_embeddings = np.vstack(layer_embeddings)
        
        # Compute intrinsic dimension
        intrinsic_dim = compute_intrinsic_dimension_twonn(layer_embeddings)
        layer_dims.append((layer_idx, intrinsic_dim))
        
        logger.debug(f"Layer {layer_idx}: intrinsic dim = {intrinsic_dim:.2f}")
    
    # Filter out invalid dimensions and avoid first/last few layers
    margin = max(1, n_layers // 6)  # Skip first/last ~15% of layers
    valid_dims = [
        (idx, dim) for idx, dim in layer_dims
        if not np.isnan(dim) and margin <= idx < n_layers - margin
    ]
    
    if len(valid_dims) < n_layers_to_select:
        logger.warning("Not enough valid layers, using middle layers as fallback")
        mid = n_layers // 2
        return list(range(mid - n_layers_to_select // 2, mid + n_layers_to_select // 2 + 1))[:n_layers_to_select]
    
    # Sort by ID
    sorted_dims = sorted(valid_dims, key=lambda x: x[1])
    
    # Paper-compliant: Select range around the minimum ID
    # The paper mentions selecting "layers where representations exhibit maximal compression"
    # and compares against uniform thirds (e.g., 10 layers). 
    # For robust selection, we take the top-k layers, but ensure they are somewhat contiguous
    # or just take the best k.
    
    # If the user specifically requested n_layers, stick to it.
    selected = [idx for idx, _ in sorted_dims[:n_layers_to_select]]
    selected.sort()
    
    # Log the ID profile for debugging
    logger.info(f"Top {n_layers_to_select} layers by ID: {selected}")
    logger.debug(f"ID Selection Profile: {[(idx, f'{dim:.2f}') for idx, dim in sorted_dims[:n_layers_to_select]]}")
    
    return selected


class KVEmbeddingExtractor:
    """
    Extracts KV-Embeddings using the method from the paper.
    
    Core mechanism:
    1. Wrap input with compression prompt
    2. Forward pass with KV manipulation
    3. Extract token-level embeddings
    4. Hybrid pooling
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        target_layers: Optional[List[int]] = None,
    ):
        """
        Initialize the KV-Embedding extractor.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            target_layers: Specific layers for KV-routing (auto-selected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        if target_layers is None:
            # Will be initialized on first use with sample texts
            self.target_layers = None
        else:
            self.target_layers = target_layers
    
    def _init_layers_if_needed(self, sample_texts: List[str]) -> None:
        """Initialize target layers using sample texts if not already set."""
        if self.target_layers is None:
            self.target_layers = select_optimal_layers(
                self.model,
                self.tokenizer,
                sample_texts[:5],  # Use up to 5 samples
                n_layers_to_select=4,
            )
    
    def _wrap_with_prompt(self, text: str) -> str:
        """Wrap text with compression prompt."""
        return COMPRESSION_PROMPT.format(context=text)
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        texts: List[str],
        return_token_embeddings: bool = True,
        use_kv_routing: bool = True,
    ) -> dict:
        """
        Extract KV-Embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            return_token_embeddings: Whether to return per-token embeddings
            use_kv_routing: If True, use paper-compliant KV-routing (two passes)
        
        Returns:
            Dictionary with:
            - pooled_embeddings: (n_texts, hidden_size)
            - token_embeddings: List of (seq_len, hidden_size) arrays (if requested)
            - token_ids: List of token ID arrays
        """
        self._init_layers_if_needed(texts)
        
        results = {
            "pooled_embeddings": [],
            "token_embeddings": [] if return_token_embeddings else None,
            "token_ids": [],
        }
        
        for text in texts:
            prompted_text = self._wrap_with_prompt(text)
            
            inputs = self.tokenizer(
                prompted_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if use_kv_routing:
                # Paper-compliant KV-routing: Two-pass approach
                # Pass 1: Get KV states of the last token
                outputs_pass1 = self.model(**inputs, output_hidden_states=True, use_cache=True)
                past_key_values = outputs_pass1.past_key_values
                
                # Create a new cache with only the last token's KV from selected layers
                # For modern transformers, past_key_values is a DynamicCache object
                try:
                    from transformers.cache_utils import DynamicCache
                    
                    # Create new cache for prefix
                    prefix_cache = DynamicCache()
                    
                    # Access the internal key/value cache
                    for layer_idx in range(len(past_key_values.key_cache)):
                        k = past_key_values.key_cache[layer_idx]
                        v = past_key_values.value_cache[layer_idx]
                        
                        if layer_idx in self.target_layers:
                            # Take only the last token's KV as prefix
                            k_last = k[:, :, -1:, :]  # (batch, heads, 1, head_dim)
                            v_last = v[:, :, -1:, :]
                        else:
                            # Empty prefix for other layers
                            k_last = k[:, :, :0, :]
                            v_last = v[:, :, :0, :]
                        
                        prefix_cache.update(k_last, v_last, layer_idx)
                    
                    # Pass 2: Forward with prefix KV (re-routing)
                    outputs = self.model(
                        **inputs,
                        past_key_values=prefix_cache,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    hidden_states = outputs.hidden_states
                    
                except (ImportError, AttributeError):
                    # Fallback: If DynamicCache not available, use simple approach
                    # Just use the hidden states from first pass
                    hidden_states = outputs_pass1.hidden_states
            else:
                # Simple forward pass (no KV-routing)
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            
            # Extract hidden states from target layers
            # hidden_states[0] is embedding layer, [1:] are transformer layers
            selected_states = [hidden_states[l + 1] for l in self.target_layers]
            combined = torch.stack(selected_states, dim=0).mean(dim=0)  # (1, seq_len, hidden)
            
            token_embs = combined[0]  # (seq_len, hidden)
            
            # Normalize token embeddings (for fair comparison with normalized static embeddings)
            token_embs = F.normalize(token_embs, p=2, dim=-1)
            
            # Store normalized token embeddings for delta comparison
            if return_token_embeddings:
                results["token_embeddings"].append(token_embs.cpu().numpy())
            
            # Hybrid pooling: (last_token + mean) / 2
            last_token = token_embs[-1]
            mean_pool = token_embs.mean(dim=0)
            pooled = (last_token + mean_pool) / 2
            pooled = F.normalize(pooled, p=2, dim=-1)
            
            results["pooled_embeddings"].append(pooled.cpu().numpy())
            results["token_ids"].append(inputs["input_ids"].squeeze(0).cpu().numpy())
        
        results["pooled_embeddings"] = np.stack(results["pooled_embeddings"])
        
        return results


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for embeddings.
    
    Args:
        embeddings: Array of shape (n_samples, hidden_size)
    
    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)
    
    # Compute similarity matrix
    similarity = normalized @ normalized.T
    
    return similarity


def hybrid_pooling(
    token_embeddings: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply hybrid pooling: (last_token + mean) / 2
    
    Args:
        token_embeddings: Array of shape (seq_len, hidden_size)
        normalize: Whether to L2-normalize the result
    
    Returns:
        Pooled embedding of shape (hidden_size,)
    """
    last_token = token_embeddings[-1]
    mean_pool = token_embeddings.mean(axis=0)
    
    pooled = (last_token + mean_pool) / 2
    
    if normalize:
        norm = np.linalg.norm(pooled)
        if norm > 1e-10:
            pooled = pooled / norm
    
    return pooled


def apply_pca_whitening(embeddings: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
    """
    Apply PCA-whitening to embeddings.
    
    Whitening removes linear correlations between features and normalizes variance.
    With small datasets (N < D), standard whitening is unstable. This uses PCA
    to project to min(N, D) dimensions before whitening.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        n_components: Number of components to keep (default: min(n_samples, n_features))
    
    Returns:
        Whitened embeddings
    """
    from sklearn.decomposition import PCA
    
    n_samples, n_features = embeddings.shape
    
    # If not specified, keep all possible components
    if n_components is None:
        n_components = min(n_samples, n_features)
        
    logger.info(f"Applying PCA-whitening (n_components={n_components})...")
    
    pca = PCA(n_components=n_components, whiten=True)
    return pca.fit_transform(embeddings)


def apply_zscore(embeddings: np.ndarray) -> np.ndarray:
    """
    Apply Z-Score normalization (standardization) to embeddings.
    
    Rescales each feature to have mean=0 and std=1.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
    
    Returns:
        Standardized embeddings
    """
    from scipy.stats import zscore
    
    logger.info("Applying Z-Score normalization...")
    
    # Handle division by zero/constant columns by replacing NaNs with 0
    standardized = zscore(embeddings, axis=0)
    return np.nan_to_num(standardized)
