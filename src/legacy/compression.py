"""
Compression utilities for KV-Embedding experiments.

Implements various compression methods for delta vectors:
- Scalar quantization (INT8/4/2)
- Product Quantization (PQ)
- Residual Vector Quantization (RVQ)
- SimHash binary projections
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


@dataclass
class QuantizationParams:
    """Parameters for dequantization."""
    min_val: float
    max_val: float
    bits: int
    dtype: np.dtype


@dataclass
class PQParams:
    """Parameters for Product Quantization."""
    n_subvectors: int
    n_centroids: int
    codebooks: np.ndarray  # (n_subvectors, n_centroids, subvector_dim)
    subvector_dim: int


@dataclass
class RVQParams:
    """Parameters for Residual Vector Quantization."""
    n_stages: int
    n_centroids: int
    codebooks: List[np.ndarray]  # List of (n_centroids, hidden_dim) arrays


@dataclass
class SimHashParams:
    """Parameters for SimHash."""
    n_bits: int
    random_planes: np.ndarray  # (n_bits, hidden_dim)


# -----------------------------------------------------------------------------
# Scalar Quantization
# -----------------------------------------------------------------------------

def scalar_quantize(
    vectors: np.ndarray,
    bits: int = 8,
    per_vector: bool = False,
) -> Tuple[np.ndarray, QuantizationParams]:
    """
    Quantize vectors to fixed-point representation.
    
    Args:
        vectors: Array of shape (n_vectors, hidden_dim) or (hidden_dim,)
        bits: Number of bits (2, 4, or 8)
        per_vector: Whether to use per-vector min/max (better quality, more storage)
    
    Returns:
        Tuple of (quantized_array, params)
    """
    if bits not in [2, 4, 8]:
        raise ValueError(f"Unsupported bit width: {bits}")
    
    dtype_map = {2: np.uint8, 4: np.uint8, 8: np.uint8}
    
    if per_vector and vectors.ndim == 2:
        # Per-vector quantization
        min_vals = vectors.min(axis=1, keepdims=True)
        max_vals = vectors.max(axis=1, keepdims=True)
    else:
        # Global quantization
        min_vals = vectors.min()
        max_vals = vectors.max()
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    if isinstance(range_vals, np.ndarray):
        range_vals = np.maximum(range_vals, 1e-10)
    else:
        range_vals = max(range_vals, 1e-10)
    
    # Quantize
    n_levels = 2 ** bits - 1
    normalized = (vectors - min_vals) / range_vals
    quantized = np.round(normalized * n_levels).astype(dtype_map[bits])
    
    # Pack 4-bit and 2-bit values
    if bits == 4:
        # Pack two 4-bit values per byte
        if quantized.ndim == 1:
            if len(quantized) % 2:
                quantized = np.append(quantized, 0)
            packed = (quantized[::2] << 4) | (quantized[1::2] & 0x0F)
        else:
            if quantized.shape[1] % 2:
                quantized = np.pad(quantized, ((0, 0), (0, 1)))
            packed = (quantized[:, ::2] << 4) | (quantized[:, 1::2] & 0x0F)
        quantized = packed
    elif bits == 2:
        # Pack four 2-bit values per byte
        if quantized.ndim == 1:
            pad_len = (4 - len(quantized) % 4) % 4
            if pad_len:
                quantized = np.append(quantized, np.zeros(pad_len, dtype=np.uint8))
            packed = (quantized[::4] << 6) | (quantized[1::4] << 4) | \
                     (quantized[2::4] << 2) | quantized[3::4]
        else:
            pad_len = (4 - quantized.shape[1] % 4) % 4
            if pad_len:
                quantized = np.pad(quantized, ((0, 0), (0, pad_len)))
            packed = (quantized[:, ::4] << 6) | (quantized[:, 1::4] << 4) | \
                     (quantized[:, 2::4] << 2) | quantized[:, 3::4]
        quantized = packed
    
    params = QuantizationParams(
        min_val=float(min_vals) if np.isscalar(min_vals) else min_vals.flatten(),
        max_val=float(max_vals) if np.isscalar(max_vals) else max_vals.flatten(),
        bits=bits,
        dtype=dtype_map[bits],
    )
    
    return quantized, params


def scalar_dequantize(
    quantized: np.ndarray,
    params: QuantizationParams,
    original_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Dequantize vectors from fixed-point to float.
    
    Args:
        quantized: Quantized array
        params: Quantization parameters
        original_shape: Original shape before packing
    
    Returns:
        Dequantized vectors
    """
    bits = params.bits
    n_levels = 2 ** bits - 1
    
    # Unpack if needed
    if bits == 4:
        if quantized.ndim == 1:
            unpacked = np.zeros(len(quantized) * 2, dtype=np.uint8)
            unpacked[::2] = (quantized >> 4) & 0x0F
            unpacked[1::2] = quantized & 0x0F
        else:
            unpacked = np.zeros((quantized.shape[0], quantized.shape[1] * 2), dtype=np.uint8)
            unpacked[:, ::2] = (quantized >> 4) & 0x0F
            unpacked[:, 1::2] = quantized & 0x0F
        quantized = unpacked
    elif bits == 2:
        if quantized.ndim == 1:
            unpacked = np.zeros(len(quantized) * 4, dtype=np.uint8)
            unpacked[::4] = (quantized >> 6) & 0x03
            unpacked[1::4] = (quantized >> 4) & 0x03
            unpacked[2::4] = (quantized >> 2) & 0x03
            unpacked[3::4] = quantized & 0x03
        else:
            unpacked = np.zeros((quantized.shape[0], quantized.shape[1] * 4), dtype=np.uint8)
            unpacked[:, ::4] = (quantized >> 6) & 0x03
            unpacked[:, 1::4] = (quantized >> 4) & 0x03
            unpacked[:, 2::4] = (quantized >> 2) & 0x03
            unpacked[:, 3::4] = quantized & 0x03
        quantized = unpacked
    
    # Trim to original shape
    if original_shape is not None:
        if quantized.ndim == 1:
            quantized = quantized[:original_shape[0]]
        else:
            quantized = quantized[:, :original_shape[1]]
    
    # Dequantize
    normalized = quantized.astype(np.float32) / n_levels
    
    min_val = params.min_val
    max_val = params.max_val
    range_val = max_val - min_val
    
    if isinstance(range_val, np.ndarray):
        min_val = min_val.reshape(-1, 1)
        range_val = range_val.reshape(-1, 1)
    
    dequantized = normalized * range_val + min_val
    
    return dequantized


# -----------------------------------------------------------------------------
# Product Quantization
# -----------------------------------------------------------------------------

def product_quantize_fit(
    vectors: np.ndarray,
    n_subvectors: int = 8,
    n_centroids: int = 256,
) -> PQParams:
    """
    Fit Product Quantization codebooks.
    
    Args:
        vectors: Training vectors of shape (n_samples, hidden_dim)
        n_subvectors: Number of subvectors to split into
        n_centroids: Number of centroids per subvector
    
    Returns:
        PQ parameters with trained codebooks
    """
    n_samples, hidden_dim = vectors.shape
    
    if hidden_dim % n_subvectors != 0:
        raise ValueError(f"hidden_dim {hidden_dim} must be divisible by n_subvectors {n_subvectors}")
    
    subvector_dim = hidden_dim // n_subvectors
    codebooks = np.zeros((n_subvectors, n_centroids, subvector_dim))
    
    logger.info(f"Fitting PQ: {n_subvectors} subvectors, {n_centroids} centroids each")
    
    for i in range(n_subvectors):
        start = i * subvector_dim
        end = (i + 1) * subvector_dim
        subvectors = vectors[:, start:end]
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_centroids,
            random_state=42,
            batch_size=min(1024, n_samples),
            n_init=1,
        )
        kmeans.fit(subvectors)
        codebooks[i] = kmeans.cluster_centers_
    
    return PQParams(
        n_subvectors=n_subvectors,
        n_centroids=n_centroids,
        codebooks=codebooks,
        subvector_dim=subvector_dim,
    )


def product_quantize(
    vectors: np.ndarray,
    params: PQParams,
) -> np.ndarray:
    """
    Quantize vectors using Product Quantization.
    
    Args:
        vectors: Vectors of shape (n_vectors, hidden_dim)
        params: Pre-fitted PQ parameters
    
    Returns:
        Codes of shape (n_vectors, n_subvectors) as uint8/uint16
    """
    n_vectors = vectors.shape[0]
    codes = np.zeros((n_vectors, params.n_subvectors), dtype=np.uint16 if params.n_centroids > 256 else np.uint8)
    
    for i in range(params.n_subvectors):
        start = i * params.subvector_dim
        end = (i + 1) * params.subvector_dim
        subvectors = vectors[:, start:end]
        
        # Find nearest centroid
        distances = np.linalg.norm(
            subvectors[:, np.newaxis, :] - params.codebooks[i][np.newaxis, :, :],
            axis=2
        )
        codes[:, i] = np.argmin(distances, axis=1)
    
    return codes


def product_dequantize(
    codes: np.ndarray,
    params: PQParams,
) -> np.ndarray:
    """
    Reconstruct vectors from PQ codes.
    
    Args:
        codes: Codes of shape (n_vectors, n_subvectors)
        params: PQ parameters with codebooks
    
    Returns:
        Reconstructed vectors of shape (n_vectors, hidden_dim)
    """
    n_vectors = codes.shape[0]
    hidden_dim = params.n_subvectors * params.subvector_dim
    reconstructed = np.zeros((n_vectors, hidden_dim))
    
    for i in range(params.n_subvectors):
        start = i * params.subvector_dim
        end = (i + 1) * params.subvector_dim
        reconstructed[:, start:end] = params.codebooks[i][codes[:, i]]
    
    return reconstructed


# -----------------------------------------------------------------------------
# Residual Vector Quantization
# -----------------------------------------------------------------------------

def residual_vq_fit(
    vectors: np.ndarray,
    n_stages: int = 4,
    n_centroids: int = 256,
) -> RVQParams:
    """
    Fit Residual Vector Quantization codebooks.
    
    Args:
        vectors: Training vectors of shape (n_samples, hidden_dim)
        n_stages: Number of quantization stages
        n_centroids: Number of centroids per stage
    
    Returns:
        RVQ parameters with trained codebooks
    """
    n_samples, hidden_dim = vectors.shape
    codebooks = []
    residuals = vectors.copy()
    
    logger.info(f"Fitting RVQ: {n_stages} stages, {n_centroids} centroids each")
    
    for stage in range(n_stages):
        kmeans = MiniBatchKMeans(
            n_clusters=n_centroids,
            random_state=42 + stage,
            batch_size=min(1024, n_samples),
            n_init=1,
        )
        kmeans.fit(residuals)
        codebooks.append(kmeans.cluster_centers_)
        
        # Compute residuals for next stage
        labels = kmeans.predict(residuals)
        residuals = residuals - kmeans.cluster_centers_[labels]
    
    return RVQParams(
        n_stages=n_stages,
        n_centroids=n_centroids,
        codebooks=codebooks,
    )


def residual_vq(
    vectors: np.ndarray,
    params: RVQParams,
) -> np.ndarray:
    """
    Quantize vectors using Residual Vector Quantization.
    
    Args:
        vectors: Vectors of shape (n_vectors, hidden_dim)
        params: Pre-fitted RVQ parameters
    
    Returns:
        Codes of shape (n_vectors, n_stages) as uint8/uint16
    """
    n_vectors = vectors.shape[0]
    codes = np.zeros((n_vectors, params.n_stages), dtype=np.uint16 if params.n_centroids > 256 else np.uint8)
    residuals = vectors.copy()
    
    for stage in range(params.n_stages):
        # Find nearest centroid
        distances = np.linalg.norm(
            residuals[:, np.newaxis, :] - params.codebooks[stage][np.newaxis, :, :],
            axis=2
        )
        codes[:, stage] = np.argmin(distances, axis=1)
        
        # Update residuals
        residuals = residuals - params.codebooks[stage][codes[:, stage]]
    
    return codes


def residual_vq_dequantize(
    codes: np.ndarray,
    params: RVQParams,
) -> np.ndarray:
    """
    Reconstruct vectors from RVQ codes.
    
    Args:
        codes: Codes of shape (n_vectors, n_stages)
        params: RVQ parameters with codebooks
    
    Returns:
        Reconstructed vectors of shape (n_vectors, hidden_dim)
    """
    n_vectors = codes.shape[0]
    hidden_dim = params.codebooks[0].shape[1]
    reconstructed = np.zeros((n_vectors, hidden_dim))
    
    for stage in range(params.n_stages):
        reconstructed += params.codebooks[stage][codes[:, stage]]
    
    return reconstructed


# -----------------------------------------------------------------------------
# SimHash / Signed Random Projections
# -----------------------------------------------------------------------------

def simhash_fit(
    hidden_dim: int,
    n_bits: int = 256,
    seed: int = 42,
) -> SimHashParams:
    """
    Create random projection matrix for SimHash.
    
    Args:
        hidden_dim: Dimension of input vectors
        n_bits: Number of hash bits
        seed: Random seed for reproducibility
    
    Returns:
        SimHash parameters with random planes
    """
    rng = np.random.RandomState(seed)
    random_planes = rng.randn(n_bits, hidden_dim).astype(np.float32)
    
    return SimHashParams(
        n_bits=n_bits,
        random_planes=random_planes,
    )


def simhash(
    vectors: np.ndarray,
    params: SimHashParams,
) -> np.ndarray:
    """
    Compute SimHash for vectors.
    
    Args:
        vectors: Vectors of shape (n_vectors, hidden_dim)
        params: SimHash parameters
    
    Returns:
        Binary hash codes of shape (n_vectors, n_bytes) as uint8
    """
    # Project and take sign
    projections = vectors @ params.random_planes.T  # (n_vectors, n_bits)
    signs = (projections > 0).astype(np.uint8)
    
    # Pack bits into bytes
    n_bytes = (params.n_bits + 7) // 8
    n_vectors = vectors.shape[0]
    hashes = np.zeros((n_vectors, n_bytes), dtype=np.uint8)
    
    for i in range(params.n_bits):
        byte_idx = i // 8
        bit_idx = i % 8
        hashes[:, byte_idx] |= signs[:, i] << bit_idx
    
    return hashes


def simhash_dequantize(
    hashes: np.ndarray,
    params: SimHashParams,
) -> np.ndarray:
    """
    Reconstruct vectors from SimHash (lossy reconstruction via pseudo-inverse).
    
    Note: This is a crude approximation. SimHash is meant for similarity
    search, not reconstruction.
    
    Args:
        hashes: Hash codes of shape (n_vectors, n_bytes)
        params: SimHash parameters
    
    Returns:
        Reconstructed vectors (very approximate)
    """
    # Unpack bits
    n_vectors = hashes.shape[0]
    signs = np.zeros((n_vectors, params.n_bits), dtype=np.float32)
    
    for i in range(params.n_bits):
        byte_idx = i // 8
        bit_idx = i % 8
        signs[:, i] = ((hashes[:, byte_idx] >> bit_idx) & 1) * 2 - 1  # Map to {-1, 1}
    
    # Pseudo-inverse reconstruction
    # signs ~ sign(v @ random_planes.T)
    # Solve least squares: random_planes @ v.T ≈ signs.T
    reconstructed, _, _, _ = np.linalg.lstsq(
        params.random_planes,
        signs.T,
        rcond=None,
    )
    
    return reconstructed.T


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def compute_compression_ratio(
    original_shape: Tuple[int, ...],
    compressed_size_bytes: int,
    original_dtype: np.dtype = np.float32,
) -> float:
    """
    Compute compression ratio.
    
    Args:
        original_shape: Shape of original array
        compressed_size_bytes: Size of compressed representation in bytes
        original_dtype: Data type of original array
    
    Returns:
        Compression ratio (original_size / compressed_size)
    """
    original_size = np.prod(original_shape) * np.dtype(original_dtype).itemsize
    return original_size / compressed_size_bytes


def compute_reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Args:
        original: Original vectors
        reconstructed: Reconstructed vectors
    
    Returns:
        Dictionary with L2 error, MSE, and cosine similarity
    """
    # L2 error (per vector, then mean)
    l2_errors = np.linalg.norm(original - reconstructed, axis=-1)
    mean_l2 = float(np.mean(l2_errors))
    
    # MSE
    mse = float(np.mean((original - reconstructed) ** 2))
    
    # Cosine similarity
    orig_norm = np.linalg.norm(original, axis=-1, keepdims=True)
    recon_norm = np.linalg.norm(reconstructed, axis=-1, keepdims=True)
    
    orig_normalized = original / np.maximum(orig_norm, 1e-10)
    recon_normalized = reconstructed / np.maximum(recon_norm, 1e-10)
    
    cosine_sim = np.sum(orig_normalized * recon_normalized, axis=-1)
    mean_cosine = float(np.mean(cosine_sim))
    
    return {
        "l2_error": mean_l2,
        "mse": mse,
        "cosine_similarity": mean_cosine,
    }
