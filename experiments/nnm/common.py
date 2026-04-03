"""Shared helpers for NNM experiment scripts."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)
    return normalized @ normalized.T


def split_intra_inter_by_category(
    similarity_matrix: np.ndarray,
    categories: Sequence[str],
) -> Tuple[List[float], List[float]]:
    intra: List[float] = []
    inter: List[float] = []
    n = len(categories)

    for i in range(n):
        for j in range(i + 1, n):
            value = float(similarity_matrix[i, j])
            if categories[i] == categories[j]:
                intra.append(value)
            else:
                inter.append(value)
    return intra, inter


def recall_at_k(
    original_embeddings: np.ndarray,
    reconstructed_embeddings: np.ndarray,
    k: int = 10,
) -> float:
    n_samples = int(original_embeddings.shape[0])
    if n_samples <= 1:
        return 1.0
    k = max(1, min(k, n_samples - 1))

    orig = original_embeddings / np.maximum(
        np.linalg.norm(original_embeddings, axis=1, keepdims=True), 1e-10
    )
    recon = reconstructed_embeddings / np.maximum(
        np.linalg.norm(reconstructed_embeddings, axis=1, keepdims=True), 1e-10
    )

    sim_orig = orig @ orig.T
    sim_recon = recon @ recon.T

    total = 0.0
    for i in range(n_samples):
        sim_orig[i, i] = -np.inf
        sim_recon[i, i] = -np.inf
        top_orig = set(np.argsort(sim_orig[i])[-k:])
        top_recon = set(np.argsort(sim_recon[i])[-k:])
        total += len(top_orig & top_recon) / k
    return float(total / n_samples)
