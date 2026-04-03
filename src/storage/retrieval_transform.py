"""Utilities for retrieval-side normalization and whitening transforms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _row_l2(x: np.ndarray, eps: float) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


@dataclass(frozen=True)
class RetrievalTransform:
    mean: np.ndarray
    std: np.ndarray
    whitening_matrix: Optional[np.ndarray]
    eps: float
    use_zscore: bool
    use_whitening: bool
    metadata: Dict[str, Any]

    def apply(self, x: np.ndarray, l2_normalize: bool = True) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y = x.astype(np.float32, copy=False)
        y = y - self.mean.reshape(1, -1)
        if self.use_zscore:
            y = y / self.std.reshape(1, -1)
        if self.use_whitening and self.whitening_matrix is not None:
            y = y @ self.whitening_matrix
        if l2_normalize:
            y = _row_l2(y, eps=self.eps)
        return y.astype(np.float32, copy=False)


def fit_retrieval_transform(
    x: np.ndarray,
    *,
    use_zscore: bool = True,
    use_whitening: bool = True,
    eps: float = 1e-5,
    metadata: Optional[Dict[str, Any]] = None,
) -> RetrievalTransform:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for fit, got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 samples to fit retrieval transform.")

    x = x.astype(np.float32, copy=False)
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    centered = x - mean.reshape(1, -1)

    if use_zscore:
        std = np.std(centered, axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < eps, 1.0, std).astype(np.float32)
        base = centered / std.reshape(1, -1)
    else:
        std = np.ones(centered.shape[1], dtype=np.float32)
        base = centered

    whitening_matrix: Optional[np.ndarray]
    if use_whitening:
        n = max(1, int(base.shape[0] - 1))
        cov = (base.T @ base) / float(n)
        evals, evecs = np.linalg.eigh(cov.astype(np.float64))
        evals = np.maximum(evals, eps)
        inv_sqrt = (1.0 / np.sqrt(evals)).astype(np.float64)
        # W = V * diag(inv_sqrt) * V^T
        whitening_matrix = (evecs * inv_sqrt) @ evecs.T
        whitening_matrix = whitening_matrix.astype(np.float32)
    else:
        whitening_matrix = None

    md: Dict[str, Any] = dict(metadata or {})
    md.setdefault("n_samples", int(x.shape[0]))
    md.setdefault("d_model", int(x.shape[1]))
    md.setdefault("use_zscore", bool(use_zscore))
    md.setdefault("use_whitening", bool(use_whitening))
    md.setdefault("eps", float(eps))

    return RetrievalTransform(
        mean=mean,
        std=std,
        whitening_matrix=whitening_matrix,
        eps=float(eps),
        use_zscore=bool(use_zscore),
        use_whitening=bool(use_whitening),
        metadata=md,
    )


def save_retrieval_transform(transform: RetrievalTransform, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    whitening = (
        transform.whitening_matrix
        if transform.whitening_matrix is not None
        else np.zeros((0, 0), dtype=np.float32)
    )
    np.savez_compressed(
        out_path,
        mean=transform.mean.astype(np.float32),
        std=transform.std.astype(np.float32),
        whitening_matrix=whitening.astype(np.float32),
        use_zscore=np.array([1 if transform.use_zscore else 0], dtype=np.int8),
        use_whitening=np.array([1 if transform.use_whitening else 0], dtype=np.int8),
        eps=np.array([transform.eps], dtype=np.float32),
        metadata_json=np.array([json.dumps(transform.metadata)], dtype=object),
    )
    return out_path


def load_retrieval_transform(path: str | Path) -> RetrievalTransform:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Retrieval transform file not found: {in_path}")
    with np.load(in_path, allow_pickle=True) as data:
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        wm = data["whitening_matrix"].astype(np.float32)
        use_zscore = bool(int(data["use_zscore"][0]))
        use_whitening = bool(int(data["use_whitening"][0]))
        eps = float(data["eps"][0])
        meta_raw = data["metadata_json"][0]
        metadata = json.loads(str(meta_raw)) if meta_raw is not None else {}
    whitening_matrix: Optional[np.ndarray] = wm if wm.size > 0 else None
    return RetrievalTransform(
        mean=mean,
        std=std,
        whitening_matrix=whitening_matrix,
        eps=eps,
        use_zscore=use_zscore,
        use_whitening=use_whitening,
        metadata=metadata,
    )
