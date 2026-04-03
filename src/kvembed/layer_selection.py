"""Intrinsic-dimensionality based layer selection for KV rerouting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class LayerSelectionResult:
    """Selection output for routing layers."""

    selected_layers: List[int]
    id_by_layer: Dict[int, float]
    used_u_shape_mode: bool


def compute_intrinsic_dimension_twonn(embeddings: np.ndarray) -> float:
    """
    Estimate intrinsic dimensionality with TwoNN (Facco et al., 2017).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if embeddings.shape[0] < 3:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    valid = r1 > 1e-10
    if not np.any(valid):
        return float("nan")

    mu = r2[valid] / r1[valid]
    denom = float(np.sum(np.log(mu)))
    if denom <= 0:
        return float("nan")

    return float(len(mu) / denom)


def _is_u_shaped(values: np.ndarray, max_violation_ratio: float = 0.20) -> bool:
    """
    Heuristic check for a U-shaped trajectory around the minimum.
    """
    if values.ndim != 1 or len(values) < 5:
        return False
    if np.any(~np.isfinite(values)):
        return False

    idx = int(np.argmin(values))
    left = values[: idx + 1]
    right = values[idx:]

    if len(left) < 2 or len(right) < 2:
        return False

    # Left should mostly decrease, right should mostly increase.
    left_viol = int(np.sum(np.diff(left) > 0))
    right_viol = int(np.sum(np.diff(right) < 0))
    total_edges = (len(left) - 1) + (len(right) - 1)
    return (left_viol + right_viol) / max(total_edges, 1) <= max_violation_ratio


def estimate_layer_intrinsic_dimensions(
    model: Any,
    texts: Sequence[str],
    prepend_bos: bool,
) -> Dict[int, float]:
    """
    Run the model and estimate intrinsic dimensionality per layer from resid_post.
    """
    n_layers = int(model.cfg.n_layers)
    per_layer: List[List[np.ndarray]] = [[] for _ in range(n_layers)]

    def names_filter(name: str) -> bool:
        return name.endswith("hook_resid_post")

    with torch.no_grad():
        for text in texts:
            _, cache = model.run_with_cache(
                text,
                return_type=None,
                prepend_bos=prepend_bos,
                names_filter=names_filter,
                remove_batch_dim=False,
            )

            for layer in range(n_layers):
                key = f"blocks.{layer}.hook_resid_post"
                if key not in cache:
                    continue
                layer_hidden = cache[key][0].detach().to(torch.float32).cpu().numpy()
                per_layer[layer].append(layer_hidden)

            del cache

    result: Dict[int, float] = {}
    for layer in range(n_layers):
        if not per_layer[layer]:
            result[layer] = float("nan")
            continue
        vectors = np.concatenate(per_layer[layer], axis=0)
        result[layer] = compute_intrinsic_dimension_twonn(vectors)
    return result


def select_rerouting_layers(
    id_by_layer: Mapping[int, float],
    n_layers: int,
    layer_window_fraction: float,
    exclude_early_fraction: float,
    detect_u_shape: bool = True,
) -> LayerSelectionResult:
    """
    Paper-near routing layer selection (Section 3.2.3 + Appendix C style heuristic).
    """
    values = np.array([id_by_layer.get(i, float("nan")) for i in range(n_layers)], dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise RuntimeError("No valid intrinsic dimension values found.")

    span = max(1, int(np.floor(layer_window_fraction * n_layers)))
    target_count = span + 1

    use_u_shape = bool(detect_u_shape and _is_u_shaped(values[finite]))

    if use_u_shape:
        min_layer = int(np.nanargmin(values))
        start = min_layer
        end = min(n_layers - 1, start + span)
        selected = list(range(start, end + 1))
        return LayerSelectionResult(
            selected_layers=selected,
            id_by_layer={int(k): float(v) for k, v in id_by_layer.items()},
            used_u_shape_mode=True,
        )

    # Multi-minima fallback: focus on low-ID middle/late regions, excluding early layers.
    early_cutoff = int(np.floor(exclude_early_fraction * n_layers))
    candidates = [i for i in range(early_cutoff, n_layers) if np.isfinite(values[i])]
    if not candidates:
        candidates = [i for i in range(n_layers) if np.isfinite(values[i])]

    ranked = sorted(candidates, key=lambda i: values[i])
    selected = sorted(ranked[: min(target_count, len(ranked))])

    return LayerSelectionResult(
        selected_layers=selected,
        id_by_layer={int(k): float(v) for k, v in id_by_layer.items()},
        used_u_shape_mode=False,
    )
