"""
llama.cpp-based embedding utilities for KV-Embedding experiments.

Provides a llama.cpp backend that mirrors the transformers logic:
1) KV re-routing (two-pass) using last-token K/V as a virtual prefix
2) Hidden-state extraction from selected layers
3) Hybrid pooling of token embeddings
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict

import numpy as np

from .embedding_utils import compute_intrinsic_dimension_twonn, COMPRESSION_PROMPT
from .llama_raw import LlamaModel, HiddenStatesExtractor

logger = logging.getLogger(__name__)


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def select_optimal_layers_llama(
    model: LlamaModel,
    sample_texts: List[str],
    n_layers_to_select: int = 4,
) -> List[int]:
    """
    Select layers with optimal semantic compression using intrinsic dimensionality.

    Mirrors embedding_utils.select_optimal_layers but uses llama.cpp hidden states.
    """
    n_layers = model.n_layer
    kv_layers = [i for i in range(n_layers) if model.kv_has_layer(i)]
    candidate_layers = kv_layers if kv_layers else list(range(n_layers))

    logger.info(
        "Analyzing %d layers for optimal selection (llama.cpp)...",
        len(candidate_layers),
    )
    if kv_layers:
        logger.info("Restricting selection to %d KV layers.", len(kv_layers))

    layer_embeddings: Dict[int, List[np.ndarray]] = {i: [] for i in candidate_layers}

    # Use a few samples for speed
    texts = sample_texts[:5] if len(sample_texts) > 5 else sample_texts
    for text in texts:
        model.kv_prefix_clear()
        model.reset(clear_data=True)
        tokens = model.tokenize(text, add_bos=True)
        with HiddenStatesExtractor(model, layers=candidate_layers) as hs:
            model.eval(tokens)
            states = hs.get_all()
        for layer_id, h in states.items():
            if layer_id in layer_embeddings:
                layer_embeddings[layer_id].append(h)

    layer_dims = []
    for layer_id in candidate_layers:
        if not layer_embeddings[layer_id]:
            continue
        emb = np.vstack(layer_embeddings[layer_id])
        intrinsic_dim = compute_intrinsic_dimension_twonn(emb)
        layer_dims.append((layer_id, intrinsic_dim))

    if not layer_dims:
        logger.warning("No layers captured; falling back to middle layers")
        base = candidate_layers if candidate_layers else list(range(n_layers))
        mid = len(base) // 2
        start = max(0, mid - n_layers_to_select // 2)
        return base[start:start + n_layers_to_select]

    # Filter out invalid dimensions and avoid first/last few layers
    margin = max(1, n_layers // 6)
    valid_dims = [
        (idx, dim) for idx, dim in layer_dims
        if not np.isnan(dim) and margin <= idx < n_layers - margin
    ]

    if len(valid_dims) < n_layers_to_select:
        logger.warning("Not enough valid layers, using middle layers as fallback")
        base = candidate_layers if candidate_layers else list(range(n_layers))
        mid = len(base) // 2
        start = max(0, mid - n_layers_to_select // 2)
        return base[start:start + n_layers_to_select]

    sorted_dims = sorted(valid_dims, key=lambda x: x[1])
    selected = [idx for idx, _ in sorted_dims[:n_layers_to_select]]
    selected.sort()
    logger.info(f"Top {n_layers_to_select} layers by ID (llama.cpp): {selected}")
    return selected


def extract_static_embeddings_llama(
    model: LlamaModel,
    texts: List[str],
    use_prompt: bool = False,
    prompt_template: str = COMPRESSION_PROMPT,
    normalize: bool = True,
) -> Dict[str, List[np.ndarray]]:
    """
    Extract static embeddings (layer 0) using llama.cpp embedding matrix.
    """
    embed_matrix, _, _ = model.get_embed_matrix()

    results = {
        "token_ids": [],
        "static_embeddings": [],
    }

    for text in texts:
        if use_prompt:
            processed_text = prompt_template.format(context=text)
        else:
            processed_text = text

        token_ids = np.array(model.tokenize(processed_text, add_bos=True), dtype=np.int64)
        static_embs = embed_matrix[token_ids]

        if normalize:
            static_embs = _l2_normalize(static_embs, axis=1)

        results["token_ids"].append(token_ids)
        results["static_embeddings"].append(static_embs)

    return results


class LlamaKVEmbeddingExtractor:
    """
    Extract KV-Embeddings using llama.cpp.

    Mirrors the transformers KVEmbeddingExtractor with two-pass KV re-routing.
    """

    def __init__(
        self,
        model: LlamaModel,
        target_layers: Optional[List[int]] = None,
    ):
        self.model = model
        self.target_layers = target_layers
        self._dummy_token_cache: Optional[List[int]] = None

    def _wrap_with_prompt(self, text: str) -> str:
        return COMPRESSION_PROMPT.format(context=text)

    def _init_layers_if_needed(self, sample_texts: List[str]) -> None:
        if self.target_layers is None:
            self.target_layers = select_optimal_layers_llama(
                self.model,
                sample_texts[:5],
                n_layers_to_select=4,
            )

    def _get_dummy_tokens(self) -> List[int]:
        if self._dummy_token_cache is None:
            # Tokenizing empty string with BOS gives a single dummy prefix token
            toks = self.model.tokenize("", add_bos=True)
            self._dummy_token_cache = toks if toks else [0]
        return self._dummy_token_cache

    def _get_layer_shapes(self) -> Dict[int, tuple[int, int]]:
        """
        Get (k_embd, v_embd) per layer after a dummy eval.
        """
        shapes: Dict[int, tuple[int, int]] = {}
        n_layers = self.model.kv_n_layers()
        for layer_id in range(n_layers):
            k, _, k_embd = self.model.kv_get_layer_k(layer_id)
            v, _, v_embd = self.model.kv_get_layer_v(layer_id)
            shapes[layer_id] = (k_embd, v_embd)
        return shapes

    def extract_embeddings(
        self,
        texts: List[str],
        return_token_embeddings: bool = True,
        use_kv_routing: bool = True,
    ) -> Dict[str, object]:
        self._init_layers_if_needed(texts)

        results = {
            "pooled_embeddings": [],
            "token_embeddings": [] if return_token_embeddings else None,
            "token_ids": [],
        }

        assert self.target_layers is not None

        for text in texts:
            # Ensure prefix masking is cleared between runs
            self.model.kv_prefix_clear()

            prompted_text = self._wrap_with_prompt(text)
            token_ids = self.model.tokenize(prompted_text, add_bos=True)

            if use_kv_routing:
                # Pass 1: populate KV cache and extract last-token K/V
                self.model.reset(clear_data=True)
                self.model.eval(token_ids, pos_offset=0)

                last_kv: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
                valid_layers: List[int] = []
                for layer_id in self.target_layers:
                    try:
                        k, n_tok, _ = self.model.kv_get_layer_k(layer_id)
                        v, _, _ = self.model.kv_get_layer_v(layer_id)
                    except ValueError:
                        logger.debug("Skipping layer %d (no KV cache)", layer_id)
                        continue

                    if n_tok <= 0:
                        logger.debug("Skipping layer %d (no tokens in KV)", layer_id)
                        continue

                    k_last = k[n_tok - 1:n_tok].copy()
                    v_last = v[n_tok - 1:n_tok].copy()
                    last_kv[layer_id] = (k_last, v_last)
                    valid_layers.append(layer_id)

                if not valid_layers:
                    raise RuntimeError("No valid KV layers found for prefix routing.")

                # Restrict to layers that actually have KV cache
                self.target_layers = valid_layers

                # Pass 2: prepare a 1-token virtual prefix
                self.model.reset(clear_data=True)
                dummy_tokens = self._get_dummy_tokens()
                self.model.eval(dummy_tokens, pos_offset=0)
                prefix_len = len(dummy_tokens)

                # Initialize all layers with zero prefix, then overwrite selected
                n_layers = self.model.kv_n_layers()
                for layer_id in range(n_layers):
                    if not self.model.kv_has_layer(layer_id):
                        continue

                    k0, _, k_embd = self.model.kv_get_layer_k(layer_id)
                    v0, _, v_embd = self.model.kv_get_layer_v(layer_id)
                    k_prefix = np.zeros((prefix_len, k_embd), dtype=np.float32)
                    v_prefix = np.zeros((prefix_len, v_embd), dtype=np.float32)
                    if layer_id in last_kv:
                        k_prefix[:, :] = last_kv[layer_id][0]
                        v_prefix[:, :] = last_kv[layer_id][1]
                    self.model.kv_set_layer_k(layer_id, k_prefix)
                    self.model.kv_set_layer_v(layer_id, v_prefix)

                # Pass 2: compute hidden states with prefix offset
                with HiddenStatesExtractor(self.model, layers=self.target_layers) as hs:
                    self.model.kv_prefix_set(self.target_layers, prefix_len)
                    self.model.eval(token_ids, pos_offset=prefix_len)
                    states = hs.get_all()
                self.model.kv_prefix_clear()
            else:
                self.model.reset(clear_data=True)
                with HiddenStatesExtractor(self.model, layers=self.target_layers) as hs:
                    self.model.eval(token_ids, pos_offset=0)
                    states = hs.get_all()

            # Combine selected layer states
            selected_states = [states[layer_id] for layer_id in self.target_layers if layer_id in states]
            if not selected_states:
                raise RuntimeError("No hidden states captured for selected layers.")

            combined = np.mean(np.stack(selected_states, axis=0), axis=0)  # (n_tokens, n_embd)
            token_embs = _l2_normalize(combined, axis=1)

            if return_token_embeddings:
                results["token_embeddings"].append(token_embs)

            last_token = token_embs[-1]
            mean_pool = token_embs.mean(axis=0)
            pooled = _l2_normalize((last_token + mean_pool) / 2.0, axis=0)

            results["pooled_embeddings"].append(pooled)
            results["token_ids"].append(np.array(token_ids, dtype=np.int64))

        results["pooled_embeddings"] = np.stack(results["pooled_embeddings"])
        return results
