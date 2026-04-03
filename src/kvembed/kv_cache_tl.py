"""Stored KV-cache utilities for TransformerLens."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class StoredTLKVCache:
    """Container for a stored TransformerLens KV cache."""

    keys: Dict[int, np.ndarray]
    values: Dict[int, np.ndarray]
    token_ids: np.ndarray
    text: str
    prefix_len: int
    metadata: Dict[str, Any]


def _key_hook_name(model, layer_idx: int) -> str:
    # TransformerLens caches pre-RoPE keys (rotation is applied after caching).
    # For cache injection we therefore must store hook_k (not hook_rot_k),
    # otherwise keys would be rotated twice on the next forward pass.
    return f"blocks.{layer_idx}.attn.hook_k"


@torch.no_grad()
def extract_kv_cache_tl(
    model,
    text: str,
    prepend_bos: bool = True,
) -> StoredTLKVCache:
    """Extract full post-RoPE KV cache from a forward pass."""
    tokens = model.to_tokens(text, prepend_bos=prepend_bos)
    prefix_len = int(tokens.shape[1])

    names = set()
    for layer in range(int(model.cfg.n_layers)):
        names.add(_key_hook_name(model, layer))
        names.add(f"blocks.{layer}.attn.hook_v")

    _, cache = model.run_with_cache(
        tokens,
        return_type=None,
        names_filter=lambda name: name in names,
        remove_batch_dim=False,
        prepend_bos=False,
    )

    keys: Dict[int, np.ndarray] = {}
    values: Dict[int, np.ndarray] = {}
    for layer in range(int(model.cfg.n_layers)):
        k_name = _key_hook_name(model, layer)
        v_name = f"blocks.{layer}.attn.hook_v"
        if k_name not in cache or v_name not in cache:
            raise RuntimeError(f"Missing cache entries for layer {layer}")

        k_tensor = cache[k_name].detach()
        v = cache[v_name].detach().cpu().numpy()

        # Qwen3 and models with QK-normalization store hook_k pre-norm.
        # abstract_attention.forward appends past_keys before applying RoPE, so
        # past keys must be in the same space as current keys at append-time
        # (i.e. post-QKnorm, pre-RoPE).  Apply the per-layer k_norm now so that
        # RoPE is applied consistently to all keys (past + current) at inference.
        if getattr(model.cfg, "use_qk_norm", False):
            attn = model.blocks[layer].attn
            if hasattr(attn, "_apply_qk_norm") and hasattr(attn, "k_norm") and attn.k_norm is not None:
                k_tensor = attn._apply_qk_norm(
                    k_tensor.to(torch.float32), attn.k_norm
                ).to(k_tensor.dtype)

        # Persist in float16 for portability/size.
        keys[layer] = k_tensor.cpu().numpy().astype(np.float16)
        values[layer] = v.astype(np.float16)

    token_ids = tokens.squeeze(0).detach().cpu().numpy().astype(np.int64)

    metadata = {
        "model": getattr(model.cfg, "model_name", "unknown"),
        "num_layers": int(model.cfg.n_layers),
        "positional_embedding_type": getattr(model.cfg, "positional_embedding_type", None),
        "dtype": str(model.cfg.dtype),
        "kv_shape": list(keys[0].shape) if keys else [],
    }

    logger.info("Extracted KV cache: %d layers, prefix_len=%d", len(keys), prefix_len)

    return StoredTLKVCache(
        keys=keys,
        values=values,
        token_ids=token_ids,
        text=text,
        prefix_len=prefix_len,
        metadata=metadata,
    )


def save_kv_cache_tl(stored_kv: StoredTLKVCache, filepath: str | Path) -> Path:
    """Save a StoredTLKVCache to .npz file."""
    filepath = Path(filepath)
    if filepath.suffix != ".npz":
        filepath = filepath.with_suffix(".npz")

    arrays: Dict[str, np.ndarray] = {}
    for layer_idx in stored_kv.keys:
        arrays[f"k_{layer_idx}"] = stored_kv.keys[layer_idx]
        arrays[f"v_{layer_idx}"] = stored_kv.values[layer_idx]

    arrays["token_ids"] = stored_kv.token_ids
    metadata_json = json.dumps(
        {
            "text": stored_kv.text,
            "prefix_len": stored_kv.prefix_len,
            "metadata": stored_kv.metadata,
        }
    )
    arrays["metadata_json"] = np.array([metadata_json], dtype="S")

    np.savez_compressed(filepath, **arrays)
    logger.info("Saved KV cache to %s", filepath)
    return filepath


def load_kv_cache_tl(filepath: str | Path) -> StoredTLKVCache:
    """Load a StoredTLKVCache from .npz file."""
    filepath = Path(filepath)
    data = np.load(filepath, allow_pickle=True)

    keys: Dict[int, np.ndarray] = {}
    values: Dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith("k_"):
            layer_idx = int(key[2:])
            keys[layer_idx] = data[key].astype(np.float16)
        elif key.startswith("v_"):
            layer_idx = int(key[2:])
            values[layer_idx] = data[key].astype(np.float16)

    token_ids = data["token_ids"].astype(np.int64)
    metadata_json = bytes(data["metadata_json"][0]).decode("utf-8")
    metadata_dict = json.loads(metadata_json)

    return StoredTLKVCache(
        keys=keys,
        values=values,
        token_ids=token_ids,
        text=metadata_dict["text"],
        prefix_len=metadata_dict["prefix_len"],
        metadata=metadata_dict.get("metadata", {}),
    )


def build_past_kv_cache_tl(
    model,
    stored_kv: StoredTLKVCache,
) -> "HookedTransformerKeyValueCache":
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    device = model.W_E.device
    dtype = model.W_E.dtype

    cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg,
        device=device,
        batch_size=1,
    )

    for layer in range(int(model.cfg.n_layers)):
        if layer not in stored_kv.keys or layer not in stored_kv.values:
            raise RuntimeError(f"Missing K/V for layer {layer}")
        k = torch.from_numpy(stored_kv.keys[layer]).to(device=device, dtype=dtype)
        v = torch.from_numpy(stored_kv.values[layer]).to(device=device, dtype=dtype)
        cache.entries[layer].past_keys = k
        cache.entries[layer].past_values = v

    cache.previous_attention_mask = torch.ones(
        (1, stored_kv.prefix_len), dtype=torch.int, device=device
    )
    return cache


@torch.no_grad()
def greedy_generate_with_cache(
    model,
    cache,
    query_tokens: torch.Tensor,
    max_new_tokens: int = 120,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generate tokens with a pre-populated cache (prefix already inside)."""
    logits, _ = model.run_with_cache(
        query_tokens,
        return_type="logits",
        past_kv_cache=cache,
        prepend_bos=False,
        remove_batch_dim=False,
    )

    generated: list[torch.Tensor] = []
    eos_token_id = getattr(model.tokenizer, "eos_token_id", None) if model.tokenizer else None

    for _ in range(max_new_tokens):
        last_logits = logits[:, -1]
        if do_sample:
            if temperature <= 0:
                temperature = 1.0
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        generated.append(next_token)
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break

        logits, _ = model.run_with_cache(
            next_token,
            return_type="logits",
            past_kv_cache=cache,
            prepend_bos=False,
            remove_batch_dim=False,
        )

    if not generated:
        return torch.zeros((1, 0), dtype=query_tokens.dtype, device=query_tokens.device)
    return torch.cat(generated, dim=1)


@torch.no_grad()
def generate_with_stored_kv_tl(
    model,
    stored_kv: StoredTLKVCache,
    query: str,
    prepend_bos: bool = True,
    max_new_tokens: int = 120,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    cache = build_past_kv_cache_tl(model, stored_kv)
    query_tokens = model.to_tokens(query, prepend_bos=prepend_bos)
    generated_tokens = greedy_generate_with_cache(
        model,
        cache,
        query_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    return model.to_string(generated_tokens[0])


@torch.no_grad()
def generate_baseline_tl(
    model,
    query: str,
    prepend_bos: bool = True,
    max_new_tokens: int = 120,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg,
        device=model.W_E.device,
        batch_size=1,
    )
    query_tokens = model.to_tokens(query, prepend_bos=prepend_bos)
    generated_tokens = greedy_generate_with_cache(
        model,
        cache,
        query_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    return model.to_string(generated_tokens[0])
