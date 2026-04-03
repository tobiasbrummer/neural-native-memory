#!/usr/bin/env python3
"""
Simplest approach: Extract K/V from the cache after a forward pass.

The cache already contains K/V with RoPE properly applied.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .model_loader import load_model

logger = logging.getLogger(__name__)


@dataclass
class CacheKVData:
    """Container for K/V pairs extracted from cache."""
    keys: Dict[int, np.ndarray]  # {layer_idx: [num_heads, seq_len, head_dim]}
    values: Dict[int, np.ndarray]
    token_ids: np.ndarray
    text: str
    metadata: Dict


def extract_kv_from_cache(
    model,
    tokenizer,
    text: str,
) -> CacheKVData:
    """
    Extract K/V pairs from cache after a forward pass.

    This is the simplest and most reliable method:
    1. Run forward pass with use_cache=True
    2. Extract K/V from the resulting cache
    3. The K/V already have RoPE properly applied

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text

    Returns:
        CacheKVData with extracted K/V pairs
    """
    device = model.device

    # Prepare inputs
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Create cache for forward pass
    cache = DynamicCache()

    # Forward pass using model.generate with 0 new tokens
    # This properly handles all the mRoPE complexity
    with torch.no_grad():
        outputs = model(
            **inputs,
            past_key_values=cache,
            max_new_tokens=0,  # Don't generate, just get cache
            output_hidden_states=False,
        )

    # Extract K/V from cache
    keys_per_layer = {}
    values_per_layer = {}

    for layer_idx, layer in enumerate(cache.layers):
        # Each layer has 'keys' and 'values' attributes
        # shape: [batch, num_heads, seq_len, head_dim]
        if layer.keys is None:
            continue

        key_tensor = layer.keys
        value_tensor = layer.values

        # Convert to numpy and remove batch dim
        keys_per_layer[layer_idx] = key_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # [num_heads, seq_len, head_dim]
        values_per_layer[layer_idx] = value_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    token_ids = inputs["input_ids"].squeeze(0).cpu().numpy()

    # Get metadata
    first_key = next(iter(keys_per_layer.values()))
    metadata = {
        "seq_len": len(token_ids),
        "num_heads": first_key.shape[0],
        "head_dim": first_key.shape[2],
        "num_layers": len(keys_per_layer),
    }

    return CacheKVData(
        keys=keys_per_layer,
        values=values_per_layer,
        token_ids=token_ids,
        text=text,
        metadata=metadata,
    )


def save_cache_kv(data: CacheKVData, path: str):
    """Save CacheKVData to numpy file."""
    np.savez(
        path,
        keys={str(k): v for k, v in data.keys.items()},
        values={str(k): v for k, v in data.values.items()},
        token_ids=data.token_ids,
        text=data.text,
        **{f"meta_{k}": v for k, v in data.metadata.items()},
    )


def load_cache_kv(path: str) -> CacheKVData:
    """Load CacheKVData from numpy file."""
    data = np.load(path, allow_pickle=True)

    keys = {int(k): v for k, v in data["keys"].item().items()}
    values = {int(k): v for k, v in data["values"].item().items()}

    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key.startswith("meta_"):
            metadata[key[5:]] = data[key]

    return CacheKVData(
        keys=keys,
        values=values,
        token_ids=data["token_ids"],
        text=str(data["text"]),
        metadata=metadata,
    )


def inject_cache_kv(
    model,
    kv_data: CacheKVData,
) -> DynamicCache:
    """Inject extracted K/V into a new cache."""
    cache = DynamicCache()
    device = model.device
    dtype = next(model.parameters()).dtype

    for layer_idx, keys_np in kv_data.keys.items():
        values_np = kv_data.values[layer_idx]

        # Convert to torch
        # keys_np: [num_heads, seq_len, head_dim]
        # cache needs: [batch, num_heads, seq_len, head_dim]
        keys_t = torch.from_numpy(keys_np).to(device).to(dtype)
        values_t = torch.from_numpy(values_np).to(device).to(dtype)

        # Add batch dimension
        keys_t = keys_t.unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        values_t = values_t.unsqueeze(0)

        with torch.no_grad():
            cache.update(keys_t, values_t, layer_idx)

    return cache


def generate_with_cache_kv(
    model,
    tokenizer,
    kv_data: CacheKVData,
    query: str,
    max_new_tokens: int = 50,
) -> str:
    """Generate with extracted K/V from cache."""
    # Inject K/V
    cache = inject_cache_kv(model, kv_data)

    # Prepare query inputs
    query_inputs = tokenizer(query, return_tensors="pt")
    query_inputs = {k: v.to(model.device) for k, v in query_inputs.items()}

    # Adjust position_ids for Qwen3-VL mRoPE
    prefix_len = kv_data.metadata["seq_len"]
    query_len = query_inputs["input_ids"].shape[1]

    position_ids = torch.arange(prefix_len, prefix_len + query_len, device=model.device)
    position_ids = position_ids.view(1, 1, -1).expand(3, 1, -1)
    query_inputs["position_ids"] = position_ids

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **query_inputs,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only new tokens
    generated_ids = outputs[0][query_inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    # Test
    model, tokenizer = load_model(load_in_4bit=True)

    kv_data = extract_kv_from_cache(model, tokenizer, "The sky is blue.")

    print(f"Extracted K/V for {len(kv_data.keys)} layers")
    print(f"Keys shape for layer 0: {kv_data.keys[0].shape}")

    # Test generation
    result = generate_with_cache_kv(model, tokenizer, kv_data, "What color is the sky?")
    print(f"Generated: {result}")
