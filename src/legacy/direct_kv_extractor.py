#!/usr/bin/env python3
"""
Alternative approach: Extract K/V directly from transformers' attention mechanism
instead of projecting ourselves. This ensures full compatibility with mRoPE.
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

from .model_loader import load_model
from .io_utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class DirectKVData:
    """Container for K/V pairs extracted directly from transformers."""
    keys: Dict[int, np.ndarray]  # {layer_idx: [seq_len, num_heads, head_dim]}
    values: Dict[int, np.ndarray]
    token_ids: np.ndarray
    text: str
    metadata: Dict


class DirectKVExtractor:
    """Extract K/V pairs directly from transformers' forward pass."""

    def __init__(self, model, tokenizer, target_layers):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.device = model.device
        self._extracted_keys = {}
        self._extracted_values = {}
        self._hooks = []

    def _create_hook(self, layer_idx):
        """Create a forward hook to capture K/V after RoPE is applied."""
        def hook(module, args, kwargs):
            # This hook runs AFTER attention's projection but BEFORE attention computation
            # We need to capture k_proj and v_proj outputs AFTER RoPE

            # Get the hidden states (input to attention)
            if args and len(args) > 0:
                hidden_states = args[0]
            else:
                return args, kwargs

            # Get position_embeddings from kwargs if available
            position_embeddings = kwargs.get("position_embeddings", None)
            if position_embeddings is None:
                return args, kwargs

            # Project to K and V
            with torch.no_grad():
                k = module.k_proj(hidden_states)
                v = module.v_proj(hidden_states)

                # Apply k_norm and v_norm (Qwen3-VL has these)
                if hasattr(module, "k_norm"):
                    k = module.k_norm(k)
                if hasattr(module, "v_norm"):
                    v = module.v_norm(v)

                # Reshape for attention
                input_shape = hidden_states.shape[:-1]
                shape = (*input_shape, -1, module.head_dim)

                k = k.view(shape).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
                v = v.view(shape).transpose(1, 2)

                # Apply RoPE using position_embeddings
                cos, sin = position_embeddings
                from modeling_qwen3_vl import apply_rotary_pos_emb

                def rotate_half(x):
                    x1 = x[..., : x.shape[-1] // 2]
                    x2 = x[..., x.shape[-1] // 2 :]
                    return torch.cat((-x2, x1), dim=-1)

                def apply_rope(q, k, cos, sin):
                    cos = cos.unsqueeze(1)
                    sin = sin.unsqueeze(1)
                    q_embed = (q * cos) + (rotate_half(q) * sin)
                    k_embed = (k * cos) + (rotate_half(k) * sin)
                    return q_embed, k_embed

                k_rot, _ = apply_rope(k, k, cos, sin)

                # Store the K/V (remove batch dim, convert to numpy)
                self._extracted_keys[layer_idx] = k_rot.squeeze(0).transpose(0, 1).cpu().numpy()
                self._extracted_values[layer_idx] = v.squeeze(0).transpose(0, 1).cpu().numpy()

            return args, kwargs

        return hook

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        # Get the language model for Qwen3-VL
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            layers = self.model.model.language_model.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        else:
            raise ValueError("Cannot find model layers")

        for layer_idx in self.target_layers:
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} >= num_layers {len(layers)}, skipping")
                continue

            layer = layers[layer_idx]
            hook = layer.self_attn.register_forward_pre_hook(
                self._create_hook(layer_idx),
                with_kwargs=True
            )
            self._hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def extract(self, text):
        """Extract K/V pairs for a text."""
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Prepare position_ids for Qwen3-VL mRoPE
        seq_len = inputs["input_ids"].shape[1]
        position_ids = torch.arange(seq_len, device=self.device)
        position_ids = position_ids.view(1, 1, -1).expand(3, 1, -1)  # [3, batch, seq_len]

        # Register hooks
        self._register_hooks()

        try:
            # Forward pass (with position_ids)
            _ = self.model.model(**inputs, position_ids=position_ids)
        finally:
            self._remove_hooks()

        if not self._extracted_keys:
            raise ValueError("No K/V pairs extracted!")

        token_ids = inputs["input_ids"].squeeze(0).cpu().numpy()

        # Get metadata
        first_key = next(iter(self._extracted_keys.values()))
        metadata = {
            "seq_len": len(token_ids),
            "num_heads": first_key.shape[1],
            "head_dim": first_key.shape[2],
        }

        return DirectKVData(
            keys=self._extracted_keys,
            values=self._extracted_values,
            token_ids=token_ids,
            text=text,
            metadata=metadata,
        )


def save_direct_kv(data: DirectKVData, path: str):
    """Save DirectKVData to numpy file."""
    np.savez(
        path,
        keys={str(k): v for k, v in data.keys.items()},
        values={str(k): v for k, v in data.values.items()},
        token_ids=data.token_ids,
        text=data.text,
        **{f"meta_{k}": v for k, v in data.metadata.items()},
    )


def load_direct_kv(path: str) -> DirectKVData:
    """Load DirectKVData from numpy file."""
    data = np.load(path, allow_pickle=True)

    keys = {int(k): v for k, v in data["keys"].item().items()}
    values = {int(k): v for k, v in data["values"].item().items()}

    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key.startswith("meta_"):
            metadata[key[5:]] = data[key]

    return DirectKVData(
        keys=keys,
        values=values,
        token_ids=data["token_ids"],
        text=str(data["text"]),
        metadata=metadata,
    )


def inject_direct_kv(
    model,
    kv_data: DirectKVData,
) -> "DynamicCache":
    """Inject directly extracted K/V into cache."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    device = model.device
    dtype = next(model.parameters()).dtype

    for layer_idx, keys_np in kv_data.keys.items():
        values_np = kv_data.values[layer_idx]

        # Convert to torch
        keys_t = torch.from_numpy(keys_np).to(device).to(dtype)
        values_t = torch.from_numpy(values_np).to(device).to(dtype)

        # Transpose to cache format: [batch, num_heads, seq_len, head_dim]
        keys_t = keys_t.transpose(0, 1).unsqueeze(0)
        values_t = values_t.transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            cache.update(keys_t, values_t, layer_idx)

    return cache


def generate_with_direct_kv(
    model,
    tokenizer,
    kv_data: DirectKVData,
    query: str,
    max_new_tokens: int = 50,
) -> str:
    """Generate with directly extracted K/V."""
    # Inject K/V
    cache = inject_direct_kv(model, kv_data)

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
    target_layers = [24]

    extractor = DirectKVExtractor(model, tokenizer, target_layers)
    kv_data = extractor.extract("The sky is blue.")

    print(f"Extracted K/V for {len(kv_data.keys)} layers")
    print(f"Keys shape: {kv_data.keys[target_layers[0]].shape}")
    print(f"Values shape: {kv_data.values[target_layers[0]].shape}")

    # Test generation
    result = generate_with_direct_kv(model, tokenizer, kv_data, "What color is the sky?")
    print(f"Generated: {result}")
