#!/usr/bin/env python3
"""
Test to compare our KV projection + RoPE with transformers' internal computation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from lib.model_loader import load_model
from lib.virtual_prefix import (
    PreRopeExtractor,
    project_hidden_to_kv,
    apply_rope_to_keys,
    _get_model_layer,
)


def main():
    print("Loading model...")
    model, tokenizer = load_model(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,
    )

    # Simple test text
    text = "The sky is blue."
    target_layers = [24]  # Just test one layer

    print(f"\nTest text: '{text}'")
    print(f"Target layer: {target_layers[0]}")

    # Step 1: Extract Pre-RoPE hidden states
    print("\n" + "="*60)
    print("Step 1: Extracting Pre-RoPE hidden states")
    print("="*60)
    extractor = PreRopeExtractor(model, tokenizer, target_layers)
    prefix_data = extractor.extract(text, use_prompt=False)

    hidden_np = prefix_data.hidden_states[target_layers[0]]
    print(f"Hidden states shape: {hidden_np.shape}")

    # Step 2: Project to K, V
    print("\n" + "="*60)
    print("Step 2: Projecting to K, V")
    print("="*60)
    keys_proj, values_proj = project_hidden_to_kv(hidden_np, target_layers[0], model)
    print(f"Projected keys shape: {keys_proj.shape}")
    print(f"Projected values shape: {values_proj.shape}")

    # Step 3: Get transformers' actual K/V for comparison
    print("\n" + "="*60)
    print("Step 3: Getting transformers' internal K/V")
    print("="*60)

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get the layer
    layer = _get_model_layer(model, target_layers[0])
    attn = layer.self_attn

    # Forward through attention to get K/V
    with torch.no_grad():
        # Get hidden states from model
        outputs = model.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[target_layers[0] - 1]  # Input to layer

        # Project through k_proj, v_proj
        k_internal = attn.k_proj(hidden_states)
        v_internal = attn.v_proj(hidden_states)

        # Reshape to match our format
        # k_internal: [batch, seq_len, num_kv_heads * head_dim]
        num_kv_heads = attn.num_key_value_heads if hasattr(attn, 'num_key_value_heads') else 8
        head_dim = attn.head_dim if hasattr(attn, 'head_dim') else 128

        k_internal = k_internal.view(1, inputs["input_ids"].shape[1], num_kv_heads, head_dim)
        v_internal = v_internal.view(1, inputs["input_ids"].shape[1], num_kv_heads, head_dim)

        # Apply RoPE using transformers
        # Get position embeddings
        position_ids = torch.arange(inputs["input_ids"].shape[1], device=model.device).view(1, -1)
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Get RoPE from language model
        rotary_emb = model.model.language_model.rotary_emb

        # Qwen3-VL expects position_ids with shape [3, batch, seq_len]
        # For text-only, we use [0, 0, 0] * seq_len for each dimension
        seq_len = inputs["input_ids"].shape[1]
        position_ids = torch.arange(seq_len, device=model.device)
        position_ids = position_ids.view(1, 1, -1).expand(3, 1, -1)  # [3, 1, seq_len]

        cos, sin = rotary_emb(hidden_states, position_ids)

        # Apply RoPE to K using transformers function
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb_single(q, k, cos, sin):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        k_internal_reshaped = k_internal.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k_rope_internal, _ = apply_rotary_pos_emb_single(k_internal_reshaped, k_internal_reshaped, cos, sin)

        # Reshape back to [seq_len, num_heads, head_dim]
        k_rope_internal = k_rope_internal.squeeze(0).transpose(0, 1).cpu().numpy()

    print(f"Internal keys (after RoPE) shape: {k_rope_internal.shape}")

    # Step 4: Apply our RoPE
    print("\n" + "="*60)
    print("Step 4: Applying our RoPE")
    print("="*60)
    positions = np.arange(keys_proj.shape[0], dtype=np.int64)
    keys_our_rope = apply_rope_to_keys(keys_proj, positions, model, target_layers[0])
    print(f"Our keys (after RoPE) shape: {keys_our_rope.shape}")

    # Step 5: Compare
    print("\n" + "="*60)
    print("Step 5: Comparison")
    print("="*60)

    k_internal_flat = k_rope_internal.flatten().astype(np.float32)  # Convert to float32
    k_our_flat = keys_our_rope.flatten()

    print(f"Internal K stats: mean={k_internal_flat.mean():.6f}, std={k_internal_flat.std():.6f}")
    print(f"Our K stats: mean={k_our_flat.mean():.6f}, std={k_our_flat.std():.6f}")

    # Compute cosine similarity (filter out inf/nan)
    mask = np.isfinite(k_internal_flat) & np.isfinite(k_our_flat)
    k_internal_clean = k_internal_flat[mask]
    k_our_clean = k_our_flat[mask]

    print(f"Valid values: {mask.sum()} / {len(mask)} (filtered inf/nan)")

    if len(k_internal_clean) > 0:
        dot_product = np.dot(k_internal_clean, k_our_clean)
        norm_internal = np.linalg.norm(k_internal_clean)
        norm_our = np.linalg.norm(k_our_clean)
        cosine_sim = dot_product / (norm_internal * norm_our)

        print(f"\nCosine similarity: {cosine_sim:.6f}")

        # Show some values
        print(f"\nArray shapes: internal={k_internal_flat.shape}, our={k_our_flat.shape}")
        print(f"Dtype: internal={k_internal_flat.dtype}, our={k_our_flat.dtype}")
        print(f"\nFirst 10 values of internal K: {k_internal_flat[:10]}")
        print(f"First 10 values of our K:    {k_our_flat[:10]}")

        # Check sign patterns
        sign_match = np.sign(k_internal_flat[:100]) == np.sign(k_our_flat[:100])
        print(f"\nSign match (first 100): {sign_match.sum()}/100")

        # Max absolute difference
        max_diff = np.max(np.abs(k_internal_flat[:100] - k_our_flat[:100]))
        print(f"Max absolute difference (first 100): {max_diff:.6f}")

        # Mean absolute difference
        mean_diff = np.mean(np.abs(k_internal_flat[:100] - k_our_flat[:100]))
        print(f"Mean absolute difference (first 100): {mean_diff:.6f}")

        # Debug cosine sim manually (first 100)
        dot_manual = np.sum(k_internal_clean[:100] * k_our_clean[:100])
        print(f"\nManual dot product (first 100): {dot_manual:.6f}")
        print(f"Norm internal (first 100): {np.linalg.norm(k_internal_clean[:100]):.6f}")
        print(f"Norm our (first 100): {np.linalg.norm(k_our_clean[:100]):.6f}")

        # Check if they match
        if cosine_sim > 0.99:
            print("\n✓ RoPE calculation is CORRECT!")
        elif cosine_sim > 0.95:
            print("\n~ RoPE calculation is CLOSE but not exact")
        else:
            print("\n✗ RoPE calculation DOES NOT MATCH!")
    else:
        print("ERROR: No valid values to compare!")


if __name__ == "__main__":
    main()
