#!/usr/bin/env python3
"""Quick diagnostic: verify that steering hooks actually fire in 4-bit mode."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "TransformerLens"))

from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder

config = KVEmbeddingConfig(
    model_name="Qwen/Qwen3-8B",
    device=None,
    dtype="float16",
    load_in_4bit=True,
)
embedder = TransformerLensKVEmbedder(config)
model = embedder.model
print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
print(f"load_in_4bit config: {model.cfg.load_in_4bit}")

# --- Check 1: Does hook_resid_pre exist in hook_dict? ---
layer = 28
hook_name = f"blocks.{layer}.hook_resid_pre"
print(f"\nHook name: {hook_name}")
print(f"In hook_dict: {hook_name in model.hook_dict}")
if hook_name in model.hook_dict:
    hp = model.hook_dict[hook_name]
    print(f"  HookPoint name: {hp.name}")
    print(f"  HookPoint type: {type(hp)}")

# --- Check 2: Do hooks fire during run_with_cache? ---
hook_fired = {"count": 0, "shapes": []}

def diag_hook(resid, hook):
    hook_fired["count"] += 1
    hook_fired["shapes"].append(tuple(resid.shape))
    return resid  # no-op

prompt = "<|im_start|>system\n/no_think\n<|im_end|>\n<|im_start|>user\nHello\n<|im_end|>\n<|im_start|>assistant\n"
tokens = model.to_tokens(prompt, prepend_bos=True)
print(f"\nTokens shape: {tokens.shape}")

with torch.no_grad():
    with model.hooks(fwd_hooks=[(hook_name, diag_hook)]):
        logits, cache = model.run_with_cache(
            tokens,
            return_type="logits",
            prepend_bos=False,
            remove_batch_dim=False,
        )

print(f"\nHook fired {hook_fired['count']} time(s)")
print(f"Activation shapes: {hook_fired['shapes']}")

# --- Check 3: Does modifying the residual actually change logits? ---
print("\n--- Check 3: Does modification propagate? ---")
# Baseline
with torch.no_grad():
    logits_base = model(tokens, prepend_bos=False, return_type="logits")
    top5_base = torch.topk(logits_base[0, -1], 5)
    print(f"Baseline top-5 tokens: {top5_base.indices.tolist()}")
    print(f"Baseline top-5 logits: {[f'{v:.3f}' for v in top5_base.values.tolist()]}")

# With steering: add a LARGE random vector to see if it has any effect
random_vec = torch.randn(1, 1, model.cfg.d_model, device=tokens.device, dtype=torch.float16) * 100.0

def big_steer_hook(resid, hook, v=random_vec):
    out = resid.clone()
    out[:, -1:, :] = out[:, -1:, :] + v.to(device=resid.device, dtype=resid.dtype)
    return out

with torch.no_grad():
    logits_steered = model.run_with_hooks(
        tokens,
        prepend_bos=False,
        return_type="logits",
        fwd_hooks=[(hook_name, big_steer_hook)],
    )
    top5_steered = torch.topk(logits_steered[0, -1], 5)
    print(f"Steered top-5 tokens: {top5_steered.indices.tolist()}")
    print(f"Steered top-5 logits: {[f'{v:.3f}' for v in top5_steered.values.tolist()]}")

logit_diff = (logits_steered[0, -1] - logits_base[0, -1]).abs().max().item()
print(f"\nMax logit diff: {logit_diff:.4f}")
if logit_diff < 0.01:
    print("*** PROBLEM: Steering has NO effect on logits! Hook modification is NOT propagating. ***")
elif logit_diff > 1.0:
    print("OK: Steering clearly changes logits.")
else:
    print("Weak: Steering has minimal effect.")

# --- Check 4: Try with run_with_cache (same as generation loop) ---
print("\n--- Check 4: run_with_cache with hooks context ---")
hook_fired2 = {"count": 0}

def diag_steer_hook(resid, hook, v=random_vec):
    hook_fired2["count"] += 1
    out = resid.clone()
    out[:, -1:, :] = out[:, -1:, :] + v.to(device=resid.device, dtype=resid.dtype)
    return out

with torch.no_grad():
    with model.hooks(fwd_hooks=[(hook_name, diag_steer_hook)]):
        logits_ctx, _ = model.run_with_cache(
            tokens,
            return_type="logits",
            prepend_bos=False,
            remove_batch_dim=False,
        )
    top5_ctx = torch.topk(logits_ctx[0, -1], 5)
    print(f"Hook fired {hook_fired2['count']} time(s)")
    print(f"Context-steered top-5 tokens: {top5_ctx.indices.tolist()}")
    logit_diff2 = (logits_ctx[0, -1] - logits_base[0, -1]).abs().max().item()
    print(f"Max logit diff: {logit_diff2:.4f}")
    if logit_diff2 < 0.01:
        print("*** PROBLEM: Hooks DON'T fire inside run_with_cache! ***")
    else:
        print("OK: Hooks fire correctly inside run_with_cache context.")

print("\n--- Done ---")
