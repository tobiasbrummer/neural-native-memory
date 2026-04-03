#!/usr/bin/env python3
"""Experiment 9 (NNM): Delta -> K/V reconstruction and injection (TransformerLens)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.io_utils import create_results_dir, save_json, setup_logging
from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder
from nnm.kvembed.prompts import build_compression_prompt
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.utils import repeat_along_head_dimension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp9: Delta->KV reconstruction + injection (TransformerLens)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--memory", type=str, default="Das Lieblingstier des Users ist der Hase.")
    parser.add_argument("--query", type=str, default="Was ist das Lieblingstier des Users?")
    parser.add_argument("--id-corpus", type=str, default=None)
    parser.add_argument("--prefix-bias", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--logits-topk", type=int, default=8)
    return parser.parse_args()


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _topk_tokens(model, logits: torch.Tensor, k: int) -> list[dict[str, object]]:
    if k <= 0:
        return []
    values, indices = torch.topk(logits, k)
    results: list[dict[str, object]] = []
    for score, token_id in zip(values[0].tolist(), indices[0].tolist()):
        token_str = model.to_string([int(token_id)])
        results.append({"token_id": int(token_id), "token": token_str, "logit": float(score)})
    return results


def _build_attention_bias_hooks(model, selected_layers: Iterable[int], prefix_bias: float, non_rerouted: float):
    selected = set(int(x) for x in selected_layers)
    hooks = []

    for layer in range(int(model.cfg.n_layers)):
        name = f"blocks.{layer}.attn.hook_attn_scores"
        bias_value = float(prefix_bias if layer in selected else non_rerouted)

        def _hook(scores: torch.Tensor, hook, b: float = bias_value) -> torch.Tensor:
            updated = scores.clone()
            updated[..., 0] = updated[..., 0] + b
            return updated

        hooks.append((name, _hook))
    return hooks


def _compute_kv_from_resid_pre(model, layer: int, resid_pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block = model.blocks[layer]
    cfg = model.cfg

    if cfg.use_split_qkv_input:
        n_kv_heads = (
            cfg.n_key_value_heads
            if cfg.n_key_value_heads is not None and not cfg.ungroup_grouped_query_attention
            else cfg.n_heads
        )
        query_input = repeat_along_head_dimension(resid_pre, n_heads=cfg.n_heads)
        key_input = repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
        value_input = repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
    elif cfg.use_attn_in:
        attn_in = repeat_along_head_dimension(resid_pre, n_heads=cfg.n_heads)
        query_input = attn_in
        key_input = attn_in
        value_input = attn_in
    else:
        query_input = resid_pre
        key_input = resid_pre
        value_input = resid_pre

    q_in = block.ln1(query_input)
    k_in = block.ln1(key_input)
    v_in = block.ln1(value_input)

    _, k, v = block.attn.calculate_qkv_matrices(q_in, k_in, v_in)
    return k, v


def _build_prefix_cache(model, selected_layers: Iterable[int], kv_by_layer: Dict[int, tuple[torch.Tensor, torch.Tensor]]):
    selected = set(int(x) for x in selected_layers)
    cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg,
        device=model.W_E.device,
        batch_size=1,
    )
    n_layers = int(model.cfg.n_layers)
    n_kv_heads = (
        int(model.cfg.n_key_value_heads)
        if model.cfg.n_key_value_heads is not None
        else int(model.cfg.n_heads)
    )
    d_head = int(model.cfg.d_head)
    dtype = model.W_E.dtype
    device = model.W_E.device

    for layer in range(n_layers):
        if layer in selected:
            k, v = kv_by_layer[layer]
            k_prefix = k.to(device=device, dtype=dtype)
            v_prefix = v.to(device=device, dtype=dtype)
        else:
            k_prefix = torch.zeros((1, 1, n_kv_heads, d_head), dtype=dtype, device=device)
            v_prefix = torch.zeros((1, 1, n_kv_heads, d_head), dtype=dtype, device=device)

        cache.entries[layer].past_keys = k_prefix
        cache.entries[layer].past_values = v_prefix

    cache.previous_attention_mask = torch.ones((1, 1), dtype=torch.int, device=device)
    cache.freeze()
    return cache


def main() -> int:
    args = parse_args()
    results_dir = create_results_dir("nnm_exp9")
    logger = setup_logging("nnm_exp9_delta_to_kv_injection_tl", results_dir)

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        prefix_bias=args.prefix_bias,
    )
    embedder = TransformerLensKVEmbedder(config)
    model = embedder.model
    prepend_bos = embedder._prepend_bos

    id_texts = _read_lines(Path(args.id_corpus)) if args.id_corpus else [args.memory]
    layer_selection = embedder.select_layers(id_texts)
    selected_layers = list(layer_selection.selected_layers)

    memory_prompt = build_compression_prompt(
        text=args.memory,
        role="context",
        template=config.prompt_template,
    )
    query_prompt = f"Query: {args.query}\nAnswer:"

    logger.info("Memory prompt: %s", memory_prompt)
    logger.info("Query prompt: %s", query_prompt)
    logger.info("Selected layers: %s", selected_layers)

    # Forward pass on memory prompt to capture resid_pre + K/V
    token_ids = model.to_tokens(memory_prompt, prepend_bos=prepend_bos)
    last_token_id = int(token_ids[0, -1].item())

    names = set()
    for layer in selected_layers:
        names.add(f"blocks.{layer}.hook_resid_pre")
        names.add(f"blocks.{layer}.attn.hook_k")
        names.add(f"blocks.{layer}.attn.hook_v")

    with torch.no_grad():
        _, cache = model.run_with_cache(
            token_ids,
            return_type=None,
            prepend_bos=False,
            names_filter=lambda name: name in names,
            remove_batch_dim=False,
        )

    # Build delta for last token at each selected layer and reconstruct K/V
    token_embed = model.W_E[last_token_id].detach().to(model.W_E.device)
    token_embed = token_embed.view(1, 1, -1)

    kv_true: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    kv_recon: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    kv_metrics: Dict[int, dict[str, float]] = {}

    for layer in selected_layers:
        resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][:, -1:, :].detach()
        k_true = cache[f"blocks.{layer}.attn.hook_k"][:, -1:, :, :].detach()
        v_true = cache[f"blocks.{layer}.attn.hook_v"][:, -1:, :, :].detach()

        delta = resid_pre - token_embed
        resid_rec = delta + token_embed

        k_rec, v_rec = _compute_kv_from_resid_pre(model, layer, resid_rec)

        diff_k = (k_rec - k_true).detach().to(torch.float32).cpu().numpy()
        diff_v = (v_rec - v_true).detach().to(torch.float32).cpu().numpy()

        kv_metrics[layer] = {
            "k_abs_mean": float(np.mean(np.abs(diff_k))),
            "k_abs_max": float(np.max(np.abs(diff_k))),
            "v_abs_mean": float(np.mean(np.abs(diff_v))),
            "v_abs_max": float(np.max(np.abs(diff_v))),
        }

        kv_true[layer] = (k_true, v_true)
        kv_recon[layer] = (k_rec, v_rec)

    # Build caches for injection comparison
    true_cache = _build_prefix_cache(model, selected_layers, kv_true)
    recon_cache = _build_prefix_cache(model, selected_layers, kv_recon)

    attn_hooks = _build_attention_bias_hooks(
        model,
        selected_layers,
        prefix_bias=args.prefix_bias,
        non_rerouted=config.non_rerouted_prefix_bias,
    )

    query_tokens = model.to_tokens(query_prompt, prepend_bos=prepend_bos)

    with torch.no_grad():
        with model.hooks(fwd_hooks=attn_hooks):
            logits_true, _ = model.run_with_cache(
                query_tokens,
                return_type="logits",
                past_kv_cache=true_cache,
                prepend_bos=False,
                remove_batch_dim=False,
            )
            logits_recon, _ = model.run_with_cache(
                query_tokens,
                return_type="logits",
                past_kv_cache=recon_cache,
                prepend_bos=False,
                remove_batch_dim=False,
            )

        logits_base, _ = model.run_with_cache(
            query_tokens,
            return_type="logits",
            past_kv_cache=None,
            prepend_bos=False,
            remove_batch_dim=False,
        )

    topk = max(0, int(args.logits_topk))
    topk_true = _topk_tokens(model, logits_true[:, -1], topk)
    topk_recon = _topk_tokens(model, logits_recon[:, -1], topk)
    topk_base = _topk_tokens(model, logits_base[:, -1], topk)

    logit_diff = (logits_recon[:, -1] - logits_true[:, -1]).detach().to(torch.float32).cpu().numpy()
    logit_metrics = {
        "logits_abs_mean": float(np.mean(np.abs(logit_diff))),
        "logits_abs_max": float(np.max(np.abs(logit_diff))),
    }

    logger.info("KV diff per layer: %s", kv_metrics)
    logger.info("Logit diff (recon vs true): %s", logit_metrics)
    logger.info("Top-%d logits (true): %s", topk, topk_true)
    logger.info("Top-%d logits (recon): %s", topk, topk_recon)
    logger.info("Top-%d logits (base): %s", topk, topk_base)

    output = {
        "experiment": "nnm_exp9_delta_to_kv_injection_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "prefix_bias": args.prefix_bias,
            "logits_topk": topk,
        },
        "memory_prompt": memory_prompt,
        "query_prompt": query_prompt,
        "selected_layers": selected_layers,
        "kv_diff": kv_metrics,
        "logit_diff": logit_metrics,
        "topk_logits": {
            "true": topk_true,
            "recon": topk_recon,
            "baseline": topk_base,
        },
    }

    out_path = results_dir / "results.json"
    save_json(output, out_path)
    logger.info("Saved results to %s", out_path)

    # Success if reconstruction is numerically close
    success = all(v["k_abs_max"] < 1e-3 and v["v_abs_max"] < 1e-3 for v in kv_metrics.values())
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
