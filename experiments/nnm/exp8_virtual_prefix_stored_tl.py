#!/usr/bin/env python3
"""Experiment 8 (NNM): Stored KV cache injection with TransformerLens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from lib.io_utils import create_results_dir, save_json, setup_logging
from nnm.kvembed import (
    KVEmbeddingConfig,
    TransformerLensKVEmbedder,
    extract_kv_cache_tl,
    generate_baseline_tl,
    generate_with_stored_kv_tl,
    load_kv_cache_tl,
    save_kv_cache_tl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp8: Stored KV cache injection (TransformerLens)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--memory", type=str, default="Faktum: Mein Lieblingstier ist der Hase.")
    parser.add_argument(
        "--query",
        type=str,
        default="Was frisst das Lieblingstier des Users?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument(
        "--expected",
        type=str,
        default="Moehren,Karotten,Gemuese,Gras,Salat,Klee,Heu",
        help="Comma-separated list of expected keywords",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="paper",
        choices=["plain", "paper"],
        help="Use paper-style Context/Query prompts or raw strings.",
    )
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--logits-topk", type=int, default=5)
    return parser.parse_args()


def _topk_tokens(model, logits: torch.Tensor, k: int) -> list[dict[str, object]]:
    if k <= 0:
        return []
    values, indices = torch.topk(logits, k)
    results: list[dict[str, object]] = []
    for score, token_id in zip(values[0].tolist(), indices[0].tolist()):
        token_str = model.to_string([int(token_id)])
        results.append({"token_id": int(token_id), "token": token_str, "logit": float(score)})
    return results


def main() -> int:
    args = parse_args()

    results_dir = create_results_dir("nnm_exp8")
    logger = setup_logging("nnm_exp8_virtual_prefix_stored_tl", results_dir)

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    embedder = TransformerLensKVEmbedder(config)
    model = embedder.model
    prepend_bos = embedder._prepend_bos

    if args.prompt_style == "paper":
        memory_text = f"Context: {args.memory}"
        query_text = f"Query: {args.query}\nAnswer:"
    else:
        memory_text = args.memory
        query_text = args.query

    logger.info("Memory: %s", memory_text)
    logger.info("Query: %s", query_text)

    logger.info("Phase 1: Extracting KV cache (no prompt)...")
    stored_kv = extract_kv_cache_tl(
        model,
        memory_text,
        prepend_bos=prepend_bos,
    )
    logger.info(
        "Stored KV cache: %d layers, prefix_len=%d",
        stored_kv.metadata.get("num_layers", len(stored_kv.keys)),
        stored_kv.prefix_len,
    )

    if not args.skip_save:
        cache_path = Path(args.cache_path) if args.cache_path else results_dir / "tl_kv_cache.npz"
        save_kv_cache_tl(stored_kv, cache_path)
        stored_kv = load_kv_cache_tl(cache_path)
        logger.info("Saved and reloaded KV cache from %s", cache_path)

    logger.info("Phase 2: Generating with stored KV cache injection...")
    injected_text = generate_with_stored_kv_tl(
        model,
        stored_kv,
        query_text,
        prepend_bos=prepend_bos,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    logger.info("Injected answer: %s", injected_text.strip())

    logger.info("Phase 3: Baseline generation (no injection)...")
    baseline_text = generate_baseline_tl(
        model,
        query_text,
        prepend_bos=prepend_bos,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    logger.info("Baseline answer: %s", baseline_text.strip())

    # Optional logits diagnostics
    topk = max(0, int(args.logits_topk))
    injection_topk = []
    baseline_topk = []
    if topk > 0:
        from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

        query_tokens = model.to_tokens(query_text, prepend_bos=prepend_bos)
        injected_cache = HookedTransformerKeyValueCache.init_cache(
            model.cfg,
            device=model.W_E.device,
            batch_size=1,
        )
        for layer in range(int(model.cfg.n_layers)):
            k = torch.from_numpy(stored_kv.keys[layer]).to(device=model.W_E.device, dtype=model.W_E.dtype)
            v = torch.from_numpy(stored_kv.values[layer]).to(device=model.W_E.device, dtype=model.W_E.dtype)
            injected_cache.entries[layer].past_keys = k
            injected_cache.entries[layer].past_values = v
        injected_cache.previous_attention_mask = torch.ones(
            (1, stored_kv.prefix_len), dtype=torch.int, device=model.W_E.device
        )

        logits_inj, _ = model.run_with_cache(
            query_tokens,
            return_type="logits",
            past_kv_cache=injected_cache,
            prepend_bos=False,
            remove_batch_dim=False,
        )
        injection_topk = _topk_tokens(model, logits_inj[:, -1], topk)

        logits_base, _ = model.run_with_cache(
            query_tokens,
            return_type="logits",
            past_kv_cache=None,
            prepend_bos=False,
            remove_batch_dim=False,
        )
        baseline_topk = _topk_tokens(model, logits_base[:, -1], topk)

        logger.info("Top-%d next-token logits (injected): %s", topk, injection_topk)
        logger.info("Top-%d next-token logits (baseline): %s", topk, baseline_topk)

    expected_keywords = [kw.strip() for kw in args.expected.split(",") if kw.strip()]
    has_expected = any(kw.lower() in injected_text.lower() for kw in expected_keywords)

    logger.info("SUCCESS: %s", has_expected)

    output = {
        "experiment": "nnm_exp8_virtual_prefix_stored_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "expected_keywords": expected_keywords,
            "prompt_style": args.prompt_style,
        },
        "memory": memory_text,
        "query": query_text,
        "stored_cache": {
            "prefix_len": stored_kv.prefix_len,
            "metadata": stored_kv.metadata,
        },
        "answers": {
            "injected": injected_text,
            "baseline": baseline_text,
        },
        "next_token_topk": {
            "k": topk,
            "injected": injection_topk,
            "baseline": baseline_topk,
        },
        "success": has_expected,
    }
    out_path = results_dir / "results.json"
    save_json(output, out_path)
    logger.info("Saved results to %s", out_path)

    return 0 if has_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
