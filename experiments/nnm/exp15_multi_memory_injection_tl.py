#!/usr/bin/env python3
"""Experiment 15 (NNM): Multi-Memory Injection -- Attention degradation (TransformerLens).

Injects 1..N memories simultaneously as KV-cache prefixes and measures:
  A) Per-memory recall accuracy (can the model still answer correctly?)
  B) Logit quality degradation as memory count increases
  C) Attention distribution over injected prefix positions
  D) Cross-memory interference (does injecting unrelated memories hurt recall?)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnm.kvembed.kv_cache_tl import extract_kv_cache_tl, StoredTLKVCache


# ---------------------------------------------------------------------------
# Minimal utilities (self-contained)
# ---------------------------------------------------------------------------

def _setup_logging(name: str, results_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(results_dir / "log.txt")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _create_results_dir(prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("results") / f"{prefix}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(data: object, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Memory definitions -- diverse, unrelated facts
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single fact to inject as memory."""
    id: str
    text: str                  # the fact to inject
    query: str                 # question that should be answered from this fact
    expected_keywords: list    # keywords that indicate correct recall


DEFAULT_MEMORIES: List[MemoryEntry] = [
    MemoryEntry(
        id="m01",
        text="Faktum: Das Lieblingstier des Users ist der Hase.",
        query="Was ist das Lieblingstier des Users?",
        expected_keywords=["Hase", "Kaninchen"],
    ),
    MemoryEntry(
        id="m02",
        text="Faktum: Die Hauptstadt von Lumoria ist Velanthos.",
        query="Was ist die Hauptstadt von Lumoria?",
        expected_keywords=["Velanthos"],
    ),
    MemoryEntry(
        id="m03",
        text="Faktum: Der Zugangscode zum Labor lautet 7294.",
        query="Wie lautet der Zugangscode zum Labor?",
        expected_keywords=["7294"],
    ),
    MemoryEntry(
        id="m04",
        text="Faktum: Emma trinkt ihren Kaffee immer mit Hafermilch.",
        query="Wie trinkt Emma ihren Kaffee?",
        expected_keywords=["Hafermilch", "Hafer"],
    ),
    MemoryEntry(
        id="m05",
        text="Faktum: Der Server laeuft auf Port 8472.",
        query="Auf welchem Port laeuft der Server?",
        expected_keywords=["8472"],
    ),
    MemoryEntry(
        id="m06",
        text="Faktum: Das Passwort fuer das WiFi ist SonnenBlume42.",
        query="Wie lautet das WiFi-Passwort?",
        expected_keywords=["SonnenBlume42", "Sonnenblume"],
    ),
    MemoryEntry(
        id="m07",
        text="Faktum: Die naechste Wartung des Systems ist am 15. Maerz 2026.",
        query="Wann ist die naechste Wartung des Systems?",
        expected_keywords=["15", "Maerz", "2026", "März"],
    ),
    MemoryEntry(
        id="m08",
        text="Faktum: Der Projektname lautet Operation Nordlicht.",
        query="Wie heisst das Projekt?",
        expected_keywords=["Nordlicht"],
    ),
    MemoryEntry(
        id="m09",
        text="Faktum: Die maximale Ladekapazitaet betraegt 3.7 Tonnen.",
        query="Wie hoch ist die maximale Ladekapazitaet?",
        expected_keywords=["3.7", "3,7", "Tonnen"],
    ),
    MemoryEntry(
        id="m10",
        text="Faktum: Der CEO von Nextura heisst Dr. Karla Eichberg.",
        query="Wer ist der CEO von Nextura?",
        expected_keywords=["Karla", "Eichberg"],
    ),
]


# ---------------------------------------------------------------------------
# Multi-memory cache building
# ---------------------------------------------------------------------------

def build_multi_memory_cache(
    model,
    stored_kvs: List[StoredTLKVCache],
) -> Tuple[object, int]:
    """
    Combine multiple stored KV caches into a single prefix cache.

    Concatenates K/V tensors along the sequence dimension.
    Returns (cache, total_prefix_len).
    """
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    device = model.W_E.device
    dtype = model.W_E.dtype
    n_layers = int(model.cfg.n_layers)

    cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg,
        device=device,
        batch_size=1,
    )

    total_prefix_len = sum(kv.prefix_len for kv in stored_kvs)

    for layer in range(n_layers):
        # Concatenate all memories along seq dimension
        k_parts = []
        v_parts = []
        for kv in stored_kvs:
            k = torch.from_numpy(kv.keys[layer]).to(device=device, dtype=dtype)
            v = torch.from_numpy(kv.values[layer]).to(device=device, dtype=dtype)
            k_parts.append(k)
            v_parts.append(v)

        cache.entries[layer].past_keys = torch.cat(k_parts, dim=1)
        cache.entries[layer].past_values = torch.cat(v_parts, dim=1)

    cache.previous_attention_mask = torch.ones(
        (1, total_prefix_len), dtype=torch.int, device=device
    )

    return cache, total_prefix_len


# ---------------------------------------------------------------------------
# Generation with multi-memory cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_multi_cache(
    model,
    cache,
    query: str,
    prepend_bos: bool = True,
    max_new_tokens: int = 60,
) -> Tuple[str, torch.Tensor]:
    """
    Generate with a multi-memory cache prefix.
    Returns (generated_text, first_token_logits).
    """
    query_tokens = model.to_tokens(query, prepend_bos=prepend_bos)

    logits, _ = model.run_with_cache(
        query_tokens,
        return_type="logits",
        past_kv_cache=cache,
        prepend_bos=False,
        remove_batch_dim=False,
    )

    first_logits = logits[:, -1].clone()

    generated = []
    eos_id = getattr(model.tokenizer, "eos_token_id", None) if model.tokenizer else None

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        generated.append(next_token)
        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break
        logits, _ = model.run_with_cache(
            next_token,
            return_type="logits",
            past_kv_cache=cache,
            prepend_bos=False,
            remove_batch_dim=False,
        )

    if not generated:
        text = ""
    else:
        text = model.to_string(torch.cat(generated, dim=1)[0])

    return text, first_logits


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_recall_test(
    model,
    memories: List[MemoryEntry],
    stored_kvs: List[StoredTLKVCache],
    target_memory: MemoryEntry,
    target_kv: StoredTLKVCache,
    prepend_bos: bool,
    max_new_tokens: int,
    logger: logging.Logger,
    no_think: bool = False,
) -> dict:
    """
    Inject all given memories and test recall of the target memory.
    """
    cache, total_prefix_len = build_multi_memory_cache(model, stored_kvs)

    query_prefix = "/no_think\n" if no_think else ""
    query_text = f"{query_prefix}Query: {target_memory.query}\nAnswer:"
    answer, first_logits = generate_with_multi_cache(
        model, cache, query_text,
        prepend_bos=prepend_bos,
        max_new_tokens=max_new_tokens,
    )

    # Check for expected keywords
    answer_lower = answer.lower()
    found_keywords = [kw for kw in target_memory.expected_keywords if kw.lower() in answer_lower]
    recall_success = len(found_keywords) > 0

    # Top-k logits for diagnostics
    topk_values, topk_indices = torch.topk(first_logits[0], 5)
    topk_tokens = []
    for score, tid in zip(topk_values.tolist(), topk_indices.tolist()):
        topk_tokens.append({
            "token": model.to_string([int(tid)]),
            "logit": score,
        })

    return {
        "target_memory_id": target_memory.id,
        "target_query": target_memory.query,
        "n_memories_injected": len(memories),
        "total_prefix_tokens": total_prefix_len,
        "answer": answer.strip(),
        "found_keywords": found_keywords,
        "recall_success": recall_success,
        "topk_first_token": topk_tokens,
        "memory_ids_injected": [m.id for m in memories],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp15: Multi-Memory Injection -- Attention degradation"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--max-memories", type=int, default=10,
                        help="Maximum number of memories to inject simultaneously")
    parser.add_argument("--test-counts", type=str, default="1,2,3,5,7,10",
                        help="Comma-separated memory counts to test")
    parser.add_argument("--no-think", action="store_true",
                        help="Prepend /no_think to queries (suppresses Qwen3 thinking mode)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit NF4 quantization (requires bitsandbytes)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = _create_results_dir("nnm_exp15")
    logger = _setup_logging("nnm_exp15_multi_memory_injection_tl", results_dir)

    # Load model
    from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        load_in_4bit=args.load_in_4bit,
    )
    embedder = TransformerLensKVEmbedder(config)
    model = embedder.model
    prepend_bos = embedder._prepend_bos

    # Limit memories
    memories = DEFAULT_MEMORIES[:args.max_memories]
    test_counts = [int(x.strip()) for x in args.test_counts.split(",")]
    test_counts = [c for c in test_counts if c <= len(memories)]

    logger.info("Model: %s", args.model)
    logger.info("Available memories: %d", len(memories))
    logger.info("Test counts: %s", test_counts)

    # -----------------------------------------------------------------------
    # Phase 1: Extract KV caches for all memories
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1: Extracting KV caches for %d memories", len(memories))
    logger.info("=" * 60)

    stored_kvs: List[StoredTLKVCache] = []
    for mem in memories:
        memory_text = f"Context: {mem.text}"
        kv = extract_kv_cache_tl(model, memory_text, prepend_bos=prepend_bos)
        stored_kvs.append(kv)
        logger.info("  %s: prefix_len=%d (%s)", mem.id, kv.prefix_len, mem.text[:50])

    # -----------------------------------------------------------------------
    # Phase 2: Baseline -- each memory alone
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2: Baseline -- single memory injection")
    logger.info("=" * 60)

    baseline_results: List[dict] = []
    for i, (mem, kv) in enumerate(zip(memories, stored_kvs)):
        result = run_recall_test(
            model,
            memories=[mem],
            stored_kvs=[kv],
            target_memory=mem,
            target_kv=kv,
            prepend_bos=prepend_bos,
            max_new_tokens=args.max_new_tokens,
            logger=logger,
            no_think=args.no_think,
        )
        baseline_results.append(result)
        status = "OK" if result["recall_success"] else "FAIL"
        logger.info(
            "  [%s] %s (prefix=%d): %s",
            status, mem.id, result["total_prefix_tokens"],
            result["answer"][:80],
        )

    baseline_recall = sum(1 for r in baseline_results if r["recall_success"])
    logger.info("Baseline recall: %d/%d", baseline_recall, len(baseline_results))

    # -----------------------------------------------------------------------
    # Phase 3: Multi-memory injection -- scaling test
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 3: Multi-memory injection scaling")
    logger.info("=" * 60)

    scaling_results: List[dict] = []

    for n_memories in test_counts:
        logger.info("--- Testing with %d memories ---", n_memories)

        # Use the first N memories
        active_memories = memories[:n_memories]
        active_kvs = stored_kvs[:n_memories]

        # Test recall for EACH injected memory
        count_results = []
        for i, (mem, kv) in enumerate(zip(active_memories, active_kvs)):
            result = run_recall_test(
                model,
                memories=active_memories,
                stored_kvs=active_kvs,
                target_memory=mem,
                target_kv=kv,
                prepend_bos=prepend_bos,
                max_new_tokens=args.max_new_tokens,
                logger=logger,
                no_think=args.no_think,
            )
            count_results.append(result)
            status = "OK" if result["recall_success"] else "FAIL"
            logger.info(
                "  [%s] %s (total_prefix=%d): %s",
                status, mem.id, result["total_prefix_tokens"],
                result["answer"][:80],
            )

        n_recalled = sum(1 for r in count_results if r["recall_success"])
        total_prefix = count_results[0]["total_prefix_tokens"] if count_results else 0

        scaling_summary = {
            "n_memories": n_memories,
            "total_prefix_tokens": total_prefix,
            "n_recalled": n_recalled,
            "n_total": n_memories,
            "recall_rate": n_recalled / n_memories if n_memories > 0 else 0,
            "per_memory_results": count_results,
        }
        scaling_results.append(scaling_summary)
        logger.info(
            "  => %d memories: %d/%d recalled (%.0f%%), prefix=%d tokens",
            n_memories, n_recalled, n_memories,
            100 * scaling_summary["recall_rate"], total_prefix,
        )

    # -----------------------------------------------------------------------
    # Phase 4: Cross-interference test
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 4: Cross-interference (target at different positions)")
    logger.info("=" * 60)

    interference_results: List[dict] = []

    if len(memories) >= 5:
        target = memories[0]
        target_kv = stored_kvs[0]

        # Test target memory at different positions in the prefix
        positions = ["first", "middle", "last"]
        n_inject = min(5, len(memories))

        for pos in positions:
            other_memories = memories[1:n_inject]
            other_kvs = stored_kvs[1:n_inject]

            if pos == "first":
                ordered_memories = [target] + other_memories
                ordered_kvs = [target_kv] + other_kvs
            elif pos == "middle":
                mid = len(other_memories) // 2
                ordered_memories = other_memories[:mid] + [target] + other_memories[mid:]
                ordered_kvs = other_kvs[:mid] + [target_kv] + other_kvs[mid:]
            else:  # last
                ordered_memories = other_memories + [target]
                ordered_kvs = other_kvs + [target_kv]

            result = run_recall_test(
                model,
                memories=ordered_memories,
                stored_kvs=ordered_kvs,
                target_memory=target,
                target_kv=target_kv,
                prepend_bos=prepend_bos,
                max_new_tokens=args.max_new_tokens,
                logger=logger,
                no_think=args.no_think,
            )
            result["target_position"] = pos
            interference_results.append(result)

            status = "OK" if result["recall_success"] else "FAIL"
            logger.info(
                "  [%s] target at %s (n=%d, prefix=%d): %s",
                status, pos, len(ordered_memories),
                result["total_prefix_tokens"],
                result["answer"][:80],
            )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # Print degradation curve
    logger.info("Recall degradation curve:")
    logger.info("  n_memories | recall_rate | prefix_tokens")
    logger.info("  -----------|-------------|---------------")
    for sr in scaling_results:
        logger.info(
            "  %10d | %10.0f%% | %13d",
            sr["n_memories"], 100 * sr["recall_rate"], sr["total_prefix_tokens"],
        )

    # Determine at what point recall starts degrading
    degradation_point = None
    for sr in scaling_results:
        if sr["recall_rate"] < 1.0:
            degradation_point = sr["n_memories"]
            break

    if degradation_point:
        logger.info("First degradation at %d memories", degradation_point)
    else:
        logger.info("No degradation observed up to %d memories", test_counts[-1] if test_counts else 0)

    # Position sensitivity
    if interference_results:
        logger.info("Position sensitivity (target=%s):", target.id)
        for ir in interference_results:
            status = "OK" if ir["recall_success"] else "FAIL"
            logger.info("  [%s] position=%s", status, ir["target_position"])

    overall_success = baseline_recall == len(baseline_results)  # at least baseline should work

    output = {
        "experiment": "nnm_exp15_multi_memory_injection_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "max_memories": args.max_memories,
            "test_counts": test_counts,
        },
        "n_memories_available": len(memories),
        "phase2_baseline": {
            "results": baseline_results,
            "recall": f"{baseline_recall}/{len(baseline_results)}",
        },
        "phase3_scaling": scaling_results,
        "phase4_interference": interference_results,
        "degradation_point": degradation_point,
        "overall_success": overall_success,
    }

    out_path = results_dir / "results.json"
    _save_json(output, out_path)
    logger.info("Saved results to %s", out_path)

    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
