#!/usr/bin/env python3
"""Search token-level retrieval vectors in Qdrant."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder
from nnm.kvembed.prompts import build_compression_prompt
from nnm.storage import NNMQdrantTokenStore, load_retrieval_transform


def _parse_int_list(raw: str) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return []
    out: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search retrieval vectors in Qdrant.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--retrieval-layer", type=int, required=True)

    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--role", type=str, choices=["context", "query"], default="query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--search-k",
        type=int,
        default=200,
        help="Number of raw token hits to fetch before optional entry grouping.",
    )
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument(
        "--retrieval-transform-file",
        type=str,
        default=None,
        help="Optional .npz retrieval transform (z-score/whitening) to apply to query vector.",
    )
    parser.add_argument(
        "--retrieval-post-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply L2 normalization after optional retrieval transform (default: true).",
    )
    parser.add_argument(
        "--exclude-token-ids",
        type=str,
        default="",
        help="Comma-separated token IDs to exclude, e.g. '25,6,7,8,9,10'.",
    )
    parser.add_argument(
        "--exclude-special-token-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude tokenizer special token IDs automatically (default: true).",
    )
    parser.add_argument(
        "--group-by-entry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aggregate token hits to entry-level ranking (default: true).",
    )
    parser.add_argument(
        "--entry-agg",
        type=str,
        choices=["max", "mean_top3"],
        default="mean_top3",
        help="Aggregation strategy for grouped entry ranking.",
    )
    parser.add_argument(
        "--min-entry-token-hits",
        type=int,
        default=1,
        help="Keep only entries with at least this many token hits after grouping.",
    )

    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-path", type=str, default=None)
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--qdrant-timeout", type=float, default=120.0)
    parser.add_argument("--qdrant-prefer-grpc", action="store_true")
    parser.add_argument(
        "--qdrant-check-compatibility",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--collection", type=str, default="nnm_token_memory")
    parser.add_argument("--filter-model-id", type=str, default=None)
    return parser.parse_args()


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n <= 1e-12:
        return x
    return x / n


def _aggregate_entries(hits: Sequence[object], agg: str) -> List[Dict[str, object]]:
    by_entry: Dict[str, List[object]] = defaultdict(list)
    for hit in hits:
        payload = hit.payload or {}
        entry_id = str(payload.get("entry_id", "") or "")
        if not entry_id:
            continue
        by_entry[entry_id].append(hit)

    rows: List[Dict[str, object]] = []
    for entry_id, group in by_entry.items():
        group_sorted = sorted(group, key=lambda h: float(h.score), reverse=True)
        top_hit = group_sorted[0]
        scores = [float(h.score) for h in group_sorted]
        if agg == "max":
            score = scores[0]
        else:
            score = float(np.mean(scores[: min(3, len(scores))]))
        payload = top_hit.payload or {}
        rows.append(
            {
                "entry_id": entry_id,
                "score": score,
                "n_token_hits": len(group_sorted),
                "top_token_id": payload.get("token_id"),
                "top_token_index": payload.get("token_index"),
                "entity_id": payload.get("entity_id"),
                "source_id": payload.get("source_id"),
                "source_title": payload.get("source_title"),
            }
        )
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows


def main() -> int:
    args = parse_args()

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    embedder = TransformerLensKVEmbedder(config)
    model = embedder.model
    prepend_bos = embedder._prepend_bos

    layer = int(args.retrieval_layer)
    n_layers = int(model.cfg.n_layers)
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"Invalid retrieval layer {layer} for n_layers={n_layers}")

    prompt = build_compression_prompt(
        text=args.query,
        role=args.role,
        template=config.prompt_template,
    )
    token_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
    retr_key = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(
            token_ids,
            return_type=None,
            prepend_bos=False,
            names_filter=lambda name: name == retr_key,
            remove_batch_dim=False,
        )
    retrieval = cache[retr_key][0].detach().to(torch.float32).cpu().numpy()
    # Query pooling: same shape-robust default as in experiments.
    q = (retrieval[-1] + retrieval.mean(axis=0)) / 2.0
    if args.retrieval_transform_file:
        transform = load_retrieval_transform(args.retrieval_transform_file)
        if int(transform.mean.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"Transform dimension mismatch: transform={transform.mean.shape[0]} query={q.shape[0]}"
            )
        q = transform.apply(q, l2_normalize=bool(args.retrieval_post_l2))[0]
    elif args.retrieval_post_l2:
        q = _l2(q)

    store = NNMQdrantTokenStore(
        url=args.qdrant_url,
        path=args.qdrant_path,
        api_key=args.qdrant_api_key,
        prefer_grpc=bool(args.qdrant_prefer_grpc),
        timeout=float(args.qdrant_timeout),
        check_compatibility=bool(args.qdrant_check_compatibility),
    )
    exclude_token_ids_set = set(_parse_int_list(args.exclude_token_ids))
    if args.exclude_special_token_ids:
        tokenizer = getattr(model, "tokenizer", None)
        all_special_ids = getattr(tokenizer, "all_special_ids", None)
        if all_special_ids:
            for tid in all_special_ids:
                exclude_token_ids_set.add(int(tid))
    exclude_token_ids = sorted(exclude_token_ids_set)
    search_k = max(int(args.top_k), int(args.search_k))
    hits = store.search_retrieval(
        collection_name=args.collection,
        query_vector=q.tolist(),
        limit=search_k,
        model_id=args.filter_model_id,
        min_score=args.score_threshold,
        exclude_token_ids=exclude_token_ids,
        with_vectors=False,
        with_payload=True,
    )

    if args.group_by_entry:
        rows = _aggregate_entries(hits, args.entry_agg)
        min_hits = max(1, int(args.min_entry_token_hits))
        rows = [r for r in rows if int(r["n_token_hits"]) >= min_hits]
        rows = rows[: args.top_k]
        print(f"Entry hits: {len(rows)} (from {len(hits)} token hits)")
        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank:02d} score={float(row['score']):.4f} "
                f"entry_id={row['entry_id']} entity_id={row['entity_id']} source_id={row['source_id']} "
                f"n_token_hits={row['n_token_hits']} top_token_id={row['top_token_id']} "
                f"title={row['source_title']}"
            )
    else:
        hits = list(hits)[: args.top_k]
        print(f"Token hits: {len(hits)}")
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            print(
                f"{rank:02d} score={hit.score:.4f} id={hit.id} "
                f"entry_id={payload.get('entry_id')} token_index={payload.get('token_index')} "
                f"token_id={payload.get('token_id')} entity_id={payload.get('entity_id')} "
                f"source_id={payload.get('source_id')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
