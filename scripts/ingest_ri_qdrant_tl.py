#!/usr/bin/env python3
"""Ingest token-level retrieval/injection vectors into Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder
from nnm.kvembed.prompts import build_compression_prompt
from nnm.storage import (
    NNMQdrantTokenStore,
    RetrievalTransform,
    TokenVectorRecord,
    fit_retrieval_transform,
    load_retrieval_transform,
    save_retrieval_transform,
)


@dataclass(frozen=True)
class InputItem:
    text: str
    entity_id: Optional[str] = None
    source_id: Optional[str] = None
    source_title: Optional[str] = None


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _build_items_from_text_inputs(args: argparse.Namespace) -> List[InputItem]:
    texts: List[str] = list(args.text)
    if args.text_file:
        texts.extend(_read_lines(Path(args.text_file)))
    return [InputItem(text=t) for t in texts if t.strip()]


def _build_items_from_jsonl(args: argparse.Namespace) -> List[InputItem]:
    if not args.jsonl_file:
        return []
    path = Path(args.jsonl_file)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    items: List[InputItem] = []
    id_field = args.jsonl_id_field
    title_field = args.jsonl_title_field
    text_field = args.jsonl_text_field
    include_title = bool(args.jsonl_include_title)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get(id_field, "") or "").strip()
            title = str(row.get(title_field, "") or "").strip()
            text = str(row.get(text_field, "") or "").strip()
            full_text = f"{title} {text}".strip() if include_title and title else text
            if not full_text:
                continue
            entity_id = source_id if (source_id and args.entity_id_from_jsonl) else None
            items.append(
                InputItem(
                    text=full_text,
                    entity_id=entity_id,
                    source_id=source_id or None,
                    source_title=title or None,
                )
            )
    return items


def _build_items(args: argparse.Namespace) -> List[InputItem]:
    items = _build_items_from_text_inputs(args)
    items.extend(_build_items_from_jsonl(args))
    if not items:
        raise ValueError("No input items provided. Use --text/--text-file/--jsonl-file.")
    if args.start_item > 0:
        items = items[args.start_item :]
    if args.max_items > 0:
        items = items[: args.max_items]
    return items


def _entry_id_for_item(
    *,
    model: str,
    retrieval_layer: int,
    injection_layer: int,
    item: InputItem,
) -> str:
    if item.source_id:
        return str(
            uuid5(
                NAMESPACE_URL,
                f"nnm:{model}:r{retrieval_layer}:i{injection_layer}:source:{item.source_id}",
            )
        )
    return str(uuid4())


def _sanitize_component(raw: str) -> str:
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "model"


def _resolve_transform_path(args: argparse.Namespace, retrieval_layer: int) -> Optional[Path]:
    if not (args.retrieval_zscore or args.retrieval_whitening):
        return None
    if args.retrieval_transform_file:
        return Path(args.retrieval_transform_file)
    safe_model = _sanitize_component(args.model)
    name = (
        f"{safe_model}_r{int(retrieval_layer)}_"
        f"z{1 if args.retrieval_zscore else 0}_w{1 if args.retrieval_whitening else 0}.npz"
    )
    return REPO_ROOT / "data" / "retrieval_transforms" / name


def _row_l2(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _fit_transform_from_items(
    *,
    items: List[InputItem],
    model: object,
    prepend_bos: bool,
    role: str,
    template: str,
    retrieval_key: str,
    fit_items: int,
    fit_max_tokens: int,
    use_zscore: bool,
    use_whitening: bool,
    eps: float,
    progress_every: int,
    model_name: str,
    retrieval_layer: int,
) -> RetrievalTransform:
    arrays: List[np.ndarray] = []
    n_items = 0
    n_tokens = 0
    limit_items = max(1, int(fit_items))
    limit_tokens = max(2, int(fit_max_tokens))
    progress_every = max(1, int(progress_every))

    for idx, item in enumerate(items, start=1):
        if n_items >= limit_items or n_tokens >= limit_tokens:
            break
        prompt = build_compression_prompt(text=item.text, role=role, template=template)
        token_ids_t = model.to_tokens(prompt, prepend_bos=prepend_bos)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                token_ids_t,
                return_type=None,
                prepend_bos=False,
                names_filter=lambda name: name == retrieval_key,
                remove_batch_dim=False,
            )
        retrieval = cache[retrieval_key][0].detach().to(torch.float32).cpu().numpy()
        del cache
        if retrieval.size == 0:
            continue
        remaining = limit_tokens - n_tokens
        if remaining <= 0:
            break
        take = retrieval[:remaining].astype(np.float32, copy=False)
        arrays.append(take)
        n_tokens += int(take.shape[0])
        n_items += 1
        if n_items == 1 or n_items % progress_every == 0:
            print(
                f"Transform fit progress: items={n_items}/{limit_items} tokens={n_tokens}/{limit_tokens}",
                flush=True,
            )

    if n_tokens < 2:
        raise RuntimeError("Not enough samples to fit retrieval transform (need at least 2 token vectors).")

    x = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    metadata = {
        "model": model_name,
        "retrieval_layer": int(retrieval_layer),
        "n_items": int(n_items),
        "n_tokens": int(x.shape[0]),
    }
    print(
        f"Fitting retrieval transform: zscore={use_zscore} whitening={use_whitening} "
        f"samples={x.shape[0]} dim={x.shape[1]}",
        flush=True,
    )
    return fit_retrieval_transform(
        x,
        use_zscore=bool(use_zscore),
        use_whitening=bool(use_whitening),
        eps=float(eps),
        metadata=metadata,
    )


def _log_progress(done: int, total: int, start_ts: float) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    elapsed = max(time.perf_counter() - start_ts, 1e-9)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta = remaining / max(rate, 1e-9)
    pct = 100.0 * done / total
    return f"{done}/{total} ({pct:.1f}%), {rate:.2f} items/s, ETA {eta:.1f}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract token-level R/I vectors and ingest into Qdrant."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--retrieval-layer", type=int, required=True)
    parser.add_argument("--injection-layer", type=int, required=True)
    parser.add_argument("--role", type=str, choices=["context", "query"], default="context")
    parser.add_argument(
        "--retrieval-zscore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply z-score normalization on retrieval vectors (default: true).",
    )
    parser.add_argument(
        "--retrieval-whitening",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply whitening on retrieval vectors (default: true).",
    )
    parser.add_argument(
        "--retrieval-post-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply L2 normalization after retrieval transform (default: true).",
    )
    parser.add_argument(
        "--retrieval-transform-file",
        type=str,
        default=None,
        help="Path to .npz retrieval transform. If missing, it can be fitted and saved.",
    )
    parser.add_argument(
        "--fit-transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit retrieval transform if file does not exist (default: true).",
    )
    parser.add_argument("--fit-transform-items", type=int, default=64)
    parser.add_argument("--fit-transform-max-tokens", type=int, default=30000)
    parser.add_argument("--fit-transform-eps", type=float, default=1e-5)

    parser.add_argument("--text", action="append", default=[], help="Input text (repeatable)")
    parser.add_argument("--text-file", type=str, default=None, help="One input text per line")

    parser.add_argument("--jsonl-file", type=str, default=None, help="JSONL input file")
    parser.add_argument("--jsonl-id-field", type=str, default="_id")
    parser.add_argument("--jsonl-title-field", type=str, default="title")
    parser.add_argument("--jsonl-text-field", type=str, default="text")
    parser.add_argument(
        "--jsonl-include-title",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include title + text when reading JSONL (default: true).",
    )
    parser.add_argument(
        "--entity-id-from-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use jsonl id field as entity_id (default: true).",
    )
    parser.add_argument("--start-item", type=int, default=0, help="Skip first N items before ingest.")
    parser.add_argument("--max-items", type=int, default=0, help="0 means unlimited")

    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-path", type=str, default=None, help="Local embedded Qdrant path")
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument(
        "--qdrant-timeout",
        type=float,
        default=300.0,
        help="Client timeout in seconds for Qdrant requests.",
    )
    parser.add_argument("--qdrant-prefer-grpc", action="store_true")
    parser.add_argument(
        "--qdrant-check-compatibility",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable client-server version compatibility warning.",
    )
    parser.add_argument("--collection", type=str, default="nnm_token_memory")
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument("--disable-int8-scalar-quantization", action="store_true")
    parser.add_argument("--on-disk-vectors", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--upsert-retries", type=int, default=6)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument(
        "--upsert-wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for server-side upsert confirmation (default: true).",
    )
    parser.add_argument(
        "--skip-existing-entry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip item if entry_id already exists in collection (default: true).",
    )

    parser.add_argument(
        "--entity-prefix",
        type=str,
        default="entity",
        help="Prefix used to generate entity IDs if none is supplied by input.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = _build_items(args)

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    embedder = TransformerLensKVEmbedder(config)
    model = embedder.model
    prepend_bos = embedder._prepend_bos

    n_layers = int(model.cfg.n_layers)
    retrieval_layer = int(args.retrieval_layer)
    injection_layer = int(args.injection_layer)
    if retrieval_layer < 0 or retrieval_layer >= n_layers:
        raise ValueError(f"Invalid retrieval layer {retrieval_layer} for n_layers={n_layers}")
    if injection_layer < 0 or injection_layer >= n_layers:
        raise ValueError(f"Invalid injection layer {injection_layer} for n_layers={n_layers}")

    d_model = int(model.cfg.d_model)
    retr_key = f"blocks.{retrieval_layer}.hook_resid_post"
    inj_key = f"blocks.{injection_layer}.hook_resid_pre"
    names = {retr_key, inj_key}

    transform_path = _resolve_transform_path(args, retrieval_layer)
    retrieval_transform: Optional[RetrievalTransform] = None
    if args.retrieval_zscore or args.retrieval_whitening:
        if transform_path is not None and transform_path.exists():
            retrieval_transform = load_retrieval_transform(transform_path)
            print(f"Loaded retrieval transform: {transform_path}", flush=True)
        else:
            if not args.fit_transform:
                raise FileNotFoundError(
                    "Retrieval transform file does not exist and --no-fit-transform is set."
                )
            retrieval_transform = _fit_transform_from_items(
                items=items,
                model=model,
                prepend_bos=prepend_bos,
                role=args.role,
                template=config.prompt_template,
                retrieval_key=retr_key,
                fit_items=args.fit_transform_items,
                fit_max_tokens=args.fit_transform_max_tokens,
                use_zscore=bool(args.retrieval_zscore),
                use_whitening=bool(args.retrieval_whitening),
                eps=float(args.fit_transform_eps),
                progress_every=args.progress_every,
                model_name=args.model,
                retrieval_layer=retrieval_layer,
            )
            if transform_path is not None:
                save_retrieval_transform(retrieval_transform, transform_path)
                print(f"Saved retrieval transform: {transform_path}", flush=True)
        if retrieval_transform.mean.shape[0] != d_model:
            raise ValueError(
                f"Transform dimension mismatch: transform={retrieval_transform.mean.shape[0]} d_model={d_model}"
            )
    elif args.retrieval_post_l2:
        print("Retrieval transform disabled. Using L2 normalization only for retrieval vectors.", flush=True)

    store = NNMQdrantTokenStore(
        url=args.qdrant_url,
        path=args.qdrant_path,
        api_key=args.qdrant_api_key,
        prefer_grpc=bool(args.qdrant_prefer_grpc),
        timeout=float(args.qdrant_timeout),
        check_compatibility=bool(args.qdrant_check_compatibility),
    )
    store.ensure_collection(
        collection_name=args.collection,
        vector_size=d_model,
        recreate=bool(args.recreate_collection),
        on_disk=bool(args.on_disk_vectors),
        use_scalar_int8=not bool(args.disable_int8_scalar_quantization),
    )

    total_tokens = 0
    skipped_items = 0
    t0 = time.perf_counter()
    progress_every = max(1, int(args.progress_every))
    total_items = len(items)

    for item_idx, item in enumerate(items, start=1):
        prompt = build_compression_prompt(
            text=item.text,
            role=args.role,
            template=config.prompt_template,
        )
        token_ids_t = model.to_tokens(prompt, prepend_bos=prepend_bos)
        token_ids = token_ids_t.squeeze(0).detach().cpu().numpy().astype("int64")

        with torch.no_grad():
            _, cache = model.run_with_cache(
                token_ids_t,
                return_type=None,
                prepend_bos=False,
                names_filter=lambda name: name in names,
                remove_batch_dim=False,
            )

        retrieval = cache[retr_key][0].detach().to(torch.float32).cpu().numpy()
        injection = cache[inj_key][0].detach().to(torch.float32).cpu().numpy()
        del cache

        if retrieval.shape != injection.shape:
            raise RuntimeError(
                f"Shape mismatch: retrieval={retrieval.shape}, injection={injection.shape}"
            )
        if retrieval.shape[0] != token_ids.shape[0]:
            raise RuntimeError(
                f"Token mismatch: vectors={retrieval.shape[0]} token_ids={token_ids.shape[0]}"
            )
        if retrieval_transform is not None:
            retrieval = retrieval_transform.apply(
                retrieval,
                l2_normalize=bool(args.retrieval_post_l2),
            )
        elif args.retrieval_post_l2:
            retrieval = _row_l2(retrieval.astype(np.float32, copy=False))

        timestamp_created = int(time.time())
        entry_id = _entry_id_for_item(
            model=args.model,
            retrieval_layer=retrieval_layer,
            injection_layer=injection_layer,
            item=item,
        )
        entity_id = item.entity_id or f"{args.entity_prefix}_{item_idx - 1:06d}"

        if args.skip_existing_entry and store.entry_exists(
            collection_name=args.collection,
            entry_id=entry_id,
        ):
            skipped_items += 1
            if item_idx == 1 or item_idx % progress_every == 0 or item_idx == total_items:
                progress = _log_progress(item_idx, total_items, t0)
                print(
                    f"Ingest progress: {progress} | skip existing entry_id={entry_id} "
                    f"entity_id={entity_id} skipped={skipped_items}",
                    flush=True,
                )
            continue

        entry_uuid = UUID(entry_id)
        records = [
            TokenVectorRecord(
                point_id=str(uuid5(entry_uuid, f"tok:{tok_idx}")),
                entry_id=entry_id,
                entity_id=entity_id,
                token_index=int(tok_idx),
                token_id=int(token_ids[tok_idx]),
                retrieval_vector=retrieval[tok_idx].tolist(),
                injection_vector=injection[tok_idx].tolist(),
                model_id=args.model,
                retrieval_layer=retrieval_layer,
                injection_layer=injection_layer,
                timestamp_created=timestamp_created,
                source_id=item.source_id,
                source_title=item.source_title,
            )
            for tok_idx in range(retrieval.shape[0])
        ]

        n_upserted = store.upsert_tokens(
            collection_name=args.collection,
            records=records,
            wait=bool(args.upsert_wait),
            batch_size=args.batch_size,
            max_retries=args.upsert_retries,
            retry_base_delay=args.retry_base_delay,
        )
        total_tokens += n_upserted

        if item_idx == 1 or item_idx % progress_every == 0 or item_idx == total_items:
            progress = _log_progress(item_idx, total_items, t0)
            print(
                f"Ingest progress: {progress} | entry_id={entry_id} entity_id={entity_id} "
                f"tokens={n_upserted} skipped={skipped_items}",
                flush=True,
            )

    elapsed = max(time.perf_counter() - t0, 1e-9)
    rate = total_tokens / elapsed
    print(
        f"Done. items={total_items} tokens={total_tokens} skipped={skipped_items} "
        f"elapsed={elapsed:.2f}s rate={rate:.2f} tok/s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
