#!/usr/bin/env python3
"""Compare baseline vs transformed retrieval collections on BEIR with Qdrant."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.io_utils import create_results_dir, save_json, setup_logging
from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder
from nnm.kvembed.prompts import build_compression_prompt
from nnm.storage import NNMQdrantTokenStore, RetrievalTransform, load_retrieval_transform


@dataclass(frozen=True)
class EvalQuery:
    qid: str
    text: str


@dataclass(frozen=True)
class EvalConfig:
    name: str
    collection: str
    transform: Optional[RetrievalTransform]
    post_l2: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs zscore+whitening retrieval on BEIR queries via Qdrant."
    )
    parser.add_argument("--dataset", type=str, default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--retrieval-layer", type=int, required=True)
    parser.add_argument("--query-role", type=str, choices=["context", "query"], default="query")

    parser.add_argument("--baseline-collection", type=str, required=True)
    parser.add_argument("--transformed-collection", type=str, required=True)
    parser.add_argument(
        "--transformed-transform-file",
        type=str,
        required=True,
        help=".npz retrieval transform used for transformed collection (z-score/whitening).",
    )
    parser.add_argument(
        "--baseline-post-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply L2 normalization to baseline query vectors (default: true).",
    )
    parser.add_argument(
        "--transformed-post-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply L2 normalization after transformed query normalization (default: true).",
    )

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--search-k", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)

    parser.add_argument(
        "--exclude-token-ids",
        type=str,
        default="25,6,7,8,9,10",
        help="Comma-separated token ids to exclude from retrieval.",
    )
    parser.add_argument(
        "--exclude-special-token-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude tokenizer special token ids automatically (default: true).",
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
        default="mean_top3",
        choices=["max", "mean_top3"],
        help="Entry aggregation strategy if grouping is enabled.",
    )
    parser.add_argument(
        "--sweep-entry-agg",
        type=str,
        default="",
        help="Optional comma list for entry aggregation sweep, e.g. 'mean_top3,max'.",
    )
    parser.add_argument(
        "--min-entry-token-hits",
        type=int,
        default=3,
        help="Minimum token hits for grouped entries (default: 3).",
    )
    parser.add_argument(
        "--sweep-min-entry-token-hits",
        type=str,
        default="",
        help="Optional comma list for min-entry-token-hits sweep, e.g. '1,2,3'.",
    )

    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-path", type=str, default=None)
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--qdrant-timeout", type=float, default=300.0)
    parser.add_argument("--qdrant-prefer-grpc", action="store_true")
    parser.add_argument(
        "--qdrant-check-compatibility",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--filter-model-id", type=str, default=None)

    parser.add_argument(
        "--query-ids-file",
        type=str,
        default=None,
        help="Optional file with fixed query ids (one per line).",
    )
    parser.add_argument(
        "--save-query-ids-file",
        type=str,
        default=None,
        help="Optional output file to save sampled query ids.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional explicit results dir (default: data/results/nnm_eval_compare_*).",
    )

    return parser.parse_args()


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


def _parse_str_list(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    out: List[str] = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _l2(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n <= eps:
        return x
    return (x / n).astype(np.float32, copy=False)


def _dcg_at_k(rels: Sequence[float], k: int) -> float:
    out = 0.0
    for i, rel in enumerate(rels[:k]):
        if rel <= 0.0:
            continue
        out += (2.0 ** float(rel) - 1.0) / math.log2(i + 2.0)
    return out


def _aggregate_entries(hits: Sequence[object], agg: str) -> List[Dict[str, object]]:
    by_entry: Dict[str, List[object]] = {}
    for hit in hits:
        payload = hit.payload or {}
        entry_id = str(payload.get("entry_id", "") or "")
        if not entry_id:
            continue
        by_entry.setdefault(entry_id, []).append(hit)

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
                "source_id": payload.get("source_id"),
                "entity_id": payload.get("entity_id"),
                "source_title": payload.get("source_title"),
            }
        )
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows


def _load_beir_dataset(dataset: str, split: str) -> tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    root = REPO_ROOT / "data" / "beir_datasets" / dataset
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}. Please place BEIR files under data/beir_datasets/<dataset>/"
        )

    queries: Dict[str, str] = {}
    with (root / "queries.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if qid and text:
                queries[qid] = text

    qrels_path = root / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"qrels file not found: {qrels_path}")

    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            rel = int(float(row["score"]))
            if rel <= 0:
                continue
            qrels.setdefault(qid, {})[did] = rel

    return queries, qrels


def _select_queries(
    *,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    max_queries: int,
    seed: int,
    query_ids_file: Optional[Path],
) -> List[EvalQuery]:
    if query_ids_file is not None:
        if not query_ids_file.exists():
            raise FileNotFoundError(f"query ids file not found: {query_ids_file}")
        out: List[EvalQuery] = []
        with query_ids_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                qid = line.strip()
                if not qid:
                    continue
                if qid not in queries:
                    continue
                rels = qrels.get(qid, {})
                if not any(int(v) > 0 for v in rels.values()):
                    continue
                out.append(EvalQuery(qid=qid, text=queries[qid]))
        if not out:
            raise RuntimeError("No valid query ids found in --query-ids-file.")
        if max_queries > 0:
            return out[: max_queries]
        return out

    rng = random.Random(seed)
    valid_qids = [
        qid
        for qid in queries.keys()
        if qid in qrels and any(int(v) > 0 for v in qrels[qid].values())
    ]
    rng.shuffle(valid_qids)
    if max_queries > 0:
        valid_qids = valid_qids[: max_queries]
    return [EvalQuery(qid=qid, text=queries[qid]) for qid in valid_qids]


def _save_query_ids(query_ids: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for qid in query_ids:
            handle.write(f"{qid}\n")


def _extract_query_vector(
    *,
    model: object,
    prepend_bos: bool,
    retrieval_layer: int,
    query_text: str,
    role: str,
    template: str,
) -> np.ndarray:
    prompt = build_compression_prompt(text=query_text, role=role, template=template)
    token_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
    retr_key = f"blocks.{int(retrieval_layer)}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(
            token_ids,
            return_type=None,
            prepend_bos=False,
            names_filter=lambda name: name == retr_key,
            remove_batch_dim=False,
        )
    retrieval = cache[retr_key][0].detach().to(torch.float32).cpu().numpy()
    del cache
    if retrieval.shape[0] == 0:
        raise RuntimeError("Empty retrieval hidden-state sequence for query.")
    pooled = (retrieval[-1] + retrieval.mean(axis=0)) / 2.0
    return pooled.astype(np.float32, copy=False)


def _query_for_config(raw_query: np.ndarray, config: EvalConfig) -> np.ndarray:
    if config.transform is not None:
        q = config.transform.apply(raw_query, l2_normalize=bool(config.post_l2))[0]
        return q.astype(np.float32, copy=False)
    if config.post_l2:
        return _l2(raw_query.astype(np.float32, copy=False))
    return raw_query.astype(np.float32, copy=False)


def _rank_doc_ids(
    *,
    hits: Sequence[object],
    top_k: int,
    group_by_entry: bool,
    entry_agg: str,
    min_entry_token_hits: int,
) -> Tuple[List[str], int, float]:
    doc_ids: List[str] = []
    seen: set[str] = set()

    if group_by_entry:
        rows = _aggregate_entries(hits, entry_agg)
        rows = [r for r in rows if int(r["n_token_hits"]) >= int(min_entry_token_hits)]
        rows = rows[: max(1, int(top_k))]
        for row in rows:
            did = str(row.get("source_id") or row.get("entity_id") or row.get("entry_id") or "").strip()
            if not did or did in seen:
                continue
            seen.add(did)
            doc_ids.append(did)
        if rows:
            return doc_ids, int(rows[0]["n_token_hits"]), float(rows[0]["score"])
        return doc_ids, 0, 0.0

    for hit in list(hits)[: max(1, int(top_k))]:
        payload = hit.payload or {}
        did = str(payload.get("source_id") or payload.get("entity_id") or payload.get("entry_id") or "").strip()
        if not did or did in seen:
            continue
        seen.add(did)
        doc_ids.append(did)
    top_score = float(hits[0].score) if hits else 0.0
    return doc_ids, 1 if doc_ids else 0, top_score


def _evaluate_ranking(
    *,
    ranked_doc_ids: Sequence[str],
    rel_docs: Mapping[str, int],
    top_k: int,
) -> Dict[str, float]:
    rel_binary = {did for did, s in rel_docs.items() if int(s) > 0}
    if not rel_binary:
        return {
            "ndcg": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "precision": 0.0,
            "hit": 0.0,
        }

    ranked = list(ranked_doc_ids)[: max(1, int(top_k))]

    gains = [float(rel_docs.get(did, 0)) for did in ranked]
    dcg = _dcg_at_k(gains, top_k)
    ideal = sorted((float(v) for v in rel_docs.values() if int(v) > 0), reverse=True)
    idcg = _dcg_at_k(ideal, top_k)
    ndcg = float(dcg / idcg) if idcg > 0 else 0.0

    retrieved_rel = sum(1 for did in ranked if did in rel_binary)
    recall = float(retrieved_rel / max(1, len(rel_binary)))
    precision = float(retrieved_rel / max(1, len(ranked)))
    hit = 1.0 if retrieved_rel > 0 else 0.0

    rr = 0.0
    for rank, did in enumerate(ranked, start=1):
        if did in rel_binary:
            rr = 1.0 / rank
            break

    return {
        "ndcg": ndcg,
        "recall": recall,
        "mrr": rr,
        "precision": precision,
        "hit": hit,
    }


def _summarize(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _compute_delta(
    *,
    baseline_metrics: Mapping[str, object],
    transformed_metrics: Mapping[str, object],
    top_k: int,
) -> Dict[str, float]:
    return {
        f"ndcg@{top_k}": float(
            float(transformed_metrics[f"ndcg@{top_k}"]) - float(baseline_metrics[f"ndcg@{top_k}"])
        ),
        f"recall@{top_k}": float(
            float(transformed_metrics[f"recall@{top_k}"]) - float(baseline_metrics[f"recall@{top_k}"])
        ),
        f"mrr@{top_k}": float(
            float(transformed_metrics[f"mrr@{top_k}"]) - float(baseline_metrics[f"mrr@{top_k}"])
        ),
        f"precision@{top_k}": float(
            float(transformed_metrics[f"precision@{top_k}"]) - float(baseline_metrics[f"precision@{top_k}"])
        ),
        f"hit_rate@{top_k}": float(
            float(transformed_metrics[f"hit_rate@{top_k}"]) - float(baseline_metrics[f"hit_rate@{top_k}"])
        ),
        "mean_top_score": float(
            float(transformed_metrics["mean_top_score"]) - float(baseline_metrics["mean_top_score"])
        ),
        "mean_top_token_hits": float(
            float(transformed_metrics["mean_top_token_hits"]) - float(baseline_metrics["mean_top_token_hits"])
        ),
    }


def _preflight_collections(
    *,
    store: NNMQdrantTokenStore,
    baseline_collection: str,
    transformed_collection: str,
) -> None:
    missing: List[str] = []
    if not store.collection_exists(baseline_collection):
        missing.append(baseline_collection)
    if not store.collection_exists(transformed_collection):
        missing.append(transformed_collection)
    if not missing:
        return

    available = store.list_collections()
    available_txt = ", ".join(available) if available else "<none>"
    missing_txt = ", ".join(missing)
    raise RuntimeError(
        "Missing required Qdrant collection(s): "
        f"{missing_txt}. Available collections: {available_txt}. "
        "Please ingest both baseline and transformed collections first."
    )


def _eval_config(
    *,
    store: NNMQdrantTokenStore,
    config: EvalConfig,
    raw_queries: Sequence[np.ndarray],
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    top_k: int,
    search_k: int,
    filter_model_id: Optional[str],
    exclude_token_ids: Sequence[int],
    group_by_entry: bool,
    entry_agg: str,
    min_entry_token_hits: int,
    logger: object,
    progress_every: int,
) -> Dict[str, object]:
    ndcgs: List[float] = []
    recalls: List[float] = []
    mrrs: List[float] = []
    precisions: List[float] = []
    hits: List[float] = []
    top_scores: List[float] = []
    top_token_counts: List[float] = []

    total = len(query_ids)
    t0 = time.perf_counter()

    for idx, (qid, raw_q) in enumerate(zip(query_ids, raw_queries), start=1):
        q = _query_for_config(raw_q, config)
        hits_raw = store.search_retrieval(
            collection_name=config.collection,
            query_vector=q.tolist(),
            limit=max(int(search_k), int(top_k)),
            model_id=filter_model_id,
            min_score=None,
            exclude_token_ids=exclude_token_ids,
            with_vectors=False,
            with_payload=True,
        )
        ranked_doc_ids, top_tok_hits, top_score = _rank_doc_ids(
            hits=hits_raw,
            top_k=top_k,
            group_by_entry=group_by_entry,
            entry_agg=entry_agg,
            min_entry_token_hits=min_entry_token_hits,
        )
        rels = qrels.get(qid, {})
        m = _evaluate_ranking(ranked_doc_ids=ranked_doc_ids, rel_docs=rels, top_k=top_k)

        ndcgs.append(float(m["ndcg"]))
        recalls.append(float(m["recall"]))
        mrrs.append(float(m["mrr"]))
        precisions.append(float(m["precision"]))
        hits.append(float(m["hit"]))
        top_scores.append(float(top_score))
        top_token_counts.append(float(top_tok_hits))

        if idx == 1 or idx % max(1, int(progress_every)) == 0 or idx == total:
            elapsed = max(time.perf_counter() - t0, 1e-9)
            rate = idx / elapsed
            rem = (total - idx) / max(rate, 1e-9)
            logger.info(
                "[%s] progress: %d/%d (%.1f%%), %.2f q/s, ETA %.1fs",
                config.name,
                idx,
                total,
                100.0 * idx / max(1, total),
                rate,
                rem,
            )

    return {
        "collection": config.collection,
        "queries_evaluated": int(total),
        f"ndcg@{top_k}": _summarize(ndcgs),
        f"recall@{top_k}": _summarize(recalls),
        f"mrr@{top_k}": _summarize(mrrs),
        f"precision@{top_k}": _summarize(precisions),
        f"hit_rate@{top_k}": _summarize(hits),
        "mean_top_score": _summarize(top_scores),
        "mean_top_token_hits": _summarize(top_token_counts),
    }


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else create_results_dir("nnm_eval_compare")
    logger = setup_logging("nnm_eval_qdrant_beir_compare_tl", results_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    store = NNMQdrantTokenStore(
        url=args.qdrant_url,
        path=args.qdrant_path,
        api_key=args.qdrant_api_key,
        prefer_grpc=bool(args.qdrant_prefer_grpc),
        timeout=float(args.qdrant_timeout),
        check_compatibility=bool(args.qdrant_check_compatibility),
    )
    _preflight_collections(
        store=store,
        baseline_collection=args.baseline_collection,
        transformed_collection=args.transformed_collection,
    )

    queries, qrels = _load_beir_dataset(args.dataset, args.split)
    query_ids_path = Path(args.query_ids_file) if args.query_ids_file else None
    eval_queries = _select_queries(
        queries=queries,
        qrels=qrels,
        max_queries=int(args.max_queries),
        seed=int(args.seed),
        query_ids_file=query_ids_path,
    )
    if not eval_queries:
        raise RuntimeError("No evaluation queries selected.")

    if args.save_query_ids_file:
        _save_query_ids([q.qid for q in eval_queries], Path(args.save_query_ids_file))

    logger.info(
        "Eval set prepared: dataset=%s split=%s queries=%d",
        args.dataset,
        args.split,
        len(eval_queries),
    )

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=bool(args.local_files_only),
    )
    embedder = TransformerLensKVEmbedder(config)
    model = embedder.model
    prepend_bos = embedder._prepend_bos

    layer = int(args.retrieval_layer)
    n_layers = int(model.cfg.n_layers)
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"Invalid retrieval layer {layer} for n_layers={n_layers}")

    transform = load_retrieval_transform(args.transformed_transform_file)
    if int(transform.mean.shape[0]) != int(model.cfg.d_model):
        raise ValueError(
            "Transform dimension mismatch: "
            f"transform={transform.mean.shape[0]} model_d_model={model.cfg.d_model}"
        )

    raw_queries: List[np.ndarray] = []
    t0 = time.perf_counter()
    total = len(eval_queries)
    for idx, q in enumerate(eval_queries, start=1):
        raw_q = _extract_query_vector(
            model=model,
            prepend_bos=prepend_bos,
            retrieval_layer=layer,
            query_text=q.text,
            role=args.query_role,
            template=config.prompt_template,
        )
        raw_queries.append(raw_q)
        if idx == 1 or idx % max(1, int(args.progress_every)) == 0 or idx == total:
            elapsed = max(time.perf_counter() - t0, 1e-9)
            rate = idx / elapsed
            rem = (total - idx) / max(rate, 1e-9)
            logger.info(
                "[embed] progress: %d/%d (%.1f%%), %.2f q/s, ETA %.1fs",
                idx,
                total,
                100.0 * idx / max(1, total),
                rate,
                rem,
            )

    exclude_token_ids_set = set(_parse_int_list(args.exclude_token_ids))
    if args.exclude_special_token_ids:
        tokenizer = getattr(model, "tokenizer", None)
        all_special_ids = getattr(tokenizer, "all_special_ids", None)
        if all_special_ids:
            for tid in all_special_ids:
                exclude_token_ids_set.add(int(tid))
    exclude_token_ids = sorted(exclude_token_ids_set)

    filter_model_id = args.filter_model_id or args.model

    baseline_cfg = EvalConfig(
        name="baseline",
        collection=args.baseline_collection,
        transform=None,
        post_l2=bool(args.baseline_post_l2),
    )
    transformed_cfg = EvalConfig(
        name="transformed",
        collection=args.transformed_collection,
        transform=transform,
        post_l2=bool(args.transformed_post_l2),
    )

    entry_aggs = _parse_str_list(args.sweep_entry_agg)
    if not entry_aggs:
        entry_aggs = [args.entry_agg]
    entry_aggs = [x for x in entry_aggs if x in {"max", "mean_top3"}]
    if not entry_aggs:
        raise ValueError("No valid entry_agg values after parsing --sweep-entry-agg.")

    min_hits_values = _parse_int_list(args.sweep_min_entry_token_hits)
    if not min_hits_values:
        min_hits_values = [int(args.min_entry_token_hits)]
    min_hits_values = sorted(set(max(1, int(x)) for x in min_hits_values))

    query_ids = [q.qid for q in eval_queries]
    k = int(args.top_k)
    sweep_results: List[Dict[str, object]] = []
    baseline_metrics: Dict[str, object] | None = None
    transformed_metrics: Dict[str, object] | None = None
    delta: Dict[str, float] | None = None

    for agg in entry_aggs:
        for min_hits in min_hits_values:
            logger.info(
                "Running evaluation combo: entry_agg=%s min_entry_token_hits=%d",
                agg,
                min_hits,
            )
            b_metrics = _eval_config(
                store=store,
                config=baseline_cfg,
                raw_queries=raw_queries,
                query_ids=query_ids,
                qrels=qrels,
                top_k=k,
                search_k=int(args.search_k),
                filter_model_id=filter_model_id,
                exclude_token_ids=exclude_token_ids,
                group_by_entry=bool(args.group_by_entry),
                entry_agg=agg,
                min_entry_token_hits=min_hits,
                logger=logger,
                progress_every=max(1, int(args.progress_every)),
            )

            t_metrics = _eval_config(
                store=store,
                config=transformed_cfg,
                raw_queries=raw_queries,
                query_ids=query_ids,
                qrels=qrels,
                top_k=k,
                search_k=int(args.search_k),
                filter_model_id=filter_model_id,
                exclude_token_ids=exclude_token_ids,
                group_by_entry=bool(args.group_by_entry),
                entry_agg=agg,
                min_entry_token_hits=min_hits,
                logger=logger,
                progress_every=max(1, int(args.progress_every)),
            )
            d_metrics = _compute_delta(
                baseline_metrics=b_metrics,
                transformed_metrics=t_metrics,
                top_k=k,
            )
            sweep_results.append(
                {
                    "entry_agg": agg,
                    "min_entry_token_hits": int(min_hits),
                    "baseline": b_metrics,
                    "transformed": t_metrics,
                    "delta_transformed_minus_baseline": d_metrics,
                }
            )

            if agg == args.entry_agg and int(min_hits) == int(args.min_entry_token_hits):
                baseline_metrics = b_metrics
                transformed_metrics = t_metrics
                delta = d_metrics

    if baseline_metrics is None or transformed_metrics is None or delta is None:
        baseline_metrics = sweep_results[0]["baseline"]  # type: ignore[assignment]
        transformed_metrics = sweep_results[0]["transformed"]  # type: ignore[assignment]
        delta = sweep_results[0]["delta_transformed_minus_baseline"]  # type: ignore[assignment]

    output = {
        "experiment": "nnm_eval_qdrant_beir_compare_tl",
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "retrieval_layer": int(args.retrieval_layer),
        "query_role": args.query_role,
        "query_count": int(len(eval_queries)),
        "query_ids": [q.qid for q in eval_queries],
        "config": {
            "top_k": int(args.top_k),
            "search_k": int(args.search_k),
            "group_by_entry": bool(args.group_by_entry),
            "entry_agg": args.entry_agg,
            "min_entry_token_hits": int(args.min_entry_token_hits),
            "sweep_entry_agg": entry_aggs,
            "sweep_min_entry_token_hits": min_hits_values,
            "exclude_token_ids": exclude_token_ids,
            "filter_model_id": filter_model_id,
            "baseline_collection": args.baseline_collection,
            "transformed_collection": args.transformed_collection,
            "transformed_transform_file": args.transformed_transform_file,
            "baseline_post_l2": bool(args.baseline_post_l2),
            "transformed_post_l2": bool(args.transformed_post_l2),
        },
        "baseline": baseline_metrics,
        "transformed": transformed_metrics,
        "delta_transformed_minus_baseline": delta,
        "sweep_results": sweep_results,
    }

    out_json = results_dir / "results.json"
    save_json(output, out_json)

    logger.info("Saved results to %s", out_json)
    logger.info(
        "Summary @%d: baseline ndcg=%.4f recall=%.4f mrr=%.4f | transformed ndcg=%.4f recall=%.4f mrr=%.4f",
        k,
        baseline_metrics[f"ndcg@{k}"],
        baseline_metrics[f"recall@{k}"],
        baseline_metrics[f"mrr@{k}"],
        transformed_metrics[f"ndcg@{k}"],
        transformed_metrics[f"recall@{k}"],
        transformed_metrics[f"mrr@{k}"],
    )
    logger.info(
        "Delta transformed-baseline @%d: ndcg=%+.4f recall=%+.4f mrr=%+.4f",
        k,
        delta[f"ndcg@{k}"],
        delta[f"recall@{k}"],
        delta[f"mrr@{k}"],
    )

    print(f"Saved results to {out_json}")
    print(
        "Delta transformed-baseline "
        f"ndcg@{k}={delta[f'ndcg@{k}']:+.4f} "
        f"recall@{k}={delta[f'recall@{k}']:+.4f} "
        f"mrr@{k}={delta[f'mrr@{k}']:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
