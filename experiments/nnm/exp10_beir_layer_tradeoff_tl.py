#!/usr/bin/env python3
"""Experiment 10 (NNM): BEIR layer tradeoff (retrieval vs injection).

This experiment compares two layer groups on BEIR data:
1) 4 ID-selected layers (paper-style) for cosine retrieval quality.
2) 4 layers in the last third (excluding very last layers) for injection quality.

The goal is to derive a practical storage methodology:
- Which layers are best for retrieval embeddings?
- Which layers are best for robust delta->KV reconstruction/injection?
"""

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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.io_utils import create_results_dir, save_json, setup_logging
from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder
from nnm.kvembed.layer_selection import compute_intrinsic_dimension_twonn, select_rerouting_layers
from nnm.kvembed.prompts import build_compression_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp10: BEIR layer tradeoff retrieval vs injection (TransformerLens)"
    )
    parser.add_argument("--dataset", type=str, default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prefix-bias", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=80)
    parser.add_argument("--max-corpus", type=int, default=800)
    parser.add_argument("--id-corpus-size", type=int, default=120)
    parser.add_argument("--injection-pairs", type=int, default=24)

    parser.add_argument("--id-layer-mode", type=str, default="paper", choices=["paper", "fixed"])
    parser.add_argument("--id-layer-count", type=int, default=4)
    parser.add_argument(
        "--late-layer-count",
        type=int,
        default=None,
        help="If omitted, uses len(id_layers).",
    )
    parser.add_argument("--tail-exclude", type=int, default=2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--no-progress-stdout",
        action="store_true",
        help="Disable direct progress prints to stdout.",
    )

    parser.add_argument("--save-layer-embeddings", action="store_true")
    return parser.parse_args()


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


def _dcg_at_k(binary_rels: Sequence[int], k: int) -> float:
    out = 0.0
    for i, rel in enumerate(binary_rels[:k]):
        if rel <= 0:
            continue
        out += 1.0 / math.log2(i + 2.0)
    return out


def _safe_text(doc: Mapping[str, object]) -> str:
    title = str(doc.get("title", "") or "").strip()
    text = str(doc.get("text", "") or "").strip()
    joined = f"{title} {text}".strip()
    return joined if joined else text


def _log_progress(
    logger: Any,
    phase: str,
    done: int,
    total: int,
    start_ts: float,
    progress_stdout: bool = True,
) -> None:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    elapsed = max(time.perf_counter() - start_ts, 1e-6)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta = remaining / max(rate, 1e-9)
    pct = 100.0 * done / total
    message = (
        f"{phase} progress: {done}/{total} ({pct:.1f}%), "
        f"{rate:.2f} items/s, ETA {eta:.1f}s"
    )
    logger.info(message)
    if progress_stdout:
        print(message, flush=True)


def load_beir_dataset(dataset: str, split: str) -> tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    root = REPO_ROOT / "data" / "beir_datasets" / dataset
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}. "
            "Please place BEIR dataset files under data/beir_datasets/<dataset>/."
        )

    corpus: Dict[str, dict] = {}
    with (root / "corpus.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row["_id"])
            corpus[doc_id] = row

    queries: Dict[str, str] = {}
    with (root / "queries.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["_id"])
            queries[qid] = str(row.get("text", ""))

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

    return corpus, queries, qrels


@dataclass(frozen=True)
class SubsetData:
    corpus_ids: List[str]
    query_ids: List[str]
    qrels: Dict[str, Dict[str, int]]


def build_subset(
    corpus: Mapping[str, dict],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    max_queries: int,
    max_corpus: int,
    seed: int,
) -> SubsetData:
    rng = random.Random(seed)

    valid_qids = []
    for qid, rels in qrels.items():
        if qid not in queries:
            continue
        if any(did in corpus for did in rels.keys()):
            valid_qids.append(qid)
    rng.shuffle(valid_qids)
    query_ids = valid_qids[: max(1, min(max_queries, len(valid_qids)))]

    qrels_sub: Dict[str, Dict[str, int]] = {}
    positive_doc_ids: set[str] = set()
    for qid in query_ids:
        rels = qrels.get(qid, {})
        filtered = {did: int(score) for did, score in rels.items() if did in corpus and int(score) > 0}
        if not filtered:
            continue
        qrels_sub[qid] = filtered
        positive_doc_ids.update(filtered.keys())

    query_ids = [qid for qid in query_ids if qid in qrels_sub]
    if not query_ids:
        raise RuntimeError("No valid query ids left after filtering qrels/corpus overlap.")

    corpus_ids = list(positive_doc_ids)
    if len(corpus_ids) < max_corpus:
        remaining = [did for did in corpus.keys() if did not in positive_doc_ids]
        rng.shuffle(remaining)
        take = max_corpus - len(corpus_ids)
        corpus_ids.extend(remaining[:take])
    else:
        rng.shuffle(corpus_ids)
        corpus_ids = corpus_ids[:max_corpus]

    return SubsetData(
        corpus_ids=sorted(corpus_ids),
        query_ids=query_ids,
        qrels=qrels_sub,
    )


def estimate_id_by_layer(
    embedder: TransformerLensKVEmbedder,
    corpus_texts: Sequence[str],
    logger: Any | None = None,
    progress_every: int = 25,
    progress_stdout: bool = True,
) -> Dict[int, float]:
    model = embedder.model
    n_layers = int(model.cfg.n_layers)
    names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    per_layer: List[List[np.ndarray]] = [[] for _ in range(n_layers)]
    progress_every = max(1, int(progress_every))
    total = len(corpus_texts)
    t0 = time.perf_counter()
    if logger is not None:
        logger.info("Starting ID estimation activation pass: texts=%d layers=%d", total, n_layers)

    with torch.no_grad():
        for idx, text in enumerate(corpus_texts, start=1):
            _, cache = model.run_with_cache(
                text,
                return_type=None,
                prepend_bos=embedder._prepend_bos,
                names_filter=lambda name: name in names,
                remove_batch_dim=False,
            )
            for layer in range(n_layers):
                key = f"blocks.{layer}.hook_resid_post"
                if key not in cache:
                    continue
                hidden = cache[key][0].detach().to(torch.float32).cpu().numpy()
                per_layer[layer].append(hidden)
            del cache
            if logger is not None and (idx == 1 or idx % progress_every == 0 or idx == total):
                _log_progress(
                    logger,
                    "ID activation pass",
                    idx,
                    total,
                    t0,
                    progress_stdout=progress_stdout,
                )

    t1 = time.perf_counter()
    if logger is not None:
        logger.info("Starting ID reduction over layers...")
    id_by_layer: Dict[int, float] = {}
    for layer in range(n_layers):
        if not per_layer[layer]:
            id_by_layer[layer] = float("nan")
            continue
        mat = np.concatenate(per_layer[layer], axis=0)
        id_by_layer[layer] = compute_intrinsic_dimension_twonn(mat)
        if logger is not None and ((layer + 1) % 4 == 0 or layer == n_layers - 1):
            _log_progress(
                logger,
                "ID layer reduction",
                layer + 1,
                n_layers,
                t1,
                progress_stdout=progress_stdout,
            )
    return id_by_layer


def _is_u_shaped(values: np.ndarray, max_violation_ratio: float = 0.20) -> bool:
    if values.ndim != 1 or len(values) < 5:
        return False
    if np.any(~np.isfinite(values)):
        return False
    idx = int(np.argmin(values))
    left = values[: idx + 1]
    right = values[idx:]
    if len(left) < 2 or len(right) < 2:
        return False
    left_viol = int(np.sum(np.diff(left) > 0))
    right_viol = int(np.sum(np.diff(right) < 0))
    total = max((len(left) - 1) + (len(right) - 1), 1)
    return (left_viol + right_viol) / total <= max_violation_ratio


def select_id_layers_fixed_count(
    id_by_layer: Mapping[int, float],
    n_layers: int,
    count: int = 4,
    exclude_early_fraction: float = 0.20,
) -> List[int]:
    values = np.array([id_by_layer.get(i, float("nan")) for i in range(n_layers)], dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise RuntimeError("No finite ID values available for ID-layer selection.")

    count = max(1, int(count))
    use_u_shape = _is_u_shaped(values[finite])
    if use_u_shape:
        min_layer = int(np.nanargmin(values))
        start = min_layer
        end = min(n_layers - 1, start + count - 1)
        selected = list(range(start, end + 1))
        if len(selected) < count:
            prepend = list(range(max(0, start - (count - len(selected))), start))
            selected = prepend + selected
        return sorted(selected)

    early_cutoff = int(np.floor(exclude_early_fraction * n_layers))
    candidates = [i for i in range(early_cutoff, n_layers) if np.isfinite(values[i])]
    if len(candidates) < count:
        candidates = [i for i in range(n_layers) if np.isfinite(values[i])]

    ranked = sorted(candidates, key=lambda i: values[i])
    return sorted(ranked[:count])


def select_late_third_layers(
    n_layers: int,
    count: int,
    tail_exclude: int = 2,
) -> List[int]:
    count = max(1, int(count))
    tail_exclude = max(0, int(tail_exclude))

    start = int(np.floor((2.0 * n_layers) / 3.0))
    end = max(start, n_layers - 1 - tail_exclude)
    raw = np.linspace(start, end, num=count)
    idx = sorted({int(round(v)) for v in raw})

    cur = start
    while len(idx) < count and cur <= end:
        if cur not in idx:
            idx.append(cur)
        cur += 1
    idx = sorted(idx)[:count]
    return idx


def extract_pooled_by_layers(
    embedder: TransformerLensKVEmbedder,
    texts: Sequence[str],
    role: str,
    layers: Sequence[int],
    logger: Any | None = None,
    progress_every: int = 25,
    progress_stdout: bool = True,
) -> Dict[int, np.ndarray]:
    model = embedder.model
    layers = sorted(set(int(x) for x in layers))
    names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    out: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    progress_every = max(1, int(progress_every))
    total = len(texts)
    t0 = time.perf_counter()
    if logger is not None:
        logger.info(
            "Starting pooled extraction: role=%s texts=%d layers=%s",
            role,
            total,
            layers,
        )

    for idx, text in enumerate(texts, start=1):
        prompt = build_compression_prompt(
            text=text,
            role=role,  # type: ignore[arg-type]
            template=embedder.config.prompt_template,
        )
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt,
                return_type=None,
                prepend_bos=embedder._prepend_bos,
                names_filter=lambda name: name in names,
                remove_batch_dim=False,
            )

        for layer in layers:
            key = f"blocks.{layer}.hook_resid_post"
            hidden = cache[key][0].detach().to(torch.float32).cpu().numpy()
            pooled = (hidden[-1] + hidden.mean(axis=0)) / 2.0
            pooled = _l2_normalize(pooled, axis=0)
            out[layer].append(pooled.astype(np.float32))
        del cache
        if logger is not None and (idx == 1 or idx % progress_every == 0 or idx == total):
            _log_progress(
                logger,
                f"Pooled extraction ({role})",
                idx,
                total,
                t0,
                progress_stdout=progress_stdout,
            )

    return {layer: np.stack(v, axis=0) for layer, v in out.items()}


def evaluate_retrieval_metrics(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    corpus_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    topk: int,
) -> Dict[str, float]:
    c = _l2_normalize(corpus_embeddings.astype(np.float32), axis=1)
    q = _l2_normalize(query_embeddings.astype(np.float32), axis=1)
    sims = q @ c.T

    id_to_col = {did: i for i, did in enumerate(corpus_ids)}
    ndcgs: List[float] = []
    recalls: List[float] = []
    mrrs: List[float] = []
    pos_cos: List[float] = []
    neg_cos: List[float] = []

    for qi, qid in enumerate(query_ids):
        rel_docs = qrels.get(qid, {})
        rel_set = {did for did, s in rel_docs.items() if s > 0 and did in id_to_col}
        if not rel_set:
            continue

        row = sims[qi]
        ranking_idx = np.argsort(row)[::-1]
        top_idx = ranking_idx[:topk]
        top_doc_ids = [corpus_ids[i] for i in top_idx]

        binary = [1 if did in rel_set else 0 for did in top_doc_ids]
        dcg = _dcg_at_k(binary, topk)
        idcg = _dcg_at_k([1] * min(len(rel_set), topk), topk)
        ndcgs.append(float(dcg / idcg) if idcg > 0 else 0.0)

        hit_count = int(sum(binary))
        recalls.append(float(hit_count / len(rel_set)))

        rr = 0.0
        for rank, did in enumerate(top_doc_ids, start=1):
            if did in rel_set:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        rel_cols = [id_to_col[did] for did in rel_set]
        pos_cos.extend([float(x) for x in row[rel_cols]])

        nonrel_cols = [i for i, did in enumerate(corpus_ids) if did not in rel_set]
        if nonrel_cols:
            sample_neg = nonrel_cols[: min(64, len(nonrel_cols))]
            neg_cos.extend([float(x) for x in row[sample_neg]])

    out = {
        "queries_evaluated": float(len(ndcgs)),
        f"ndcg@{topk}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"recall@{topk}": float(np.mean(recalls)) if recalls else 0.0,
        f"mrr@{topk}": float(np.mean(mrrs)) if mrrs else 0.0,
        "mean_positive_cosine": float(np.mean(pos_cos)) if pos_cos else 0.0,
        "mean_negative_cosine": float(np.mean(neg_cos)) if neg_cos else 0.0,
    }
    out["cosine_margin"] = out["mean_positive_cosine"] - out["mean_negative_cosine"]
    return out


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def evaluate_injection_layers(
    embedder: TransformerLensKVEmbedder,
    late_layers: Sequence[int],
    corpus: Mapping[str, dict],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    query_ids: Sequence[str],
    max_pairs: int,
    seed: int,
    logger: Any | None = None,
    progress_every: int = 25,
    progress_stdout: bool = True,
) -> Dict[int, Dict[str, float]]:
    rng = random.Random(seed)
    model = embedder.model
    prepend_bos = embedder._prepend_bos

    pairs: List[Tuple[str, str]] = []
    for qid in query_ids:
        rels = qrels.get(qid, {})
        pos = [did for did, s in rels.items() if s > 0 and did in corpus]
        if not pos:
            continue
        rng.shuffle(pos)
        pairs.append((qid, pos[0]))
    rng.shuffle(pairs)
    pairs = pairs[: max(1, min(max_pairs, len(pairs)))]
    progress_every = max(1, int(progress_every))
    total_pairs = len(pairs)
    t0 = time.perf_counter()
    if logger is not None:
        logger.info(
            "Starting injection evaluation: pairs=%d layers=%s",
            total_pairs,
            sorted(int(x) for x in late_layers),
        )

    layer_metrics: Dict[int, Dict[str, List[float]]] = {}
    for layer in late_layers:
        layer_metrics[int(layer)] = {
            "kv_k_abs_mean": [],
            "kv_k_abs_max": [],
            "kv_v_abs_mean": [],
            "kv_v_abs_max": [],
            "logit_recon_true_abs_mean": [],
            "logit_recon_true_abs_max": [],
            "logit_true_base_abs_mean": [],
            "logit_true_base_abs_max": [],
            "top1_recon_true_match": [],
            "top5_jaccard_recon_true": [],
        }

    names: set[str] = set()
    for layer in late_layers:
        names.add(f"blocks.{layer}.hook_resid_pre")
        names.add(f"blocks.{layer}.attn.hook_k")
        names.add(f"blocks.{layer}.attn.hook_v")

    for pair_idx, (qid, did) in enumerate(pairs, start=1):
        memory_text = _safe_text(corpus[did])
        query_text = str(queries[qid])

        memory_prompt = build_compression_prompt(
            text=memory_text,
            role="context",
            template=embedder.config.prompt_template,
        )
        query_prompt = f"Query: {query_text}\nAnswer:"

        token_ids = model.to_tokens(memory_prompt, prepend_bos=prepend_bos)
        last_token_id = int(token_ids[0, -1].item())

        with torch.no_grad():
            _, mem_cache = model.run_with_cache(
                token_ids,
                return_type=None,
                prepend_bos=False,
                names_filter=lambda name: name in names,
                remove_batch_dim=False,
            )

        token_embed = model.W_E[last_token_id].detach().to(model.W_E.device).to(torch.float32).view(1, 1, -1)

        query_tokens = model.to_tokens(query_prompt, prepend_bos=prepend_bos)
        with torch.no_grad():
            logits_base, _ = model.run_with_cache(
                query_tokens,
                return_type="logits",
                past_kv_cache=None,
                prepend_bos=False,
                remove_batch_dim=False,
            )
        base_last = logits_base[:, -1].detach().to(torch.float32)

        for layer in late_layers:
            layer = int(layer)
            resid_pre = mem_cache[f"blocks.{layer}.hook_resid_pre"][:, -1:, :].detach()
            k_true = mem_cache[f"blocks.{layer}.attn.hook_k"][:, -1:, :, :].detach()
            v_true = mem_cache[f"blocks.{layer}.attn.hook_v"][:, -1:, :, :].detach()

            delta = resid_pre - token_embed
            resid_rec = token_embed + delta
            k_rec, v_rec = embedder._compute_kv_from_resid_pre(layer, resid_rec)

            diff_k = (k_rec - k_true).detach().to(torch.float32).cpu().numpy()
            diff_v = (v_rec - v_true).detach().to(torch.float32).cpu().numpy()

            lm = layer_metrics[layer]
            lm["kv_k_abs_mean"].append(float(np.mean(np.abs(diff_k))))
            lm["kv_k_abs_max"].append(float(np.max(np.abs(diff_k))))
            lm["kv_v_abs_mean"].append(float(np.mean(np.abs(diff_v))))
            lm["kv_v_abs_max"].append(float(np.max(np.abs(diff_v))))

            true_cache = embedder.build_prefix_cache_from_kv({layer: (k_true, v_true)}, [layer])
            recon_cache = embedder.build_prefix_cache_from_kv({layer: (k_rec, v_rec)}, [layer])
            attn_hooks = embedder._build_attention_bias_hooks([layer])

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

            true_last = logits_true[:, -1].detach().to(torch.float32)
            recon_last = logits_recon[:, -1].detach().to(torch.float32)

            diff_recon_true = (recon_last - true_last).detach().cpu().numpy()
            diff_true_base = (true_last - base_last).detach().cpu().numpy()

            lm["logit_recon_true_abs_mean"].append(float(np.mean(np.abs(diff_recon_true))))
            lm["logit_recon_true_abs_max"].append(float(np.max(np.abs(diff_recon_true))))
            lm["logit_true_base_abs_mean"].append(float(np.mean(np.abs(diff_true_base))))
            lm["logit_true_base_abs_max"].append(float(np.max(np.abs(diff_true_base))))

            top1_true = int(torch.argmax(true_last, dim=-1).item())
            top1_recon = int(torch.argmax(recon_last, dim=-1).item())
            lm["top1_recon_true_match"].append(float(1.0 if top1_true == top1_recon else 0.0))

            k = min(5, int(true_last.shape[-1]))
            _, top5_true = torch.topk(true_last, k=k, dim=-1)
            _, top5_recon = torch.topk(recon_last, k=k, dim=-1)
            set_true = set(int(x) for x in top5_true[0].tolist())
            set_recon = set(int(x) for x in top5_recon[0].tolist())
            inter = len(set_true & set_recon)
            union = len(set_true | set_recon)
            lm["top5_jaccard_recon_true"].append(float(inter / union) if union > 0 else 0.0)

        del mem_cache
        if logger is not None and (
            pair_idx == 1 or pair_idx % progress_every == 0 or pair_idx == total_pairs
        ):
            _log_progress(
                logger,
                "Injection evaluation",
                pair_idx,
                total_pairs,
                t0,
                progress_stdout=progress_stdout,
            )

    summary: Dict[int, Dict[str, float]] = {}
    for layer, values in layer_metrics.items():
        s = {k: _safe_mean(v) for k, v in values.items()}
        # Composite score: high if reconstruction is close and injection has non-trivial effect.
        s["injectability_score"] = (
            (1.0 / (1.0 + s["logit_recon_true_abs_mean"]))
            * (1.0 + s["logit_true_base_abs_mean"])
            * (0.5 + 0.5 * s["top1_recon_true_match"])
        )
        s["pairs_evaluated"] = float(len(pairs))
        summary[layer] = s

    return summary


def main() -> int:
    args = parse_args()
    results_dir = create_results_dir("nnm_exp10")
    logger = setup_logging("nnm_exp10_beir_layer_tradeoff_tl", results_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus_all, queries_all, qrels_all = load_beir_dataset(args.dataset, args.split)
    subset = build_subset(
        corpus=corpus_all,
        queries=queries_all,
        qrels=qrels_all,
        max_queries=args.max_queries,
        max_corpus=args.max_corpus,
        seed=args.seed,
    )

    corpus_texts = [_safe_text(corpus_all[did]) for did in subset.corpus_ids]
    query_texts = [queries_all[qid] for qid in subset.query_ids]

    logger.info(
        "Subset prepared: dataset=%s split=%s corpus=%d queries=%d",
        args.dataset,
        args.split,
        len(subset.corpus_ids),
        len(subset.query_ids),
    )

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        prefix_bias=args.prefix_bias,
    )
    embedder = TransformerLensKVEmbedder(config)
    n_layers = int(embedder.model.cfg.n_layers)

    id_texts = corpus_texts[: max(1, min(args.id_corpus_size, len(corpus_texts)))]
    progress_stdout = not bool(args.no_progress_stdout)
    id_by_layer = estimate_id_by_layer(
        embedder,
        id_texts,
        logger=logger,
        progress_every=args.progress_every,
        progress_stdout=progress_stdout,
    )

    if args.id_layer_mode == "paper":
        paper_sel = select_rerouting_layers(
            id_by_layer=id_by_layer,
            n_layers=n_layers,
            layer_window_fraction=embedder.config.layer_window_fraction,
            exclude_early_fraction=embedder.config.exclude_early_fraction,
            detect_u_shape=embedder.config.detect_u_shape,
        )
        id_layers = list(sorted(int(x) for x in paper_sel.selected_layers))
    else:
        id_layers = select_id_layers_fixed_count(
            id_by_layer=id_by_layer,
            n_layers=n_layers,
            count=args.id_layer_count,
            exclude_early_fraction=embedder.config.exclude_early_fraction,
        )

    late_count = int(args.late_layer_count) if args.late_layer_count is not None else len(id_layers)
    late_count = max(1, late_count)
    late_layers = select_late_third_layers(
        n_layers=n_layers,
        count=late_count,
        tail_exclude=args.tail_exclude,
    )

    logger.info("ID layers (mode=%s): %s", args.id_layer_mode, id_layers)
    logger.info("Late-third layers (count=%d): %s", late_count, late_layers)

    retrieval_layers = sorted(set(id_layers))
    corpus_by_layer = extract_pooled_by_layers(
        embedder,
        corpus_texts,
        role="context",
        layers=retrieval_layers,
        logger=logger,
        progress_every=args.progress_every,
        progress_stdout=progress_stdout,
    )
    query_by_layer = extract_pooled_by_layers(
        embedder,
        query_texts,
        role="query",
        layers=retrieval_layers,
        logger=logger,
        progress_every=args.progress_every,
        progress_stdout=progress_stdout,
    )

    retrieval_results: Dict[str, Dict[str, float]] = {}
    for layer in retrieval_layers:
        metrics = evaluate_retrieval_metrics(
            corpus_embeddings=corpus_by_layer[layer],
            query_embeddings=query_by_layer[layer],
            corpus_ids=subset.corpus_ids,
            query_ids=subset.query_ids,
            qrels=subset.qrels,
            topk=max(1, int(args.topk)),
        )
        retrieval_results[str(layer)] = metrics
        logger.info(
            "Retrieval layer %d: NDCG@%d=%.4f Recall@%d=%.4f MRR@%d=%.4f margin=%.4f",
            layer,
            args.topk,
            metrics[f"ndcg@{args.topk}"],
            args.topk,
            metrics[f"recall@{args.topk}"],
            args.topk,
            metrics[f"mrr@{args.topk}"],
            metrics["cosine_margin"],
        )

    # Optional aggregation across ID layers (simple average of layer embeddings).
    corpus_stack = np.stack([corpus_by_layer[l] for l in retrieval_layers], axis=0)
    query_stack = np.stack([query_by_layer[l] for l in retrieval_layers], axis=0)
    corpus_mean = _l2_normalize(corpus_stack.mean(axis=0), axis=1)
    query_mean = _l2_normalize(query_stack.mean(axis=0), axis=1)
    retrieval_results["id_layer_mean"] = evaluate_retrieval_metrics(
        corpus_embeddings=corpus_mean,
        query_embeddings=query_mean,
        corpus_ids=subset.corpus_ids,
        query_ids=subset.query_ids,
        qrels=subset.qrels,
        topk=max(1, int(args.topk)),
    )

    injection_results = evaluate_injection_layers(
        embedder=embedder,
        late_layers=late_layers,
        corpus=corpus_all,
        queries=queries_all,
        qrels=subset.qrels,
        query_ids=subset.query_ids,
        max_pairs=args.injection_pairs,
        seed=args.seed,
        logger=logger,
        progress_every=max(1, min(args.progress_every, 10)),
        progress_stdout=progress_stdout,
    )
    for layer in late_layers:
        m = injection_results[int(layer)]
        logger.info(
            "Injection layer %d: kv_max(k/v)=%.6f/%.6f recon_true_logit_mean=%.6f effect_true_base=%.6f score=%.6f",
            layer,
            m["kv_k_abs_max"],
            m["kv_v_abs_max"],
            m["logit_recon_true_abs_mean"],
            m["logit_true_base_abs_mean"],
            m["injectability_score"],
        )

    # Suggestion heuristics for methodology.
    best_retrieval_layer = max(
        retrieval_layers,
        key=lambda l: retrieval_results[str(l)].get(f"ndcg@{args.topk}", 0.0),
    )
    best_injection_layer = max(
        late_layers,
        key=lambda l: injection_results[int(l)].get("injectability_score", 0.0),
    )

    output = {
        "experiment": "nnm_exp10_beir_layer_tradeoff_tl",
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "prefix_bias": args.prefix_bias,
            "seed": args.seed,
            "max_queries": args.max_queries,
            "max_corpus": args.max_corpus,
            "id_corpus_size": args.id_corpus_size,
            "injection_pairs": args.injection_pairs,
            "topk": args.topk,
            "progress_every": args.progress_every,
            "progress_stdout": progress_stdout,
            "id_layer_mode": args.id_layer_mode,
            "id_layer_count": args.id_layer_count,
            "late_layer_count": args.late_layer_count,
            "late_layer_count_effective": late_count,
            "tail_exclude": args.tail_exclude,
        },
        "subset": {
            "n_corpus": len(subset.corpus_ids),
            "n_queries": len(subset.query_ids),
        },
        "layer_selection": {
            "n_layers": n_layers,
            "id_by_layer": {int(k): float(v) for k, v in id_by_layer.items()},
            "id_layers": id_layers,
            "late_layers": late_layers,
        },
        "retrieval": retrieval_results,
        "injection": {int(k): v for k, v in injection_results.items()},
        "methodology_hint": {
            "best_retrieval_layer_by_ndcg": int(best_retrieval_layer),
            "best_injection_layer_by_score": int(best_injection_layer),
            "recommended_split_storage": {
                "retrieval_layers": id_layers,
                "injection_layers": late_layers,
            },
        },
    }

    if args.save_layer_embeddings:
        np.savez_compressed(
            results_dir / "layer_embeddings.npz",
            corpus_ids=np.array(subset.corpus_ids, dtype=object),
            query_ids=np.array(subset.query_ids, dtype=object),
            id_layers=np.array(retrieval_layers, dtype=np.int64),
            corpus_embeddings=np.array([corpus_by_layer[l] for l in retrieval_layers], dtype=np.float32),
            query_embeddings=np.array([query_by_layer[l] for l in retrieval_layers], dtype=np.float32),
        )
        logger.info("Saved layer embeddings to %s", results_dir / "layer_embeddings.npz")

    out_path = results_dir / "results.json"
    save_json(output, out_path)
    logger.info("Saved results to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
