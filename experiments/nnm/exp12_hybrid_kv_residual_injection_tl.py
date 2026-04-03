#!/usr/bin/env python3
"""Experiment 12 (NNM): Hybrid knowledge+steering injection (KV vs residual vs hybrid).

Compares three inference-time interventions on identical BEIR pairs:
1) `kv_only`: Inject memory as a virtual prefix in K/V cache (knowledge path).
2) `resid_only`: Inject a residual steering vector at one layer (steering path).
3) `hybrid`: Apply both K/V memory and residual steering together.

Goal:
- Quantify effect size vs baseline (no injection).
- Quantify agreement with KV behavior (as "knowledge target").
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.io_utils import create_results_dir, save_json, setup_logging
from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder
from nnm.kvembed.prompts import build_compression_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp12: compare kv_only vs resid_only vs hybrid injection (TransformerLens)"
    )
    parser.add_argument("--dataset", type=str, default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prefix-bias", type=float, default=1.0)

    parser.add_argument("--injection-layer", type=int, default=18)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--max-corpus", type=int, default=2000)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--kv-alpha",
        type=float,
        default=1.0,
        help="Scale factor for injected V in KV cache (knowledge strength).",
    )
    parser.add_argument(
        "--residual-alpha",
        type=float,
        default=0.10,
        help="Scale factor for residual steering vector.",
    )
    parser.add_argument(
        "--residual-source",
        type=str,
        default="delta",
        choices=["delta", "full"],
        help=(
            "Steering source vector. "
            "'delta' = resid_pre_last - token_embed(last_token), "
            "'full' = resid_pre_last."
        ),
    )
    parser.add_argument(
        "--residual-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize residual steering vector before applying alpha (default: true).",
    )
    parser.add_argument(
        "--residual-apply",
        type=str,
        default="all",
        choices=["all", "last"],
        help="Apply residual steering to all query positions or only last position.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-progress-stdout", action="store_true")
    return parser.parse_args()


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
    msg = f"{phase} progress: {done}/{total} ({pct:.1f}%), {rate:.2f} items/s, ETA {eta:.1f}s"
    logger.info(msg)
    if progress_stdout:
        print(msg, flush=True)


def _safe_text(doc: Mapping[str, object]) -> str:
    title = str(doc.get("title", "") or "").strip()
    text = str(doc.get("text", "") or "").strip()
    joined = f"{title} {text}".strip()
    return joined if joined else text


def load_beir_dataset(dataset: str, split: str) -> tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    root = REPO_ROOT / "data" / "beir_datasets" / dataset
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    corpus: Dict[str, dict] = {}
    with (root / "corpus.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            corpus[str(row["_id"])] = row

    queries: Dict[str, str] = {}
    with (root / "queries.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            queries[str(row["_id"])] = str(row.get("text", ""))

    qrels_path = root / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"qrels file not found: {qrels_path}")

    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            score = int(float(row["score"]))
            if score <= 0:
                continue
            qrels.setdefault(qid, {})[did] = score

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
        kept = {did: int(s) for did, s in rels.items() if did in corpus and int(s) > 0}
        if not kept:
            continue
        qrels_sub[qid] = kept
        positive_doc_ids.update(kept.keys())

    query_ids = [qid for qid in query_ids if qid in qrels_sub]
    if not query_ids:
        raise RuntimeError("No valid queries after overlap filtering.")

    corpus_ids = list(positive_doc_ids)
    if len(corpus_ids) < max_corpus:
        others = [did for did in corpus.keys() if did not in positive_doc_ids]
        rng.shuffle(others)
        corpus_ids.extend(others[: (max_corpus - len(corpus_ids))])
    else:
        rng.shuffle(corpus_ids)
        corpus_ids = corpus_ids[:max_corpus]

    return SubsetData(
        corpus_ids=sorted(corpus_ids),
        query_ids=query_ids,
        qrels=qrels_sub,
    )


def build_eval_pairs(
    subset: SubsetData,
    corpus: Mapping[str, dict],
    max_pairs: int,
    seed: int,
) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    for qid in subset.query_ids:
        rels = subset.qrels.get(qid, {})
        pos = [did for did, s in rels.items() if s > 0 and did in corpus]
        if not pos:
            continue
        rng.shuffle(pos)
        pairs.append((qid, pos[0]))
    rng.shuffle(pairs)
    return pairs[: max(1, min(max_pairs, len(pairs)))]


def _as_set_topk(x: torch.Tensor, k: int = 5) -> set[int]:
    kk = min(k, int(x.shape[-1]))
    _, topk = torch.topk(x, k=kk, dim=-1)
    return set(int(v) for v in topk[0].tolist())


def _jaccard(a: set[int], b: set[int]) -> float:
    u = len(a | b)
    if u == 0:
        return 0.0
    return float(len(a & b) / u)


def _l2(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)


def _build_residual_hook(
    *,
    hook_name: str,
    steering_vec: torch.Tensor,
    alpha: float,
    apply_mode: str,
) -> List[Tuple[str, Any]]:
    # steering_vec: [d_model], float32 on any device
    vec = steering_vec.detach().to(torch.float32).view(1, 1, -1)
    aa = float(alpha)
    mode = str(apply_mode)

    def _hook(resid: torch.Tensor, hook: Any, v: torch.Tensor = vec, a: float = aa) -> torch.Tensor:
        add = (v * a).to(device=resid.device, dtype=resid.dtype)
        out = resid.clone()
        if mode == "last":
            out[:, -1:, :] = out[:, -1:, :] + add
        else:
            out = out + add
        return out

    return [(hook_name, _hook)]


def main() -> int:
    args = parse_args()
    results_dir = create_results_dir("nnm_exp12")
    logger = setup_logging("nnm_exp12_hybrid_kv_residual_injection_tl", results_dir)
    progress_stdout = not bool(args.no_progress_stdout)

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
    pairs = build_eval_pairs(subset, corpus_all, max_pairs=args.max_pairs, seed=args.seed)
    logger.info(
        "Subset prepared: dataset=%s split=%s corpus=%d queries=%d eval_pairs=%d",
        args.dataset,
        args.split,
        len(subset.corpus_ids),
        len(subset.query_ids),
        len(pairs),
    )

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

    n_layers = int(model.cfg.n_layers)
    injection_layer = int(args.injection_layer)
    if injection_layer < 0 or injection_layer >= n_layers:
        raise ValueError(f"Invalid injection layer: {injection_layer} for n_layers={n_layers}")

    key_name = embedder._key_hook_name(injection_layer)
    inj_resid_key = f"blocks.{injection_layer}.hook_resid_pre"
    inj_v_key = f"blocks.{injection_layer}.attn.hook_v"
    names = {inj_resid_key, key_name, inj_v_key}
    attn_hooks = embedder._build_attention_bias_hooks([injection_layer])
    resid_hook_name = f"blocks.{injection_layer}.hook_resid_pre"

    metrics: Dict[str, Dict[str, List[float]]] = {
        "kv_only": {
            "logit_vs_base_abs_mean": [],
            "logit_vs_base_abs_max": [],
            "top1_change_vs_base": [],
            "top5_jaccard_vs_base": [],
            "logit_vs_kv_abs_mean": [],
            "logit_vs_kv_abs_max": [],
            "top1_match_vs_kv": [],
            "top5_jaccard_vs_kv": [],
        },
        "resid_only": {
            "logit_vs_base_abs_mean": [],
            "logit_vs_base_abs_max": [],
            "top1_change_vs_base": [],
            "top5_jaccard_vs_base": [],
            "logit_vs_kv_abs_mean": [],
            "logit_vs_kv_abs_max": [],
            "top1_match_vs_kv": [],
            "top5_jaccard_vs_kv": [],
        },
        "hybrid": {
            "logit_vs_base_abs_mean": [],
            "logit_vs_base_abs_max": [],
            "top1_change_vs_base": [],
            "top5_jaccard_vs_base": [],
            "logit_vs_kv_abs_mean": [],
            "logit_vs_kv_abs_max": [],
            "top1_match_vs_kv": [],
            "top5_jaccard_vs_kv": [],
        },
    }
    steering_norms: List[float] = []

    t0 = time.perf_counter()
    progress_every = max(1, int(args.progress_every))
    for idx, (qid, did) in enumerate(pairs, start=1):
        memory_text = _safe_text(corpus_all[did])
        query_text = str(queries_all[qid])
        memory_prompt = build_compression_prompt(
            text=memory_text,
            role="context",
            template=config.prompt_template,
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
        resid_mem_last = mem_cache[inj_resid_key][:, -1:, :].detach().to(torch.float32)  # [1,1,d]
        k_true = mem_cache[key_name][:, -1:, :, :].detach()
        v_true = mem_cache[inj_v_key][:, -1:, :, :].detach()
        del mem_cache

        if float(args.kv_alpha) != 1.0:
            v_true = v_true * float(args.kv_alpha)

        if args.residual_source == "delta":
            token_embed = model.W_E[last_token_id].detach().to(torch.float32).view(1, 1, -1)
            steering = (resid_mem_last - token_embed)[0, 0, :].detach().to(torch.float32)
        else:
            steering = resid_mem_last[0, 0, :].detach().to(torch.float32)
        if args.residual_normalize:
            steering = _l2(steering.view(1, -1))[0]
        steering_norms.append(float(torch.norm(steering).item()))
        resid_hooks = _build_residual_hook(
            hook_name=resid_hook_name,
            steering_vec=steering,
            alpha=float(args.residual_alpha),
            apply_mode=args.residual_apply,
        )

        query_tokens = model.to_tokens(query_prompt, prepend_bos=prepend_bos)
        kv_cache = embedder.build_prefix_cache_from_kv({injection_layer: (k_true, v_true)}, [injection_layer])

        with torch.no_grad():
            logits_base, _ = model.run_with_cache(
                query_tokens,
                return_type="logits",
                past_kv_cache=None,
                prepend_bos=False,
                remove_batch_dim=False,
            )
            with model.hooks(fwd_hooks=attn_hooks):
                logits_kv, _ = model.run_with_cache(
                    query_tokens,
                    return_type="logits",
                    past_kv_cache=kv_cache,
                    prepend_bos=False,
                    remove_batch_dim=False,
                )
            with model.hooks(fwd_hooks=resid_hooks):
                logits_resid, _ = model.run_with_cache(
                    query_tokens,
                    return_type="logits",
                    past_kv_cache=None,
                    prepend_bos=False,
                    remove_batch_dim=False,
                )
            with model.hooks(fwd_hooks=(attn_hooks + resid_hooks)):
                logits_hybrid, _ = model.run_with_cache(
                    query_tokens,
                    return_type="logits",
                    past_kv_cache=kv_cache,
                    prepend_bos=False,
                    remove_batch_dim=False,
                )

        base_last = logits_base[:, -1].detach().to(torch.float32)
        kv_last = logits_kv[:, -1].detach().to(torch.float32)
        resid_last = logits_resid[:, -1].detach().to(torch.float32)
        hybrid_last = logits_hybrid[:, -1].detach().to(torch.float32)

        top1_base = int(torch.argmax(base_last, dim=-1).item())
        top1_kv = int(torch.argmax(kv_last, dim=-1).item())
        top1_resid = int(torch.argmax(resid_last, dim=-1).item())
        top1_hybrid = int(torch.argmax(hybrid_last, dim=-1).item())
        top5_base = _as_set_topk(base_last, k=5)
        top5_kv = _as_set_topk(kv_last, k=5)
        top5_resid = _as_set_topk(resid_last, k=5)
        top5_hybrid = _as_set_topk(hybrid_last, k=5)

        rows = {
            "kv_only": (kv_last, top1_kv, top5_kv),
            "resid_only": (resid_last, top1_resid, top5_resid),
            "hybrid": (hybrid_last, top1_hybrid, top5_hybrid),
        }
        for mode, (last_logits, top1_mode, top5_mode) in rows.items():
            d_base = (last_logits - base_last).abs()
            d_kv = (last_logits - kv_last).abs()
            m = metrics[mode]
            m["logit_vs_base_abs_mean"].append(float(torch.mean(d_base).item()))
            m["logit_vs_base_abs_max"].append(float(torch.max(d_base).item()))
            m["top1_change_vs_base"].append(float(1.0 if top1_mode != top1_base else 0.0))
            m["top5_jaccard_vs_base"].append(_jaccard(top5_mode, top5_base))
            m["logit_vs_kv_abs_mean"].append(float(torch.mean(d_kv).item()))
            m["logit_vs_kv_abs_max"].append(float(torch.max(d_kv).item()))
            m["top1_match_vs_kv"].append(float(1.0 if top1_mode == top1_kv else 0.0))
            m["top5_jaccard_vs_kv"].append(_jaccard(top5_mode, top5_kv))

        if idx == 1 or idx % progress_every == 0 or idx == len(pairs):
            _log_progress(
                logger,
                "Exp12 pair loop",
                idx,
                len(pairs),
                t0,
                progress_stdout=progress_stdout,
            )

    summary: Dict[str, Dict[str, float]] = {}
    for mode, vals in metrics.items():
        out: Dict[str, float] = {}
        for key, arr in vals.items():
            out[key] = float(np.mean(arr)) if arr else 0.0
        summary[mode] = out

    output = {
        "experiment": "nnm_exp12_hybrid_kv_residual_injection_tl",
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "prefix_bias": args.prefix_bias,
            "seed": args.seed,
            "max_queries": args.max_queries,
            "max_corpus": args.max_corpus,
            "max_pairs": args.max_pairs,
            "injection_layer": injection_layer,
            "kv_alpha": float(args.kv_alpha),
            "residual_alpha": float(args.residual_alpha),
            "residual_source": args.residual_source,
            "residual_normalize": bool(args.residual_normalize),
            "residual_apply": args.residual_apply,
            "progress_every": args.progress_every,
            "progress_stdout": progress_stdout,
        },
        "subset": {
            "n_corpus": len(subset.corpus_ids),
            "n_queries": len(subset.query_ids),
            "n_eval_pairs": len(pairs),
        },
        "results": summary,
        "diagnostics": {
            "mean_steering_norm": float(np.mean(steering_norms)) if steering_norms else 0.0,
            "min_steering_norm": float(np.min(steering_norms)) if steering_norms else 0.0,
            "max_steering_norm": float(np.max(steering_norms)) if steering_norms else 0.0,
        },
        "interpretation_hint": {
            "note": (
                "kv_only is treated as the knowledge reference in this experiment. "
                "logit_vs_kv_* and top*_vs_kv show how close resid_only/hybrid stay to the KV effect."
            )
        },
    }

    out_path = results_dir / "results.json"
    save_json(output, out_path)
    logger.info("Saved results to %s", out_path)

    logger.info(
        "Summary (vs baseline): kv_only=%.6f resid_only=%.6f hybrid=%.6f [logit_abs_mean]",
        summary["kv_only"]["logit_vs_base_abs_mean"],
        summary["resid_only"]["logit_vs_base_abs_mean"],
        summary["hybrid"]["logit_vs_base_abs_mean"],
    )
    logger.info(
        "Summary (vs kv_only): resid_only=%.6f hybrid=%.6f [logit_abs_mean]",
        summary["resid_only"]["logit_vs_kv_abs_mean"],
        summary["hybrid"]["logit_vs_kv_abs_mean"],
    )
    logger.info(
        "Top1 match vs kv_only: resid_only=%.4f hybrid=%.4f",
        summary["resid_only"]["top1_match_vs_kv"],
        summary["hybrid"]["top1_match_vs_kv"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
