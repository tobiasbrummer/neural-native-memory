#!/usr/bin/env python3
"""Experiment 11 (NNM): Compare factorized storage strategies for R/I states.

Compares three token-level memory strategies:
1) Store R + I (full retrieval-layer and injection-layer states)
2) Store d_retr + d_inj + token_ids, with:
     d_retr = R - E_t, d_inj = I - R
3) Store d_retr + token_ids, reconstruct R, then forward to injection layer

Where:
- E_t = static token embeddings looked up via token ids
- R = token-level residual state at retrieval layer (resid_post)
- I = token-level residual state at injection layer (resid_pre)
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
        description="NNM Exp11: factorized storage strategies (TransformerLens)"
    )
    parser.add_argument("--dataset", type=str, default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prefix-bias", type=float, default=1.0)

    parser.add_argument("--retrieval-layer", type=int, default=26)
    parser.add_argument("--injection-layer", type=int, default=18)

    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--max-corpus", type=int, default=2000)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--quant-modes",
        type=str,
        default=None,
        help="Comma-separated quantization modes, e.g. fp16,int8,int4. "
        "If omitted, uses --storage-dtype as single mode.",
    )
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "int8", "int4"],
        help="Backward-compatible single quant mode if --quant-modes is not set.",
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


def _mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: [T, d]
    an = a / torch.clamp(torch.norm(a, dim=-1, keepdim=True), min=1e-10)
    bn = b / torch.clamp(torch.norm(b, dim=-1, keepdim=True), min=1e-10)
    return float(torch.mean(torch.sum(an * bn, dim=-1)).item())


def _parse_quant_modes(args: argparse.Namespace) -> List[str]:
    if args.quant_modes:
        modes = [m.strip().lower() for m in args.quant_modes.split(",") if m.strip()]
    else:
        single = str(args.storage_dtype).lower()
        mapping = {"float16": "fp16", "float32": "fp32", "int8": "int8", "int4": "int4"}
        modes = [mapping.get(single, single)]
    allowed = {"fp32", "fp16", "int8", "int4"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported quant modes: {bad}. Allowed: {sorted(allowed)}")
    dedup: List[str] = []
    for m in modes:
        if m not in dedup:
            dedup.append(m)
    return dedup


def _quantize_roundtrip_rowwise(x: torch.Tensor, mode: str) -> tuple[torch.Tensor, int]:
    """Quantize-dequantize tensor and return (reconstructed_tensor, storage_bytes)."""
    x = x.to(torch.float32)
    numel = int(x.numel())
    if mode == "fp32":
        return x, numel * np.dtype(np.float32).itemsize
    if mode == "fp16":
        return x.to(torch.float16).to(torch.float32), numel * np.dtype(np.float16).itemsize

    # Row-wise symmetric quantization over last dimension.
    # For tensors [T, d], this gives one scale per token vector.
    max_abs = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    if mode == "int8":
        qmax = 127.0
        scale = torch.where(max_abs > 1e-12, max_abs / qmax, torch.ones_like(max_abs))
        q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        x_hat = q.to(torch.float32) * scale
        bytes_q = numel  # int8
        bytes_scale = int(scale.numel()) * np.dtype(np.float16).itemsize
        return x_hat, int(bytes_q + bytes_scale)

    if mode == "int4":
        qmax = 7.0
        scale = torch.where(max_abs > 1e-12, max_abs / qmax, torch.ones_like(max_abs))
        q = torch.round(x / scale).clamp(-7, 7).to(torch.int8)
        x_hat = q.to(torch.float32) * scale
        bytes_q = (numel + 1) // 2  # packed int4
        bytes_scale = int(scale.numel()) * np.dtype(np.float16).itemsize
        return x_hat, int(bytes_q + bytes_scale)

    raise ValueError(f"Unsupported quant mode: {mode}")


def main() -> int:
    args = parse_args()
    results_dir = create_results_dir("nnm_exp11")
    logger = setup_logging("nnm_exp11_factorized_storage_tl", results_dir)
    progress_stdout = not bool(args.no_progress_stdout)
    quant_modes = _parse_quant_modes(args)

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
    retrieval_layer = int(args.retrieval_layer)
    injection_layer = int(args.injection_layer)
    if retrieval_layer < 0 or retrieval_layer >= n_layers:
        raise ValueError(f"Invalid retrieval layer: {retrieval_layer} for n_layers={n_layers}")
    if injection_layer < 0 or injection_layer >= n_layers:
        raise ValueError(f"Invalid injection layer: {injection_layer} for n_layers={n_layers}")
    strategy_c_available = injection_layer > retrieval_layer
    if not strategy_c_available:
        logger.warning(
            "Strategy C disabled: injection_layer (%d) <= retrieval_layer (%d). "
            "Forward reconstruction from retrieval->injection is only possible for injection_layer > retrieval_layer.",
            injection_layer,
            retrieval_layer,
        )

    retr_key = f"blocks.{retrieval_layer}.hook_resid_post"
    inj_resid_key = f"blocks.{injection_layer}.hook_resid_pre"
    inj_k_key = f"blocks.{injection_layer}.attn.hook_k"
    inj_v_key = f"blocks.{injection_layer}.attn.hook_v"
    names = {retr_key, inj_resid_key, inj_k_key, inj_v_key}

    int32_size = np.dtype(np.int32).itemsize

    strategies = ["ri_full", "dr_di_tokenid"]
    if strategy_c_available:
        strategies.append("dr_tokenid_forward")
    metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        qmode: {
            s: {
                "storage_bytes": [],
                "i_l2_mean": [],
                "i_l2_max": [],
                "i_cosine_mean": [],
                "kv_k_abs_mean": [],
                "kv_k_abs_max": [],
                "kv_v_abs_mean": [],
                "kv_v_abs_max": [],
                "logit_recon_true_abs_mean": [],
                "logit_recon_true_abs_max": [],
                "top1_recon_true_match": [],
                "top5_jaccard_recon_true": [],
                "extra_forward_ms": [],
            }
            for s in strategies
        }
        for qmode in quant_modes
    }

    attn_hooks = embedder._build_attention_bias_hooks([injection_layer])
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
        token_ids_np = token_ids.squeeze(0).detach().cpu().numpy().astype(np.int32)
        T = int(token_ids.shape[1])
        d_model = int(model.cfg.d_model)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                token_ids,
                return_type=None,
                prepend_bos=False,
                names_filter=lambda name: name in names,
                remove_batch_dim=False,
            )

        R_true = cache[retr_key][0].detach().to(torch.float32)  # [T, d]
        I_true = cache[inj_resid_key][0].detach().to(torch.float32)  # [T, d]
        k_true = cache[inj_k_key][:, -1:, :, :].detach()
        v_true = cache[inj_v_key][:, -1:, :, :].detach()
        del cache

        E_t = model.W_E[token_ids[0]].detach().to(torch.float32)  # [T, d]
        d_retr = R_true - E_t
        d_inj = I_true - R_true

        # Run true injection logits once for this pair.
        query_tokens = model.to_tokens(query_prompt, prepend_bos=prepend_bos)
        true_cache = embedder.build_prefix_cache_from_kv({injection_layer: (k_true, v_true)}, [injection_layer])
        with torch.no_grad():
            with model.hooks(fwd_hooks=attn_hooks):
                logits_true, _ = model.run_with_cache(
                    query_tokens,
                    return_type="logits",
                    past_kv_cache=true_cache,
                    prepend_bos=False,
                    remove_batch_dim=False,
                )
        true_last = logits_true[:, -1].detach().to(torch.float32)
        _, top5_true = torch.topk(true_last, k=min(5, int(true_last.shape[-1])), dim=-1)
        set_true = set(int(x) for x in top5_true[0].tolist())
        top1_true = int(torch.argmax(true_last, dim=-1).item())

        for qmode in quant_modes:
            # Strategy A: store R + I
            R_a, bytes_r = _quantize_roundtrip_rowwise(R_true, qmode)
            I_a, bytes_i = _quantize_roundtrip_rowwise(I_true, qmode)
            bytes_a = int(bytes_r + bytes_i)

            # Strategy B: store d_retr + d_inj + token_ids
            dr_b, bytes_dr = _quantize_roundtrip_rowwise(d_retr, qmode)
            di_b, bytes_di = _quantize_roundtrip_rowwise(d_inj, qmode)
            R_b = E_t + dr_b
            I_b = R_b + di_b
            bytes_b = int(bytes_dr + bytes_di + T * int32_size)

            per_strategy: Dict[str, Tuple[torch.Tensor, torch.Tensor, int, float]] = {
                "ri_full": (R_a, I_a, bytes_a, 0.0),
                "dr_di_tokenid": (R_b, I_b, bytes_b, 0.0),
            }
            if strategy_c_available:
                # Strategy C: store d_retr + token_ids, then forward retrieval->injection
                dr_c, bytes_drc = _quantize_roundtrip_rowwise(d_retr, qmode)
                R_c = E_t + dr_c
                forward_t0 = time.perf_counter()
                with torch.no_grad():
                    I_c = model.forward(
                        R_c.unsqueeze(0).to(device=model.W_E.device, dtype=model.W_E.dtype),
                        return_type=None,
                        prepend_bos=False,
                        start_at_layer=retrieval_layer + 1,
                        stop_at_layer=injection_layer,
                    )
                I_c = I_c[0].detach().to(torch.float32)
                forward_ms = (time.perf_counter() - forward_t0) * 1000.0
                bytes_c = int(bytes_drc + T * int32_size)
                per_strategy["dr_tokenid_forward"] = (R_c, I_c, bytes_c, float(forward_ms))

            for name, (_R_hat, I_hat, storage_bytes, extra_ms) in per_strategy.items():
                m = metrics[qmode][name]
                m["storage_bytes"].append(float(storage_bytes))
                m["extra_forward_ms"].append(float(extra_ms))

                diff_i = (I_hat - I_true).detach()
                l2_i = torch.norm(diff_i, dim=-1)
                m["i_l2_mean"].append(float(torch.mean(l2_i).item()))
                m["i_l2_max"].append(float(torch.max(l2_i).item()))
                m["i_cosine_mean"].append(_mean_cosine(I_hat, I_true))

                I_last = I_hat[-1].view(1, 1, -1).to(device=model.W_E.device, dtype=torch.float32)
                k_hat, v_hat = embedder._compute_kv_from_resid_pre(injection_layer, I_last)

                dk = (k_hat - k_true).detach().to(torch.float32).abs()
                dv = (v_hat - v_true).detach().to(torch.float32).abs()
                m["kv_k_abs_mean"].append(float(torch.mean(dk).item()))
                m["kv_k_abs_max"].append(float(torch.max(dk).item()))
                m["kv_v_abs_mean"].append(float(torch.mean(dv).item()))
                m["kv_v_abs_max"].append(float(torch.max(dv).item()))

                recon_cache = embedder.build_prefix_cache_from_kv(
                    {injection_layer: (k_hat, v_hat)},
                    [injection_layer],
                )
                with torch.no_grad():
                    with model.hooks(fwd_hooks=attn_hooks):
                        logits_recon, _ = model.run_with_cache(
                            query_tokens,
                            return_type="logits",
                            past_kv_cache=recon_cache,
                            prepend_bos=False,
                            remove_batch_dim=False,
                        )
                recon_last = logits_recon[:, -1].detach().to(torch.float32)
                dlogit = (recon_last - true_last).detach().abs()
                m["logit_recon_true_abs_mean"].append(float(torch.mean(dlogit).item()))
                m["logit_recon_true_abs_max"].append(float(torch.max(dlogit).item()))

                top1_recon = int(torch.argmax(recon_last, dim=-1).item())
                m["top1_recon_true_match"].append(float(1.0 if top1_recon == top1_true else 0.0))

                _, top5_recon = torch.topk(recon_last, k=min(5, int(recon_last.shape[-1])), dim=-1)
                set_recon = set(int(x) for x in top5_recon[0].tolist())
                inter = len(set_true & set_recon)
                union = len(set_true | set_recon)
                m["top5_jaccard_recon_true"].append(float(inter / union) if union > 0 else 0.0)

        if idx == 1 or idx % progress_every == 0 or idx == len(pairs):
            _log_progress(
                logger,
                "Exp11 pair loop",
                idx,
                len(pairs),
                t0,
                progress_stdout=progress_stdout,
            )

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for qmode in quant_modes:
        summary[qmode] = {}
        baseline_vals = metrics[qmode]["ri_full"]["storage_bytes"]
        baseline_bytes = float(np.mean(baseline_vals)) if baseline_vals else 0.0
        for name in strategies:
            out: Dict[str, float] = {}
            for key, values in metrics[qmode][name].items():
                out[key] = float(np.mean(values)) if values else 0.0
            if baseline_bytes > 0:
                out["compression_vs_ri_full"] = float(baseline_bytes / max(out["storage_bytes"], 1e-9))
            else:
                out["compression_vs_ri_full"] = 0.0
            summary[qmode][name] = out

    output = {
        "experiment": "nnm_exp11_factorized_storage_strategies_tl",
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "quant_modes": quant_modes,
            "storage_dtype_fallback": args.storage_dtype,
            "prefix_bias": args.prefix_bias,
            "seed": args.seed,
            "max_queries": args.max_queries,
            "max_corpus": args.max_corpus,
            "max_pairs": args.max_pairs,
            "retrieval_layer": retrieval_layer,
            "injection_layer": injection_layer,
            "progress_every": args.progress_every,
            "progress_stdout": progress_stdout,
        },
        "subset": {
            "n_corpus": len(subset.corpus_ids),
            "n_queries": len(subset.query_ids),
            "n_eval_pairs": len(pairs),
        },
        "results": summary,
        "strategy_c_available": strategy_c_available,
        "strategy_c_skip_reason": (
            None
            if strategy_c_available
            else "injection_layer <= retrieval_layer"
        ),
        "methodology_hint": {
            "note": (
                "I_t is not projected by only K/V weights in isolation; in TransformerLens path "
                "it goes through layer norm and attention projection logic "
                "(_compute_kv_from_resid_pre)."
            )
        },
    }

    out_path = results_dir / "results.json"
    save_json(output, out_path)
    logger.info("Saved results to %s", out_path)

    # Human-readable one-liner summary
    for qmode in quant_modes:
        if strategy_c_available:
            logger.info(
                "[%s] Mean storage bytes: R+I=%.1f, dR+dI+tid=%.1f, dR+tid+fwd=%.1f",
                qmode,
                summary[qmode]["ri_full"]["storage_bytes"],
                summary[qmode]["dr_di_tokenid"]["storage_bytes"],
                summary[qmode]["dr_tokenid_forward"]["storage_bytes"],
            )
        else:
            logger.info(
                "[%s] Mean storage bytes: R+I=%.1f, dR+dI+tid=%.1f (strategy C skipped)",
                qmode,
                summary[qmode]["ri_full"]["storage_bytes"],
                summary[qmode]["dr_di_tokenid"]["storage_bytes"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
