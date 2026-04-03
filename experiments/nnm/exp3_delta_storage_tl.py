#!/usr/bin/env python3
"""Experiment 3 (NNM): token-level delta storage and compression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.compression import (
    compute_compression_ratio,
    compute_reconstruction_metrics,
    product_dequantize,
    product_quantize,
    product_quantize_fit,
    residual_vq,
    residual_vq_dequantize,
    residual_vq_fit,
    scalar_dequantize,
    scalar_quantize,
    simhash,
    simhash_dequantize,
    simhash_fit,
)
from lib.io_utils import create_results_dir, load_test_documents, save_json, setup_logging
from nnm.experiments.common import recall_at_k
from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NNM Exp3: Delta storage (TransformerLens)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--id-corpus", type=str, default=None)
    parser.add_argument("--prefix-bias", type=float, default=1.0)
    parser.add_argument("--quality-threshold", type=float, default=0.95)
    return parser.parse_args()


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> int:
    args = parse_args()

    results_dir = create_results_dir("nnm_exp3")
    logger = setup_logging("nnm_exp3_delta_storage_tl", results_dir)

    documents = (
        load_test_documents(Path(args.data_file))
        if args.data_file
        else load_test_documents()
    )
    texts = [doc["text"] for doc in documents]
    id_texts = _read_lines(Path(args.id_corpus)) if args.id_corpus else texts

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        prefix_bias=args.prefix_bias,
    )
    embedder = TransformerLensKVEmbedder(config)
    layer_selection = embedder.select_layers(id_texts)
    result = embedder.extract_embeddings(
        texts=texts,
        roles=["context"] * len(texts),
        return_token_embeddings=True,
        return_token_deltas=True,
    )

    static_list = [np.asarray(x, dtype=np.float32) for x in result["static_embeddings"]]
    contextual_list = [np.asarray(x, dtype=np.float32) for x in result["token_embeddings"]]
    delta_list = [np.asarray(x, dtype=np.float32) for x in result["token_deltas"]]

    # Part 3a
    token_alignment_ok = True
    recon_errors = []
    for static, contextual, delta in zip(static_list, contextual_list, delta_list):
        rec = static + delta
        err = np.linalg.norm(rec - contextual, axis=1)
        recon_errors.append(float(np.mean(err)))

    avg_recon_error = float(np.mean(recon_errors)) if recon_errors else 0.0
    max_recon_error = float(np.max(recon_errors)) if recon_errors else 0.0
    part3a_success = token_alignment_ok and max_recon_error < 1e-4

    all_static = np.vstack(static_list)
    all_contextual = np.vstack(contextual_list)
    all_delta = np.vstack(delta_list)

    original_shape = all_delta.shape
    original_size = all_delta.nbytes
    total_tokens = int(all_delta.shape[0])

    logger.info("Selected layers: %s", layer_selection.selected_layers)
    logger.info("Token-level shape: %s", original_shape)
    logger.info("Part 3a avg L2 error: %.2e", avg_recon_error)
    logger.info("Part 3a max L2 error: %.2e", max_recon_error)

    # Part 3b
    compression_results = []

    # FP16 baseline
    fp16 = all_delta.astype(np.float16)
    fp16_delta = fp16.astype(np.float32)
    fp16_recon = all_static + fp16_delta
    fp16_metrics = compute_reconstruction_metrics(all_contextual, fp16_recon)
    fp16_ratio = compute_compression_ratio(original_shape, fp16.nbytes)
    fp16_recall = recall_at_k(all_contextual, fp16_recon, k=min(10, total_tokens - 1))
    compression_results.append(
        {
            "method": "fp16",
            "compression_ratio": fp16_ratio,
            "compressed_size_kb": fp16.nbytes / 1024.0,
            **fp16_metrics,
            "recall_at_k": fp16_recall,
        }
    )

    # Scalar quantization
    for bits in (8, 4, 2):
        q, params = scalar_quantize(all_delta, bits=bits)
        size = q.nbytes + 16
        dq = scalar_dequantize(q, params, original_shape)
        recon = all_static + dq
        metrics = compute_reconstruction_metrics(all_contextual, recon)
        ratio = compute_compression_ratio(original_shape, size)
        recall = recall_at_k(all_contextual, recon, k=min(10, total_tokens - 1))
        compression_results.append(
            {
                "method": f"int{bits}",
                "compression_ratio": ratio,
                "compressed_size_kb": size / 1024.0,
                **metrics,
                "recall_at_k": recall,
            }
        )

    # Product quantization
    try:
        n_subvectors = max(1, min(8, all_delta.shape[1] // 32))
        usable_dim = (all_delta.shape[1] // n_subvectors) * n_subvectors
        d = all_delta[:, :usable_dim]
        s = all_static[:, :usable_dim]
        c = all_contextual[:, :usable_dim]
        n_centroids = max(8, min(256, total_tokens // 2))
        pq_params = product_quantize_fit(d, n_subvectors=n_subvectors, n_centroids=n_centroids)
        codes = product_quantize(d, pq_params)
        dq = product_dequantize(codes, pq_params)
        recon = s + dq
        size = pq_params.codebooks.nbytes + codes.nbytes
        metrics = compute_reconstruction_metrics(c, recon)
        ratio = original_size / size
        recall = recall_at_k(c, recon, k=min(10, total_tokens - 1))
        compression_results.append(
            {
                "method": f"pq_{n_subvectors}x{n_centroids}",
                "compression_ratio": ratio,
                "compressed_size_kb": size / 1024.0,
                **metrics,
                "recall_at_k": recall,
            }
        )
    except Exception as err:
        logger.warning("PQ failed: %s", err)

    # RVQ
    try:
        n_centroids = max(8, min(256, total_tokens // 2))
        rvq_params = residual_vq_fit(all_delta, n_stages=4, n_centroids=n_centroids)
        codes = residual_vq(all_delta, rvq_params)
        dq = residual_vq_dequantize(codes, rvq_params)
        recon = all_static + dq
        size = sum(cb.nbytes for cb in rvq_params.codebooks) + codes.nbytes
        metrics = compute_reconstruction_metrics(all_contextual, recon)
        ratio = compute_compression_ratio(original_shape, size)
        recall = recall_at_k(all_contextual, recon, k=min(10, total_tokens - 1))
        compression_results.append(
            {
                "method": f"rvq_4x{n_centroids}",
                "compression_ratio": ratio,
                "compressed_size_kb": size / 1024.0,
                **metrics,
                "recall_at_k": recall,
            }
        )
    except Exception as err:
        logger.warning("RVQ failed: %s", err)

    # SimHash
    try:
        params = simhash_fit(all_delta.shape[1], n_bits=256)
        codes = simhash(all_delta, params)
        dq = simhash_dequantize(codes, params)
        recon = all_static + dq
        metrics = compute_reconstruction_metrics(all_contextual, recon)
        ratio = compute_compression_ratio(original_shape, codes.nbytes)
        recall = recall_at_k(all_contextual, recon, k=min(10, total_tokens - 1))
        compression_results.append(
            {
                "method": "simhash_256",
                "compression_ratio": ratio,
                "compressed_size_kb": codes.nbytes / 1024.0,
                **metrics,
                "recall_at_k": recall,
            }
        )
    except Exception as err:
        logger.warning("SimHash failed: %s", err)

    best_result = None
    for item in compression_results:
        compression_ok = item["compression_ratio"] >= 2.0
        quality_ok = (
            item["cosine_similarity"] >= args.quality_threshold
            or item["recall_at_k"] >= args.quality_threshold
        )
        if compression_ok and quality_ok:
            if best_result is None or item["compression_ratio"] > best_result["compression_ratio"]:
                best_result = item

    part3b_success = best_result is not None
    overall_success = part3a_success and part3b_success

    logger.info("Part 3a success: %s", part3a_success)
    logger.info("Part 3b success: %s", part3b_success)
    logger.info("Overall success: %s", overall_success)
    if best_result:
        logger.info("Best compression method: %s", best_result["method"])

    output = {
        "experiment": "nnm_exp3_delta_storage_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "prefix_bias": args.prefix_bias,
            "quality_threshold": args.quality_threshold,
        },
        "selected_layers": layer_selection.selected_layers,
        "used_u_shape_mode": layer_selection.used_u_shape_mode,
        "n_documents": len(documents),
        "total_tokens": total_tokens,
        "part3a": {
            "avg_reconstruction_error": avg_recon_error,
            "max_reconstruction_error": max_recon_error,
            "success": part3a_success,
        },
        "part3b": {
            "original_shape": list(original_shape),
            "original_size_kb": original_size / 1024.0,
            "compression_results": compression_results,
            "best_result": best_result,
            "success": part3b_success,
        },
        "overall_success": overall_success,
    }
    out_path = results_dir / "results.json"
    save_json(output, out_path)
    logger.info("Saved results to %s", out_path)
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
