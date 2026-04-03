#!/usr/bin/env python3
"""Experiment 2 (NNM): Token-id reconstruction from static embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.io_utils import create_results_dir, load_test_documents, save_json, setup_logging
from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NNM Exp2: Token decode (TransformerLens)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--id-corpus", type=str, default=None)
    parser.add_argument("--prefix-bias", type=float, default=1.0)
    parser.add_argument("--success-threshold", type=float, default=0.95)
    return parser.parse_args()


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _nearest_ids(
    queries: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-10)
    m = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-10)
    sims = q @ m.T
    idx = np.argmax(sims, axis=1)
    best = sims[np.arange(len(idx)), idx]
    return idx.astype(np.int64), best.astype(np.float32)


def main() -> int:
    args = parse_args()

    results_dir = create_results_dir("nnm_exp2")
    logger = setup_logging("nnm_exp2_token_decode_tl", results_dir)

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
        return_token_embeddings=False,
        return_token_deltas=False,
    )

    embed_matrix = embedder.model.W_E.detach().to(torch.float32).cpu().numpy()

    total_tokens = 0
    total_correct = 0
    per_doc = []
    all_mismatches = []

    for i, doc in enumerate(documents):
        token_ids = np.asarray(result["token_ids"][i], dtype=np.int64)
        static_embeddings = np.asarray(result["static_embeddings"][i], dtype=np.float32)

        pred_ids, similarities = _nearest_ids(static_embeddings, embed_matrix)
        matches = token_ids == pred_ids
        accuracy = float(np.mean(matches)) if len(matches) > 0 else 0.0

        mismatches = []
        for pos, ok in enumerate(matches):
            if not ok:
                mm = {
                    "position": int(pos),
                    "original": int(token_ids[pos]),
                    "reconstructed": int(pred_ids[pos]),
                }
                mismatches.append(mm)
                all_mismatches.append({"doc_id": doc["doc_id"], **mm})

        total_tokens += int(len(token_ids))
        total_correct += int(np.sum(matches))

        per_doc.append(
            {
                "doc_id": doc["doc_id"],
                "n_tokens": int(len(token_ids)),
                "accuracy": accuracy,
                "mismatch_count": int(len(mismatches)),
                "mismatches_preview": mismatches[:10],
                "mean_similarity": float(np.mean(similarities)) if len(similarities) > 0 else 0.0,
            }
        )

    overall_accuracy = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    success = overall_accuracy >= args.success_threshold

    logger.info("Selected layers: %s", layer_selection.selected_layers)
    logger.info("Total tokens: %d", total_tokens)
    logger.info("Overall accuracy: %.4f", overall_accuracy)
    logger.info("SUCCESS: %s", success)

    output = {
        "experiment": "nnm_exp2_token_decode_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "prefix_bias": args.prefix_bias,
            "success_threshold": args.success_threshold,
        },
        "selected_layers": layer_selection.selected_layers,
        "used_u_shape_mode": layer_selection.used_u_shape_mode,
        "metrics": {
            "total_tokens": total_tokens,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
        },
        "documents": per_doc,
        "all_mismatches_preview": all_mismatches[:100],
        "success": success,
    }

    out_path = results_dir / "results.json"
    save_json(output, out_path)
    logger.info("Saved results to %s", out_path)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
