#!/usr/bin/env python3
"""Experiment 1 (NNM): Contextual embedding quality with TransformerLens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.io_utils import create_results_dir, load_test_documents, save_json, setup_logging
from nnm.experiments.common import cosine_similarity_matrix, split_intra_inter_by_category
from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NNM Exp1: Contextual embeddings (TransformerLens)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--id-corpus", type=str, default=None)
    parser.add_argument("--prefix-bias", type=float, default=1.0)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    return parser.parse_args()


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> int:
    args = parse_args()

    results_dir = create_results_dir("nnm_exp1")
    logger = setup_logging("nnm_exp1_contextual_embeddings_tl", results_dir)

    documents = (
        load_test_documents(Path(args.data_file))
        if args.data_file
        else load_test_documents()
    )
    texts = [doc["text"] for doc in documents]
    categories = [doc.get("category", "unknown") for doc in documents]
    doc_ids = [doc["doc_id"] for doc in documents]

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
    pooled = result["pooled_embeddings"]
    assert isinstance(pooled, np.ndarray)

    sim = cosine_similarity_matrix(pooled)
    intra, inter = split_intra_inter_by_category(sim, categories)
    avg_intra = float(np.mean(intra)) if intra else 0.0
    avg_inter = float(np.mean(inter)) if inter else 0.0
    margin = avg_intra - avg_inter
    success = margin > args.margin_threshold

    logger.info("Selected layers: %s", layer_selection.selected_layers)
    logger.info("Avg intra similarity: %.4f", avg_intra)
    logger.info("Avg inter similarity: %.4f", avg_inter)
    logger.info("Margin: %.4f", margin)
    logger.info("SUCCESS: %s", success)

    output = {
        "experiment": "nnm_exp1_contextual_embeddings_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "prefix_bias": args.prefix_bias,
            "margin_threshold": args.margin_threshold,
        },
        "selected_layers": layer_selection.selected_layers,
        "used_u_shape_mode": layer_selection.used_u_shape_mode,
        "id_by_layer": layer_selection.id_by_layer,
        "documents": [
            {
                "doc_id": doc["doc_id"],
                "category": doc.get("category", "unknown"),
                "text": doc["text"],
            }
            for doc in documents
        ],
        "similarity_matrix": sim.tolist(),
        "metrics": {
            "avg_intra_category_similarity": avg_intra,
            "avg_inter_category_similarity": avg_inter,
            "separation_margin": margin,
        },
        "success": success,
    }
    out_path = results_dir / "results.json"
    save_json(output, out_path)
    logger.info("Saved results to %s", out_path)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
