#!/usr/bin/env python3
"""Small CLI to run paper-near KV-Embedding with TransformerLens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-near KV-Embedding (TransformerLens backend)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--prefix-bias", type=float, default=1.0)
    parser.add_argument("--id-corpus", type=str, default=None, help="Text file for ID layer selection")

    parser.add_argument("--text", action="append", default=[], help="Input text (repeatable)")
    parser.add_argument("--text-file", type=str, default=None, help="Text file with one input per line")

    parser.add_argument("--save-npz", type=str, default=None, help="Optional output .npz")
    parser.add_argument("--save-meta", type=str, default=None, help="Optional output metadata JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    texts: List[str] = list(args.text)
    if args.text_file:
        texts.extend(_read_lines(Path(args.text_file)))
    if not texts:
        texts = [
            "Die Katze sitzt auf der Fensterbank und beobachtet den Regen.",
            "Maschinelles Lernen verbessert semantische Suche in Wissenssystemen.",
        ]

    id_texts = _read_lines(Path(args.id_corpus)) if args.id_corpus else texts

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        prefix_bias=args.prefix_bias,
    )
    embedder = TransformerLensKVEmbedder(config)
    selection = embedder.select_layers(id_texts)

    result = embedder.extract_embeddings(
        texts=texts,
        roles=["context"] * len(texts),
        return_token_embeddings=True,
        return_token_deltas=True,
    )

    pooled = result["pooled_embeddings"]
    assert isinstance(pooled, np.ndarray)

    print("Model:", args.model)
    print("Selected layers:", selection.selected_layers)
    print("Use u-shape mode:", selection.used_u_shape_mode)
    print("Pooled shape:", tuple(pooled.shape))
    print("Num texts:", len(texts))

    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            pooled_embeddings=result["pooled_embeddings"],
            token_ids=np.array(result["token_ids"], dtype=object),
            token_embeddings=np.array(result["token_embeddings"], dtype=object),
            token_deltas=np.array(result["token_deltas"], dtype=object),
            static_embeddings=np.array(result["static_embeddings"], dtype=object),
        )
        print("Saved arrays to:", args.save_npz)

    if args.save_meta:
        metadata = {
            "model_name": result["model_name"],
            "selected_layers": result["selected_layers"],
            "layer_selection": {
                "used_u_shape_mode": selection.used_u_shape_mode,
                "id_by_layer": selection.id_by_layer,
            },
            "n_texts": len(texts),
        }
        with Path(args.save_meta).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print("Saved metadata to:", args.save_meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
