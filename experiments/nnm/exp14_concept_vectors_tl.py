#!/usr/bin/env python3
"""Experiment 14 (NNM): Concept Vector extraction via Contrast Pairs (TransformerLens).

Tests:
  A) Extract concept vectors from minimal contrast pairs (e.g. "Samstag" vs "Dienstag")
  B) Validate arithmetic: sentence_with_A - v_concept ~ sentence_with_B
  C) Test orthogonality between different concept vectors
  D) Test combined concept arithmetic (multiple concepts)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder


# ---------------------------------------------------------------------------
# Minimal utilities (self-contained, no lib dependency)
# ---------------------------------------------------------------------------

def _setup_logging(name: str, results_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(results_dir / "log.txt")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _create_results_dir(prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("results") / f"{prefix}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(data: object, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / max(np.linalg.norm(a), 1e-10)
    b_n = b / max(np.linalg.norm(b), 1e-10)
    return float(np.dot(a_n, b_n))


def _l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Concept Vector definitions
# ---------------------------------------------------------------------------

@dataclass
class ContrastPair:
    """A minimal contrast pair for concept vector extraction."""
    concept_name: str
    positive: str          # word/phrase representing the concept
    negative: str          # word/phrase WITHOUT the concept
    description: str       # human-readable description


@dataclass
class ValidationSentence:
    """A sentence pair for validating concept vector arithmetic."""
    concept_name: str
    sentence_pos: str      # sentence containing the positive concept
    sentence_neg: str      # same sentence with negative concept
    description: str


# Default contrast pairs and validation sentences
DEFAULT_CONTRASTS: List[ContrastPair] = [
    ContrastPair(
        concept_name="wochenende",
        positive="Samstag",
        negative="Dienstag",
        description="Wochenende vs. Werktag",
    ),
    ContrastPair(
        concept_name="sommer",
        positive="Juli",
        negative="Januar",
        description="Sommer vs. Winter",
    ),
    ContrastPair(
        concept_name="monatsende",
        positive="am 28. des Monats",
        negative="am 3. des Monats",
        description="Ende vs. Anfang des Monats",
    ),
    ContrastPair(
        concept_name="abend",
        positive="um 20 Uhr abends",
        negative="um 8 Uhr morgens",
        description="Abend vs. Morgen",
    ),
]

DEFAULT_VALIDATIONS: List[ValidationSentence] = [
    # Wochenende
    ValidationSentence(
        concept_name="wochenende",
        sentence_pos="Die Muellabfuhr kommt jeden Samstag um 7 Uhr.",
        sentence_neg="Die Muellabfuhr kommt jeden Dienstag um 7 Uhr.",
        description="Muellabfuhr: Samstag vs. Dienstag",
    ),
    ValidationSentence(
        concept_name="wochenende",
        sentence_pos="Das Meeting findet am Samstag statt.",
        sentence_neg="Das Meeting findet am Dienstag statt.",
        description="Meeting: Samstag vs. Dienstag",
    ),
    # Sommer
    ValidationSentence(
        concept_name="sommer",
        sentence_pos="Die Konferenz ist im Juli geplant.",
        sentence_neg="Die Konferenz ist im Januar geplant.",
        description="Konferenz: Juli vs. Januar",
    ),
    ValidationSentence(
        concept_name="sommer",
        sentence_pos="Wir starten das Projekt im Juli.",
        sentence_neg="Wir starten das Projekt im Januar.",
        description="Projekt: Juli vs. Januar",
    ),
    # Monatsende
    ValidationSentence(
        concept_name="monatsende",
        sentence_pos="Die Rechnung wird am 28. des Monats faellig.",
        sentence_neg="Die Rechnung wird am 3. des Monats faellig.",
        description="Rechnung: 28. vs. 3.",
    ),
    # Abend
    ValidationSentence(
        concept_name="abend",
        sentence_pos="Der Termin ist um 20 Uhr abends.",
        sentence_neg="Der Termin ist um 8 Uhr morgens.",
        description="Termin: abends vs. morgens",
    ),
]


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------

def extract_pooled_delta(
    embedder: TransformerLensKVEmbedder,
    text: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract pooled contextual embedding, static embedding mean, and delta.

    Returns: (pooled_contextual, pooled_static, pooled_delta)
    """
    result = embedder.extract_embeddings(
        texts=[text],
        roles=["context"],
        return_token_embeddings=True,
        return_token_deltas=True,
    )
    contextual = np.asarray(result["token_embeddings"][0], dtype=np.float32)
    static = np.asarray(result["static_embeddings"][0], dtype=np.float32)
    delta = np.asarray(result["token_deltas"][0], dtype=np.float32)

    # Pooled: (last + mean) / 2 to match KV-Embedding paper
    pooled_ctx = (contextual[-1] + contextual.mean(axis=0)) / 2.0
    pooled_static = (static[-1] + static.mean(axis=0)) / 2.0
    pooled_delta = (delta[-1] + delta.mean(axis=0)) / 2.0

    return pooled_ctx, pooled_static, pooled_delta


def extract_concept_vector(
    embedder: TransformerLensKVEmbedder,
    pair: ContrastPair,
) -> Tuple[np.ndarray, dict]:
    """
    Extract concept vector from a contrast pair.

    concept_vector = delta(positive) - delta(negative)
    """
    _, _, delta_pos = extract_pooled_delta(embedder, pair.positive)
    _, _, delta_neg = extract_pooled_delta(embedder, pair.negative)

    concept_vec = delta_pos - delta_neg
    norm = float(np.linalg.norm(concept_vec))

    metrics = {
        "concept": pair.concept_name,
        "positive": pair.positive,
        "negative": pair.negative,
        "concept_vector_norm": norm,
        "delta_pos_norm": float(np.linalg.norm(delta_pos)),
        "delta_neg_norm": float(np.linalg.norm(delta_neg)),
        "delta_cosine_sim": _cosine_sim(delta_pos, delta_neg),
    }

    return concept_vec, metrics


def validate_concept_arithmetic(
    embedder: TransformerLensKVEmbedder,
    concept_vectors: Dict[str, np.ndarray],
    validation: ValidationSentence,
) -> dict:
    """
    Test: delta(sentence_pos) - concept_vector ~ delta(sentence_neg)?

    Measures:
    - cosine(delta_pos, delta_neg) -- baseline similarity (without arithmetic)
    - cosine(delta_pos - v_concept, delta_neg) -- after subtracting concept
    - If arithmetic works, the second value should be HIGHER.
    """
    v_concept = concept_vectors[validation.concept_name]

    _, _, delta_pos = extract_pooled_delta(embedder, validation.sentence_pos)
    _, _, delta_neg = extract_pooled_delta(embedder, validation.sentence_neg)

    # Baseline: how similar are pos and neg deltas without any arithmetic?
    baseline_cosine = _cosine_sim(delta_pos, delta_neg)
    baseline_l2 = _l2_dist(delta_pos, delta_neg)

    # After subtracting concept vector from positive
    adjusted_pos = delta_pos - v_concept
    adjusted_cosine = _cosine_sim(adjusted_pos, delta_neg)
    adjusted_l2 = _l2_dist(adjusted_pos, delta_neg)

    # Improvement: adjusted should be MORE similar (higher cosine, lower L2)
    cosine_improvement = adjusted_cosine - baseline_cosine
    l2_improvement = baseline_l2 - adjusted_l2  # positive = better

    return {
        "concept": validation.concept_name,
        "description": validation.description,
        "sentence_pos": validation.sentence_pos,
        "sentence_neg": validation.sentence_neg,
        "baseline_cosine": baseline_cosine,
        "adjusted_cosine": adjusted_cosine,
        "cosine_improvement": cosine_improvement,
        "baseline_l2": baseline_l2,
        "adjusted_l2": adjusted_l2,
        "l2_improvement": l2_improvement,
        "success": cosine_improvement > 0,
    }


def test_orthogonality(
    concept_vectors: Dict[str, np.ndarray],
) -> List[dict]:
    """Test pairwise orthogonality between concept vectors."""
    names = sorted(concept_vectors.keys())
    results = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            sim = _cosine_sim(concept_vectors[name_a], concept_vectors[name_b])
            results.append({
                "pair": f"{name_a} vs {name_b}",
                "cosine_similarity": sim,
                "approximately_orthogonal": abs(sim) < 0.3,
            })
    return results


def test_combined_arithmetic(
    embedder: TransformerLensKVEmbedder,
    concept_vectors: Dict[str, np.ndarray],
) -> List[dict]:
    """
    Test combining multiple concept vectors.

    E.g.: sentence with (Samstag + Juli) - v_wochenende - v_sommer ~ sentence with (Dienstag + Januar)
    """
    combined_tests = [
        {
            "description": "Wochenende + Sommer entfernen",
            "subtract": ["wochenende", "sommer"],
            "sentence_pos": "Das Grillfest ist am Samstag im Juli.",
            "sentence_neg": "Das Grillfest ist am Dienstag im Januar.",
        },
        {
            "description": "Abend + Wochenende entfernen",
            "subtract": ["abend", "wochenende"],
            "sentence_pos": "Die Party ist am Samstag um 20 Uhr abends.",
            "sentence_neg": "Die Party ist am Dienstag um 8 Uhr morgens.",
        },
    ]

    results = []
    for test in combined_tests:
        subtract_names = test["subtract"]
        # Check all needed concept vectors exist
        missing = [n for n in subtract_names if n not in concept_vectors]
        if missing:
            results.append({
                "description": test["description"],
                "skipped": True,
                "reason": f"Missing concept vectors: {missing}",
            })
            continue

        _, _, delta_pos = extract_pooled_delta(embedder, test["sentence_pos"])
        _, _, delta_neg = extract_pooled_delta(embedder, test["sentence_neg"])

        baseline_cosine = _cosine_sim(delta_pos, delta_neg)

        # Subtract all concept vectors
        adjusted_pos = delta_pos.copy()
        for name in subtract_names:
            adjusted_pos = adjusted_pos - concept_vectors[name]

        adjusted_cosine = _cosine_sim(adjusted_pos, delta_neg)
        cosine_improvement = adjusted_cosine - baseline_cosine

        results.append({
            "description": test["description"],
            "concepts_subtracted": subtract_names,
            "sentence_pos": test["sentence_pos"],
            "sentence_neg": test["sentence_neg"],
            "baseline_cosine": baseline_cosine,
            "adjusted_cosine": adjusted_cosine,
            "cosine_improvement": cosine_improvement,
            "success": cosine_improvement > 0,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp14: Concept Vector extraction via Contrast Pairs"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--id-corpus", type=str, default=None)
    parser.add_argument("--prefix-bias", type=float, default=1.0)
    parser.add_argument("--rerouting-layers", type=str, default=None,
                        help="Comma-separated layer indices (skip ID selection)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = _create_results_dir("nnm_exp14")
    logger = _setup_logging("nnm_exp14_concept_vectors_tl", results_dir)

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        load_in_4bit=args.load_in_4bit,
        prefix_bias=args.prefix_bias,
    )
    embedder = TransformerLensKVEmbedder(config)

    # Layer selection
    if args.rerouting_layers:
        layers = [int(x.strip()) for x in args.rerouting_layers.split(",")]
        from nnm.kvembed.layer_selection import LayerSelectionResult
        embedder.layer_selection = LayerSelectionResult(
            selected_layers=layers,
            id_by_layer={l: 0.0 for l in layers},
            used_u_shape_mode=False,
        )
        logger.info("Using manual rerouting layers: %s", layers)
    else:
        id_texts = []
        if args.id_corpus:
            with open(args.id_corpus, "r", encoding="utf-8") as f:
                id_texts = [line.strip() for line in f if line.strip()]
        if not id_texts:
            # Use contrast pair texts + validation sentences as ID corpus
            id_texts = [c.positive for c in DEFAULT_CONTRASTS]
            id_texts += [c.negative for c in DEFAULT_CONTRASTS]
            id_texts += [v.sentence_pos for v in DEFAULT_VALIDATIONS]
        layer_selection = embedder.select_layers(id_texts)
        logger.info("Selected layers: %s", layer_selection.selected_layers)

    # -----------------------------------------------------------------------
    # Part A: Extract concept vectors from contrast pairs
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART A: Concept Vector Extraction")
    logger.info("=" * 60)

    concept_vectors: Dict[str, np.ndarray] = {}
    extraction_metrics: List[dict] = []

    for pair in DEFAULT_CONTRASTS:
        logger.info("Extracting: %s (%s vs %s)", pair.concept_name, pair.positive, pair.negative)
        vec, metrics = extract_concept_vector(embedder, pair)
        concept_vectors[pair.concept_name] = vec
        extraction_metrics.append(metrics)
        logger.info(
            "  norm=%.4f, delta_cosine=%.4f",
            metrics["concept_vector_norm"],
            metrics["delta_cosine_sim"],
        )

    # -----------------------------------------------------------------------
    # Part B: Validate arithmetic (single concept)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART B: Single Concept Arithmetic Validation")
    logger.info("=" * 60)

    validation_results: List[dict] = []
    for val in DEFAULT_VALIDATIONS:
        if val.concept_name not in concept_vectors:
            logger.warning("Skipping %s: concept vector not available", val.description)
            continue

        result = validate_concept_arithmetic(embedder, concept_vectors, val)
        validation_results.append(result)

        status = "OK" if result["success"] else "FAIL"
        logger.info(
            "  [%s] %s: baseline=%.4f -> adjusted=%.4f (improvement=%.4f)",
            status,
            result["description"],
            result["baseline_cosine"],
            result["adjusted_cosine"],
            result["cosine_improvement"],
        )

    n_success = sum(1 for r in validation_results if r["success"])
    n_total = len(validation_results)
    part_b_success = n_success > n_total / 2  # majority should improve
    logger.info("Part B: %d/%d validations improved (success=%s)", n_success, n_total, part_b_success)

    # -----------------------------------------------------------------------
    # Part C: Orthogonality between concept vectors
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART C: Concept Vector Orthogonality")
    logger.info("=" * 60)

    orthogonality_results = test_orthogonality(concept_vectors)
    for r in orthogonality_results:
        status = "ORTH" if r["approximately_orthogonal"] else "CORR"
        logger.info("  [%s] %s: cosine=%.4f", status, r["pair"], r["cosine_similarity"])

    n_orthogonal = sum(1 for r in orthogonality_results if r["approximately_orthogonal"])
    part_c_info = f"{n_orthogonal}/{len(orthogonality_results)} approximately orthogonal"
    logger.info("Part C: %s", part_c_info)

    # -----------------------------------------------------------------------
    # Part D: Combined concept arithmetic
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART D: Combined Concept Arithmetic")
    logger.info("=" * 60)

    combined_results = test_combined_arithmetic(embedder, concept_vectors)
    for r in combined_results:
        if r.get("skipped"):
            logger.info("  [SKIP] %s: %s", r["description"], r["reason"])
            continue
        status = "OK" if r["success"] else "FAIL"
        logger.info(
            "  [%s] %s: baseline=%.4f -> adjusted=%.4f (improvement=%.4f)",
            status,
            r["description"],
            r["baseline_cosine"],
            r["adjusted_cosine"],
            r["cosine_improvement"],
        )

    n_combined_success = sum(1 for r in combined_results if r.get("success", False))
    n_combined_total = sum(1 for r in combined_results if not r.get("skipped", False))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    overall_success = part_b_success

    output = {
        "experiment": "nnm_exp14_concept_vectors_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "prefix_bias": args.prefix_bias,
            "load_in_4bit": args.load_in_4bit,
        },
        "selected_layers": (
            embedder.layer_selection.selected_layers
            if embedder.layer_selection else None
        ),
        "part_a_extraction": extraction_metrics,
        "part_b_single_arithmetic": {
            "results": validation_results,
            "n_success": n_success,
            "n_total": n_total,
            "success": part_b_success,
        },
        "part_c_orthogonality": {
            "results": orthogonality_results,
            "n_orthogonal": n_orthogonal,
            "n_total": len(orthogonality_results),
        },
        "part_d_combined_arithmetic": {
            "results": combined_results,
            "n_success": n_combined_success,
            "n_total": n_combined_total,
        },
        "overall_success": overall_success,
    }

    out_path = results_dir / "results.json"
    _save_json(output, out_path)
    logger.info("Saved results to %s", out_path)
    logger.info("Overall success: %s", overall_success)

    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
