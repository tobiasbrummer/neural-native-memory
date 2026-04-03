#!/usr/bin/env python3
"""Experiment 14b (NNM): Concept Vector arithmetic WITH whitening/normalization.

Repeats exp14 with four normalization modes to test whether whitening fixes
the failed arithmetic from exp14 (1/6 success on raw deltas).

Modes:
  raw       -- no normalization (baseline, same as exp14)
  zscore    -- centering + std normalization
  whitened  -- zscore + PCA whitening
  whitened_l2 -- zscore + PCA whitening + L2 normalization

For each mode:
  A) Fit normalization parameters on a diverse delta corpus
  B) Extract concept vectors in normalized space
  C) Test orthogonality
  D) Validate single-concept arithmetic (6 pairs)
  E) Validate combined arithmetic (2 pairs)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder


# ---------------------------------------------------------------------------
# Minimal utilities (self-contained)
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


# ---------------------------------------------------------------------------
# Inline whitening transform (simplified from storage/retrieval_transform.py)
# ---------------------------------------------------------------------------

@dataclass
class WhiteningTransform:
    mean: np.ndarray
    std: np.ndarray
    whitening_matrix: Optional[np.ndarray]
    eps: float = 1e-5

    def apply(self, x: np.ndarray, mode: str = "whitened_l2") -> np.ndarray:
        """Apply normalization.

        Modes:
          raw         -- return x unchanged
          zscore      -- center + std normalize
          whitened    -- zscore + PCA whitening
          whitened_l2 -- zscore + PCA whitening + L2 normalize
        """
        if mode == "raw":
            return x.copy()

        y = x.astype(np.float32, copy=True)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        # Center
        y = y - self.mean.reshape(1, -1)
        # Z-Score
        y = y / self.std.reshape(1, -1)

        if mode == "zscore":
            return y.squeeze()

        # PCA Whitening
        if self.whitening_matrix is not None:
            y = y @ self.whitening_matrix

        if mode == "whitened":
            return y.squeeze()

        # L2 normalize
        norms = np.linalg.norm(y, axis=1, keepdims=True)
        y = y / np.maximum(norms, self.eps)
        return y.squeeze()


def fit_whitening(deltas: np.ndarray, eps: float = 1e-5) -> WhiteningTransform:
    """Fit Z-Score + PCA Whitening transform on a matrix of delta vectors."""
    x = deltas.astype(np.float32)
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    centered = x - mean.reshape(1, -1)

    std = np.std(centered, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    base = centered / std.reshape(1, -1)

    n = max(1, int(base.shape[0] - 1))
    cov = (base.T @ base) / float(n)
    evals, evecs = np.linalg.eigh(cov.astype(np.float64))
    evals = np.maximum(evals, eps)
    inv_sqrt = (1.0 / np.sqrt(evals)).astype(np.float64)
    whitening_matrix = ((evecs * inv_sqrt) @ evecs.T).astype(np.float32)

    return WhiteningTransform(
        mean=mean,
        std=std,
        whitening_matrix=whitening_matrix,
        eps=eps,
    )


# ---------------------------------------------------------------------------
# Contrast pairs and validation sentences (same as exp14)
# ---------------------------------------------------------------------------

@dataclass
class ContrastPair:
    concept_name: str
    positive: str
    negative: str
    description: str


@dataclass
class ValidationSentence:
    concept_name: str
    sentence_pos: str
    sentence_neg: str
    description: str


DEFAULT_CONTRASTS: List[ContrastPair] = [
    ContrastPair("wochenende", "Samstag", "Dienstag", "Wochenende vs. Werktag"),
    ContrastPair("sommer", "Juli", "Januar", "Sommer vs. Winter"),
    ContrastPair("monatsende", "am 28. des Monats", "am 3. des Monats", "Ende vs. Anfang des Monats"),
    ContrastPair("abend", "um 20 Uhr abends", "um 8 Uhr morgens", "Abend vs. Morgen"),
]

DEFAULT_VALIDATIONS: List[ValidationSentence] = [
    ValidationSentence("wochenende", "Die Muellabfuhr kommt jeden Samstag um 7 Uhr.", "Die Muellabfuhr kommt jeden Dienstag um 7 Uhr.", "Muellabfuhr: Samstag vs. Dienstag"),
    ValidationSentence("wochenende", "Das Meeting findet am Samstag statt.", "Das Meeting findet am Dienstag statt.", "Meeting: Samstag vs. Dienstag"),
    ValidationSentence("sommer", "Die Konferenz ist im Juli geplant.", "Die Konferenz ist im Januar geplant.", "Konferenz: Juli vs. Januar"),
    ValidationSentence("sommer", "Wir starten das Projekt im Juli.", "Wir starten das Projekt im Januar.", "Projekt: Juli vs. Januar"),
    ValidationSentence("monatsende", "Die Rechnung wird am 28. des Monats faellig.", "Die Rechnung wird am 3. des Monats faellig.", "Rechnung: 28. vs. 3."),
    ValidationSentence("abend", "Der Termin ist um 20 Uhr abends.", "Der Termin ist um 8 Uhr morgens.", "Termin: abends vs. morgens"),
]

# ---------------------------------------------------------------------------
# BEIR corpus loading for whitening fit
# ---------------------------------------------------------------------------

def _safe_text(doc: dict) -> str:
    """Extract text from a BEIR corpus document."""
    title = str(doc.get("title", "") or "").strip()
    text = str(doc.get("text", "") or "").strip()
    joined = f"{title} {text}".strip()
    return joined if joined else text


def load_fitting_texts(
    beir_dataset: str,
    beir_path: Optional[str],
    max_samples: int,
    logger: logging.Logger,
) -> List[str]:
    """Load corpus texts for whitening fit.

    Priority:
      1. Local JSONL from --beir-path or data/beir_datasets/{dataset}/corpus.jsonl
      2. HuggingFace datasets library (auto-download)
      3. Fail with clear error (too few samples = broken PCA)
    """
    texts: List[str] = []

    # --- Try local JSONL ---
    if beir_path:
        local_jsonl = Path(beir_path)
    else:
        local_jsonl = REPO_ROOT / "data" / "beir_datasets" / beir_dataset / "corpus.jsonl"

    if local_jsonl.exists():
        logger.info("Loading fitting corpus from local JSONL: %s", local_jsonl)
        with local_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                t = _safe_text(doc)
                if t:
                    texts.append(t)
        logger.info("Loaded %d texts from local JSONL", len(texts))
    else:
        logger.info("Local JSONL not found at %s, trying HuggingFace datasets...", local_jsonl)

    # --- Try HuggingFace datasets ---
    if not texts:
        try:
            from datasets import load_dataset
            ds = load_dataset("BeIR/" + beir_dataset, "corpus", split="corpus")
            for row in ds:
                t = _safe_text(row)
                if t:
                    texts.append(t)
            logger.info("Loaded %d texts from HuggingFace BeIR/%s", len(texts), beir_dataset)
        except ImportError:
            logger.error("HuggingFace 'datasets' library not installed. "
                         "Install with: pip install datasets")
            raise
        except Exception as e:
            logger.error("Failed to load BeIR/%s from HuggingFace: %s", beir_dataset, e)
            raise

    if len(texts) < max_samples:
        logger.warning("Corpus has only %d texts (requested %d). Using all.", len(texts), max_samples)
    else:
        random.seed(42)
        texts = random.sample(texts, max_samples)
        logger.info("Sampled %d texts for fitting corpus", max_samples)

    return texts


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------

def extract_pooled_delta(
    embedder: TransformerLensKVEmbedder,
    text: str,
) -> np.ndarray:
    """Extract pooled delta for a text. Returns pooled_delta."""
    result = embedder.extract_embeddings(
        texts=[text],
        roles=["context"],
        return_token_embeddings=True,
        return_token_deltas=True,
    )
    delta = np.asarray(result["token_deltas"][0], dtype=np.float32)
    # Pool: (last + mean) / 2
    return (delta[-1] + delta.mean(axis=0)) / 2.0


def collect_fitting_corpus_deltas(
    embedder: TransformerLensKVEmbedder,
    fitting_texts: List[str],
    logger: logging.Logger,
) -> np.ndarray:
    """Collect deltas from BEIR corpus texts for fitting the whitening transform.

    Uses only the external fitting corpus (BEIR/SciFact), NOT the contrast pairs
    or validation sentences — those are evaluation data and must not leak into
    the normalization transform.
    """
    logger.info("Collecting deltas from %d BEIR texts for whitening fit...", len(fitting_texts))
    deltas = []
    for i, text in enumerate(fitting_texts):
        d = extract_pooled_delta(embedder, text)
        deltas.append(d)
        if (i + 1) % 100 == 0:
            logger.info("  ... %d/%d deltas collected", i + 1, len(fitting_texts))

    corpus_deltas = np.stack(deltas, axis=0)
    logger.info("Fitting corpus shape: %s", corpus_deltas.shape)
    return corpus_deltas


MODES = ["raw", "zscore", "whitened", "whitened_l2"]


def run_extraction(
    embedder: TransformerLensKVEmbedder,
    transform: WhiteningTransform,
    mode: str,
    logger: logging.Logger,
) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """Extract concept vectors in a given normalization mode."""
    concept_vectors: Dict[str, np.ndarray] = {}
    metrics: List[dict] = []

    for pair in DEFAULT_CONTRASTS:
        delta_pos = extract_pooled_delta(embedder, pair.positive)
        delta_neg = extract_pooled_delta(embedder, pair.negative)

        # Normalize deltas (without L2 for subtraction, then re-check)
        # For concept vector extraction: use mode without L2 to keep linearity,
        # then the concept vector lives in the same space.
        extract_mode = mode if mode != "whitened_l2" else "whitened"
        d_pos_t = transform.apply(delta_pos, mode=extract_mode)
        d_neg_t = transform.apply(delta_neg, mode=extract_mode)

        concept_vec = d_pos_t - d_neg_t
        norm = float(np.linalg.norm(concept_vec))

        metrics.append({
            "concept": pair.concept_name,
            "positive": pair.positive,
            "negative": pair.negative,
            "concept_vector_norm": norm,
            "delta_pos_norm": float(np.linalg.norm(d_pos_t)),
            "delta_neg_norm": float(np.linalg.norm(d_neg_t)),
            "delta_cosine_sim": _cosine_sim(d_pos_t, d_neg_t),
        })
        concept_vectors[pair.concept_name] = concept_vec
        logger.info("  [%s] %s: norm=%.4f, delta_cosine=%.4f",
                     mode, pair.concept_name, norm, metrics[-1]["delta_cosine_sim"])

    return concept_vectors, metrics


def run_arithmetic(
    embedder: TransformerLensKVEmbedder,
    transform: WhiteningTransform,
    concept_vectors: Dict[str, np.ndarray],
    mode: str,
    logger: logging.Logger,
) -> List[dict]:
    """Validate single-concept arithmetic in a given normalization mode."""
    results = []
    for val in DEFAULT_VALIDATIONS:
        v_concept = concept_vectors[val.concept_name]

        delta_pos = extract_pooled_delta(embedder, val.sentence_pos)
        delta_neg = extract_pooled_delta(embedder, val.sentence_neg)

        # Apply same normalization as used for concept vector extraction
        compare_mode = mode if mode != "whitened_l2" else "whitened"
        d_pos_t = transform.apply(delta_pos, mode=compare_mode)
        d_neg_t = transform.apply(delta_neg, mode=compare_mode)

        # Baseline cosine (in normalized space)
        baseline_cosine = _cosine_sim(d_pos_t, d_neg_t)

        # After subtracting concept vector
        adjusted_pos = d_pos_t - v_concept
        adjusted_cosine = _cosine_sim(adjusted_pos, d_neg_t)
        cosine_improvement = adjusted_cosine - baseline_cosine

        # Also measure with L2 norm applied post-hoc (if whitened_l2 mode)
        if mode == "whitened_l2":
            d_pos_l2 = d_pos_t / max(np.linalg.norm(d_pos_t), 1e-10)
            d_neg_l2 = d_neg_t / max(np.linalg.norm(d_neg_t), 1e-10)
            adj_l2 = adjusted_pos / max(np.linalg.norm(adjusted_pos), 1e-10)
            baseline_cosine_l2 = _cosine_sim(d_pos_l2, d_neg_l2)
            adjusted_cosine_l2 = _cosine_sim(adj_l2, d_neg_l2)
            cosine_improvement_l2 = adjusted_cosine_l2 - baseline_cosine_l2
        else:
            baseline_cosine_l2 = None
            adjusted_cosine_l2 = None
            cosine_improvement_l2 = None

        success = cosine_improvement > 0
        result = {
            "concept": val.concept_name,
            "description": val.description,
            "baseline_cosine": baseline_cosine,
            "adjusted_cosine": adjusted_cosine,
            "cosine_improvement": cosine_improvement,
            "success": success,
        }
        if mode == "whitened_l2":
            result["baseline_cosine_l2"] = baseline_cosine_l2
            result["adjusted_cosine_l2"] = adjusted_cosine_l2
            result["cosine_improvement_l2"] = cosine_improvement_l2
            result["success_l2"] = cosine_improvement_l2 > 0 if cosine_improvement_l2 is not None else None

        results.append(result)
        status = "OK" if success else "FAIL"
        logger.info("  [%s] [%s] %s: baseline=%.4f -> adjusted=%.4f (improvement=%+.4f)",
                     mode, status, val.description, baseline_cosine, adjusted_cosine, cosine_improvement)

    return results


def run_combined_arithmetic(
    embedder: TransformerLensKVEmbedder,
    transform: WhiteningTransform,
    concept_vectors: Dict[str, np.ndarray],
    mode: str,
    logger: logging.Logger,
) -> List[dict]:
    """Test combined concept vector subtraction."""
    tests = [
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
    compare_mode = mode if mode != "whitened_l2" else "whitened"

    for test in tests:
        delta_pos = extract_pooled_delta(embedder, test["sentence_pos"])
        delta_neg = extract_pooled_delta(embedder, test["sentence_neg"])
        d_pos_t = transform.apply(delta_pos, mode=compare_mode)
        d_neg_t = transform.apply(delta_neg, mode=compare_mode)

        baseline_cosine = _cosine_sim(d_pos_t, d_neg_t)

        adjusted_pos = d_pos_t.copy()
        for name in test["subtract"]:
            adjusted_pos = adjusted_pos - concept_vectors[name]

        adjusted_cosine = _cosine_sim(adjusted_pos, d_neg_t)
        cosine_improvement = adjusted_cosine - baseline_cosine

        results.append({
            "description": test["description"],
            "concepts_subtracted": test["subtract"],
            "baseline_cosine": baseline_cosine,
            "adjusted_cosine": adjusted_cosine,
            "cosine_improvement": cosine_improvement,
            "success": cosine_improvement > 0,
        })
        status = "OK" if cosine_improvement > 0 else "FAIL"
        logger.info("  [%s] [%s] %s: baseline=%.4f -> adjusted=%.4f (%+.4f)",
                     mode, status, test["description"], baseline_cosine, adjusted_cosine, cosine_improvement)

    return results


def test_orthogonality(concept_vectors: Dict[str, np.ndarray]) -> List[dict]:
    names = sorted(concept_vectors.keys())
    results = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sim = _cosine_sim(concept_vectors[a], concept_vectors[b])
            results.append({
                "pair": f"{a} vs {b}",
                "cosine_similarity": sim,
                "approximately_orthogonal": abs(sim) < 0.3,
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp14b: Concept Vectors with whitening normalization"
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
    # BEIR corpus for whitening fit
    parser.add_argument("--beir-dataset", type=str, default="scifact",
                        help="BEIR dataset name (default: scifact)")
    parser.add_argument("--beir-path", type=str, default=None,
                        help="Path to local corpus.jsonl (overrides --beir-dataset auto-detect)")
    parser.add_argument("--max-fitting-samples", type=int, default=500,
                        help="Max corpus samples for whitening fit (default: 500)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = _create_results_dir("nnm_exp14b")
    logger = _setup_logging("nnm_exp14b_concept_vectors_whitened_tl", results_dir)

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
            id_texts = [c.positive for c in DEFAULT_CONTRASTS]
            id_texts += [c.negative for c in DEFAULT_CONTRASTS]
            id_texts += [v.sentence_pos for v in DEFAULT_VALIDATIONS]
        layer_selection = embedder.select_layers(id_texts)
        logger.info("Selected layers: %s", layer_selection.selected_layers)

    # -----------------------------------------------------------------------
    # Phase 0: Load BEIR corpus and fit whitening transform
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 0: Load BEIR corpus and fit whitening transform")
    logger.info("=" * 60)

    fitting_texts = load_fitting_texts(
        beir_dataset=args.beir_dataset,
        beir_path=args.beir_path,
        max_samples=args.max_fitting_samples,
        logger=logger,
    )
    logger.info("Fitting corpus: %d texts from BEIR/%s", len(fitting_texts), args.beir_dataset)

    corpus_deltas = collect_fitting_corpus_deltas(embedder, fitting_texts, logger)
    transform = fit_whitening(corpus_deltas)
    logger.info("Whitening transform fitted on %d samples, dim=%d",
                corpus_deltas.shape[0], corpus_deltas.shape[1])

    # -----------------------------------------------------------------------
    # Run all modes
    # -----------------------------------------------------------------------
    all_results: Dict[str, Any] = {
        "experiment": "nnm_exp14b_concept_vectors_whitened_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "prefix_bias": args.prefix_bias,
            "load_in_4bit": args.load_in_4bit,
            "beir_dataset": args.beir_dataset,
            "max_fitting_samples": args.max_fitting_samples,
            "fitting_corpus_size": int(corpus_deltas.shape[0]),
            "fitting_corpus_dim": int(corpus_deltas.shape[1]),
        },
        "selected_layers": (
            embedder.layer_selection.selected_layers
            if embedder.layer_selection else []
        ),
        "modes": {},
    }

    for mode in MODES:
        logger.info("=" * 60)
        logger.info("MODE: %s", mode)
        logger.info("=" * 60)

        # Extract concept vectors
        logger.info("--- Extraction ---")
        concept_vectors, extraction_metrics = run_extraction(
            embedder, transform, mode, logger
        )

        # Orthogonality
        logger.info("--- Orthogonality ---")
        ortho_results = test_orthogonality(concept_vectors)
        n_ortho = sum(1 for r in ortho_results if r["approximately_orthogonal"])
        logger.info("  [%s] Orthogonal: %d/%d", mode, n_ortho, len(ortho_results))

        # Single arithmetic
        logger.info("--- Single Arithmetic ---")
        arith_results = run_arithmetic(
            embedder, transform, concept_vectors, mode, logger
        )
        n_arith_success = sum(1 for r in arith_results if r["success"])
        logger.info("  [%s] Arithmetic success: %d/%d", mode, n_arith_success, len(arith_results))

        # Combined arithmetic
        logger.info("--- Combined Arithmetic ---")
        combined_results = run_combined_arithmetic(
            embedder, transform, concept_vectors, mode, logger
        )
        n_combined_success = sum(1 for r in combined_results if r["success"])
        logger.info("  [%s] Combined success: %d/%d", mode, n_combined_success, len(combined_results))

        all_results["modes"][mode] = {
            "extraction": extraction_metrics,
            "orthogonality": {
                "results": ortho_results,
                "n_orthogonal": n_ortho,
                "n_total": len(ortho_results),
            },
            "single_arithmetic": {
                "results": arith_results,
                "n_success": n_arith_success,
                "n_total": len(arith_results),
            },
            "combined_arithmetic": {
                "results": combined_results,
                "n_success": n_combined_success,
                "n_total": len(combined_results),
            },
        }

    # -----------------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    summary = {}
    for mode in MODES:
        m = all_results["modes"][mode]
        single = m["single_arithmetic"]
        combined = m["combined_arithmetic"]
        ortho = m["orthogonality"]
        total_success = single["n_success"] + combined["n_success"]
        total_tests = single["n_total"] + combined["n_total"]

        avg_improvement = np.mean([r["cosine_improvement"] for r in single["results"]])

        summary[mode] = {
            "single_success": f"{single['n_success']}/{single['n_total']}",
            "combined_success": f"{combined['n_success']}/{combined['n_total']}",
            "total_success": f"{total_success}/{total_tests}",
            "orthogonal_pairs": f"{ortho['n_orthogonal']}/{ortho['n_total']}",
            "avg_cosine_improvement": float(avg_improvement),
        }
        logger.info(
            "  %-12s  single=%s  combined=%s  total=%s  ortho=%s  avg_improvement=%+.4f",
            mode,
            summary[mode]["single_success"],
            summary[mode]["combined_success"],
            summary[mode]["total_success"],
            summary[mode]["orthogonal_pairs"],
            avg_improvement,
        )

    all_results["summary"] = summary

    # Determine best mode
    best_mode = max(
        MODES,
        key=lambda m: (
            all_results["modes"][m]["single_arithmetic"]["n_success"],
            all_results["modes"][m]["combined_arithmetic"]["n_success"],
        ),
    )
    all_results["best_mode"] = best_mode
    logger.info("Best mode: %s", best_mode)

    _save_json(all_results, results_dir / "results.json")
    logger.info("Results saved to %s", results_dir / "results.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
