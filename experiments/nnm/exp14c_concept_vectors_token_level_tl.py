#!/usr/bin/env python3
"""Experiment 14c (NNM): Concept Vectors via Template-averaged Token-level Deltas.

Addresses the core weaknesses of exp14/14b:
  - exp14:  pooled sentence delta loses concept signal in surrounding context noise
  - exp14b: whitening fitted on wrong domain (SciFact vs German text), n << D

Fix:
  - Extract delta at the concept word's token position(s) in context (not pooled)
  - Average over multiple template sentences per concept for robustness
  - No whitening needed -- the signal is cleaner without it

Concept vector = mean over N templates of:
    delta_at_slot(template.format(positive)) - delta_at_slot(template.format(negative))

Slot positions are found by comparing the two tokenizations and identifying the
differing token span (common prefix + suffix approach).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder


# ---------------------------------------------------------------------------
# Utilities
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
    a_n = a / max(float(np.linalg.norm(a)), 1e-10)
    b_n = b / max(float(np.linalg.norm(b)), 1e-10)
    return float(np.dot(a_n, b_n))


# ---------------------------------------------------------------------------
# Token span finding
# ---------------------------------------------------------------------------

def find_slot_span(
    tokens_a: List[int],
    tokens_b: List[int],
) -> Tuple[List[int], List[int]]:
    """Find the differing token positions between two sequences.

    The two sequences share a common prefix (before the slot) and a common
    suffix (after the slot). Returns (span_a, span_b) as lists of indices.

    Returns ([], []) if the sequences are identical or the span cannot be
    determined unambiguously.
    """
    # Common prefix
    prefix_len = 0
    for i in range(min(len(tokens_a), len(tokens_b))):
        if tokens_a[i] == tokens_b[i]:
            prefix_len += 1
        else:
            break

    # Common suffix (not overlapping with prefix region)
    suffix_len = 0
    max_suffix = min(len(tokens_a), len(tokens_b)) - prefix_len
    for i in range(1, max_suffix + 1):
        if tokens_a[-i] == tokens_b[-i]:
            suffix_len += 1
        else:
            break

    span_a = list(range(prefix_len, len(tokens_a) - suffix_len))
    span_b = list(range(prefix_len, len(tokens_b) - suffix_len))

    # Empty span means identical sequences or degenerate split
    if not span_a or not span_b:
        return [], []

    return span_a, span_b


# ---------------------------------------------------------------------------
# Delta extraction at specific token positions
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_slot_delta(
    model,
    text: str,
    token_positions: List[int],
    prepend_bos: bool,
) -> np.ndarray:
    """Return mean delta (contextual - static) across the given token positions.

    Uses the final residual stream (no KV-rerouting -- we want the raw
    contextual representation for concept arithmetic).

    Shape of output: [d_model].
    """
    n_layers = int(model.cfg.n_layers)
    final_name = f"blocks.{n_layers - 1}.hook_resid_post"

    # Run forward pass and capture final residual stream
    tokens = model.to_tokens(text, prepend_bos=prepend_bos)
    _, cache = model.run_with_cache(
        tokens,
        return_type=None,
        names_filter=[final_name],
        remove_batch_dim=False,
        prepend_bos=False,  # BOS already included in `tokens` if needed
    )

    contextual = cache[final_name][0].detach().to(torch.float32).cpu().numpy()  # [seq, d]
    static = model.W_E[tokens[0]].detach().to(torch.float32).cpu().numpy()       # [seq, d]
    delta = contextual - static  # [seq, d]  -- positional cancels in pos-neg diff

    return delta[token_positions].mean(axis=0)  # [d]



# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------

@dataclass
class TemplateContrastPair:
    concept_name: str
    positive_slot: str   # word/phrase for the positive concept
    negative_slot: str   # word/phrase for the negative concept
    templates: List[str] # each contains {SLOT}
    description: str


@dataclass
class CombinedConceptTest:
    description: str
    sentence_pos: str
    sentence_neg: str
    # (concept_name, positive_slot_in_sentence_pos, negative_slot_in_sentence_neg)
    concept_slots: List[Tuple[str, str, str]]


@dataclass
class ValidationSentencePair:
    concept_name: str
    sentence_pos: str
    sentence_neg: str
    description: str


DEFAULT_CONTRASTS: List[TemplateContrastPair] = [
    TemplateContrastPair(
        concept_name="wochenende",
        positive_slot="Samstag",
        negative_slot="Dienstag",
        templates=[
            "Das Meeting findet am {SLOT} statt.",
            "Die Muellabfuhr kommt jeden {SLOT} um 7 Uhr.",
            "Er geht jeden {SLOT} ins Fitnessstudio.",
            "Die Konferenz ist fuer {SLOT} geplant.",
            "Sie hatten am {SLOT} einen wichtigen Termin.",
            "Der Kurs findet jeden {SLOT} statt.",
            "Das Treffen wurde auf {SLOT} verschoben.",
        ],
        description="Wochenende vs. Werktag",
    ),
    TemplateContrastPair(
        concept_name="sommer",
        positive_slot="Juli",
        negative_slot="Januar",
        templates=[
            "Die Konferenz ist im {SLOT} geplant.",
            "Wir starten das Projekt im {SLOT}.",
            "Die Ferien beginnen im {SLOT}.",
            "Die Veranstaltung findet im {SLOT} statt.",
            "Der Urlaub ist fuer {SLOT} gebucht.",
            "Das Festival findet im {SLOT} statt.",
            "Sie beginnen im {SLOT} mit der Renovierung.",
        ],
        description="Sommer vs. Winter",
    ),
    TemplateContrastPair(
        concept_name="monatsende",
        positive_slot="28.",
        negative_slot="3.",
        templates=[
            "Die Rechnung wird am {SLOT} des Monats faellig.",
            "Das Gehalt kommt am {SLOT} des Monats.",
            "Der Vertrag laeuft am {SLOT} des Monats aus.",
            "Die Miete ist am {SLOT} des Monats faellig.",
            "Der Bericht muss am {SLOT} des Monats eingereicht werden.",
            "Die Zahlung ist am {SLOT} des Monats abzubuchen.",
        ],
        description="Ende vs. Anfang des Monats",
    ),
    TemplateContrastPair(
        concept_name="abend",
        positive_slot="20 Uhr abends",
        negative_slot="8 Uhr morgens",
        templates=[
            "Der Termin ist um {SLOT}.",
            "Das Treffen beginnt um {SLOT}.",
            "Die Veranstaltung startet um {SLOT}.",
            "Sie kommen um {SLOT} an.",
            "Das Konzert beginnt um {SLOT}.",
            "Der Flug geht um {SLOT}.",
        ],
        description="Abend vs. Morgen",
    ),
]

DEFAULT_VALIDATIONS: List[ValidationSentencePair] = [
    ValidationSentencePair(
        "wochenende",
        "Die Muellabfuhr kommt jeden Samstag um 7 Uhr.",
        "Die Muellabfuhr kommt jeden Dienstag um 7 Uhr.",
        "Muellabfuhr: Samstag vs. Dienstag",
    ),
    ValidationSentencePair(
        "wochenende",
        "Das Meeting findet am Samstag statt.",
        "Das Meeting findet am Dienstag statt.",
        "Meeting: Samstag vs. Dienstag",
    ),
    ValidationSentencePair(
        "sommer",
        "Die Konferenz ist im Juli geplant.",
        "Die Konferenz ist im Januar geplant.",
        "Konferenz: Juli vs. Januar",
    ),
    ValidationSentencePair(
        "sommer",
        "Wir starten das Projekt im Juli.",
        "Wir starten das Projekt im Januar.",
        "Projekt: Juli vs. Januar",
    ),
    ValidationSentencePair(
        "monatsende",
        "Die Rechnung wird am 28. des Monats faellig.",
        "Die Rechnung wird am 3. des Monats faellig.",
        "Rechnung: 28. vs. 3.",
    ),
    ValidationSentencePair(
        "abend",
        "Der Termin ist um 20 Uhr abends.",
        "Der Termin ist um 8 Uhr morgens.",
        "Termin: abends vs. morgens",
    ),
]


DEFAULT_COMBINED_TESTS: List[CombinedConceptTest] = [
    CombinedConceptTest(
        description="Wochenende + Sommer entfernen",
        sentence_pos="Das Grillfest ist am Samstag im Juli.",
        sentence_neg="Das Grillfest ist am Dienstag im Januar.",
        concept_slots=[
            ("wochenende", "Samstag", "Dienstag"),
            ("sommer", "Juli", "Januar"),
        ],
    ),
    CombinedConceptTest(
        description="Abend + Wochenende entfernen",
        sentence_pos="Die Party ist am Samstag um 20 Uhr abends.",
        sentence_neg="Die Party ist am Dienstag um 8 Uhr morgens.",
        concept_slots=[
            ("wochenende", "Samstag", "Dienstag"),
            ("abend", "20 Uhr abends", "8 Uhr morgens"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Part A: Concept vector extraction
# ---------------------------------------------------------------------------

def extract_concept_vector(
    model,
    pair: TemplateContrastPair,
    prepend_bos: bool,
    logger: logging.Logger,
) -> Tuple[np.ndarray, dict]:
    """Extract concept vector as template-averaged token-level delta difference."""
    diffs: List[np.ndarray] = []
    skipped = 0
    span_info = []

    for template in pair.templates:
        text_pos = template.replace("{SLOT}", pair.positive_slot)
        text_neg = template.replace("{SLOT}", pair.negative_slot)

        tokens_pos = model.to_tokens(text_pos, prepend_bos=prepend_bos)[0].tolist()
        tokens_neg = model.to_tokens(text_neg, prepend_bos=prepend_bos)[0].tolist()

        pos_span, neg_span = find_slot_span(tokens_pos, tokens_neg)

        if not pos_span or not neg_span:
            logger.warning("    No slot span found, skipping: %s", template[:60])
            skipped += 1
            continue

        span_info.append({"pos_len": len(pos_span), "neg_len": len(neg_span)})

        d_pos = extract_slot_delta(model, text_pos, pos_span, prepend_bos)
        d_neg = extract_slot_delta(model, text_neg, neg_span, prepend_bos)
        diffs.append(d_pos - d_neg)

    if not diffs:
        raise RuntimeError(f"No valid templates for concept '{pair.concept_name}'")

    concept_vec = np.stack(diffs, axis=0).mean(axis=0)

    # Cross-template consistency: cosine similarity between individual diff vectors
    if len(diffs) > 1:
        sims = [
            _cosine_sim(diffs[i], diffs[j])
            for i in range(len(diffs))
            for j in range(i + 1, len(diffs))
        ]
        template_consistency = float(np.mean(sims))
    else:
        template_consistency = float("nan")

    metrics = {
        "concept": pair.concept_name,
        "positive_slot": pair.positive_slot,
        "negative_slot": pair.negative_slot,
        "n_templates_used": len(diffs),
        "n_templates_skipped": skipped,
        "concept_vector_norm": float(np.linalg.norm(concept_vec)),
        "template_consistency": template_consistency,
        "span_info": span_info,
    }
    return concept_vec, metrics


# ---------------------------------------------------------------------------
# Part B: Single-concept arithmetic validation
# ---------------------------------------------------------------------------

def validate_concept_arithmetic(
    model,
    val: ValidationSentencePair,
    concept_vec: np.ndarray,
    prepend_bos: bool,
) -> dict:
    """Test: delta_slot(pos) - concept_vec ~ delta_slot(neg).

    Finds the differing token span between the sentence pair and measures
    cosine similarity before and after subtracting the concept vector.
    """
    tokens_pos = model.to_tokens(val.sentence_pos, prepend_bos=prepend_bos)[0].tolist()
    tokens_neg = model.to_tokens(val.sentence_neg, prepend_bos=prepend_bos)[0].tolist()

    pos_span, neg_span = find_slot_span(tokens_pos, tokens_neg)

    if not pos_span or not neg_span:
        return {
            "concept": val.concept_name,
            "description": val.description,
            "skipped": True,
            "reason": "Could not find differing token span",
        }

    d_pos = extract_slot_delta(model, val.sentence_pos, pos_span, prepend_bos)
    d_neg = extract_slot_delta(model, val.sentence_neg, neg_span, prepend_bos)

    baseline_cosine = _cosine_sim(d_pos, d_neg)
    adjusted_cosine = _cosine_sim(d_pos - concept_vec, d_neg)
    improvement = adjusted_cosine - baseline_cosine

    return {
        "concept": val.concept_name,
        "description": val.description,
        "pos_span_len": len(pos_span),
        "neg_span_len": len(neg_span),
        "baseline_cosine": baseline_cosine,
        "adjusted_cosine": adjusted_cosine,
        "cosine_improvement": improvement,
        "success": improvement > 0,
    }


# ---------------------------------------------------------------------------
# Part C: Orthogonality
# ---------------------------------------------------------------------------

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
# Part D: Combined-concept arithmetic
# ---------------------------------------------------------------------------

def find_slot_in_sentence(
    model,
    sentence: str,
    positive_slot: str,
    negative_slot: str,
    prepend_bos: bool,
) -> List[int]:
    """Find token positions of positive_slot in sentence.

    Replaces positive_slot with negative_slot to create a contrast, then uses
    find_slot_span to locate the differing tokens. Returns [] on failure.
    """
    sentence_swapped = sentence.replace(positive_slot, negative_slot, 1)
    if sentence_swapped == sentence:
        return []  # slot string not found in sentence

    tokens_orig = model.to_tokens(sentence, prepend_bos=prepend_bos)[0].tolist()
    tokens_swap = model.to_tokens(sentence_swapped, prepend_bos=prepend_bos)[0].tolist()

    pos_span, _ = find_slot_span(tokens_orig, tokens_swap)
    return pos_span


def test_combined_arithmetic(
    model,
    tests: List[CombinedConceptTest],
    concept_vectors: Dict[str, np.ndarray],
    prepend_bos: bool,
    logger: logging.Logger,
) -> List[dict]:
    """Per-concept token-level slot arithmetic for multi-concept sentences.

    For each concept in the test, finds the slot token positions in the
    respective sentence individually (by swapping only that concept's slot),
    then measures cosine improvement after subtracting the concept vector.
    This avoids the scale mismatch of the old pooled-delta approach.
    """
    results = []

    for test in tests:
        missing = [name for name, _, _ in test.concept_slots if name not in concept_vectors]
        if missing:
            results.append({
                "description": test.description,
                "skipped": True,
                "reason": f"Missing concept vectors: {missing}",
            })
            continue

        per_concept = []
        all_success = True

        for concept_name, pos_slot, neg_slot in test.concept_slots:
            # Find slot in positive sentence (swap pos→neg to locate)
            pos_span = find_slot_in_sentence(
                model, test.sentence_pos, pos_slot, neg_slot, prepend_bos
            )
            # Find slot in negative sentence (swap neg→pos to locate)
            neg_span = find_slot_in_sentence(
                model, test.sentence_neg, neg_slot, pos_slot, prepend_bos
            )

            if not pos_span or not neg_span:
                logger.warning(
                    "    [%s] slot not found in sentence (pos_span=%s neg_span=%s)",
                    concept_name, pos_span, neg_span,
                )
                per_concept.append({
                    "concept": concept_name,
                    "skipped": True,
                    "reason": "slot not found in sentence",
                })
                all_success = False
                continue

            d_pos = extract_slot_delta(model, test.sentence_pos, pos_span, prepend_bos)
            d_neg = extract_slot_delta(model, test.sentence_neg, neg_span, prepend_bos)

            baseline = _cosine_sim(d_pos, d_neg)
            adjusted = _cosine_sim(d_pos - concept_vectors[concept_name], d_neg)
            improvement = adjusted - baseline
            success = improvement > 0
            if not success:
                all_success = False

            per_concept.append({
                "concept": concept_name,
                "pos_slot": pos_slot,
                "neg_slot": neg_slot,
                "pos_span_len": len(pos_span),
                "neg_span_len": len(neg_span),
                "baseline_cosine": baseline,
                "adjusted_cosine": adjusted,
                "cosine_improvement": improvement,
                "success": success,
            })
            status = "OK" if success else "FAIL"
            logger.info(
                "  [%s] %s / %s: baseline=%.4f -> adjusted=%.4f (%+.4f)",
                status, test.description, concept_name, baseline, adjusted, improvement,
            )

        results.append({
            "description": test.description,
            "per_concept": per_concept,
            "success": all_success,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp14c: Concept Vectors via template-averaged token-level deltas"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = _create_results_dir("nnm_exp14c")
    logger = _setup_logging("nnm_exp14c_concept_vectors_tl", results_dir)

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        load_in_4bit=args.load_in_4bit,
    )
    embedder = TransformerLensKVEmbedder(config)
    model = embedder.model
    prepend_bos = embedder._prepend_bos
    logger.info("Model: %s | prepend_bos=%s", args.model, prepend_bos)

    # -----------------------------------------------------------------------
    # Part A: Extract concept vectors
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART A: Template-averaged Token-level Concept Vector Extraction")
    logger.info("=" * 60)

    concept_vectors: Dict[str, np.ndarray] = {}
    extraction_metrics = []

    for pair in DEFAULT_CONTRASTS:
        logger.info(
            "Extracting: %s  ('%s' vs '%s', %d templates)",
            pair.concept_name, pair.positive_slot, pair.negative_slot, len(pair.templates),
        )
        vec, metrics = extract_concept_vector(model, pair, prepend_bos, logger)
        concept_vectors[pair.concept_name] = vec
        extraction_metrics.append(metrics)
        logger.info(
            "  norm=%.4f  template_consistency=%.4f  (%d/%d templates used)",
            metrics["concept_vector_norm"],
            metrics["template_consistency"] if not np.isnan(metrics["template_consistency"]) else 0.0,
            metrics["n_templates_used"],
            len(pair.templates),
        )

    # -----------------------------------------------------------------------
    # Part B: Validate single-concept arithmetic
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART B: Single Concept Arithmetic Validation")
    logger.info("=" * 60)

    validation_results = []
    for val in DEFAULT_VALIDATIONS:
        if val.concept_name not in concept_vectors:
            logger.warning("Skipping %s: concept vector missing", val.description)
            continue
        result = validate_concept_arithmetic(
            model, val, concept_vectors[val.concept_name], prepend_bos
        )
        validation_results.append(result)
        if result.get("skipped"):
            logger.info("  [SKIP] %s: %s", result["description"], result["reason"])
        else:
            status = "OK" if result["success"] else "FAIL"
            logger.info(
                "  [%s] %s: baseline=%.4f -> adjusted=%.4f (improvement=%+.4f)",
                status, result["description"],
                result["baseline_cosine"], result["adjusted_cosine"], result["cosine_improvement"],
            )

    n_success = sum(1 for r in validation_results if r.get("success", False))
    n_total = sum(1 for r in validation_results if not r.get("skipped", False))
    part_b_success = n_success > n_total / 2
    logger.info("Part B: %d/%d improved (success=%s)", n_success, n_total, part_b_success)

    # -----------------------------------------------------------------------
    # Part C: Orthogonality
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART C: Concept Vector Orthogonality")
    logger.info("=" * 60)

    ortho_results = test_orthogonality(concept_vectors)
    for r in ortho_results:
        tag = "ORTH" if r["approximately_orthogonal"] else "CORR"
        logger.info("  [%s] %s: cosine=%.4f", tag, r["pair"], r["cosine_similarity"])
    n_ortho = sum(1 for r in ortho_results if r["approximately_orthogonal"])
    logger.info("Part C: %d/%d approximately orthogonal", n_ortho, len(ortho_results))

    # -----------------------------------------------------------------------
    # Part D: Combined concept arithmetic
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART D: Combined Concept Arithmetic (pooled delta fallback)")
    logger.info("=" * 60)

    combined_results = test_combined_arithmetic(
        model, DEFAULT_COMBINED_TESTS, concept_vectors, prepend_bos, logger
    )
    n_combined_success = sum(1 for r in combined_results if r.get("success", False))
    n_combined_total = sum(1 for r in combined_results if not r.get("skipped", False))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    output = {
        "experiment": "nnm_exp14c_concept_vectors_token_level_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "load_in_4bit": args.load_in_4bit,
        },
        "part_a_extraction": extraction_metrics,
        "part_b_single_arithmetic": {
            "results": validation_results,
            "n_success": n_success,
            "n_total": n_total,
            "success": part_b_success,
        },
        "part_c_orthogonality": {
            "results": ortho_results,
            "n_orthogonal": n_ortho,
            "n_total": len(ortho_results),
        },
        "part_d_combined_arithmetic": {
            "results": combined_results,
            "n_success": n_combined_success,
            "n_total": n_combined_total,
        },
        "overall_success": part_b_success,
    }

    out_path = results_dir / "results.json"
    _save_json(output, out_path)
    logger.info("Results saved to %s", out_path)
    logger.info("Overall success: %s", part_b_success)

    return 0 if part_b_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
