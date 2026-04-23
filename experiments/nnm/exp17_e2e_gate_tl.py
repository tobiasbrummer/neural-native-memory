#!/usr/bin/env python3
"""Experiment 17 (NNM): E2E Gate -- NNMA KV-Injection vs RAG-Prepend.

SELF-CONTAINED VERSION.
No nnm.* imports. Drop next to a requirements.txt, run:
    pip install -r requirements.txt
    python exp17_e2e_gate_tl.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit \
        --max-queries 20 --seeds 42 --no-think

Tests the core NNMA hypothesis on BEIR/SciFact claim verification:
Does KV-Injection of retrieved passage beat text-prepending (RAG) on
the same information?

Four arms per query (same retrieved passage):
  1. cold   : query only, no context (sanity baseline)
  2. rag    : retrieved passage as text context (honest baseline)
  3. nnma   : retrieved passage as KV-cache prefix
  4. random : random unrelated passage as KV-cache prefix (noise control)

Labels are single-letter A/B/C multiple-choice to make logit extraction robust:
  A = supports, B = refutes, C = not enough information

Gold labels come from the original allenai/scifact dataset (claim.evidence).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Logging / IO helpers
# ---------------------------------------------------------------------------

def _setup_logging(name: str, results_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
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


# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

LABEL_MAP_TO_LETTER = {
    "SUPPORT": "A",
    "SUPPORTS": "A",
    "CONTRADICT": "B",
    "REFUTES": "B",
    "REFUTE": "B",
    "NEI": "C",
    "NOT_ENOUGH_INFO": "C",
    "NOT ENOUGH INFO": "C",
}
LETTER_TO_LABEL = {"A": "supports", "B": "refutes", "C": "not_enough_info"}
# Forced-choice: only A/B offered in the prompt. NEI-claims are dropped by the
# loader; Gold is always A or B. Models over-hedged with "C" available, so we
# strip it from the choice set.
LETTERS = ["A", "B"]


# ---------------------------------------------------------------------------
# Model loading (TransformerLens)
# ---------------------------------------------------------------------------

def load_hooked_model(
    model_name: str,
    device: Optional[str],
    dtype: str,
    local_files_only: bool,
    load_in_4bit: bool,
    prepend_bos: bool,
) -> Tuple[Any, bool]:
    """Load a HookedTransformer. Returns (model, effective_prepend_bos).

    Falls back to prepend_bos=False if the tokenizer refuses add_bos_token=True.
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError as e:
        raise RuntimeError(
            "transformer_lens not installed. `pip install transformer_lens`"
        ) from e

    common_kwargs: Dict[str, Any] = dict(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )

    hf_model = None
    if load_in_4bit:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        hf_token = os.environ.get("HF_TOKEN") or None
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            local_files_only=local_files_only,
        )

    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            default_prepend_bos=prepend_bos,
            **common_kwargs,
        )
        effective_prepend_bos = prepend_bos
    except ValueError as exc:
        if "add_bos_token = True but bos_token = None" not in str(exc):
            raise
        from transformers import AutoTokenizer

        hf_token = os.environ.get("HF_TOKEN") or None
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_bos_token=False,
            trust_remote_code=True,
            use_fast=True,
            token=hf_token,
            local_files_only=local_files_only,
        )
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            default_prepend_bos=False,
            **common_kwargs,
        )
        effective_prepend_bos = False

    model.eval()
    return model, effective_prepend_bos


# ---------------------------------------------------------------------------
# BEIR data loading (HuggingFace fallback if no local JSONL)
# ---------------------------------------------------------------------------

def _safe_text(doc: Mapping[str, object]) -> str:
    title = str(doc.get("title", "") or "").strip()
    text = str(doc.get("text", "") or "").strip()
    joined = f"{title} {text}".strip()
    return joined if joined else text


def load_beir_corpus(dataset: str, beir_path: Optional[str]) -> Dict[str, str]:
    """Load BEIR corpus as {doc_id: text}. Local JSONL > HuggingFace fallback."""
    if beir_path:
        corpus_path = Path(beir_path)
    else:
        corpus_path = Path("data/beir_datasets") / dataset / "corpus.jsonl"

    corpus: Dict[str, str] = {}
    if corpus_path.exists():
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = _safe_text(row)
                if text:
                    corpus[str(row["_id"])] = text
        return corpus

    from datasets import load_dataset

    ds = load_dataset("BeIR/" + dataset, "corpus", split="corpus")
    for row in ds:
        text = _safe_text(row)
        if text:
            corpus[str(row["_id"])] = text
    return corpus


def load_beir_queries(dataset: str, beir_path: Optional[str]) -> Dict[str, str]:
    if beir_path:
        queries_path = Path(beir_path).parent / "queries.jsonl"
    else:
        queries_path = Path("data/beir_datasets") / dataset / "queries.jsonl"

    queries: Dict[str, str] = {}
    if queries_path.exists():
        with queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get("text", "")).strip()
                if text:
                    queries[str(row["_id"])] = text
        return queries

    from datasets import load_dataset

    ds = load_dataset("BeIR/" + dataset, "queries", split="queries")
    for row in ds:
        text = str(row.get("text", "")).strip()
        if text:
            queries[str(row["_id"])] = text
    return queries


def load_beir_qrels(dataset: str, split: str, beir_path: Optional[str]) -> Dict[str, Dict[str, int]]:
    if beir_path:
        qrels_path = Path(beir_path).parent / "qrels" / f"{split}.tsv"
    else:
        qrels_path = Path("data/beir_datasets") / dataset / "qrels" / f"{split}.tsv"

    qrels: Dict[str, Dict[str, int]] = {}
    if qrels_path.exists():
        import csv
        with qrels_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid = str(row["query-id"])
                did = str(row["corpus-id"])
                score = int(float(row["score"]))
                if score > 0:
                    qrels.setdefault(qid, {})[did] = score
        return qrels

    from datasets import load_dataset

    ds = load_dataset("BeIR/" + dataset + "-qrels", split=split)
    for row in ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(float(row["score"]))
        if score > 0:
            qrels.setdefault(qid, {})[did] = score
    return qrels


# ---------------------------------------------------------------------------
# Scifact claim-label loader (from allenai/scifact, not BEIR)
# ---------------------------------------------------------------------------

def load_scifact_claim_labels(split: str = "test") -> Dict[str, str]:
    """Load gold claim labels from allenai/scifact across ALL splits.

    BEIR/scifact queries.jsonl contains claims from every split, so we merge
    train+validation+test claims to maximise the overlap. The `split` param
    is kept for API-compat but ignored; we always union.

    Returns {claim_id_str: A|B|C}.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("`pip install datasets` to fetch allenai/scifact labels") from e

    lg = logging.getLogger(__name__)
    labels: Dict[str, str] = {}
    votes_by_cid: Dict[str, List[str]] = defaultdict(list)
    loaded_splits: List[str] = []
    schema_logged = False

    for sp in ("train", "validation", "test"):
        try:
            ds = load_dataset("allenai/scifact", "claims", split=sp)
        except Exception as exc:
            lg.warning("scifact split %s unavailable (%s)", sp, exc)
            continue
        loaded_splits.append(sp)

        if not schema_logged and len(ds) > 0:
            lg.info("scifact claim schema: %s", list(ds.features.keys()))
            lg.info("scifact claim example: %s", {k: ds[0][k] for k in ds.features.keys()})
            schema_logged = True

        feat_keys = set(ds.features.keys())

        for row in ds:
            cid = str(row["id"])
            raw = ""
            # Schema v1: flat `evidence_label` string column
            if "evidence_label" in feat_keys:
                raw = str(row.get("evidence_label", "") or "").strip().upper()
            # Schema v2: nested `evidence` dict {doc_id: [{label: ...}]}
            elif "evidence" in feat_keys:
                evidence = row.get("evidence", {}) or {}
                if isinstance(evidence, dict):
                    for _doc_id, ev_list in evidence.items():
                        if not ev_list:
                            continue
                        for ev in ev_list:
                            sub = str(ev.get("label", "")).strip().upper()
                            if sub in LABEL_MAP_TO_LETTER:
                                votes_by_cid[cid].append(LABEL_MAP_TO_LETTER[sub])
            if raw in LABEL_MAP_TO_LETTER:
                votes_by_cid[cid].append(LABEL_MAP_TO_LETTER[raw])

    for cid, votes in votes_by_cid.items():
        if votes:
            labels[cid] = Counter(votes).most_common(1)[0][0]

    lg.info(
        "scifact labels loaded: %d claims across splits=%s "
        "(only claims with explicit evidence)",
        len(labels), loaded_splits,
    )
    return labels


# ---------------------------------------------------------------------------
# BM25 index (subword-token-id based)
# ---------------------------------------------------------------------------

@dataclass
class BM25Index:
    doc_ids: List[str]
    doc_token_counts: List[Counter]
    doc_lengths: List[int]
    avg_dl: float
    idf: Dict[int, float]
    n_docs: int

    @staticmethod
    def build(
        doc_ids: List[str],
        doc_token_ids: List[List[int]],
        exclude_token_ids: Optional[set] = None,
    ) -> "BM25Index":
        if exclude_token_ids is None:
            exclude_token_ids = set()
        n = len(doc_ids)
        doc_counts = []
        doc_lengths = []
        df: Counter = Counter()
        for tokens in doc_token_ids:
            filtered = [t for t in tokens if t not in exclude_token_ids]
            counts = Counter(filtered)
            doc_counts.append(counts)
            doc_lengths.append(len(filtered))
            for token_id in counts:
                df[token_id] += 1
        avg_dl = sum(doc_lengths) / max(n, 1)
        idf = {}
        for token_id, freq in df.items():
            idf[token_id] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
        return BM25Index(
            doc_ids=doc_ids,
            doc_token_counts=doc_counts,
            doc_lengths=doc_lengths,
            avg_dl=avg_dl,
            idf=idf,
            n_docs=n,
        )

    def score_query(
        self,
        query_token_ids: List[int],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Tuple[str, float]]:
        query_counts = Counter(query_token_ids)
        scores = []
        for i in range(self.n_docs):
            score = 0.0
            dl = self.doc_lengths[i]
            doc_counts = self.doc_token_counts[i]
            for token_id, _qf in query_counts.items():
                if token_id not in self.idf:
                    continue
                tf = doc_counts.get(token_id, 0)
                if tf == 0:
                    continue
                idf_val = self.idf[token_id]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avg_dl))
                score += idf_val * tf_norm
            scores.append((self.doc_ids[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ---------------------------------------------------------------------------
# Dense delta extraction + whitening
# ---------------------------------------------------------------------------

def extract_token_deltas(
    model,
    text: str,
    retrieval_layer: int,
    prepend_bos: bool,
    max_tokens: int = 128,
) -> np.ndarray:
    """Per-token (contextual - static) deltas at retrieval_layer. (seq, d)."""
    layer_name = f"blocks.{retrieval_layer}.hook_resid_post"
    tokens = model.to_tokens(text, prepend_bos=prepend_bos)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            return_type=None,
            names_filter=[layer_name],
            remove_batch_dim=False,
            prepend_bos=False,
        )
    contextual = cache[layer_name][0].detach().to(torch.float32).cpu().numpy()
    static = model.W_E[tokens[0]].detach().to(torch.float32).cpu().numpy()
    delta = (contextual - static).astype(np.float32)
    if delta.shape[0] > max_tokens:
        delta = delta[:max_tokens]
    return delta


def fit_whitening_transform(
    deltas: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = deltas.astype(np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    std = centered.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    normed = centered / std
    n = max(1, int(normed.shape[0] - 1))
    cov = (normed.T @ normed) / float(n)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, eps)
    inv_sqrt = 1.0 / np.sqrt(evals)
    W = (evecs * inv_sqrt) @ evecs.T
    return mean.astype(np.float32), std.astype(np.float32), W.astype(np.float32)


def apply_whitening(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    W: np.ndarray,
    l2_normalize: bool = True,
) -> np.ndarray:
    was_1d = (x.ndim == 1)
    y = x.astype(np.float32, copy=True)
    if was_1d:
        y = y.reshape(1, -1)
    y = (y - mean.reshape(1, -1)) / std.reshape(1, -1)
    y = y @ W
    if l2_normalize:
        norms = np.linalg.norm(y, axis=1, keepdims=True)
        y = y / np.maximum(norms, 1e-10)
    if was_1d:
        return y.squeeze(0)
    return y


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def hybrid_linear(sparse: np.ndarray, dense: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * min_max_normalize(sparse) + (1.0 - alpha) * min_max_normalize(dense)


# ---------------------------------------------------------------------------
# TwoNN layer selection (paper)
# ---------------------------------------------------------------------------

def compute_intrinsic_dimension_twonn(embeddings: np.ndarray) -> float:
    """TwoNN intrinsic dimensionality (Facco et al., 2017)."""
    from sklearn.neighbors import NearestNeighbors

    n = embeddings.shape[0]
    if n < 4:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(embeddings)
    dists, _ = nn.kneighbors(embeddings)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mask = (r1 > 0) & (r2 > r1)
    if not np.any(mask):
        return float("nan")
    mu = r2[mask] / r1[mask]
    log_mu = np.log(mu)
    sum_log = float(np.sum(log_mu))
    if sum_log <= 0:
        return float("nan")
    return float(len(log_mu) / sum_log)


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


def estimate_layer_intrinsic_dimensions(
    model,
    texts: Sequence[str],
    prepend_bos: bool,
) -> Dict[int, float]:
    n_layers = int(model.cfg.n_layers)
    per_layer: List[List[np.ndarray]] = [[] for _ in range(n_layers)]

    def names_filter(name: str) -> bool:
        return name.endswith("hook_resid_post")

    with torch.no_grad():
        for text in texts:
            _, cache = model.run_with_cache(
                text,
                return_type=None,
                prepend_bos=prepend_bos,
                names_filter=names_filter,
                remove_batch_dim=False,
            )
            for layer in range(n_layers):
                key = f"blocks.{layer}.hook_resid_post"
                if key not in cache:
                    continue
                layer_hidden = cache[key][0].detach().to(torch.float32).cpu().numpy()
                per_layer[layer].append(layer_hidden)
            del cache

    result: Dict[int, float] = {}
    for layer in range(n_layers):
        if not per_layer[layer]:
            result[layer] = float("nan")
            continue
        vectors = np.concatenate(per_layer[layer], axis=0)
        result[layer] = compute_intrinsic_dimension_twonn(vectors)
    return result


@dataclass
class LayerSelectionResult:
    selected_layers: List[int]
    id_by_layer: Dict[int, float]
    used_u_shape_mode: bool


def select_rerouting_layers(
    id_by_layer: Mapping[int, float],
    n_layers: int,
    layer_window_fraction: float = 0.10,
    exclude_early_fraction: float = 0.20,
    detect_u_shape: bool = True,
) -> LayerSelectionResult:
    values = np.array(
        [id_by_layer.get(i, float("nan")) for i in range(n_layers)], dtype=np.float64
    )
    finite = np.isfinite(values)
    if not np.any(finite):
        raise RuntimeError("No valid intrinsic dimension values found.")

    span = max(1, int(np.floor(layer_window_fraction * n_layers)))
    target_count = span + 1
    use_u_shape = bool(detect_u_shape and _is_u_shaped(values[finite]))

    if use_u_shape:
        min_layer = int(np.nanargmin(values))
        start = min_layer
        end = min(n_layers - 1, start + span)
        selected = list(range(start, end + 1))
        return LayerSelectionResult(
            selected_layers=selected,
            id_by_layer={int(k): float(v) for k, v in id_by_layer.items()},
            used_u_shape_mode=True,
        )

    early_cutoff = int(np.floor(exclude_early_fraction * n_layers))
    candidates = [i for i in range(early_cutoff, n_layers) if np.isfinite(values[i])]
    if not candidates:
        candidates = [i for i in range(n_layers) if np.isfinite(values[i])]
    ranked = sorted(candidates, key=lambda i: values[i])
    selected = sorted(ranked[: min(target_count, len(ranked))])
    return LayerSelectionResult(
        selected_layers=selected,
        id_by_layer={int(k): float(v) for k, v in id_by_layer.items()},
        used_u_shape_mode=False,
    )


# ---------------------------------------------------------------------------
# KV cache extraction (post-QKnorm, pre-RoPE for Qwen3)
# ---------------------------------------------------------------------------

@dataclass
class StoredTLKVCache:
    keys: Dict[int, np.ndarray]
    values: Dict[int, np.ndarray]
    token_ids: np.ndarray
    text: str
    prefix_len: int


@torch.no_grad()
def extract_kv_cache_tl(model, text: str, prepend_bos: bool = True) -> StoredTLKVCache:
    tokens = model.to_tokens(text, prepend_bos=prepend_bos)
    prefix_len = int(tokens.shape[1])

    names = set()
    for layer in range(int(model.cfg.n_layers)):
        names.add(f"blocks.{layer}.attn.hook_k")
        names.add(f"blocks.{layer}.attn.hook_v")

    _, cache = model.run_with_cache(
        tokens,
        return_type=None,
        names_filter=lambda name: name in names,
        remove_batch_dim=False,
        prepend_bos=False,
    )

    keys: Dict[int, np.ndarray] = {}
    values: Dict[int, np.ndarray] = {}
    for layer in range(int(model.cfg.n_layers)):
        k_name = f"blocks.{layer}.attn.hook_k"
        v_name = f"blocks.{layer}.attn.hook_v"
        if k_name not in cache or v_name not in cache:
            raise RuntimeError(f"Missing cache entries for layer {layer}")
        k_tensor = cache[k_name].detach()
        v = cache[v_name].detach().cpu().numpy()

        # Qwen3-style QK-normalization must be applied before RoPE is added
        if getattr(model.cfg, "use_qk_norm", False):
            attn = model.blocks[layer].attn
            if (
                hasattr(attn, "_apply_qk_norm")
                and hasattr(attn, "k_norm")
                and attn.k_norm is not None
            ):
                k_tensor = attn._apply_qk_norm(
                    k_tensor.to(torch.float32), attn.k_norm
                ).to(k_tensor.dtype)

        keys[layer] = k_tensor.cpu().numpy().astype(np.float16)
        values[layer] = v.astype(np.float16)

    token_ids = tokens.squeeze(0).detach().cpu().numpy().astype(np.int64)
    return StoredTLKVCache(
        keys=keys,
        values=values,
        token_ids=token_ids,
        text=text,
        prefix_len=prefix_len,
    )


# ---------------------------------------------------------------------------
# Multi-memory KV cache builder (for injection) + generate-with-cache
# ---------------------------------------------------------------------------

def build_multi_memory_cache(model, stored_kvs: List[StoredTLKVCache]):
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    device = model.W_E.device
    dtype = model.W_E.dtype
    n_layers = int(model.cfg.n_layers)

    cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, device=device, batch_size=1,
    )
    total_prefix_len = sum(kv.prefix_len for kv in stored_kvs)

    for layer in range(n_layers):
        k_parts = []
        v_parts = []
        for kv in stored_kvs:
            k = torch.from_numpy(kv.keys[layer]).to(device=device, dtype=dtype)
            v = torch.from_numpy(kv.values[layer]).to(device=device, dtype=dtype)
            k_parts.append(k)
            v_parts.append(v)
        cache.entries[layer].past_keys = torch.cat(k_parts, dim=1)
        cache.entries[layer].past_values = torch.cat(v_parts, dim=1)

    cache.previous_attention_mask = torch.ones(
        (1, total_prefix_len), dtype=torch.int, device=device,
    )
    return cache, total_prefix_len


@torch.no_grad()
def generate_with_multi_cache(
    model,
    cache,
    query: str,
    prepend_bos: bool = True,
    max_new_tokens: int = 4,
) -> Tuple[str, torch.Tensor]:
    """Generate greedily with a pre-filled multi-memory KV cache.

    Returns (generated_text, first_token_logits).
    """
    query_tokens = model.to_tokens(query, prepend_bos=prepend_bos)
    logits, _ = model.run_with_cache(
        query_tokens,
        return_type="logits",
        past_kv_cache=cache,
        prepend_bos=False,
        remove_batch_dim=False,
    )
    first_logits = logits[:, -1].clone()

    generated = []
    eos_id = getattr(model.tokenizer, "eos_token_id", None) if model.tokenizer else None
    for _ in range(max_new_tokens):
        nxt = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        generated.append(nxt)
        if eos_id is not None and int(nxt.item()) == int(eos_id):
            break
        logits, _ = model.run_with_cache(
            nxt,
            return_type="logits",
            past_kv_cache=cache,
            prepend_bos=False,
            remove_batch_dim=False,
        )
    text = model.to_string(torch.cat(generated, dim=1)[0]) if generated else ""
    return text, first_logits


@torch.no_grad()
def generate_plain(
    model,
    prompt: str,
    prepend_bos: bool,
    max_new_tokens: int = 4,
) -> Tuple[str, torch.Tensor]:
    """Plain forward: first-token logits + short greedy tail."""
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    logits = model(tokens, return_type="logits")
    first_logits = logits[:, -1].clone()

    generated = []
    eos_id = getattr(model.tokenizer, "eos_token_id", None) if model.tokenizer else None
    for _ in range(max_new_tokens):
        nxt = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        generated.append(nxt)
        if eos_id is not None and int(nxt.item()) == int(eos_id):
            break
        tokens = torch.cat([tokens, nxt], dim=1)
        logits = model(tokens, return_type="logits")
    text = model.to_string(torch.cat(generated, dim=1)[0]) if generated else ""
    return text, first_logits


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_SYSTEM = (
    "You classify scientific claims based on evidence. "
    "Answer with exactly one letter."
)

PROMPT_QUESTION = (
    "Claim: {claim}\n\n"
    "Based on the evidence, is the claim supported or refuted? "
    "Choose the direction it leans toward; do not refuse to answer.\n"
    "A) supports\n"
    "B) refutes\n\n"
    "Answer (single letter):"
)

PROMPT_RAG_EVIDENCE = "Evidence: {passage}\n\n"


def build_cold_prompt(claim: str, no_think: bool) -> str:
    prefix = "/no_think\n" if no_think else ""
    return f"{prefix}{PROMPT_SYSTEM}\n\n{PROMPT_QUESTION.format(claim=claim)} "


def build_rag_prompt(claim: str, passage: str, no_think: bool) -> str:
    prefix = "/no_think\n" if no_think else ""
    body = PROMPT_RAG_EVIDENCE.format(passage=passage) + PROMPT_QUESTION.format(claim=claim)
    return f"{prefix}{PROMPT_SYSTEM}\n\n{body} "


def build_injection_prompt(claim: str, no_think: bool) -> str:
    prefix = "/no_think\n" if no_think else ""
    return f"{prefix}{PROMPT_SYSTEM}\n\n{PROMPT_QUESTION.format(claim=claim)} "


def build_memory_context(passage: str) -> str:
    return f"Context: {passage}"


# ---------------------------------------------------------------------------
# Label-token resolution
# ---------------------------------------------------------------------------

@dataclass
class LabelTokens:
    letters: List[str]
    token_ids: List[int]


def resolve_label_tokens(tokenizer) -> LabelTokens:
    """First-token id for ' A', ' B', ' C' (falls back to bare letters)."""
    token_ids: List[int] = []
    for letter in LETTERS:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(letter, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"Could not tokenize label '{letter}'")
        token_ids.append(int(ids[0]))
    return LabelTokens(letters=list(LETTERS), token_ids=token_ids)


def extract_label_logits(
    first_logits: torch.Tensor,
    label_tokens: LabelTokens,
) -> Dict[str, float]:
    vec = first_logits[0] if first_logits.dim() == 2 else first_logits
    return {
        letter: float(vec[tid].item())
        for letter, tid in zip(label_tokens.letters, label_tokens.token_ids)
    }


def pick_label_from_logits(logit_map: Dict[str, float]) -> Tuple[str, float]:
    ordered = sorted(logit_map.items(), key=lambda kv: kv[1], reverse=True)
    top_letter, top_logit = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else top_logit - 1.0
    return top_letter, top_logit - runner_up


# ---------------------------------------------------------------------------
# Arm runners
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    arm: str
    query_id: str
    gold_label: str
    predicted_label: str
    correct: bool
    label_logits: Dict[str, float]
    logit_gap: float
    free_text: str
    retrieved_doc_id: Optional[str] = None
    distractor_doc_id: Optional[str] = None
    seed: int = 0


def _make_result(
    arm: str, qid: str, gold: str,
    first_logits: torch.Tensor, free_text: str,
    label_tokens: LabelTokens, seed: int,
    retrieved_doc_id: Optional[str] = None,
    distractor_doc_id: Optional[str] = None,
) -> ArmResult:
    logit_map = extract_label_logits(first_logits, label_tokens)
    pred, gap = pick_label_from_logits(logit_map)
    return ArmResult(
        arm=arm,
        query_id=qid,
        gold_label=gold,
        predicted_label=pred,
        correct=(pred == gold),
        label_logits=logit_map,
        logit_gap=gap,
        free_text=free_text.strip(),
        retrieved_doc_id=retrieved_doc_id,
        distractor_doc_id=distractor_doc_id,
        seed=seed,
    )


def run_cold(model, claim, qid, gold, label_tokens, prepend_bos, no_think, seed) -> ArmResult:
    prompt = build_cold_prompt(claim, no_think=no_think)
    text, first_logits = generate_plain(model, prompt, prepend_bos, max_new_tokens=4)
    return _make_result("cold", qid, gold, first_logits, text, label_tokens, seed)


def run_rag(model, claim, passage, qid, gold, retrieved_doc_id,
            label_tokens, prepend_bos, no_think, seed) -> ArmResult:
    prompt = build_rag_prompt(claim, passage, no_think=no_think)
    text, first_logits = generate_plain(model, prompt, prepend_bos, max_new_tokens=4)
    return _make_result(
        "rag", qid, gold, first_logits, text, label_tokens, seed,
        retrieved_doc_id=retrieved_doc_id,
    )


def run_injection(
    arm_name: str,
    model,
    claim: str,
    passage: str,
    qid: str,
    gold: str,
    label_tokens: LabelTokens,
    prepend_bos: bool,
    no_think: bool,
    seed: int,
    retrieved_doc_id: Optional[str] = None,
    distractor_doc_id: Optional[str] = None,
) -> ArmResult:
    memory_text = build_memory_context(passage)
    stored_kv = extract_kv_cache_tl(model, memory_text, prepend_bos=prepend_bos)
    cache, _prefix_len = build_multi_memory_cache(model, [stored_kv])
    query_text = build_injection_prompt(claim, no_think=no_think)
    text, first_logits = generate_with_multi_cache(
        model, cache, query_text,
        prepend_bos=prepend_bos,
        max_new_tokens=4,
    )
    return _make_result(
        arm_name, qid, gold, first_logits, text, label_tokens, seed,
        retrieved_doc_id=retrieved_doc_id,
        distractor_doc_id=distractor_doc_id,
    )


# ---------------------------------------------------------------------------
# Retrieval driver
# ---------------------------------------------------------------------------

@dataclass
class RetrievalBundle:
    corpus_ids: List[str]
    corpus_texts: List[str]
    query_ids: List[str]
    query_texts: List[str]
    top1_doc_ids: Dict[str, str]


def _tokenize_for_bm25(tokenizer, texts: List[str]) -> List[List[int]]:
    return [list(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def run_retrieval(
    model,
    tokenizer,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    retrieval_layer: int,
    prepend_bos: bool,
    max_corpus: int,
    max_queries: int,
    alpha_sparse: float,
    max_delta_tokens: int,
    logger: logging.Logger,
) -> RetrievalBundle:
    corpus_ids_all = list(corpus.keys())
    corpus_ids = corpus_ids_all[:max_corpus] if max_corpus < len(corpus_ids_all) else corpus_ids_all
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    query_ids_all = list(queries.keys())
    query_ids = query_ids_all[:max_queries] if max_queries < len(query_ids_all) else query_ids_all
    query_texts = [queries[qid] for qid in query_ids]

    logger.info("Retrieval: %d queries over %d corpus docs", len(query_ids), len(corpus_ids))

    # --- Sparse: BM25 ---
    t0 = time.time()
    corpus_tokens = _tokenize_for_bm25(tokenizer, corpus_texts)
    query_tokens_sparse = _tokenize_for_bm25(tokenizer, query_texts)
    special_ids = set()
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        v = getattr(tokenizer, name, None)
        if v is not None:
            special_ids.add(int(v))
    bm25 = BM25Index.build(
        doc_ids=corpus_ids,
        doc_token_ids=corpus_tokens,
        exclude_token_ids=special_ids,
    )
    logger.info("BM25 built in %.1fs", time.time() - t0)

    # --- Dense: whitened-delta mean-pool ---
    t0 = time.time()
    logger.info("Extracting corpus deltas at layer %d...", retrieval_layer)
    corpus_vecs: List[np.ndarray] = []
    for i, text in enumerate(corpus_texts):
        d = extract_token_deltas(
            model, text, retrieval_layer,
            prepend_bos=prepend_bos, max_tokens=max_delta_tokens,
        )
        if d.shape[0] == 0:
            corpus_vecs.append(np.zeros(model.cfg.d_model, dtype=np.float32))
        else:
            corpus_vecs.append(d.mean(axis=0))
        if (i + 1) % 100 == 0:
            logger.info("  corpus delta %d/%d", i + 1, len(corpus_texts))
    corpus_matrix = np.stack(corpus_vecs, axis=0)

    logger.info("Fitting whitening transform on %d corpus vectors", corpus_matrix.shape[0])
    mean, std, W = fit_whitening_transform(corpus_matrix)
    corpus_white = apply_whitening(corpus_matrix, mean, std, W, l2_normalize=True)

    logger.info("Extracting query deltas...")
    query_vecs: List[np.ndarray] = []
    for q in query_texts:
        d = extract_token_deltas(
            model, q, retrieval_layer,
            prepend_bos=prepend_bos, max_tokens=max_delta_tokens,
        )
        if d.shape[0] == 0:
            query_vecs.append(np.zeros(model.cfg.d_model, dtype=np.float32))
        else:
            query_vecs.append(d.mean(axis=0))
    query_matrix = np.stack(query_vecs, axis=0)
    query_white = apply_whitening(query_matrix, mean, std, W, l2_normalize=True)

    dense_sim = query_white @ corpus_white.T
    logger.info("Dense retrieval in %.1fs", time.time() - t0)

    top1: Dict[str, str] = {}
    for qi, qid in enumerate(query_ids):
        sparse_pairs = bm25.score_query(query_tokens_sparse[qi])
        sparse_score_map = dict(sparse_pairs)
        sparse_vec = np.array(
            [sparse_score_map.get(cid, 0.0) for cid in corpus_ids], dtype=np.float32,
        )
        dense_vec = dense_sim[qi].astype(np.float32)
        fused = hybrid_linear(sparse_vec, dense_vec, alpha=alpha_sparse)
        best_idx = int(np.argmax(fused))
        top1[qid] = corpus_ids[best_idx]

    return RetrievalBundle(
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        top1_doc_ids=top1,
    )


def auto_select_retrieval_layer(
    model,
    prepend_bos: bool,
    sample_texts: List[str],
    layer_window_fraction: float,
    exclude_early_fraction: float,
    logger: logging.Logger,
) -> Tuple[int, Dict[str, Any]]:
    logger.info("Auto-selecting retrieval layer via TwoNN on %d texts...", len(sample_texts))
    id_by_layer = estimate_layer_intrinsic_dimensions(
        model=model, texts=sample_texts, prepend_bos=prepend_bos,
    )
    n_layers = int(model.cfg.n_layers)
    sel = select_rerouting_layers(
        id_by_layer=id_by_layer,
        n_layers=n_layers,
        layer_window_fraction=layer_window_fraction,
        exclude_early_fraction=exclude_early_fraction,
    )
    selected = sorted(int(x) for x in sel.selected_layers)
    if not selected:
        raise RuntimeError("select_rerouting_layers returned empty layer set")
    chosen = selected[0]
    logger.info(
        "TwoNN layer selection: u_shape=%s, candidate_layers=%s, chosen=%d",
        sel.used_u_shape_mode, selected, chosen,
    )
    meta = {
        "id_by_layer": {str(k): float(v) for k, v in id_by_layer.items()},
        "candidate_layers": selected,
        "chosen_layer": chosen,
        "used_u_shape": bool(sel.used_u_shape_mode),
        "layer_window_fraction": layer_window_fraction,
        "exclude_early_fraction": exclude_early_fraction,
        "n_sample_texts": len(sample_texts),
    }
    return chosen, meta


# ---------------------------------------------------------------------------
# Sampling + aggregation
# ---------------------------------------------------------------------------

def balanced_sample_qids(
    labels: Dict[str, str],
    query_ids: List[str],
    n_per_class: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    by_label: Dict[str, List[str]] = {lbl: [] for lbl in LETTERS}
    for qid in query_ids:
        lbl = labels.get(qid)
        if lbl in by_label:
            by_label[lbl].append(qid)
    sampled: List[str] = []
    for lbl in LETTERS:
        pool = by_label[lbl]
        rng.shuffle(pool)
        sampled.extend(pool[:n_per_class])
    rng.shuffle(sampled)
    return sampled


def pick_distractor(
    qid: str,
    gold_doc_id: Optional[str],
    corpus_ids: List[str],
    rng: random.Random,
) -> str:
    while True:
        cand = rng.choice(corpus_ids)
        if cand != gold_doc_id:
            return cand


def aggregate_arm(results: List[ArmResult]) -> Dict[str, Any]:
    if not results:
        return {}
    n = len(results)
    acc = sum(1 for r in results if r.correct) / n
    mean_gap = float(np.mean([r.logit_gap for r in results]))
    confusion: Dict[str, Dict[str, int]] = {lbl: {l: 0 for l in LETTERS} for lbl in LETTERS}
    for r in results:
        if r.gold_label in confusion:
            confusion[r.gold_label][r.predicted_label] = (
                confusion[r.gold_label].get(r.predicted_label, 0) + 1
            )
    return {
        "n": n,
        "accuracy": acc,
        "mean_logit_gap": mean_gap,
        "confusion_gold_to_pred": confusion,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NNM Exp17: E2E Gate -- NNMA vs RAG")
    # Model
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--prepend-bos", action="store_true", default=True)
    # Data
    p.add_argument("--beir-dataset", type=str, default="scifact")
    p.add_argument("--beir-path", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max-corpus", type=int, default=2000)
    p.add_argument("--max-queries", type=int, default=100)
    p.add_argument("--n-per-class", type=int, default=None)
    # Retrieval
    p.add_argument("--retrieval-layer", type=int, default=None,
                   help="If omitted, auto-selected via TwoNN.")
    p.add_argument("--id-sample-size", type=int, default=120)
    p.add_argument("--layer-window-fraction", type=float, default=0.10)
    p.add_argument("--exclude-early-fraction", type=float, default=0.20)
    p.add_argument("--alpha-sparse", type=float, default=0.4)
    p.add_argument("--max-delta-tokens", type=int, default=128)
    # Generation
    p.add_argument("--no-think", action="store_true")
    # Arms / seeds
    p.add_argument("--arms", type=str, default="cold,rag,nnma,random")
    p.add_argument("--seeds", type=str, default="42")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = _create_results_dir("nnm_exp17")
    logger = _setup_logging("nnm_exp17_e2e_gate_tl", results_dir)

    logger.info("args: %s", vars(args))

    arms_to_run = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Model
    logger.info("Loading model %s (4bit=%s)", args.model, args.load_in_4bit)
    model, prepend_bos = load_hooked_model(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        load_in_4bit=args.load_in_4bit,
        prepend_bos=args.prepend_bos,
    )
    tokenizer = model.tokenizer
    logger.info("Model loaded. d_model=%d, n_layers=%d, prepend_bos=%s",
                model.cfg.d_model, model.cfg.n_layers, prepend_bos)

    label_tokens = resolve_label_tokens(tokenizer)
    logger.info("Label token ids: %s",
                dict(zip(label_tokens.letters, label_tokens.token_ids)))

    # Data
    logger.info("Loading BEIR/%s ...", args.beir_dataset)
    corpus = load_beir_corpus(args.beir_dataset, args.beir_path)
    queries = load_beir_queries(args.beir_dataset, args.beir_path)
    qrels = load_beir_qrels(args.beir_dataset, args.split, args.beir_path)
    logger.info("Corpus=%d, Queries=%d, Qrels=%d",
                len(corpus), len(queries), len(qrels))

    logger.info("Loading scifact claim labels (allenai/scifact, %s split)...", args.split)
    gold_labels = load_scifact_claim_labels(split=args.split)
    logger.info("Gold labels loaded: %d", len(gold_labels))

    common_qids = [qid for qid in queries.keys() if qid in gold_labels]
    logger.info("Queries with gold labels: %d / %d", len(common_qids), len(queries))

    if len(common_qids) == 0:
        # Diagnose: show a handful of IDs from each side so we can eyeball formats
        sample_q = list(queries.keys())[:5]
        sample_g = list(gold_labels.keys())[:5]
        raise RuntimeError(
            f"No overlap between BEIR queries and scifact labels. "
            f"Sample BEIR qids: {sample_q}  vs. sample label qids: {sample_g}. "
            f"Likely ID format mismatch."
        )
    if len(common_qids) < 20:
        logger.warning(
            "Only %d common queries -- balanced sampling will be thin.",
            len(common_qids),
        )

    # Layer selection
    layer_meta: Dict[str, Any] = {}
    if args.retrieval_layer is None:
        corpus_ids_for_id = list(corpus.keys())
        sample_n = min(args.id_sample_size, len(corpus_ids_for_id))
        sample_rng = random.Random(seeds[0])
        sample_ids = sample_rng.sample(corpus_ids_for_id, sample_n)
        sample_texts = [corpus[cid] for cid in sample_ids]
        chosen_layer, layer_meta = auto_select_retrieval_layer(
            model=model,
            prepend_bos=prepend_bos,
            sample_texts=sample_texts,
            layer_window_fraction=args.layer_window_fraction,
            exclude_early_fraction=args.exclude_early_fraction,
            logger=logger,
        )
        retrieval_layer = chosen_layer
    else:
        retrieval_layer = int(args.retrieval_layer)
        logger.info("Using manual retrieval layer: %d", retrieval_layer)
        layer_meta = {"chosen_layer": retrieval_layer, "manual_override": True}
    _save_json(layer_meta, results_dir / "layer_selection.json")

    # Retrieval
    retrieval = run_retrieval(
        model=model,
        tokenizer=tokenizer,
        corpus=corpus,
        queries={qid: queries[qid] for qid in common_qids},
        retrieval_layer=retrieval_layer,
        prepend_bos=prepend_bos,
        max_corpus=args.max_corpus,
        max_queries=len(common_qids),
        alpha_sparse=args.alpha_sparse,
        max_delta_tokens=args.max_delta_tokens,
        logger=logger,
    )

    # Sample
    # Forced-choice uses 2 classes (A/B); drop the "/3" that assumed A/B/C
    n_per_class = args.n_per_class or max(1, args.max_queries // max(1, len(LETTERS)))
    sampled_qids = balanced_sample_qids(
        gold_labels, retrieval.query_ids, n_per_class=n_per_class, seed=seeds[0],
    )
    logger.info("Sampled %d queries (target %d per class)", len(sampled_qids), n_per_class)
    dist = Counter(gold_labels[q] for q in sampled_qids)
    logger.info("Sample label distribution: %s", dict(dist))

    # Run arms
    all_results: Dict[str, List[ArmResult]] = {a: [] for a in arms_to_run}
    for seed in seeds:
        logger.info("=" * 60)
        logger.info("SEED %d", seed)
        logger.info("=" * 60)
        rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        for qid in sampled_qids:
            claim = queries[qid]
            gold = gold_labels[qid]
            top1_doc_id = retrieval.top1_doc_ids.get(qid)
            top1_text = corpus.get(top1_doc_id, "") if top1_doc_id else ""
            distractor_id = pick_distractor(qid, top1_doc_id, retrieval.corpus_ids, rng)
            distractor_text = corpus.get(distractor_id, "")

            if "cold" in arms_to_run:
                all_results["cold"].append(
                    run_cold(model, claim, qid, gold, label_tokens, prepend_bos, args.no_think, seed)
                )
            if "rag" in arms_to_run and top1_text:
                all_results["rag"].append(
                    run_rag(model, claim, top1_text, qid, gold, top1_doc_id,
                            label_tokens, prepend_bos, args.no_think, seed)
                )
            if "nnma" in arms_to_run and top1_text:
                all_results["nnma"].append(
                    run_injection("nnma", model, claim, top1_text, qid, gold,
                                  label_tokens, prepend_bos, args.no_think, seed,
                                  retrieved_doc_id=top1_doc_id)
                )
            if "random" in arms_to_run and distractor_text:
                all_results["random"].append(
                    run_injection("random", model, claim, distractor_text, qid, gold,
                                  label_tokens, prepend_bos, args.no_think, seed,
                                  distractor_doc_id=distractor_id)
                )

        per_seed = {a: aggregate_arm([r for r in all_results[a] if r.seed == seed])
                    for a in arms_to_run}
        logger.info("Seed %d summary: %s", seed, json.dumps(per_seed, indent=2))
        _save_json(per_seed, results_dir / f"summary_seed_{seed}.json")

    # Final aggregate
    logger.info("=" * 60)
    logger.info("FINAL AGGREGATE (across %d seed(s))", len(seeds))
    logger.info("=" * 60)
    final_summary = {
        "config": vars(args),
        "seeds": seeds,
        "n_sampled_queries": len(sampled_qids),
        "label_distribution": dict(dist),
        "retrieval_layer": retrieval_layer,
        "layer_selection": layer_meta,
        "arms": {a: aggregate_arm(all_results[a]) for a in arms_to_run},
    }
    logger.info(json.dumps(final_summary["arms"], indent=2))
    _save_json(final_summary, results_dir / "summary.json")
    _save_json(
        {a: [asdict(r) for r in all_results[a]] for a in arms_to_run},
        results_dir / "per_item.json",
    )
    logger.info("Results saved to %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
