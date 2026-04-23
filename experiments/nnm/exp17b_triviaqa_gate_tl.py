#!/usr/bin/env python3
"""Experiment 17b (NNM): E2E Gate on TriviaQA-rc (open-ended QA).

SELF-CONTAINED. Drop next to exp17_requirements.txt and run:
    pip install -r exp17_requirements.txt
    python3 exp17b_triviaqa_gate_tl.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --max-queries 100 --seeds 42,43,44 \
        --retrieval-layer 26 --no-think

Tests NNMA vs RAG on open-ended trivia Q&A with ORACLE retrieval:
we pick a gold passage per question (from TriviaQA search_results that
actually contain an answer alias), so retrieval quality is fixed and
we isolate the Injection-vs-Prepending effect.

Four arms:
  cold    : question only, no context
  rag     : gold passage as text context (honest baseline)
  nnma    : gold passage as KV-cache injection
  random  : random other-question passage as KV-cache injection (noise)

Metrics: Exact Match (SQuAD-style normalization) + Token-F1 against
the official TriviaQA answer aliases.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    try:
        from transformer_lens import HookedTransformer
    except ImportError as e:
        raise RuntimeError("`pip install transformer_lens`") from e

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
            model_name, hf_model=hf_model,
            default_prepend_bos=prepend_bos, **common_kwargs,
        )
        effective = prepend_bos
    except ValueError as exc:
        if "add_bos_token = True but bos_token = None" not in str(exc):
            raise
        from transformers import AutoTokenizer
        hf_token = os.environ.get("HF_TOKEN") or None
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_bos_token=False, trust_remote_code=True,
            use_fast=True, token=hf_token, local_files_only=local_files_only,
        )
        model = HookedTransformer.from_pretrained(
            model_name, hf_model=hf_model, tokenizer=tokenizer,
            default_prepend_bos=False, **common_kwargs,
        )
        effective = False

    model.eval()
    return model, effective


# ---------------------------------------------------------------------------
# TriviaQA oracle loader
# ---------------------------------------------------------------------------

@dataclass
class TQItem:
    qid: str
    question: str
    aliases: List[str]
    passage: str  # oracle: a search_context that contains at least one alias


def _contains_alias(text: str, aliases: Sequence[str]) -> bool:
    t = text.lower()
    for a in aliases:
        a = a.strip().lower()
        if a and a in t:
            return True
    return False


def load_triviaqa_oracle(
    split: str,
    max_queries: int,
    seed: int,
    passage_char_limit: int,
    logger: logging.Logger,
) -> List[TQItem]:
    """Load TriviaQA-rc, filter to queries with an answer-containing passage."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("`pip install datasets`") from e

    logger.info("Loading mandarjoshi/trivia_qa rc split=%s ...", split)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)
    logger.info("Full size: %d", len(ds))

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    picked: List[TQItem] = []
    seen = 0
    for idx in indices:
        row = ds[idx]
        question = str(row["question"]).strip()
        ans = row["answer"]
        aliases: List[str] = []
        for key in ("aliases", "normalized_aliases", "value", "normalized_value"):
            v = ans.get(key) if isinstance(ans, dict) else None
            if isinstance(v, list):
                aliases.extend(str(a) for a in v if a)
            elif isinstance(v, str) and v:
                aliases.append(v)
        aliases = sorted(set(a.strip() for a in aliases if a.strip()))
        if not aliases:
            continue

        # Find an answer-containing search_result
        chosen_passage: Optional[str] = None
        srs = row.get("search_results") or {}
        contexts: List[str] = []
        if isinstance(srs, dict):
            # HF dataset lists contexts as parallel lists
            ctx_list = srs.get("search_context") or srs.get("context") or []
            if isinstance(ctx_list, list):
                contexts.extend(str(c or "") for c in ctx_list)
        elif isinstance(srs, list):
            for s in srs:
                if isinstance(s, dict):
                    contexts.append(str(s.get("search_context", "") or ""))

        for ctx in contexts:
            if _contains_alias(ctx, aliases):
                chosen_passage = ctx.strip()
                break

        # Fallback: entity_pages wiki_context first N chars
        if chosen_passage is None:
            ep = row.get("entity_pages") or {}
            wiki_contexts: List[str] = []
            if isinstance(ep, dict):
                wc_list = ep.get("wiki_context") or []
                if isinstance(wc_list, list):
                    wiki_contexts.extend(str(c or "") for c in wc_list)
            elif isinstance(ep, list):
                for p in ep:
                    if isinstance(p, dict):
                        wiki_contexts.append(str(p.get("wiki_context", "") or ""))
            for ctx in wiki_contexts:
                if _contains_alias(ctx[: 4 * passage_char_limit], aliases):
                    chosen_passage = ctx.strip()
                    break

        seen += 1
        if chosen_passage is None:
            continue

        # Truncate to a manageable size (whitened-delta extraction truncates at
        # --max-delta-tokens anyway; this keeps the RAG prompt bounded too).
        chosen_passage = chosen_passage[:passage_char_limit].strip()

        picked.append(
            TQItem(
                qid=str(row.get("question_id", f"idx{idx}")),
                question=question,
                aliases=aliases,
                passage=chosen_passage,
            )
        )
        if len(picked) >= max_queries:
            break

    logger.info(
        "Picked %d queries with oracle passage (scanned %d)", len(picked), seen
    )
    return picked


# ---------------------------------------------------------------------------
# SQuAD-style normalization + EM + Token-F1
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = _ARTICLES.sub(" ", s)
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, aliases: Sequence[str]) -> int:
    p = normalize_answer(pred)
    if not p:
        return 0
    for a in aliases:
        if normalize_answer(a) == p:
            return 1
    return 0


def token_f1(pred: str, aliases: Sequence[str]) -> float:
    p_toks = normalize_answer(pred).split()
    if not p_toks:
        return 0.0
    best = 0.0
    for a in aliases:
        a_toks = normalize_answer(a).split()
        if not a_toks:
            continue
        common = Counter(p_toks) & Counter(a_toks)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec = n_same / len(p_toks)
        rec = n_same / len(a_toks)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best = f1
    return best


# ---------------------------------------------------------------------------
# KV cache extraction (post-QKnorm, pre-RoPE)
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
        tokens, return_type=None,
        names_filter=lambda name: name in names,
        remove_batch_dim=False, prepend_bos=False,
    )
    keys: Dict[int, np.ndarray] = {}
    values: Dict[int, np.ndarray] = {}
    for layer in range(int(model.cfg.n_layers)):
        k_name = f"blocks.{layer}.attn.hook_k"
        v_name = f"blocks.{layer}.attn.hook_v"
        k_tensor = cache[k_name].detach()
        v = cache[v_name].detach().cpu().numpy()
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
        keys=keys, values=values, token_ids=token_ids,
        text=text, prefix_len=prefix_len,
    )


def _import_kv_cache_cls():
    """TL renamed the KV cache module between versions; try known paths."""
    candidates = [
        ("transformer_lens.past_key_value_caching", "HookedTransformerKeyValueCache"),
        ("transformer_lens", "HookedTransformerKeyValueCache"),
        ("transformer_lens.components", "HookedTransformerKeyValueCache"),
        ("transformer_lens.cache", "HookedTransformerKeyValueCache"),
        ("transformer_lens.model.caching", "HookedTransformerKeyValueCache"),
    ]
    errors = []
    for mod_path, cls_name in candidates:
        try:
            import importlib
            m = importlib.import_module(mod_path)
            cls = getattr(m, cls_name)
            return cls
        except (ImportError, AttributeError) as e:
            errors.append(f"{mod_path}.{cls_name}: {e}")
    raise ImportError(
        "Could not locate HookedTransformerKeyValueCache. "
        "Tried:\n  " + "\n  ".join(errors)
    )


def build_multi_memory_cache(model, stored_kvs: List[StoredTLKVCache]):
    HookedTransformerKeyValueCache = _import_kv_cache_cls()

    device = model.W_E.device
    dtype = model.W_E.dtype
    n_layers = int(model.cfg.n_layers)
    cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, device=device, batch_size=1,
    )
    total_prefix_len = sum(kv.prefix_len for kv in stored_kvs)
    for layer in range(n_layers):
        k_parts, v_parts = [], []
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _collect_stop_ids(tokenizer) -> set:
    """Stop on every special chat/eos token common chat models emit."""
    stops: set = set()
    for attr in ("eos_token_id", "pad_token_id"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            stops.add(int(v))
    # Cover Qwen ChatML, Llama-3 / OAI, Gemma, Mistral specials
    for name in (
        "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>",
        "<end_of_turn>", "<eos>", "<|eot|>", "</s>",
    ):
        try:
            ids = tokenizer.encode(name, add_special_tokens=False)
            if ids and len(ids) == 1:
                stops.add(int(ids[0]))
        except Exception:
            pass
    stops.discard(None)
    return stops


def _decode_clean(tokenizer, token_ids: List[int], stop_on_newline: bool) -> str:
    """Decode with skip_special_tokens=True and optional newline trim."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    if stop_on_newline:
        text = text.split("\n", 1)[0]
    return text


def _to_tokens_smart_bos(model, prompt: str, prepend_bos: bool):
    """Avoid double-BOS when apply_chat_template already inserted one."""
    bos_tok = getattr(model.tokenizer, "bos_token", None)
    if prepend_bos and bos_tok and prompt.startswith(bos_tok):
        return model.to_tokens(prompt, prepend_bos=False)
    return model.to_tokens(prompt, prepend_bos=prepend_bos)


# Mutable diagnostic state — prints what the model *wants* to emit when output
# ends up empty. Only active for the first _DIAG_BUDGET empty cases to avoid
# flooding the log.
_DIAG_BUDGET = 3
_DIAG_SEEN = {"count": 0}


def _diagnose_empty(model, first_logits: torch.Tensor, stops: set, prompt_tail: str) -> None:
    if _DIAG_SEEN["count"] >= _DIAG_BUDGET:
        return
    _DIAG_SEEN["count"] += 1
    vec = first_logits[0] if first_logits.dim() == 2 else first_logits
    topk = torch.topk(vec, 5)
    lines = []
    for rank, (tid, lg) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
        tok_str = model.tokenizer.decode([int(tid)]).replace("\n", "\\n")
        marker = "<STOP>" if int(tid) in stops else ""
        lines.append(f"  rank{rank} id={tid} logit={lg:.2f} tok={tok_str!r} {marker}")
    logging.getLogger(__name__).warning(
        "Empty generation. Prompt tail (last 80 chars): %r\n%s",
        prompt_tail[-80:], "\n".join(lines),
    )


@torch.no_grad()
def generate_with_multi_cache(
    model, cache, query: str,
    prepend_bos: bool, max_new_tokens: int,
    stop_on_newline: bool = True,
) -> str:
    query_tokens = _to_tokens_smart_bos(model, query, prepend_bos)
    logits, _ = model.run_with_cache(
        query_tokens, return_type="logits",
        past_kv_cache=cache, prepend_bos=False, remove_batch_dim=False,
    )
    generated_ids: List[int] = []
    stops = _collect_stop_ids(model.tokenizer) if model.tokenizer else set()
    first_logits_for_diag = logits[:, -1].clone()
    for _ in range(max_new_tokens):
        nxt_id = int(torch.argmax(logits[:, -1], dim=-1).item())
        if nxt_id in stops:
            break
        generated_ids.append(nxt_id)
        nxt = torch.tensor([[nxt_id]], device=query_tokens.device, dtype=query_tokens.dtype)
        logits, _ = model.run_with_cache(
            nxt, return_type="logits",
            past_kv_cache=cache, prepend_bos=False, remove_batch_dim=False,
        )
    text = _decode_clean(model.tokenizer, generated_ids, stop_on_newline)
    if not text.strip():
        _diagnose_empty(model, first_logits_for_diag, stops, query)
    return text


@torch.no_grad()
def generate_plain(
    model, prompt: str, prepend_bos: bool,
    max_new_tokens: int, stop_on_newline: bool = True,
) -> str:
    tokens = _to_tokens_smart_bos(model, prompt, prepend_bos)
    logits = model(tokens, return_type="logits")
    generated_ids: List[int] = []
    stops = _collect_stop_ids(model.tokenizer) if model.tokenizer else set()
    first_logits_for_diag = logits[:, -1].clone()
    for _ in range(max_new_tokens):
        nxt_id = int(torch.argmax(logits[:, -1], dim=-1).item())
        if nxt_id in stops:
            break
        generated_ids.append(nxt_id)
        nxt = torch.tensor([[nxt_id]], device=tokens.device, dtype=tokens.dtype)
        tokens = torch.cat([tokens, nxt], dim=1)
        logits = model(tokens, return_type="logits")
    text = _decode_clean(model.tokenizer, generated_ids, stop_on_newline)
    if not text.strip():
        _diagnose_empty(model, first_logits_for_diag, stops, prompt)
    return text


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM = (
    "You answer trivia questions with a short factual answer. "
    "Do not add explanations. Give only the answer."
)
Q_ONLY = "Question: {question}\nAnswer:"
Q_WITH_CTX = "Context: {passage}\n\nQuestion: {question}\nAnswer:"
MEM_CTX_TEMPLATE = "Context: {passage}"  # used for KV injection


def _wrap_as_chat(tokenizer, body: str, no_think: bool) -> str:
    """If tokenizer has a chat template, use it. Else fall back to plain text.

    Needed for Gemma-3/Llama-3: they emit an EOS immediately without the
    canonical <start_of_turn>user ... format. Qwen2.5 is more tolerant but
    also accepts its own ChatML template.
    """
    no_think_prefix = "/no_think\n" if no_think else ""
    user_msg = no_think_prefix + body
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl and hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return text
        except Exception:
            pass
    # Fallback: plain text (bare completion style)
    return f"{user_msg} "


def build_cold_prompt(q: str, no_think: bool, tokenizer=None) -> str:
    body = f"{SYSTEM}\n\n{Q_ONLY.format(question=q)}"
    if tokenizer is None:
        prefix = "/no_think\n" if no_think else ""
        return f"{prefix}{body} "
    return _wrap_as_chat(tokenizer, body, no_think)


def build_rag_prompt(q: str, p: str, no_think: bool, tokenizer=None) -> str:
    body = f"{SYSTEM}\n\n{Q_WITH_CTX.format(passage=p, question=q)}"
    if tokenizer is None:
        prefix = "/no_think\n" if no_think else ""
        return f"{prefix}{body} "
    return _wrap_as_chat(tokenizer, body, no_think)


def build_injection_prompt(q: str, no_think: bool, tokenizer=None) -> str:
    # Same body as cold; context lives in the KV prefix
    body = f"{SYSTEM}\n\n{Q_ONLY.format(question=q)}"
    if tokenizer is None:
        prefix = "/no_think\n" if no_think else ""
        return f"{prefix}{body} "
    return _wrap_as_chat(tokenizer, body, no_think)


# ---------------------------------------------------------------------------
# Arm runners
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    arm: str
    qid: str
    question: str
    gold_aliases: List[str]
    prediction: str
    em: int
    f1: float
    seed: int
    used_passage_qid: Optional[str] = None
    used_passage_snippet: Optional[str] = None


def _score_and_result(
    arm: str, item: TQItem, pred: str, seed: int,
    used_passage_qid: Optional[str] = None,
    used_passage_snippet: Optional[str] = None,
) -> ArmResult:
    em = exact_match(pred, item.aliases)
    f1 = token_f1(pred, item.aliases)
    return ArmResult(
        arm=arm,
        qid=item.qid,
        question=item.question,
        gold_aliases=list(item.aliases[:5]),
        prediction=pred.strip(),
        em=em,
        f1=f1,
        seed=seed,
        used_passage_qid=used_passage_qid,
        used_passage_snippet=(used_passage_snippet[:120] if used_passage_snippet else None),
    )


def run_cold(model, item: TQItem, prepend_bos, no_think, seed, max_new) -> ArmResult:
    prompt = build_cold_prompt(item.question, no_think, tokenizer=model.tokenizer)
    pred = generate_plain(model, prompt, prepend_bos, max_new_tokens=max_new)
    return _score_and_result("cold", item, pred, seed)


def run_rag(model, item: TQItem, prepend_bos, no_think, seed, max_new) -> ArmResult:
    prompt = build_rag_prompt(item.question, item.passage, no_think, tokenizer=model.tokenizer)
    pred = generate_plain(model, prompt, prepend_bos, max_new_tokens=max_new)
    return _score_and_result(
        "rag", item, pred, seed,
        used_passage_qid=item.qid, used_passage_snippet=item.passage,
    )


def run_injection(
    arm_name: str, model, item: TQItem, passage: str,
    prepend_bos, no_think, seed, max_new,
    used_passage_qid: Optional[str] = None,
) -> ArmResult:
    mem_text = MEM_CTX_TEMPLATE.format(passage=passage)
    stored_kv = extract_kv_cache_tl(model, mem_text, prepend_bos=prepend_bos)
    cache, _ = build_multi_memory_cache(model, [stored_kv])
    qprompt = build_injection_prompt(item.question, no_think, tokenizer=model.tokenizer)
    pred = generate_with_multi_cache(
        model, cache, qprompt,
        prepend_bos=prepend_bos, max_new_tokens=max_new,
    )
    return _score_and_result(
        arm_name, item, pred, seed,
        used_passage_qid=used_passage_qid, used_passage_snippet=passage,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_arm(results: List[ArmResult]) -> Dict[str, Any]:
    if not results:
        return {}
    n = len(results)
    em = sum(r.em for r in results) / n
    f1 = float(np.mean([r.f1 for r in results]))
    empty_rate = sum(1 for r in results if not r.prediction.strip()) / n
    return {
        "n": n,
        "exact_match": em,
        "token_f1": f1,
        "empty_prediction_rate": empty_rate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NNM Exp17b: TriviaQA E2E Gate")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--prepend-bos", action="store_true", default=True)

    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--max-queries", type=int, default=100)
    p.add_argument("--passage-char-limit", type=int, default=1500,
                   help="Char cap for oracle passages (keeps RAG prompt bounded).")

    p.add_argument("--retrieval-layer", type=int, default=26,
                   help="Only used by extract_kv_cache_tl indirectly; kept for symmetry.")
    p.add_argument("--max-new-tokens", type=int, default=32)

    p.add_argument("--no-think", action="store_true")
    p.add_argument("--arms", type=str, default="cold,rag,nnma,random")
    p.add_argument("--seeds", type=str, default="42")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = _create_results_dir("nnm_exp17b")
    logger = _setup_logging("nnm_exp17b_triviaqa_gate_tl", results_dir)
    logger.info("args: %s", vars(args))

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Model
    logger.info("Loading model %s (4bit=%s)", args.model, args.load_in_4bit)
    model, prepend_bos = load_hooked_model(
        model_name=args.model, device=args.device, dtype=args.dtype,
        local_files_only=args.local_files_only,
        load_in_4bit=args.load_in_4bit, prepend_bos=args.prepend_bos,
    )
    logger.info("Model loaded. d_model=%d, n_layers=%d, prepend_bos=%s",
                model.cfg.d_model, model.cfg.n_layers, prepend_bos)

    # Data (oracle filter already applied)
    items = load_triviaqa_oracle(
        split=args.split,
        max_queries=args.max_queries,
        seed=seeds[0],
        passage_char_limit=args.passage_char_limit,
        logger=logger,
    )
    if not items:
        raise RuntimeError("No TriviaQA queries with oracle passages found.")

    # Precompute distractor passage pool (one per other item)
    passages_pool: List[Tuple[str, str]] = [(it.qid, it.passage) for it in items]

    all_results: Dict[str, List[ArmResult]] = {a: [] for a in arms}

    for seed in seeds:
        logger.info("=" * 60)
        logger.info("SEED %d", seed)
        logger.info("=" * 60)
        rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        for i, it in enumerate(items):
            # Random distractor: another query's passage, not this one's
            other_idx = rng.randrange(len(passages_pool))
            tries = 0
            while passages_pool[other_idx][0] == it.qid and tries < 5:
                other_idx = rng.randrange(len(passages_pool))
                tries += 1
            distractor_qid, distractor_passage = passages_pool[other_idx]

            if "cold" in arms:
                all_results["cold"].append(
                    run_cold(model, it, prepend_bos, args.no_think, seed, args.max_new_tokens)
                )
            if "rag" in arms:
                all_results["rag"].append(
                    run_rag(model, it, prepend_bos, args.no_think, seed, args.max_new_tokens)
                )
            if "nnma" in arms:
                all_results["nnma"].append(
                    run_injection(
                        "nnma", model, it, it.passage,
                        prepend_bos, args.no_think, seed, args.max_new_tokens,
                        used_passage_qid=it.qid,
                    )
                )
            if "random" in arms:
                all_results["random"].append(
                    run_injection(
                        "random", model, it, distractor_passage,
                        prepend_bos, args.no_think, seed, args.max_new_tokens,
                        used_passage_qid=distractor_qid,
                    )
                )

            if (i + 1) % 10 == 0:
                logger.info("  seed=%d, progressed %d/%d queries", seed, i + 1, len(items))

        per_seed = {a: aggregate_arm([r for r in all_results[a] if r.seed == seed])
                    for a in arms}
        logger.info("Seed %d: %s", seed, json.dumps(per_seed, indent=2))
        _save_json(per_seed, results_dir / f"summary_seed_{seed}.json")

    logger.info("=" * 60)
    logger.info("FINAL AGGREGATE (%d seed(s))", len(seeds))
    logger.info("=" * 60)
    final_summary = {
        "config": vars(args),
        "seeds": seeds,
        "n_queries": len(items),
        "arms": {a: aggregate_arm(all_results[a]) for a in arms},
    }
    logger.info(json.dumps(final_summary["arms"], indent=2))
    _save_json(final_summary, results_dir / "summary.json")
    _save_json(
        {a: [asdict(r) for r in all_results[a]] for a in arms},
        results_dir / "per_item.json",
    )
    logger.info("Saved to %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
