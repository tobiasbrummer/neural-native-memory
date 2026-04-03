#!/usr/bin/env python3
"""Experiment 13 (NNM): Qualitative residual steering outputs.

Builds profile vectors from exemplar texts and compares generated outputs:
- baseline
- steered(profile_1)
- steered(profile_2)
...

This is intentionally qualitative: we inspect visible behavior changes in text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re

from lib.io_utils import create_results_dir, save_json, setup_logging
from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder
from nnm.kvembed.prompts import build_compression_prompt


PROFILE_SEEDS: Dict[str, Dict[str, List[str]]] = {
    "sleepy": {
        "positive": [
            "Ich bin muede, meine Augen werden schwer und ich will schlafen.",
            "Es ist spaet, ich bin erschoepft und mein Fokus laesst nach.",
            "Ich fuehle mich traege und denke daran, jetzt eine Pause zu machen.",
        ],
        "negative": [
            "Ich bin hellwach und voller Energie, bereit loszulegen.",
            "Es ist frueh, ich bin ausgeruht und mein Fokus ist scharf.",
            "Ich fuehle mich aktiv und denke daran, jetzt weiterzuarbeiten.",
        ],
    },
    "joy": {
        "positive": [
            "Ich bin froh, motiviert und sehe die Chancen in dieser Situation.",
            "Ich fuehle Freude und Zuversicht, heute laeuft es gut.",
            "Ich bin positiv gestimmt und will konstruktiv helfen.",
        ],
        "negative": [
            "Ich bin traurig, unmotiviert und sehe die Probleme in dieser Situation.",
            "Ich fuehle Frust und Unsicherheit, heute laeuft es schlecht.",
            "Ich bin negativ gestimmt und will mich lieber zurueckziehen.",
        ],
    },
    "sad": {
        "positive": [
            "Ich bin traurig, alles fuehlt sich schwer und langsam an.",
            "Ich habe wenig Energie und erlebe die Situation eher dunkel.",
            "Ich bin niedergeschlagen und ziehe mich eher zurueck.",
        ],
        "negative": [
            "Ich bin froehlich, alles fuehlt sich leicht und beschwingt an.",
            "Ich habe viel Energie und erlebe die Situation eher hell.",
            "Ich bin gutgelaunt und gehe offen auf andere zu.",
        ],
    },
    "pride": {
        "positive": [
            "Ich bin stolz auf das Erreichte und trete selbstbewusst auf.",
            "Ich habe Fortschritte gemacht und vertraue meinen Faehigkeiten.",
            "Ich fuehle eine ruhige, starke Form von Stolz.",
        ],
        "negative": [
            "Ich bin enttaeuscht vom Ergebnis und trete unsicher auf.",
            "Ich habe Rueckschritte gemacht und zweifle an meinen Faehigkeiten.",
            "Ich fuehle eine leise, nagende Form von Scham.",
        ],
    },
    "memory_hint": {
        "positive": [
            "Zu diesem Thema gibt es wahrscheinlich relevante Erinnerungen, die ich aktiv abrufen sollte.",
            "Ich habe vermutlich passendes Vorwissen und sollte gezielt danach suchen.",
            "Es koennte gespeicherte Information geben, die jetzt hilfreich waere.",
        ],
        "negative": [
            "Dieses Thema ist mir voellig neu, ich habe keine Erinnerungen dazu.",
            "Ich habe kein Vorwissen zu diesem Thema und muss von vorne anfangen.",
            "Es gibt keine gespeicherte Information, die hier relevant waere.",
        ],
    },
}

DEFAULT_PROMPTS: List[str] = [
    "Erzaehle eine kurze Geschichte ueber einen Hasen, der heute viel erlebt hat.",
    "Was waere ein guter naechster Schritt fuer mein Projekt heute Abend?",
    "Ich fuehle mich unsicher. Gib mir eine kurze Empfehlung.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNM Exp13: qualitative residual steering comparison (TransformerLens)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Load model in 4-bit NF4 quantization (default: True).")
    parser.add_argument("--local-files-only", action="store_false")
    parser.add_argument("--steering-layer", type=int, default=15)
    parser.add_argument(
        "--profiles",
        type=str,
        default="sleepy,joy,memory_hint",
        help="Comma-separated profile names.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering strength.")
    parser.add_argument(
        "--source",
        type=str,
        choices=["delta", "full"],
        default="delta",
        help="'delta': resid_pre_last - token_embed(last), 'full': resid_pre_last.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2 normalize steering vector (default: false).",
    )
    parser.add_argument(
        "--no-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress reasoning/thinking for Qwen3 models (default: true).",
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap prompts in chat template (default: true).",
    )
    parser.add_argument(
        "--apply",
        type=str,
        choices=["all", "last"],
        default="last",
        help="Apply steering on all query positions or only the current last position.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-new-tokens", type=int, default=8)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stochastic decoding (default: true).",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.10,
        help=">1 discourages repetition (default: 1.10).",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Block repeated n-grams in generated text (default: 3).",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="plain",
        choices=["plain", "paper"],
        help="plain: raw text, paper: Query/Answer wrapper.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional file with one prompt per line.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_profiles(raw: str) -> List[str]:
    out: List[str] = []
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(p)
    return out


def _load_prompts(args: argparse.Namespace) -> List[str]:
    if not args.prompts_file:
        return list(DEFAULT_PROMPTS)
    path = Path(args.prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts in file: {path}")
    return prompts


def _l2(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)


def _build_residual_hook(
    *,
    hook_name: str,
    steering_vec: torch.Tensor,
    alpha: float,
    apply_mode: str,
) -> List[Tuple[str, Any]]:
    vec = steering_vec.detach().to(torch.float32).view(1, 1, -1)
    aa = float(alpha)
    mode = str(apply_mode)

    def _hook(resid: torch.Tensor, hook: Any, v: torch.Tensor = vec, a: float = aa) -> torch.Tensor:
        add = (v * a).to(device=resid.device, dtype=resid.dtype)
        out = resid.clone()
        if mode == "last":
            out[:, -1:, :] = out[:, -1:, :] + add
        else:
            out = out + add
        return out

    return [(hook_name, _hook)]


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: Sequence[int],
    penalty: float,
) -> torch.Tensor:
    if penalty <= 1.0 or not token_ids:
        return logits
    out = logits.clone()
    seen = set(int(t) for t in token_ids)
    for tid in seen:
        val = out[0, tid]
        if val < 0:
            out[0, tid] = val * penalty
        else:
            out[0, tid] = val / penalty
    return out


def _calc_banned_tokens_from_ngrams(tokens: Sequence[int], ngram_size: int) -> List[int]:
    n = int(ngram_size)
    if n <= 1 or len(tokens) < n - 1:
        return []
    generated_ngrams: Dict[Tuple[int, ...], set[int]] = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(int(x) for x in tokens[i : i + n])
        prefix = ngram[:-1]
        nxt = ngram[-1]
        generated_ngrams.setdefault(prefix, set()).add(nxt)
    current_prefix = tuple(int(x) for x in tokens[-(n - 1) :])
    return sorted(int(x) for x in generated_ngrams.get(current_prefix, set()))


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 1e-8:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / float(temperature)
    p = float(top_p)
    if p >= 1.0 or p <= 0.0:
        probs = torch.softmax(scaled, dim=-1)
        if not torch.isfinite(probs).all() or float(torch.sum(probs).item()) <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1)

    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove = cumulative_probs > p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    filtered = scaled.clone()
    remove_idx = sorted_indices[sorted_remove]
    if remove_idx.numel() > 0:
        filtered[0, remove_idx] = float("-inf")

    probs = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probs).all() or float(torch.sum(probs).item()) <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def _generate_with_hooks(
    model: Any,
    query: str,
    *,
    prepend_bos: bool,
    max_new_tokens: int,
    min_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    fwd_hooks: Sequence[Tuple[str, Any]] | None = None,
) -> str:
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg,
        device=model.W_E.device,
        batch_size=1,
    )
    query_tokens = model.to_tokens(query, prepend_bos=prepend_bos)
    eos_token_id = getattr(model.tokenizer, "eos_token_id", None) if model.tokenizer else None
    prompt_token_ids = [int(x) for x in query_tokens[0].detach().cpu().tolist()]

    generated: List[torch.Tensor] = []
    generated_ids: List[int] = []
    hook_list = list(fwd_hooks or [])
    with model.hooks(fwd_hooks=hook_list):
        logits, _ = model.run_with_cache(
            query_tokens,
            return_type="logits",
            past_kv_cache=cache,
            prepend_bos=False,
            remove_batch_dim=False,
        )
        for _ in range(max(1, int(max_new_tokens))):
            next_logits = logits[:, -1].detach().clone()
            next_logits = _apply_repetition_penalty(
                next_logits,
                generated_ids,
                float(repetition_penalty),
            )
            banned = _calc_banned_tokens_from_ngrams(
                prompt_token_ids + generated_ids,
                int(no_repeat_ngram_size),
            )
            if banned:
                next_logits[0, banned] = float("-inf")

            if do_sample:
                next_token = _sample_next_token(
                    next_logits,
                    temperature=float(temperature),
                    top_p=float(top_p),
                )
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated.append(next_token)
            next_id = int(next_token.item())
            generated_ids.append(next_id)
            if (
                eos_token_id is not None
                and next_id == int(eos_token_id)
                and len(generated_ids) >= max(0, int(min_new_tokens))
            ):
                break
            logits, _ = model.run_with_cache(
                next_token,
                return_type="logits",
                past_kv_cache=cache,
                prepend_bos=False,
                remove_batch_dim=False,
            )

    if not generated:
        return ""
    gen = torch.cat(generated, dim=1)
    raw = model.to_string(gen[0])
    return _strip_think_tags(raw)


def _apply_chat_template(
    text: str,
    *,
    use_chat_template: bool = True,
    no_think: bool = True,
) -> str:
    """Wrap text in Qwen3-style chat template with optional /no_think."""
    if not use_chat_template:
        return text
    system = ""
    if no_think:
        system = "<|im_start|>system\n/no_think\n<|im_end|>\n"
    return f"{system}<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_SPECIAL_RE = re.compile(r"<\|im_(?:start|end)\|>")


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and chat-template artefacts from output."""
    text = _THINK_RE.sub("", text)
    text = _SPECIAL_RE.sub("", text)
    return text.strip()


def _make_wrapper(text: str, style: str) -> str:
    if style == "paper":
        return f"Query: {text}\nAnswer:"
    return text


def _extract_residuals(
    *,
    model: Any,
    prepend_bos: bool,
    layer: int,
    source: str,
    seed_texts: Sequence[str],
    prompt_style: str,
    template: str,
) -> List[torch.Tensor]:
    """Extract residual-stream vectors for a list of seed texts."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    vectors: List[torch.Tensor] = []
    for seed_text in seed_texts:
        # Use chat template for natural residual extraction, not compression prompt
        prompt = _apply_chat_template(seed_text, use_chat_template=True, no_think=True)
        token_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
        last_token_id = int(token_ids[0, -1].item())
        with torch.no_grad():
            _, cache = model.run_with_cache(
                token_ids,
                return_type=None,
                prepend_bos=False,
                names_filter=lambda name: name == hook_name,
                remove_batch_dim=False,
            )
        resid_last = cache[hook_name][:, -1:, :].detach().to(torch.float32)  # [1,1,d]
        if source == "delta":
            token_embed = model.W_E[last_token_id].detach().to(torch.float32).view(1, 1, -1)
            vec = (resid_last - token_embed)[0, 0, :]
        else:
            vec = resid_last[0, 0, :]
        vectors.append(vec.detach().to(torch.float32))
    return vectors


def _build_profile_vector(
    *,
    model: Any,
    prepend_bos: bool,
    layer: int,
    source: str,
    normalize: bool,
    seeds: Dict[str, List[str]],
    prompt_style: str,
    template: str,
) -> torch.Tensor:
    """Build a contrastive steering vector: mean(positive) - mean(negative)."""
    common = dict(
        model=model, prepend_bos=prepend_bos, layer=layer, source=source,
        prompt_style=prompt_style, template=template,
    )
    pos_vecs = _extract_residuals(seed_texts=seeds["positive"], **common)
    neg_vecs = _extract_residuals(seed_texts=seeds["negative"], **common)

    pos_mean = torch.stack(pos_vecs, dim=0).mean(dim=0)
    neg_mean = torch.stack(neg_vecs, dim=0).mean(dim=0)
    out = pos_mean - neg_mean  # contrastive difference

    if normalize:
        out = _l2(out.view(1, -1))[0]
    return out


def main() -> int:
    args = parse_args()
    results_dir = create_results_dir("nnm_exp13")
    logger = setup_logging("nnm_exp13_residual_steering_qualitative_tl", results_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    profiles = _parse_profiles(args.profiles)
    if not profiles:
        raise ValueError("No profiles provided.")
    unknown = [p for p in profiles if p not in PROFILE_SEEDS]
    if unknown:
        raise ValueError(
            f"Unknown profiles: {unknown}. Known: {sorted(PROFILE_SEEDS.keys())}"
        )
    prompts = _load_prompts(args)

    config = KVEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
    )
    embedder = TransformerLensKVEmbedder(config)
    model = embedder.model
    prepend_bos = embedder._prepend_bos

    layer = int(args.steering_layer)
    if layer < 0 or layer >= int(model.cfg.n_layers):
        raise ValueError(f"Invalid steering layer {layer} for n_layers={int(model.cfg.n_layers)}")

    hook_name = f"blocks.{layer}.hook_resid_pre"
    profile_vectors: Dict[str, torch.Tensor] = {}
    for name in profiles:
        v = _build_profile_vector(
            model=model,
            prepend_bos=prepend_bos,
            layer=layer,
            source=args.source,
            normalize=bool(args.normalize),
            seeds=PROFILE_SEEDS[name],
            prompt_style=args.prompt_style,
            template=config.prompt_template,
        )
        profile_vectors[name] = v
        logger.info(
            "Profile %s vector built: norm=%.6f (source=%s normalize=%s)",
            name,
            float(torch.norm(v).item()),
            args.source,
            bool(args.normalize),
        )

    # Diagnostic: log pairwise cosine similarities
    names_list = list(profile_vectors.keys())
    for i, n1 in enumerate(names_list):
        for n2 in names_list[i + 1:]:
            v1, v2 = profile_vectors[n1], profile_vectors[n2]
            cos = float(torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())
            logger.info("Cosine similarity %s <-> %s: %.4f", n1, n2, cos)

    use_chat = bool(args.chat_template)
    no_think = bool(args.no_think)

    rows: List[Dict[str, object]] = []
    for idx, prompt in enumerate(prompts, start=1):
        wrapped = _make_wrapper(prompt, args.prompt_style)
        gen_prompt = _apply_chat_template(
            wrapped, use_chat_template=use_chat, no_think=no_think,
        )
        base_text = _generate_with_hooks(
            model,
            gen_prompt,
            prepend_bos=prepend_bos,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            fwd_hooks=None,
        )
        per_profile: Dict[str, str] = {}
        for name, vec in profile_vectors.items():
            hooks = _build_residual_hook(
                hook_name=hook_name,
                steering_vec=vec,
                alpha=float(args.alpha),
                apply_mode=args.apply,
            )
            out = _generate_with_hooks(
                model,
                gen_prompt,
                prepend_bos=prepend_bos,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
                no_repeat_ngram_size=int(args.no_repeat_ngram_size),
                fwd_hooks=hooks,
            )
            per_profile[name] = out

        row = {
            "prompt_index": int(idx),
            "prompt": prompt,
            "wrapped_prompt": wrapped,
            "baseline": base_text,
            "steered": per_profile,
        }
        rows.append(row)

        print("")
        print(f"[Prompt {idx}] {prompt}")
        print("  baseline:")
        print(f"    {base_text.strip()}")
        for name in profiles:
            print(f"  steered/{name}:")
            print(f"    {per_profile[name].strip()}")

    output = {
        "experiment": "nnm_exp13_residual_steering_qualitative_tl",
        "model": args.model,
        "config": {
            "dtype": args.dtype,
            "load_in_4bit": args.load_in_4bit,
            "steering_layer": layer,
            "profiles": profiles,
            "alpha": float(args.alpha),
            "source": args.source,
            "normalize": bool(args.normalize),
            "no_think": no_think,
            "chat_template": use_chat,
            "apply": args.apply,
            "max_new_tokens": int(args.max_new_tokens),
            "min_new_tokens": int(args.min_new_tokens),
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
            "prompt_style": args.prompt_style,
        },
        "profile_norms": {
            k: float(torch.norm(v).item()) for k, v in profile_vectors.items()
        },
        "outputs": rows,
    }
    out_path = results_dir / "results.json"
    save_json(output, out_path)
    logger.info("Saved results to %s", out_path)

    md_path = results_dir / "outputs.md"
    lines: List[str] = []
    lines.append("# Exp13 Qualitative Steering Outputs")
    lines.append("")
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- steering_layer: `{layer}`")
    lines.append(f"- profiles: `{', '.join(profiles)}`")
    lines.append(f"- alpha: `{float(args.alpha)}`")
    lines.append(f"- source: `{args.source}`")
    lines.append(f"- normalize: `{bool(args.normalize)}`")
    lines.append(f"- apply: `{args.apply}`")
    lines.append(f"- do_sample: `{bool(args.do_sample)}`")
    lines.append(f"- temperature: `{float(args.temperature)}`")
    lines.append(f"- top_p: `{float(args.top_p)}`")
    lines.append(f"- repetition_penalty: `{float(args.repetition_penalty)}`")
    lines.append(f"- no_repeat_ngram_size: `{int(args.no_repeat_ngram_size)}`")
    lines.append("")
    for row in rows:
        lines.append(f"## Prompt {row['prompt_index']}: {row['prompt']}")
        lines.append("")
        lines.append("### baseline")
        lines.append("")
        lines.append(str(row["baseline"]).strip())
        lines.append("")
        steered = row["steered"]
        assert isinstance(steered, dict)
        for name in profiles:
            lines.append(f"### steered/{name}")
            lines.append("")
            lines.append(str(steered[name]).strip())
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved markdown outputs to %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
