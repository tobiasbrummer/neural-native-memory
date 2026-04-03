#!/usr/bin/env python3
"""Layer × Alpha Sweep: systematically find the sweet spot.

Measures residual stream norms, then sweeps layers and alpha values.
Key insight: alpha must be proportional to residual norms for meaningful effect.

Usage:
    python nnm/experiments/exp13_layer_sweep.py [--profile sleepy]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.io_utils import create_results_dir, save_json, setup_logging
from nnm.kvembed import KVEmbeddingConfig, TransformerLensKVEmbedder

# ── Contrastive Seeds ─────────────────────────────────────────────
PROFILE_SEEDS: Dict[str, Dict[str, List[str]]] = {
    "sleepy": {
        "positive": [
            "Ich bin müde, meine Augen werden schwer und ich möchte schlafen.",
            "Es ist spät, ich bin erschöpft und mein Fokus lässt nach.",
            "Ich fühle mich träge und denke daran, jetzt eine Pause zu machen.",
            "Mir fallen die Augen zu, ich kann mich kaum noch konzentrieren.",
            "Ich gähne ständig und sehne mich nach meinem Bett.",
            "Meine Gedanken werden langsamer, ich bin schläfrig und kraftlos.",
            "Ich bin total kaputt und möchte mich einfach nur hinlegen.",
            "Alles fühlt sich schwer an, ich brauche dringend Schlaf.",
        ],
        "negative": [
            "Ich bin hellwach und voller Energie, bereit loszulegen.",
            "Es ist früh, ich bin ausgeruht und mein Fokus ist scharf.",
            "Ich fühle mich aktiv und denke daran, jetzt weiterzuarbeiten.",
            "Meine Augen sind weit offen, ich kann mich gut konzentrieren.",
            "Ich bin voller Tatendrang und freue mich auf die Aufgabe.",
            "Meine Gedanken sind klar und schnell, ich bin wach und fit.",
            "Ich bin total aufgedreht und möchte am liebsten sofort anfangen.",
            "Alles fühlt sich leicht an, ich bin voller Kraft und Elan.",
        ],
    },
    "joy": {
        "positive": [
            "Ich bin froh, motiviert und sehe die Chancen in dieser Situation.",
            "Ich fühle Freude und Zuversicht, heute läuft es gut.",
            "Ich bin positiv gestimmt und will konstruktiv helfen.",
            "Ich lächle und bin dankbar für diesen schönen Moment.",
            "Mein Herz ist leicht, ich bin glücklich und zufrieden.",
            "Ich spüre eine tiefe Freude und Begeisterung in mir.",
            "Alles fühlt sich wunderbar an, ich genieße den Augenblick.",
            "Ich bin voller Optimismus und sehe überall Möglichkeiten.",
        ],
        "negative": [
            "Ich bin traurig, unmotiviert und sehe die Probleme in dieser Situation.",
            "Ich fühle Frust und Unsicherheit, heute läuft es schlecht.",
            "Ich bin negativ gestimmt und will mich lieber zurückziehen.",
            "Ich runzle die Stirn und bin enttäuscht von diesem Moment.",
            "Mein Herz ist schwer, ich bin unglücklich und unzufrieden.",
            "Ich spüre eine tiefe Traurigkeit und Resignation in mir.",
            "Alles fühlt sich schrecklich an, ich will nur noch weg.",
            "Ich bin voller Pessimismus und sehe überall nur Hindernisse.",
        ],
    },
}

TEST_PROMPT = "Erzaehle eine kurze Geschichte ueber einen Hasen, der heute viel erlebt hat."

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_SPECIAL_RE = re.compile(r"<\|im_(?:start|end)\|>")


def _strip(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _SPECIAL_RE.sub("", text)
    return text.strip()


def _chat(text: str) -> str:
    return f"<|im_start|>system\n/no_think\n<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-10)


# ── Phase 1: Measure residual norms ─────────────────────────────────
@torch.no_grad()
def measure_residual_norms(model: Any, prompt: str, prepend_bos: bool) -> Dict[int, float]:
    """Measure the L2 norm of residual stream at each layer for a prompt."""
    tokens = model.to_tokens(_chat(prompt), prepend_bos=prepend_bos)
    _, cache = model.run_with_cache(
        tokens, return_type=None, prepend_bos=False, remove_batch_dim=False,
    )
    norms = {}
    for l in range(int(model.cfg.n_layers)):
        key = f"blocks.{l}.hook_resid_pre"
        if key in cache:
            resid = cache[key][:, -1, :]  # last token position
            norms[l] = float(torch.norm(resid).item())
    return norms


# ── Phase 2: Build contrastive vectors ────────────────────────────
def _extract_residuals(model, texts, layer, prepend_bos):
    hook_name = f"blocks.{layer}.hook_resid_pre"
    vecs = []
    for t in texts:
        tokens = model.to_tokens(_chat(t), prepend_bos=prepend_bos)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, return_type=None, prepend_bos=False,
                names_filter=lambda n: n == hook_name, remove_batch_dim=False,
            )
        vecs.append(cache[hook_name][:, -1, :].detach().to(torch.float32)[0])
    return vecs


def build_contrastive_vector(model, seeds, layer, prepend_bos):
    pos = _extract_residuals(model, seeds["positive"], layer, prepend_bos)
    neg = _extract_residuals(model, seeds["negative"], layer, prepend_bos)
    diff = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
    raw_norm = float(torch.norm(diff).item())
    unit = _l2(diff.unsqueeze(0))[0]
    return unit, raw_norm


# ── Phase 3: Generate with steering ──────────────────────────────
@torch.no_grad()
def generate_steered(model, prompt, hook_name, vec, alpha, prepend_bos, max_tokens=60):
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    v = vec.view(1, 1, -1)

    def hook_fn(resid, hook):
        add = (v * alpha).to(device=resid.device, dtype=resid.dtype)
        out = resid.clone()
        out[:, :, :] += add  # apply to ALL positions
        return out

    kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, device=model.W_E.device, batch_size=1,
    )
    tokens = model.to_tokens(_chat(prompt), prepend_bos=prepend_bos)
    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    hooks = [(hook_name, hook_fn)] if alpha != 0.0 else []
    generated = []

    with model.hooks(fwd_hooks=hooks):
        logits, _ = model.run_with_cache(
            tokens, return_type="logits", past_kv_cache=kv_cache,
            prepend_bos=False, remove_batch_dim=False,
        )
        for _ in range(max_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            generated.append(next_token)
            if eos_id is not None and int(next_token.item()) == int(eos_id):
                break
            logits, _ = model.run_with_cache(
                next_token, return_type="logits", past_kv_cache=kv_cache,
                prepend_bos=False, remove_batch_dim=False,
            )

    if not generated:
        return ""
    return _strip(model.to_string(torch.cat(generated, dim=1)[0]))


# ── Main ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Layer × Alpha Sweep")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--profile", default="sleepy", choices=list(PROFILE_SEEDS.keys()))
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--layers", type=str, default="15",
                    help="Layers to sweep.")
    p.add_argument("--alphas", type=str, default="5,10,15,20",
                    help="Alpha values to sweep (applied to unit-norm vectors).")
    p.add_argument("--prompt", type=str, default=TEST_PROMPT)
    p.add_argument("--max-tokens", type=int, default=60)
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = create_results_dir("nnm_exp13_sweep")
    logger = setup_logging("exp13_layer_sweep", results_dir)

    layers = [int(x.strip()) for x in args.layers.split(",")]
    alphas = [float(x.strip()) for x in args.alphas.split(",")]
    seeds = PROFILE_SEEDS[args.profile]

    # Load model
    config = KVEmbeddingConfig(
        model_name=args.model, device=None, dtype="float16",
        load_in_4bit=args.load_in_4bit,
    )
    embedder = TransformerLensKVEmbedder(config)
    model = embedder.model
    prepend_bos = embedder._prepend_bos
    n_layers = int(model.cfg.n_layers)
    layers = [l for l in layers if 0 <= l < n_layers]

    print(f"{'='*90}")
    print(f"LAYER × ALPHA SWEEP — {args.model} ({n_layers} layers, d={model.cfg.d_model})")
    print(f"Profile: {args.profile}, Layers: {layers}, Alphas: {alphas}")
    print(f"{'='*90}\n")

    # Phase 1: Measure residual norms
    print("Phase 1: Measuring residual stream norms...")
    norms = measure_residual_norms(model, args.prompt, prepend_bos)
    print(f"  Layer |  Resid Norm")
    print(f"  ------+-----------")
    for l in sorted(norms.keys()):
        marker = " <--" if l in layers else ""
        print(f"  {l:>4}  | {norms[l]:>9.1f}{marker}")
    print()

    # Phase 2: Baseline
    baseline = generate_steered(
        model, args.prompt, "blocks.0.hook_resid_pre",
        torch.zeros(model.cfg.d_model), alpha=0.0,
        prepend_bos=prepend_bos, max_tokens=args.max_tokens,
    )
    print(f"[BASELINE] {baseline}\n")

    # Phase 3: Layer × Alpha sweep
    from difflib import SequenceMatcher
    results = {
        "baseline": baseline,
        "resid_norms": {str(k): v for k, v in norms.items()},
        "sweep": {},
    }

    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_pre"
        vec, raw_norm = build_contrastive_vector(model, seeds, layer, prepend_bos)
        resid_norm = norms.get(layer, 0.0)
        print(f"--- Layer {layer} (resid_norm={resid_norm:.1f}, raw_vec_norm={raw_norm:.1f}) ---")

        layer_results = {"resid_norm": resid_norm, "raw_vec_norm": raw_norm, "alphas": {}}

        for alpha in alphas:
            # Also test negative
            out_pos = generate_steered(
                model, args.prompt, hook_name, vec,
                alpha=alpha, prepend_bos=prepend_bos, max_tokens=args.max_tokens,
            )
            out_neg = generate_steered(
                model, args.prompt, hook_name, vec,
                alpha=-alpha, prepend_bos=prepend_bos, max_tokens=args.max_tokens,
            )

            sim_pos = SequenceMatcher(None, baseline, out_pos).ratio()
            sim_neg = SequenceMatcher(None, baseline, out_neg).ratio()

            # Ratio of alpha to residual norm (how much % we're perturbing)
            pct = (alpha / resid_norm * 100) if resid_norm > 0 else 0

            layer_results["alphas"][str(alpha)] = {
                "positive": out_pos,
                "negative": out_neg,
                "sim_pos": sim_pos,
                "sim_neg": sim_neg,
            }

            print(f"  α={alpha:>5.0f} ({pct:>5.1f}% of resid)"
                  f"  +α sim={sim_pos:.3f}  -α sim={sim_neg:.3f}")
            print(f"    +α: {out_pos[:100]}")
            print(f"    -α: {out_neg[:100]}")

        results["sweep"][str(layer)] = layer_results
        print()

    # Summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY: Similarity to baseline (1.0 = identical, 0.0 = completely different)")
    print(f"{'='*90}")
    header = f"{'Layer':>6} | {'ResNorm':>8}"
    for a in alphas:
        header += f" | α={a:>5.0f}(+/-)"
    print(header)
    print("-" * len(header))
    for layer in layers:
        lr = results["sweep"][str(layer)]
        row = f"{layer:>6} | {lr['resid_norm']:>8.1f}"
        for a in alphas:
            ar = lr["alphas"][str(a)]
            row += f" | {ar['sim_pos']:.2f} / {ar['sim_neg']:.2f}"
        print(row)

    save_json(results, results_dir / "layer_sweep.json")
    print(f"\nResults saved to: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
