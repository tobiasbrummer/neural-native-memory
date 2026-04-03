"""TransformerLens implementation of paper-near KV-Embedding."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from .config import KVEmbeddingConfig, PromptRole
from .layer_selection import (
    LayerSelectionResult,
    estimate_layer_intrinsic_dimensions,
    select_rerouting_layers,
)
from .prompts import build_compression_prompts

logger = logging.getLogger(__name__)


def _import_transformerlens() -> tuple[Any, Any]:
    """
    Import TransformerLens, falling back to local clone at ./TransformerLens.
    """
    try:
        from transformer_lens import HookedTransformer
        from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
        return HookedTransformer, HookedTransformerKeyValueCache
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        local_tl = repo_root / "TransformerLens"
        if local_tl.exists():
            sys.path.insert(0, str(local_tl))
            from transformer_lens import HookedTransformer
            from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
            return HookedTransformer, HookedTransformerKeyValueCache
        raise


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


class TransformerLensKVEmbedder:
    """
    Paper-near KV-Embedding implementation on top of TransformerLens.
    """

    def __init__(self, config: KVEmbeddingConfig):
        self.config = config
        self._prepend_bos = bool(config.prepend_bos)
        self._HookedTransformer, self._KVCache = _import_transformerlens()
        self.model = self._load_model()
        self.layer_selection: Optional[LayerSelectionResult] = None

    def _load_model(self):
        logger.info(
            "Loading TransformerLens model %s (dtype=%s, device=%s, 4bit=%s)",
            self.config.model_name,
            self.config.dtype,
            self.config.device,
            self.config.load_in_4bit,
        )
        common_kwargs = dict(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            device=self.config.device,
            dtype=self.config.dtype,
            local_files_only=self.config.local_files_only,
        )

        # Pre-load HF model in 4-bit if requested
        hf_model = None
        if self.config.load_in_4bit:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            huggingface_token = os.environ.get("HF_TOKEN", "")
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=huggingface_token if huggingface_token else None,
                local_files_only=self.config.local_files_only,
            )
            logger.info("Pre-loaded HF model in 4-bit NF4 quantization.")

        try:
            model = self._HookedTransformer.from_pretrained(
                self.config.model_name,
                hf_model=hf_model,
                default_prepend_bos=self._prepend_bos,
                **common_kwargs,
            )
        except ValueError as exc:
            if "add_bos_token = True but bos_token = None" not in str(exc):
                raise

            logger.warning(
                "Tokenizer does not define a BOS token with add_bos_token=True. "
                "Retrying with add_bos_token=False and prepend_bos=False."
            )
            from transformers import AutoTokenizer

            huggingface_token = os.environ.get("HF_TOKEN", "")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                add_bos_token=False,
                trust_remote_code=True,
                use_fast=True,
                token=huggingface_token if huggingface_token else None,
                local_files_only=self.config.local_files_only,
            )
            model = self._HookedTransformer.from_pretrained(
                self.config.model_name,
                hf_model=hf_model,
                tokenizer=tokenizer,
                default_prepend_bos=False,
                **common_kwargs,
            )
            self._prepend_bos = False
        model.eval()
        return model

    def select_layers(self, corpus_texts: Sequence[str]) -> LayerSelectionResult:
        """
        Select rerouting layers via TwoNN intrinsic dimensionality.
        """
        if not corpus_texts:
            raise ValueError("corpus_texts must not be empty")

        sample_texts = list(corpus_texts[: self.config.id_sample_size])
        logger.info("Estimating intrinsic dimensionality on %d texts...", len(sample_texts))

        id_by_layer = estimate_layer_intrinsic_dimensions(
            self.model,
            sample_texts,
            prepend_bos=self._prepend_bos,
        )

        result = select_rerouting_layers(
            id_by_layer=id_by_layer,
            n_layers=int(self.model.cfg.n_layers),
            layer_window_fraction=self.config.layer_window_fraction,
            exclude_early_fraction=self.config.exclude_early_fraction,
            detect_u_shape=self.config.detect_u_shape,
        )
        self.layer_selection = result

        logger.info(
            "Selected rerouting layers: %s (u-shape mode=%s)",
            result.selected_layers,
            result.used_u_shape_mode,
        )
        return result

    def extract_embeddings(
        self,
        texts: Sequence[str],
        roles: Optional[Sequence[PromptRole]] = None,
        rerouting_layers: Optional[Sequence[int]] = None,
        id_corpus_texts: Optional[Sequence[str]] = None,
        return_token_embeddings: bool = True,
        return_token_deltas: bool = True,
        return_layer_deltas: bool = False,
    ) -> Dict[str, object]:
        """
        Run paper-near KV-Embedding extraction with internal K/V rerouting.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        if roles is None:
            roles = ["context"] * len(texts)
        if len(roles) != len(texts):
            raise ValueError("roles must have the same length as texts")

        if rerouting_layers is None:
            if self.layer_selection is None:
                if id_corpus_texts:
                    self.select_layers(id_corpus_texts)
                else:
                    logger.warning(
                        "No explicit rerouting layers/corpus provided. Falling back to input texts for ID."
                    )
                    self.select_layers(texts)
            assert self.layer_selection is not None
            active_layers = list(self.layer_selection.selected_layers)
        else:
            active_layers = sorted(int(x) for x in rerouting_layers)

        prompts = build_compression_prompts(
            texts=texts,
            roles=roles,
            template=self.config.prompt_template,
        )

        pooled_outputs: List[np.ndarray] = []
        token_outputs: List[np.ndarray] = []
        token_ids_outputs: List[np.ndarray] = []
        static_outputs: List[np.ndarray] = []
        delta_outputs: List[np.ndarray] = []
        layer_delta_outputs: List[Dict[int, np.ndarray]] = []

        final_layer = int(self.model.cfg.n_layers - 1)
        final_hidden_name = f"blocks.{final_layer}.hook_resid_post"

        for prompt in prompts:
            token_ids_t = self.model.to_tokens(prompt, prepend_bos=self._prepend_bos)
            token_ids = token_ids_t.squeeze(0).detach().cpu().numpy().astype(np.int64)

            pass1_cache = self._capture_pass1_kv(
                prompt,
                active_layers,
                include_resid_pre=return_layer_deltas,
            )
            prefix_cache = self._build_prefix_cache(pass1_cache, active_layers)

            attn_hooks = self._build_attention_bias_hooks(active_layers)

            with torch.no_grad():
                with self.model.hooks(fwd_hooks=attn_hooks):
                    _, pass2_cache = self.model.run_with_cache(
                        prompt,
                        return_type=None,
                        prepend_bos=self._prepend_bos,
                        names_filter=[final_hidden_name],
                        remove_batch_dim=False,
                        past_kv_cache=prefix_cache,
                    )

            contextual = pass2_cache[final_hidden_name][0].detach().to(torch.float32).cpu().numpy()
            if self.config.normalize_token_embeddings:
                contextual = _l2_normalize(contextual, axis=1)

            pooled = (contextual[-1] + contextual.mean(axis=0)) / 2.0
            if self.config.normalize_pooled_embedding:
                pooled = _l2_normalize(pooled, axis=0)

            static = self._get_static_embeddings(token_ids_t)
            delta = contextual - static

            pooled_outputs.append(pooled.astype(np.float32))
            token_ids_outputs.append(token_ids)
            static_outputs.append(static.astype(np.float32))

            if return_token_embeddings:
                token_outputs.append(contextual.astype(np.float32))
            if return_token_deltas:
                delta_outputs.append(delta.astype(np.float32))
            if return_layer_deltas:
                last_token_id = int(token_ids[-1])
                token_embed = (
                    self.model.W_E[last_token_id]
                    .detach()
                    .to(torch.float32)
                    .view(1, 1, -1)
                )
                layer_deltas: Dict[int, np.ndarray] = {}
                for layer in active_layers:
                    resid_pre_name = f"blocks.{layer}.hook_resid_pre"
                    if resid_pre_name not in pass1_cache:
                        raise RuntimeError(
                            f"Missing resid_pre cache for layer {layer}. "
                            "Set return_layer_deltas=True to capture it."
                        )
                    resid_pre = (
                        pass1_cache[resid_pre_name][:, -1:, :]
                        .detach()
                        .to(torch.float32)
                    )
                    delta_layer = resid_pre - token_embed
                    layer_deltas[layer] = delta_layer.squeeze(0).squeeze(0).cpu().numpy()
                layer_delta_outputs.append(layer_deltas)

            del pass1_cache
            del pass2_cache

        result: Dict[str, object] = {
            "model_name": self.config.model_name,
            "selected_layers": active_layers,
            "pooled_embeddings": np.stack(pooled_outputs, axis=0),
            "token_ids": token_ids_outputs,
            "static_embeddings": static_outputs,
            "token_embeddings": token_outputs if return_token_embeddings else None,
            "token_deltas": delta_outputs if return_token_deltas else None,
            "layer_deltas": layer_delta_outputs if return_layer_deltas else None,
        }
        if self.layer_selection is not None:
            result["layer_selection"] = self.layer_selection
        return result

    def _capture_pass1_kv(
        self,
        prompt: str,
        selected_layers: Sequence[int],
        include_resid_pre: bool = False,
    ) -> Mapping[str, torch.Tensor]:
        names: Set[str] = set()
        for layer in selected_layers:
            names.add(self._key_hook_name(layer))
            names.add(f"blocks.{layer}.attn.hook_v")
            if include_resid_pre:
                names.add(f"blocks.{layer}.hook_resid_pre")

        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                prompt,
                return_type=None,
                prepend_bos=self._prepend_bos,
                names_filter=lambda name: name in names,
                remove_batch_dim=False,
            )
        return cache

    def _compute_kv_from_resid_pre(
        self, layer: int, resid_pre: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from transformer_lens.utils import repeat_along_head_dimension

        block = self.model.blocks[layer]
        cfg = self.model.cfg

        if cfg.use_split_qkv_input:
            n_kv_heads = (
                cfg.n_key_value_heads
                if cfg.n_key_value_heads is not None and not cfg.ungroup_grouped_query_attention
                else cfg.n_heads
            )
            query_input = repeat_along_head_dimension(resid_pre, n_heads=cfg.n_heads)
            key_input = repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
            value_input = repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
        elif cfg.use_attn_in:
            attn_in = repeat_along_head_dimension(resid_pre, n_heads=cfg.n_heads)
            query_input = attn_in
            key_input = attn_in
            value_input = attn_in
        else:
            query_input = resid_pre
            key_input = resid_pre
            value_input = resid_pre

        q_in = block.ln1(query_input)
        k_in = block.ln1(key_input)
        v_in = block.ln1(value_input)

        _, k, v = block.attn.calculate_qkv_matrices(q_in, k_in, v_in)
        return k, v

    def reconstruct_kv_from_layer_deltas(
        self,
        layer_deltas: Mapping[int, np.ndarray],
        token_id: int,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        token_embed = (
            self.model.W_E[int(token_id)]
            .detach()
            .to(self.model.W_E.device)
            .to(torch.float32)
            .view(1, 1, -1)
        )
        kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, delta in layer_deltas.items():
            delta_t = (
                torch.from_numpy(np.asarray(delta))
                .to(self.model.W_E.device)
                .to(torch.float32)
                .view(1, 1, -1)
            )
            resid_pre = token_embed + delta_t
            k, v = self._compute_kv_from_resid_pre(int(layer), resid_pre)
            kv_by_layer[int(layer)] = (k, v)
        return kv_by_layer

    def build_prefix_cache_from_kv(
        self,
        kv_by_layer: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
        selected_layers: Sequence[int],
    ):
        cache = self._KVCache.init_cache(
            self.model.cfg,
            device=self.model.cfg.device,
            batch_size=1,
        )
        selected = set(int(x) for x in selected_layers)

        n_layers = int(self.model.cfg.n_layers)
        n_kv_heads = (
            int(self.model.cfg.n_key_value_heads)
            if self.model.cfg.n_key_value_heads is not None
            else int(self.model.cfg.n_heads)
        )
        d_head = int(self.model.cfg.d_head)
        dtype = self.model.W_E.dtype
        device = self.model.W_E.device

        for layer in range(n_layers):
            if layer in selected:
                k, v = kv_by_layer[int(layer)]
                k_prefix = k.to(device=device, dtype=dtype)
                v_prefix = v.to(device=device, dtype=dtype)
            else:
                k_prefix = torch.zeros((1, 1, n_kv_heads, d_head), dtype=dtype, device=device)
                v_prefix = torch.zeros((1, 1, n_kv_heads, d_head), dtype=dtype, device=device)

            cache.entries[layer].past_keys = k_prefix
            cache.entries[layer].past_values = v_prefix

        cache.previous_attention_mask = torch.ones((1, 1), dtype=torch.int, device=device)
        cache.freeze()
        return cache

    def build_prefix_cache_from_layer_deltas(
        self,
        layer_deltas: Mapping[int, np.ndarray],
        token_id: int,
        selected_layers: Optional[Sequence[int]] = None,
    ):
        if selected_layers is None:
            if self.layer_selection is None:
                raise ValueError("selected_layers not provided and no layer_selection available.")
            selected_layers = self.layer_selection.selected_layers
        kv_by_layer = self.reconstruct_kv_from_layer_deltas(layer_deltas, token_id)
        return self.build_prefix_cache_from_kv(kv_by_layer, selected_layers)

    def _build_prefix_cache(self, pass1_cache: Mapping[str, torch.Tensor], selected_layers: Sequence[int]):
        cache = self._KVCache.init_cache(
            self.model.cfg,
            device=self.model.cfg.device,
            batch_size=1,
        )
        selected = set(int(x) for x in selected_layers)

        n_layers = int(self.model.cfg.n_layers)
        n_kv_heads = (
            int(self.model.cfg.n_key_value_heads)
            if self.model.cfg.n_key_value_heads is not None
            else int(self.model.cfg.n_heads)
        )
        d_head = int(self.model.cfg.d_head)
        dtype = self.model.W_E.dtype
        device = self.model.W_E.device

        for layer in range(n_layers):
            if layer in selected:
                k_name = self._key_hook_name(layer)
                v_name = f"blocks.{layer}.attn.hook_v"
                if k_name not in pass1_cache or v_name not in pass1_cache:
                    raise RuntimeError(f"Missing pass1 cache entries for layer {layer}")
                k_prefix = pass1_cache[k_name][:, -1:, :, :].detach().to(device=device, dtype=dtype)
                v_prefix = pass1_cache[v_name][:, -1:, :, :].detach().to(device=device, dtype=dtype)
            else:
                k_prefix = torch.zeros((1, 1, n_kv_heads, d_head), dtype=dtype, device=device)
                v_prefix = torch.zeros((1, 1, n_kv_heads, d_head), dtype=dtype, device=device)

            cache.entries[layer].past_keys = k_prefix
            cache.entries[layer].past_values = v_prefix

        # Prefix attention mask (virtual position 0)
        cache.previous_attention_mask = torch.ones((1, 1), dtype=torch.int, device=device)
        cache.freeze()
        return cache

    def _build_attention_bias_hooks(self, selected_layers: Sequence[int]):
        selected = set(int(x) for x in selected_layers)
        hooks = []

        for layer in range(int(self.model.cfg.n_layers)):
            name = f"blocks.{layer}.attn.hook_attn_scores"
            if layer in selected:
                bias_value = float(self.config.prefix_bias)
            else:
                bias_value = float(self.config.non_rerouted_prefix_bias)

            def _hook(scores: torch.Tensor, hook: Any, b: float = bias_value) -> torch.Tensor:
                updated = scores.clone()
                updated[..., 0] = updated[..., 0] + b
                return updated

            hooks.append((name, _hook))
        return hooks

    def _get_static_embeddings(self, token_ids_t: torch.Tensor) -> np.ndarray:
        token_ids = token_ids_t.to(self.model.W_E.device)
        static = self.model.W_E[token_ids[0]]
        return static.detach().to(torch.float32).cpu().numpy()

    def _key_hook_name(self, layer: int) -> str:
        """
        Select key hook depending on positional embedding type.

        For rotary models, paper-near behavior uses post-RoPE keys (`hook_rot_k`).
        For non-rotary models (e.g. GPT-2), use standard key projection (`hook_k`).
        """
        if getattr(self.model.cfg, "positional_embedding_type", None) == "rotary":
            return f"blocks.{layer}.attn.hook_rot_k"
        return f"blocks.{layer}.attn.hook_k"
