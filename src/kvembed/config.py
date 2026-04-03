"""Configuration objects for KV-Embedding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptRole = Literal["context", "query"]


@dataclass(frozen=True)
class KVEmbeddingConfig:
    """Runtime configuration for the paper-near KV-Embedding pipeline."""

    model_name: str = "Qwen/Qwen2-1.5B"
    device: str | None = None
    dtype: str = "float16"
    local_files_only: bool = False
    load_in_4bit: bool = False

    prepend_bos: bool = True

    # Paper Section 3.2.2 / Appendix G
    prefix_bias: float = 1.0
    # Non-rerouted layers get a strong negative bias on prefix position (effectively masked out)
    non_rerouted_prefix_bias: float = -1e4

    # Paper Section 3.2.3
    id_sample_size: int = 1000
    layer_window_fraction: float = 0.10
    exclude_early_fraction: float = 0.20
    detect_u_shape: bool = True

    # Paper Section 3.2.1
    prompt_template: str = '"{role}: {text}" Compress the {role} in one word:'

    # Embedding post-processing
    normalize_token_embeddings: bool = False
    normalize_pooled_embedding: bool = True
