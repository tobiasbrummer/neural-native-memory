"""Paper-near KV-Embedding core for Neural Native Memory."""

from .config import KVEmbeddingConfig, PromptRole
from .kv_cache_tl import (
    StoredTLKVCache,
    extract_kv_cache_tl,
    generate_baseline_tl,
    generate_with_stored_kv_tl,
    load_kv_cache_tl,
    save_kv_cache_tl,
)
from .layer_selection import LayerSelectionResult
from .transformerlens_backend import TransformerLensKVEmbedder

__all__ = [
    "KVEmbeddingConfig",
    "LayerSelectionResult",
    "PromptRole",
    "StoredTLKVCache",
    "TransformerLensKVEmbedder",
    "extract_kv_cache_tl",
    "generate_baseline_tl",
    "generate_with_stored_kv_tl",
    "load_kv_cache_tl",
    "save_kv_cache_tl",
]
